import pandas as pd
import numpy as np
import ast
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm

import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm

import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

import os

MAIN_DIR='../data/images_and_masks/'
IMG_DIR=f'{MAIN_DIR}images'
MASK_DIR=f'{MAIN_DIR}masks'

target_dict={
    'background':0,
    'gray-matter':1,
    'white-matter':2
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_fn=torch.nn.CrossEntropyLoss()

EPOCHS = 100

train_transform = A.Compose([
    A.PadIfNeeded(min_height=520, min_width=520, p=1),
    A.RandomCrop(520, 520, always_apply=True), 
    A.VerticalFlip(p=0.5),  
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ChannelShuffle(p=1),
    A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True)
])

val_transform = A.Compose([
    A.CenterCrop(520,520), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(transpose_mask=True)
])

class HippAI_aperio_Dataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, target_dict, transform=None):
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_dict = target_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name=self.df.image_name.iloc[idx]
        blob_name=f"{self.img_dir}/{img_name}"
        img = Image.open(blob_name)
        img = np.array(img.convert('RGB'))
        slide_name=img_name.split('.')[0]
        self.n_targets = int(len(self.target_dict))
        
        blob_name=f"{self.mask_dir}/{img_name}"
        mask = [np.array(Image.open(blob_name))]

        labels=torch.as_tensor([i+1 for i in range(self.n_targets)], dtype=torch.int64)

        if self.transform is not None:
            transformed = self.transform(
                image=np.array(img), 
                masks=mask,
                class_labels=labels
            )
            img = transformed["image"]
            
            masks=np.array(transformed['masks'])
            masks=torch.tensor(masks,dtype=torch.float32)

        return img, masks

for fold in range(0,5):
    df = pd.read_csv(f'../data/train_gw_{fold}.csv')
    
    train_dataset=HippAI_aperio_Dataset(df,IMG_DIR,MASK_DIR,target_dict,transform=train_transform)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=28, shuffle=True, num_workers=4,)
    
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=3e-2, )

    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs['out'],labels.squeeze(1).long().to(device))

            loss.backward()

            optimizer.step()

        return loss.item()

    epoch_number=0
    for epoch in tqdm(range(EPOCHS)):
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)

        model_path = f'../models/model_level_2_ce_{fold}'
        torch.save(model.state_dict(), model_path)
        epoch_number += 1
