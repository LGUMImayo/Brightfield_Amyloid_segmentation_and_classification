from tangle_ai_bbox_utils import get_model_instance_segmentation_mobile,get_model_instance_segmentation, TangleAIDataset
import pandas as pd
import numpy as np
import ast
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2

from torchvision.ops import nms
from sklearn.metrics import average_precision_score


class_dict = {
    1:'cgp',
    2:'ccp',
    3:'caa 1',
    4:'caa 2'
}


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

def find_max_iou_bboxes(ground_truths, predictions):
    max_ious = []
    
    for gt in ground_truths:
        max_iou = 0
        max_iou_pred = None
        
        for pred in predictions:
            iou = calculate_iou(gt, pred)
            if iou > max_iou:
                max_iou = iou
                max_iou_pred = pred
        
        if max_iou > 0.5:
            max_ious.append((gt, max_iou_pred, max_iou))
        else:
            max_ious.append((gt, None, 0))
    
    return max_ious



scaler={
    'loss_classifier': torch.tensor(1.),
    'loss_box_reg': torch.tensor(1.),
    'loss_objectness': torch.tensor(1.),
    'loss_rpn_box_reg':torch.tensor(1.)
}

remove_names=[
    '921359_(38679,20021)',
    '1393007_(16041,20612)',
    '921359_(1259,16269)',
    '921359_(38679,20021)',
    '1192759_(17921,34811)',
    '1303685_(34348,4815)',
    '1273113_(34197,18875)',
    '1303685_(34463,4352)',
    '1391974_(32031,3657)',
    '1440407_(13617,3773)',
    '1469937_(26028,4217)',
    '1192761_(33112,17755)',
    '866782_(12806,15412)',
    '1391983_(32321,36177)',
    '30598_(12288,30720)',
    '866782_(12806,15412)',
    '888220_(11264,3072)',
    '888220_(31744,13312)',
    '888385_(46080,19456)',
    '910064_(22534,1920)',
    '1190787_(36864,4096)',
    '1271370_(40439,-312)',
    '1405366_(20480,31744)',
    '1440407_(12806,16860)',
    '1462050_(38277,7764)',

]
"""
    '1190708_(34425,9235)',
    '1190719_(30720,30720)',
    '1190733_(5880,28395)',
    '1190786_(25160,5649)',
    '1248351_(24576,8192)',
    '1391975_(20480,23552)',
    '1391986_(5120,34816)',
    
    '1391986_(5120,34816)',
    '1236346_(23110,16952)',
    '1393013_(27648,6144)',
    
    '913295_(13312,30720)',
    '1273116_(9586,4120)',
    '1278597_(49152,28672)',
    '1391975_(13312,4096)',
    '1458298_(10240,10240)'
]
"""

#add_in=['961385_(14343,19662)','1391982_(35374,29429)']

class PlaqueAIDataset(Dataset):
    def __init__(self, df,target_dict, transform=None):
        self.df = df
        self.transform = transform
        self.prefix = prefix
        self.target_dict = target_dict

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        blob_name = f"{self.df.loc[idx, 'image_name']}"
        img = Image.open(blob_name).convert('RGB')

        image_id = torch.as_tensor(idx)
        if 'none' in self.df['labels'][idx]:
            labels = torch.as_tensor([])
            bboxes = torch.as_tensor(np.empty((0, 4)))
        if 'no label' in self.df['labels'][idx]:
            labels = torch.as_tensor([])
            bboxes = torch.as_tensor(np.empty((0, 4)))
        
        else:
            bboxes = torch.as_tensor(np.array(self.df.bboxes[idx], dtype=float))
            labels = torch.as_tensor([self.target_dict[l] for l in self.df.labels[idx]])

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            transformed = self.transform(
                image=np.array(img),
                bboxes=target['boxes'],
                class_labels=labels
            )
            img = transformed["image"]
            target['boxes'] = torch.tensor(np.array(transformed['bboxes']))
            target['labels'] = torch.tensor(transformed['class_labels'])
        if 'none' in self.df['labels'][idx]:
            target['labels'] = torch.as_tensor([], dtype=torch.int64)
            target['boxes'] = torch.as_tensor(np.empty((0, 4)))
        if target['boxes'].shape[0]==0:
            target['labels'] = torch.as_tensor([], dtype=torch.int64)
            target['boxes'] = torch.as_tensor(np.empty((0, 4)))

        return img, target

    
    
target_dict = {
    'cgp': 1,
    'ccp': 2,
    'caa 1': 3,
    'caa 2': 4
}

prefix='../../data/train_'

pretrained_path='../models/torchvision_fasterrcnn_resnet50_fpn.pt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model=get_model_instance_segmentation(len(target_dict)+1,pretrained_path)

from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection import FasterRCNN


prefix='../../data/train_'

train_df=pd.read_csv('./train_df.csv')
test_df=pd.read_csv('./test_df.csv')

train_df=train_df[~train_df.labels.isna()].reset_index(drop=True)
test_df=test_df[~test_df.labels.isna()].reset_index(drop=True)

train_df['pid']=[i.split('_')[0] for i in train_df.image_name]
test_df['pid']=[i.split('_')[0] for i in test_df.image_name]

train_df['labels']=[ast.literal_eval(i) for i in train_df.labels]
train_df['bboxes']=[ast.literal_eval(i) for i in train_df.bboxes]

test_df['labels']=[ast.literal_eval(i) for i in test_df.labels]
test_df['bboxes']=[ast.literal_eval(i) for i in test_df.bboxes]


train_df=train_df[[i not in remove_names for i in train_df.image_name ]].reset_index(drop=True)
test_df=test_df[[i not in remove_names for i in test_df.image_name ]].reset_index(drop=True)


train_df['image_name']=[f'{prefix}{j}/{i}.jpg' for i,j in zip(train_df.image_name,train_df.set)]
test_df['image_name']=[f'{prefix}{j}/{i}.jpg' for i,j in zip(test_df.image_name,test_df.set)]


train_transform = A.Compose([
    A.Resize(1024, 1024, always_apply=True),
    A.RandomResizedCrop(1024,1024,scale=(.9,1.1),ratio=(.9,1.1)),
    A.VerticalFlip(p=0.5),  
    A.HorizontalFlip(p=0.5), 
    A.RandomRotate90(p=0.5), 
    A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True)
],
    bbox_params=A.BboxParams(format='pascal_voc',min_visibility=.5,label_fields=['class_labels'])
)

val_transform = A.Compose([
    A.Resize(1024,1024, always_apply=True), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(transpose_mask=True)
],
    bbox_params=A.BboxParams(format='pascal_voc',min_visibility=.5,label_fields=['class_labels'])
)


train_dataset=PlaqueAIDataset(train_df,target_dict,transform=train_transform)
test_dataset=PlaqueAIDataset(test_df,target_dict,transform=val_transform)

import utils

data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)


model.to(device)
print('model ready')


from engine import train_one_epoch, evaluate
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=2e-5)
#optimizer = torch.optim.SGD(params, lr=2e-4, momentum=.9)


scheduler=None

num_epochs=100

best_loss=100

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=2e-5, 
    steps_per_epoch=len(data_loader), epochs=num_epochs)

best_loss=0
for epoch in range(num_epochs):
    model.train()
    model,scheduler=train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler=scheduler,print_freq=10,scaler=scaler)

    torch.save(model.state_dict(), f'../models/plaque_ai_final.pt')
        
    model.eval()
    
    coco_eval = evaluate(model,data_loader_test,device=device)
    eval_loss = coco_eval.coco_eval['bbox'].stats[1]
    print(eval_loss)
    if eval_loss > best_loss:
        best_loss = eval_loss
        torch.save(model.state_dict(), f'../models/plaque_ai_es.pt')
    print(eval_loss)
    
