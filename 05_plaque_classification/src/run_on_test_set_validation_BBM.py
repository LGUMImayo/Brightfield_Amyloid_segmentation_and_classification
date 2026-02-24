from tangle_ai_bbox_utils import get_model_instance_segmentation, TangleAIDataset
import pandas as pd
import numpy as np
import ast
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import io

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import Dataset, DataLoader

import tifffile
import imagecodecs

import math
from tqdm import tqdm
import torch
import numpy as np
from skimage import measure

from torchvision.ops import nms

import os

def create_weight_matrix(size):
    matrix=np.zeros((size,size))
    max_layer=size//2
    for layer in range(max_layer+1):
        value=(layer+1)/max_layer
        matrix[layer:size-layer,layer:size-layer]=value
    return matrix

def calculate_mask(sub_image,model,bbox,transform,weight_matrix,step=2,n_channels=3,image_size=520):
    print(image_size)
    n_x=math.floor((bbox[3]-bbox[1])/image_size*step)
    n_y=math.floor((bbox[2]-bbox[0])/image_size*step)
    final_mask=np.zeros([n_channels,sub_image.shape[0],sub_image.shape[1]])
    count_mask=np.zeros([n_channels,sub_image.shape[0],sub_image.shape[1]])
    for i in tqdm(range(int(n_x/2))):
        for j in range(int(n_y/2)):
            patch=sub_image[int(i*image_size/step):int((i+step)*image_size/step),int(j*image_size/step):int((j+step)*image_size/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0))
            sm=torch.nn.functional.sigmoid(predictions['out'])
            pred_mask=sm[0].detach().numpy()
            #pred_mask=rotation_pred(patch,model,transform)
            final_mask[:,int(i*image_size/step):int((i+step)*image_size/step),int(j*image_size/step):int((j+step)*image_size/step)]+=pred_mask*weight_matrix
            count_mask[:,int(i*image_size/step):int((i+step)*image_size/step),int(j*image_size/step):int((j+step)*image_size/step)]+=1*weight_matrix
            
    for i in tqdm(range(int(n_x/2))):
        for j in range(int(n_y/2)):
            if (i==0)&(j==0):
                patch=sub_image[-image_size:,-image_size:]
            elif (i==0)&(j!=0):
                patch=sub_image[-image_size:,-int(image_size*(j+step)/step):-int(image_size*j/step)]
            elif (i!=0)&(j==0):
                patch=sub_image[-int(image_size*(i+step)/step):-int(image_size*i/step),-image_size:]
            else:
                patch=sub_image[-int(image_size*(i+step)/step):-int(image_size*i/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0))
            sm=torch.nn.functional.sigmoid(predictions['out'])
            pred_mask=sm[0].detach().numpy()
            #pred_mask=rotation_pred(patch,model,transform)
            if (i==0)&(j==0):
                final_mask[:,-image_size:,-image_size:]+=pred_mask*weight_matrix
                count_mask[:,-image_size:,-image_size:]+=1*weight_matrix
            elif (i==0)&(j!=0):
                final_mask[:,-image_size:,-int(image_size*(j+step)/step):-int(image_size*j/step)]+=pred_mask*weight_matrix
                count_mask[:,-image_size:,-int(image_size*(j+step)/step):-int(image_size*j/step)]+=1*weight_matrix
            elif (i!=0)&(j==0):
                final_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-image_size:]+=pred_mask*weight_matrix
                count_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-image_size:]+=1*weight_matrix
            else:
                final_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=pred_mask*weight_matrix
                count_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=1*weight_matrix

    for i in tqdm(range(int(n_x/2))):
        for j in range(int(n_y/2)):
            if (i==0):
                patch=sub_image[-image_size:,int(j*image_size/step):int((j+step)*image_size/step)]
            else:
                patch=sub_image[-int(image_size*(i+step)/step):-int(image_size*i/step),int(j*image_size/step):int((j+step)*image_size/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0))
            sm=torch.nn.functional.sigmoid(predictions['out'])
            pred_mask=sm[0].detach().numpy()
            #pred_mask=rotation_pred(patch,model,transform)
            if (i==0):
                final_mask[:,-image_size:,int(j*image_size/step):int((j+step)*image_size/step)]+=pred_mask*weight_matrix
                count_mask[:,-image_size:,int(j*image_size/step):int((j+step)*image_size/step)]+=1*weight_matrix
            else:
                final_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),int(j*image_size/step):int((j+step)*image_size/step)]+=pred_mask*weight_matrix
                count_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),int(j*image_size/step):int((j+step)*image_size/step)]+=1*weight_matrix

    for i in tqdm(range(int(n_x/2))):
        for j in range(int(n_y/2)):
            if (j==0):
                patch=sub_image[int(i*image_size/step):int((i+step)*image_size/step),-image_size:]
            else:
                patch=sub_image[int(i*image_size/step):int((i+step)*image_size/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0))
            sm=torch.nn.functional.sigmoid(predictions['out'])
            pred_mask=sm[0].detach().numpy()
            #pred_mask=rotation_pred(patch,model,transform)
            if (j==0):
                final_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-image_size:]+=pred_mask*weight_matrix
                count_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-image_size:]+=1*weight_matrix
            else:
                final_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=pred_mask*weight_matrix
                count_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=1*weight_matrix
    final_mask=final_mask/count_mask

    return final_mask

class PlaqueAIDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df,target_dict, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
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

        """
        if self.transform:
            img, target = self.transform(img,target)
        """
        if self.transform is not None:
            transformed = self.transform(
                image=np.array(img),
                bboxes=target['boxes'],
                class_labels=labels
            )
            img = transformed["image"]
            target['boxes'] = torch.tensor(np.array(transformed['bboxes']))
            target['labels'] = torch.tensor(np.array(transformed['class_labels']))
        if 'none' in self.df['labels'][idx]:
            target['labels'] = torch.as_tensor([], dtype=torch.int64)
            target['boxes'] = torch.as_tensor(np.empty((0, 4)))

        return img, target

    
target_dict = {
    'cgp': 1,
    'ccp': 2,
    'caa 1': 3,
    'caa 2': 4
}

pretrained_path='../models/torchvision_fasterrcnn_resnet50_fpn.pt'

class_dict = {v: k for k, v in target_dict.items()}

model=get_model_instance_segmentation(len(target_dict)+1,pretrained_path)

df=pd.read_csv('../data/6f3d_validation.csv')

val_transform = A.Compose([
    A.Resize(1024, 1024, always_apply=True), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(transpose_mask=True)
])

#'../models/tangle_ai_5_28_1024_rrc_rotate_rotate_full_2.pt'
import torch
model.load_state_dict(
    torch.load(
        '../models/plaque_ai_full_final.pt',
        map_location=torch.device('cpu')
    ),
    strict=False
)

model.eval()

print('model loaded')

# --- CHANGE 1: Update the path to the gray/white matter model if needed ---
# Ensure this path is correct on your system, or update it if the model is elsewhere.
gray_white_path='../../../murray-structural-segmentation-dev/models/nmarkers/model_level_2_2_pseudo_label_bce'

import torchvision
gray_model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
gray_model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
gray_model=gray_model.to('cpu')

gray_model.load_state_dict(
    torch.load(
        gray_white_path,
        map_location='cpu'
    )
)

gray_model.eval()
print('model ready')


image_size=1024

weight_mat=create_weight_matrix(image_size)


transform = A.Compose([
    A.Resize(image_size, image_size, always_apply=True), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(transpose_mask=True)
])

# --- CHANGE 2: Define your source directory ---
SOURCE_TIFF_DIR = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/tiff'
OUTPUT_DIR = './validation_predictions/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

processed_files = [i.split('.')[0] for i in os.listdir(OUTPUT_DIR)]

# --- CHANGE 3: Get list of TIFF files from the directory instead of a CSV ---
import glob
tiff_files = glob.glob(os.path.join(SOURCE_TIFF_DIR, '*.tif')) + \
             glob.glob(os.path.join(SOURCE_TIFF_DIR, '*.tiff')) + \
             glob.glob(os.path.join(SOURCE_TIFF_DIR, '*.svs')) # Added .svs just in case

print(f"Found {len(tiff_files)} images in {SOURCE_TIFF_DIR}")

# Loop through the found files
for full_file_path in tqdm(tiff_files):
    
    # Extract filename and ID
    filename = os.path.basename(full_file_path)
    image_id = filename.split('.')[0]

    if image_id not in processed_files:
        try:
            image_widths=[]
            # Use the full path found by glob
            with tifffile.TiffFile(full_file_path) as tif:
                print(len(tif.pages))
                for page in tif.pages:
                    tags={tag.name: tag.value for tag in page.tags.values()}
                    image_widths.append(tags['ImageWidth'])

            with tifffile.TiffFile(full_file_path) as tif:
                for page in tif.pages:
                    tags={tag.name: tag.value for tag in page.tags.values()}
                    # Logic to find the low-res image (approx 1/16th size)
                    if tags['ImageWidth']==int(max(image_widths)/16):
                        print('!')
                        image=page.asarray()

            gray_image=image

            final_mask=calculate_mask(
                gray_image,
                gray_model,
                [0,0,gray_image.shape[1],gray_image.shape[0]],
                transform,
                weight_matrix=weight_mat,
                step=2,
                n_channels=2,
                image_size=image_size
            )

            gray_mask=final_mask[1]*final_mask[0]>.5

            with tifffile.TiffFile(full_file_path) as tif:
                for page in tif.pages:
                    tags={tag.name: tag.value for tag in page.tags.values()}
                    if tags['ImageWidth']==max(image_widths):
                        print('!')
                        full_image=page.asarray()

            image_size=1024
            overlap=512
            stride=image_size-overlap
            n_x=int(full_image.shape[0]/(stride))-1
            n_y=int(full_image.shape[1]/(stride))-1
            gray_stride=int(stride/16)
            gray_image_size=int(image_size/16)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model.to(device)

            output_df=pd.DataFrame({})
            for i in tqdm(range(n_x)):
                for j in range(n_y):
                    # Check if the corresponding low-res area has tissue (gray mask)
                    if gray_mask[i*gray_stride:(i*gray_stride+gray_image_size),j*gray_stride:(j*gray_stride+gray_image_size)].sum()>0:
                        tile=full_image[i*stride:(i*stride+image_size),j*stride:(j*stride+image_size)]
                        output=model(val_transform(image=tile)['image'].unsqueeze(0).to(device))
                        pred_idx=list(nms(output[0]['boxes'],output[0]['scores'],.5).detach().cpu().numpy())
                        output=output[0]
                        output['labels']=output['labels'][pred_idx]
                        output['scores']=output['scores'][pred_idx]
                        output['boxes']=output['boxes'][pred_idx]
                        for w in range(len(output['labels'])):
                            k=len(output_df)+1
                            output_df.loc[k,'label']=output['labels'].detach().cpu().numpy()[w]
                            output_df.loc[k,'score']=output['scores'].detach().cpu().numpy()[w]
                            output_df.loc[k,'x_0']=output['boxes'].detach().cpu().numpy()[w,0]
                            output_df.loc[k,'y_0']=output['boxes'].detach().cpu().numpy()[w,1]
                            output_df.loc[k,'x_1']=output['boxes'].detach().cpu().numpy()[w,2]
                            output_df.loc[k,'y_1']=output['boxes'].detach().cpu().numpy()[w,3]
                            output_df.loc[k,'x_start']=i*stride
                            output_df.loc[k,'y_start']=j*stride


            output_df['gray_area']=np.sum(gray_mask)
            output_df.to_csv(f'{OUTPUT_DIR}/{image_id}.csv',index=False)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")