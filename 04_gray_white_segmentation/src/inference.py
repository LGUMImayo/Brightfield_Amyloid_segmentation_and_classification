import torch
from tqdm import tqdm

import uuid

import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

import tifffile
import imagecodecs

import os

import torch.nn.functional as F

from skimage import measure

import matplotlib.pyplot as plt

from tqdm import tqdm

from datetime import date

import argparse

import json

import numpy as np
from scipy.ndimage import label, find_objects

from inference_utils import calculate_mask

bbm_home_mount_path = '/home/auto/s017156/tmp/img_for_jonathan/'
#where the models are stored
MODEL_DIR = '/home/auto/s017156/tmp/Matt_codes/s311590_gray_white/gray_white_segmentation/models'

config={
    'model_0_path':f'{MODEL_DIR}/model_level_2_ce_0',
    'model_1_path':f'{MODEL_DIR}/model_level_2_ce_1',
    'model_2_path':f'{MODEL_DIR}/model_level_2_ce_2',
    'model_3_path':f'{MODEL_DIR}/model_level_2_ce_3',
    'model_4_path':f'{MODEL_DIR}/model_level_2_ce_4',
    'image_size':520,
}

JOB_ID = str(uuid.uuid4())
MAIN_SAVE_PATH = f'{bbm_home_mount_path}/{JOB_ID}'
MASK_SAVE_PATH = f'{MAIN_SAVE_PATH}/prediction_masks'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

val_transform = A.Compose([
    A.Resize(config['image_size'],config['image_size'],interpolation=cv2.INTER_NEAREST), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(transpose_mask=True)
])

target_dict={
    'tissue':0,
    'gray-matter':1,
    'white-matter':2
}


model_0 = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
model_0.classifier[4] = torch.nn.Conv2d(256, len(target_dict), kernel_size=(1, 1), stride=(1, 1))

model_0.load_state_dict(
    torch.load(
        config['model_0_path'],
        map_location='cpu'
    )
)

model_0 = model_0.to(device)
model_0.eval()

model_1 = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
model_1.classifier[4] = torch.nn.Conv2d(256, len(target_dict), kernel_size=(1, 1), stride=(1, 1))

model_1.load_state_dict(
    torch.load(
        config['model_1_path'],
        map_location='cpu'
    )
)

model_1 = model_1.to(device)
model_1.eval()

model_2 = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
model_2.classifier[4] = torch.nn.Conv2d(256, len(target_dict), kernel_size=(1, 1), stride=(1, 1))

model_2.load_state_dict(
    torch.load(
        config['model_2_path'],
        map_location='cpu'
    )
)

model_2 = model_2.to(device)
model_2.eval()

model_3 = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
model_3.classifier[4] = torch.nn.Conv2d(256, len(target_dict), kernel_size=(1, 1), stride=(1, 1))

model_3.load_state_dict(
    torch.load(
        config['model_3_path'],
        map_location='cpu'
    )
)

model_3 = model_3.to(device)
model_3.eval()

model_4 = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
model_4.classifier[4] = torch.nn.Conv2d(256, len(target_dict), kernel_size=(1, 1), stride=(1, 1))

model_4.load_state_dict(
    torch.load(
        config['model_4_path'],
        map_location='cpu'
    )
)

model_4 = model_4.to(device)
model_4.eval()

print('models loaded')


def get_gray_white_mask(image):
    probs_0 = calculate_mask(image, model_0, val_transform,step=2,n_channels=3,image_size=config['image_size'])
    probs_1 = calculate_mask(image, model_1, val_transform)
    probs_2 = calculate_mask(image, model_2, val_transform)
    probs_3 = calculate_mask(image, model_3, val_transform)
    probs_4 = calculate_mask(image, model_4, val_transform)

    final_probs = (probs_0+probs_1+probs_2+probs_3+probs_4)/5
    final_preds = np.argmax(final_probs,axis=0)

    return final_preds

def get_gray_white_mask_from_path(file_location):
    """Get the gray-white mask from a file path"""
    all_widths = []
    with tifffile.TiffFile(file_location) as tif:
        for page in tif.pages:
            tags={tag.name: tag.value for tag in page.tags.values()}
            all_widths.append(tags['ImageWidth'])

    index = np.argsort(all_widths)[-3]

    with tifffile.TiffFile(file_location) as tif:
        image=tif.pages[index].asarray()
        
    gray_white_mask = get_gray_white_mask(image)
    return gray_white_mask
        
def save_prediction_mask(gray_white_mask, image_id, save_path):
    """Save the prediction mask as a PNG image"""
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Convert mask to 8-bit image (0, 85, 170 for classes 0, 1, 2)
    mask_8bit = (gray_white_mask * 85).astype(np.uint8)
    
    # Save as PNG
    mask_filename = os.path.join(save_path, f"{image_id}_prediction_mask.png")
    cv2.imwrite(mask_filename, mask_8bit)
    
    print(f"Saved prediction mask: {mask_filename}")
    
    return mask_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process gray-white segmentation from a CSV file.")
    parser.add_argument(
        '--csv_path', type=str, required=True, 
        help="Path to the input CSV file containing image locations."
    )
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)

    df['JOB_ID'] = JOB_ID
    df['DATE'] = date.today()

    print(f'Processing job: {JOB_ID}')

    for i in tqdm(range(len(df))):
        try:
            file_location = df.loc[i,'File Location']
            image_id = str(int(df.loc[i,'Image ID']))
            
            # Get the prediction mask and save it
            gray_white_mask = get_gray_white_mask_from_path(file_location)
            save_prediction_mask(gray_white_mask, image_id, MASK_SAVE_PATH)

        except Exception as e:
            print(e)
            print(f"Could not process image: {image_id}")
            continue