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

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

import math
from tqdm import tqdm
import torch
import numpy as np
from skimage import measure

from torchvision.ops import nms

import os
import cv2 # Added import for morphological operations
import matplotlib.patches as patches # Required for visualization

# --- NEW FUNCTION: Visualization Helper ---
def plot_predictions_only(img, bboxes, labels, scores, score_cutoff=0.5, save_name=None):
    """
    Plots predictions on the image and saves it.
    Adapted from training_test_view_output_retina_cortical.ipynb but removed Ground Truth plotting.
    """
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)

    has_prediction = False

    for j in range(bboxes.shape[0]):
        if scores[j] < score_cutoff:
            continue
        
        has_prediction = True
        bbox = bboxes[j]
        label = labels[j]
        
        # BBox coordinates: [x_min, y_min, x_max, y_max]
        # Matplotlib Rectangle takes (x, y) (top-left), width, height
        # Note: The notebook code swapped x and y in some places, but standard format is usually x=col, y=row.
        # Let's stick to standard: x_min, y_min, w, h
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2] - x_min
        h = bbox[3] - y_min

        if label == 0:
            continue

        # Label text
        class_name = class_dict.get(label, str(label))
        s = f'{class_name}-{int(scores[j]*100)}%'
        
        # Draw text
        ax.text(x_min, y_min - 5, s, fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Draw rectangle
        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax.set_axis_off()
    
    if save_name and has_prediction:
        plt.tight_layout()
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close(fig) # Close to free memory
        return True
    
    plt.close(fig)
    return False

# --- NEW FUNCTION: Tissue Overlay Helper (Adapted from inference_test.py) ---
def save_overlay_with_stats(original_image_rgb, prediction_mask, image_id, save_path):
    """Overlays the prediction mask on the original image, adds stats, and saves it."""
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # --- Create a color mask for the overlay ---
    # We are working in RGB here
    # Blue for Gray Matter [0, 0, 255], Red for White Matter [255, 0, 0]
    color_mask = np.zeros_like(original_image_rgb)
    color_mask[prediction_mask == 1] = [0, 0, 255]  # Blue
    color_mask[prediction_mask == 2] = [255, 0, 0]  # Red

    # --- Blend the original image with the color mask ---
    # The overlay is 70% original image, 30% color mask
    overlay_image = cv2.addWeighted(original_image_rgb, 0.7, color_mask, 0.3, 0)

    # --- Calculate percentages ---
    gray_matter_pixels = np.sum(prediction_mask == 1)
    white_matter_pixels = np.sum(prediction_mask == 2)
    total_tissue_pixels = gray_matter_pixels + white_matter_pixels

    if total_tissue_pixels > 0:
        gray_percentage = (gray_matter_pixels / total_tissue_pixels) * 100
        white_percentage = (white_matter_pixels / total_tissue_pixels) * 100
    else:
        gray_percentage = 0
        white_percentage = 0

    # --- Add text to the image ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255) # White
    thickness = 2
    
    # Position text at the top-left corner
    cv2.putText(overlay_image, f"Gray Matter: {gray_percentage:.1f}%", (50, 80), font, font_scale, font_color, thickness)
    cv2.putText(overlay_image, f"White Matter: {white_percentage:.1f}%", (50, 160), font, font_scale, font_color, thickness)

    # --- Save the final image ---
    # Convert RGB to BGR for OpenCV saving
    overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
    
    output_filename = os.path.join(save_path, f"{image_id}_tissue_overlay.jpg")
    cv2.imwrite(output_filename, overlay_image_bgr)
    
    print(f"Saved tissue overlay image to: {output_filename}")

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

pretrained_path='/fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_plaque_ai/s311590_plaque_ai/plaque_ai/models/torchvision_fasterrcnn_resnet50_fpn.pt'

class_dict = {v: k for k, v in target_dict.items()}

model=get_model_instance_segmentation(len(target_dict)+1,pretrained_path)

#df=pd.read_csv('../data/6f3d_validation.csv')

val_transform = A.Compose([
    A.Resize(1024, 1024, always_apply=True), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(transpose_mask=True)
])

#'../models/tangle_ai_5_28_1024_rrc_rotate_rotate_full_2.pt'
import torch
model.load_state_dict(
    torch.load(
        '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_plaque_ai/s311590_plaque_ai/plaque_ai/models/plaque_ai_final.pt',
        map_location=torch.device('cpu')
    ),
    strict=False
)

model.eval()

print('model loaded')

# --- CHANGE 1: Update the path to the gray/white matter model ---
# Updated to match inference_test.py implementation
gray_white_path = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_gray_white/gray_white_segmentation/models/model_level_2_ce_0'

# --- CHANGE 2: Switch Model Architecture to ResNet50 (matching inference_test.py) ---
gray_model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

# The updated model has 3 output classes (tissue, gray-matter, white-matter) instead of 2
gray_model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
gray_model = gray_model.to('cpu')

gray_model.load_state_dict(
    torch.load(
        gray_white_path,
        map_location='cpu'
    )
)

gray_model.eval()
print('model ready')


# --- CHANGE 3: Add configuration from inference_test.py ---
gray_config = {
    'image_size': 520,
    'post_processing_method': 'weighted_argmax',
    'white_matter_threshold': 0.2,
    'gray_matter_threshold': 0.2,
    'white_matter_weight': 1.5,
    'clean_noise': True,
    'noise_clean_kernel_size': 21,
    'downsample_factor': 26.6666667 # Factor to match inference_test.py resizing
}

# Setup specific transform and weight matrix for the gray model using its config size
gray_weight_mat = create_weight_matrix(gray_config['image_size'])

gray_transform = A.Compose([
    A.Resize(gray_config['image_size'], gray_config['image_size'], always_apply=True), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(transpose_mask=True)
])

# Keep the original image_size variable for the plaque model (used later in the loop)
image_size=1024

# --- CHANGE 4: Define your source directory ---
SOURCE_TIFF_DIR = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_tiff_test'

# Default Output Directories
OUTPUT_DIR = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_prediction/'
VISUALIZATION_DIR = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_prediction_visuals/' 
STATS_DIR = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_statistics/' 

# --- CHANGE 5: Get list of TIFF files from the directory instead of a CSV ---
import glob
import sys # Added sys

# --- CHANGE: Logic to handle CLI argument vs Directory Search ---
if len(sys.argv) > 1:
    # If a file path is passed (from SLURM), use only that file
    tiff_files = [sys.argv[1]]
    print(f"Processing single file from CLI: {tiff_files[0]}")
    
    # Check for optional output directory argument
    if len(sys.argv) > 2:
        BASE_OUTPUT_DIR = sys.argv[2]
        print(f"Using custom output directory: {BASE_OUTPUT_DIR}")
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'prediction')
        VISUALIZATION_DIR = os.path.join(BASE_OUTPUT_DIR, 'visuals')
        STATS_DIR = os.path.join(BASE_OUTPUT_DIR, 'statistics')
else:
    # Otherwise, search the directory (default behavior)
    tiff_files = glob.glob(os.path.join(SOURCE_TIFF_DIR, '*.tif')) + \
                 glob.glob(os.path.join(SOURCE_TIFF_DIR, '*.tiff')) + \
                 glob.glob(os.path.join(SOURCE_TIFF_DIR, '*.svs'))
    print(f"Found {len(tiff_files)} images in {SOURCE_TIFF_DIR}")

# Create directories if they don't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR)

if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)

# CHANGED: Check STATS_DIR so we re-process files that don't have statistics yet
processed_files = [i.split('_stats')[0] for i in os.listdir(STATS_DIR)]

# Loop through the found files (now just 1 if run via SLURM)
for full_file_path in tqdm(tiff_files):
    
    # Extract filename and ID
    filename = os.path.basename(full_file_path)
    image_id = filename.split('.')[0]

    if image_id not in processed_files:
        try:
            # --- CHANGE 6: Load Full Image and Resize (Matching inference_test.py) ---
            # Instead of searching for a pyramid layer, we load the full image and resize it.
            # This ensures the input to the gray model is exactly what it expects.
            
            # 1. Load the full resolution image using memmap to avoid OOM
            # Use tifffile.memmap to access the image data without loading it all into RAM
            full_image = tifffile.memmap(full_file_path)
            
            with tifffile.TiffFile(full_file_path) as tif:
                # Check for resolution tags to determine pixel size
                # XResolution (282), YResolution (283), ResolutionUnit (296)
                page = tif.pages[0]
                x_res = page.tags.get(282)
                unit = page.tags.get(296)
                
                # Default pixel size (from original code)
                PIXEL_SIZE_MM = 0.0002827 
                
                if x_res and unit:
                    try:
                        res_val = x_res.value[0] / x_res.value[1] # numerator / denominator
                        if unit.value == 2: # Inch
                            PIXEL_SIZE_MM = 25.4 / res_val / 1000 
                            print(f"Detected Pixel Size: {PIXEL_SIZE_MM} mm (from Inch)")
                        elif unit.value == 3: # cm
                            PIXEL_SIZE_MM = 10.0 / res_val / 1000 
                            print(f"Detected Pixel Size: {PIXEL_SIZE_MM} mm (from cm)")
                        else:
                            print(f"Unknown unit {unit.value}, using default pixel size: {PIXEL_SIZE_MM} mm")
                    except Exception as e:
                        print(f"Error parsing resolution tags: {e}, using default pixel size: {PIXEL_SIZE_MM} mm")
                else:
                    print(f"No resolution tags found, using default pixel size: {PIXEL_SIZE_MM} mm")

            # --- NEW: Calculate Scaling Factor ---
            TARGET_PIXEL_SIZE_MM = 0.0002827
            scale_factor = PIXEL_SIZE_MM / TARGET_PIXEL_SIZE_MM
            print(f"Scaling Factor: {scale_factor:.4f} (Source: {PIXEL_SIZE_MM:.7f} vs Target: {TARGET_PIXEL_SIZE_MM:.7f})")
            
            # The model input is fixed at 1024
            model_input_size = 1024
            
            # We calculate how much of YOUR image corresponds to the physical area of a 1024x1024 training tile.
            # If scale_factor > 1 (your pixels are bigger), we need FEWER of them to cover the same area.
            extraction_size = int(model_input_size / scale_factor)
            print(f"Adjusting extraction tile size to {extraction_size}x{extraction_size} (will be resized to 1024x1024)")
            # -------------------------------------

            # --- FIX: Handle Channel-First Images (C, H, W) ---
            # Note: memmap is read-only by default, so we can't transpose in place.
            # We will handle the transposition during the resize step.
            is_channel_first = False
            if full_image.ndim == 3 and full_image.shape[0] < 10 and full_image.shape[1] > 100:
                print(f"Detected Channel-First Image {full_image.shape}")
                is_channel_first = True
                original_h = full_image.shape[1]
                original_w = full_image.shape[2]
            else:
                original_h = full_image.shape[0]
                original_w = full_image.shape[1]
            # --------------------------------------------------

            # 2. Calculate new dimensions based on downsample factor
            new_w = int(original_w / gray_config['downsample_factor'])
            new_h = int(original_h / gray_config['downsample_factor'])
            
            # 3. Resize for the gray model (Memory Efficient Way)
            # Instead of loading the full image, we subsample it first using slicing
            stride_factor = int(gray_config['downsample_factor'])
            
            if is_channel_first:
                # Slice: [:, ::stride, ::stride] -> (C, H_small, W_small)
                small_view = full_image[:, ::stride_factor, ::stride_factor]
                # Transpose to (H, W, C) for OpenCV
                small_view = np.transpose(small_view, (1, 2, 0))
            else:
                # Slice: [::stride, ::stride, :] -> (H_small, W_small, C)
                small_view = full_image[::stride_factor, ::stride_factor, :]
            
            # Now 'small_view' is a much smaller array in memory.
            # We can safely resize this to the exact target dimensions.
            gray_image = cv2.resize(small_view, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # --- FIX: Ensure image is RGB (3 channels) before processing ---
            if gray_image.shape[-1] == 4:
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGBA2RGB)
            # ---------------------------------------------------------------

            # --- Updated calculate_mask call with new config ---
            final_mask=calculate_mask(
                gray_image,
                gray_model,
                [0,0,gray_image.shape[1],gray_image.shape[0]],
                gray_transform, 
                weight_matrix=gray_weight_mat, 
                step=2,
                n_channels=3, 
                image_size=gray_config['image_size']
            )

            # --- FIX: Extract Gray Matter Channel and Create 2D Mask ---
            # final_mask shape is (3, H, W). Channel 1 is Gray Matter.
            # We use argmax to get the most likely class per pixel.
            
            # Apply weight to white matter (channel 2) to reduce false positives if needed
            weighted_probs = np.copy(final_mask)
            weighted_probs[2, :, :] *= gray_config['white_matter_weight']
            
            # Get predictions (0=Tissue/Background, 1=Gray, 2=White)
            final_preds = np.argmax(weighted_probs, axis=0)
            
            # --- NEW: Save Tissue Overlay ---
            # This generates the visual showing Gray vs White matter distribution
            save_overlay_with_stats(gray_image, final_preds, image_id, VISUALIZATION_DIR)
            # --------------------------------
            
            # Create binary gray matter mask (Class 1) - This results in a 2D array (H, W)
            gray_mask = (final_preds == 1).astype(np.uint8)

            # Optional: Clean noise
            if gray_config['clean_noise']:
                kernel_size = gray_config['noise_clean_kernel_size']
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
                
            # Ensure mask is boolean/binary for the logic below
            # Now gray_mask is (H, W), which works with the slicing logic later
            # -----------------------------------------------------------

            # --- NEW: Prepare for Stitched Visualization ---
            # We use the downsampled 'gray_image' as the base. 
            visualization_image = gray_image.copy()
            
            # Ensure it's RGB so we can draw colored boxes
            if len(visualization_image.shape) == 2:
                visualization_image = cv2.cvtColor(visualization_image, cv2.COLOR_GRAY2RGB)
            elif visualization_image.shape[2] == 4:
                visualization_image = cv2.cvtColor(visualization_image, cv2.COLOR_RGBA2RGB)

            # --- CHANGED: Lists to collect RAW global predictions ---
            global_boxes_list = []
            global_scores_list = []
            global_labels_list = []
            # -----------------------------------------------

            image_size=1024
            overlap=512
            stride=image_size-overlap
            
            # Adjust loop limits based on channel-first or channel-last
            if is_channel_first:
                # Shape is (C, H, W)
                n_x=int(full_image.shape[1]/(stride))-1
                n_y=int(full_image.shape[2]/(stride))-1
                # Ratios for mapping to gray_image (H_gray, W_gray)
                ratio_x = gray_image.shape[0] / full_image.shape[1]
                ratio_y = gray_image.shape[1] / full_image.shape[2]
            else:
                # Shape is (H, W, C)
                n_x=int(full_image.shape[0]/(stride))-1
                n_y=int(full_image.shape[1]/(stride))-1
                ratio_x = gray_image.shape[0] / full_image.shape[0]
                ratio_y = gray_image.shape[1] / full_image.shape[1]
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            for i in tqdm(range(n_x)):
                for j in range(n_y):
                    # Check if the corresponding low-res area has tissue (gray mask)
                    # We use the calculated ratios to map the high-res tile coordinates to the low-res mask
                    y_start_low = int(i * stride * ratio_x)
                    y_end_low = int((i * stride + image_size) * ratio_x)
                    x_start_low = int(j * stride * ratio_y)
                    x_end_low = int((j * stride + image_size) * ratio_y)
                    
                    # Ensure indices are within bounds
                    y_start_low = max(0, y_start_low)
                    x_start_low = max(0, x_start_low)
                    y_end_low = min(gray_mask.shape[0], y_end_low)
                    x_end_low = min(gray_mask.shape[1], x_end_low)

                    if gray_mask[y_start_low:y_end_low, x_start_low:x_end_low].sum() > 0:
                        # Slice the tile from the memmap
                        if is_channel_first:
                            # Slice: [:, y:y+size, x:x+size] -> (C, H, W)
                            tile = full_image[:, i*stride:(i*stride+image_size), j*stride:(j*stride+image_size)]
                            # Transpose to (H, W, C) for model
                            tile = np.transpose(tile, (1, 2, 0))
                        else:
                            # Slice: [y:y+size, x:x+size, :] -> (H, W, C)
                            tile = full_image[i*stride:(i*stride+image_size), j*stride:(j*stride+image_size), :]
                        
                        # Ensure contiguous array for PyTorch/OpenCV
                        tile = np.ascontiguousarray(tile)
                        
                        if tile.shape[-1] == 4:
                            tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)

                        # Run inference
                        output=model(val_transform(image=tile)['image'].unsqueeze(0).to(device))
                        
                        # Get raw outputs from this tile
                        # We do NOT run NMS here, or we run a weak one just to save memory.
                        # Let's grab everything above a low score threshold to be safe.
                        boxes = output[0]['boxes'].detach().cpu()
                        scores = output[0]['scores'].detach().cpu()
                        labels = output[0]['labels'].detach().cpu()
                        
                        # Filter low confidence immediately to save memory
                        keep_idxs = scores > 0.05
                        boxes = boxes[keep_idxs]
                        scores = scores[keep_idxs]
                        labels = labels[keep_idxs]

                        if len(boxes) > 0:
                            # --- NEW: Scale boxes back to original image coordinates ---
                            # The model outputs boxes in 1024x1024 space.
                            # We need to map them back to 'extraction_size' space before adding global offsets.
                            box_scale_ratio = extraction_size / model_input_size
                            boxes = boxes * box_scale_ratio
                            # -----------------------------------------------------------

                            # Convert to Global Coordinates
                            # boxes format: x1, y1, x2, y2
                            boxes[:, 0] += j * stride # x1
                            boxes[:, 1] += i * stride # y1
                            boxes[:, 2] += j * stride # x2
                            boxes[:, 3] += i * stride # y2
                            
                            global_boxes_list.append(boxes)
                            global_scores_list.append(scores)
                            global_labels_list.append(labels)

            # --- NEW: GLOBAL NMS STEP ---
            print("Running Global NMS...")
            output_df = pd.DataFrame({})
            all_predictions = []

            if len(global_boxes_list) > 0:
                # Concatenate all predictions from all tiles
                all_boxes = torch.cat(global_boxes_list)
                all_scores = torch.cat(global_scores_list)
                all_labels = torch.cat(global_labels_list)
                
                # Apply Global NMS
                # iou_threshold=0.3 means if two boxes overlap by >30%, keep only the one with higher score
                keep_indices = nms(all_boxes, all_scores, iou_threshold=0.3)
                
                final_boxes = all_boxes[keep_indices].numpy()
                final_scores = all_scores[keep_indices].numpy()
                final_labels = all_labels[keep_indices].numpy()
                
                # Build the DataFrame and Visualization List from the CLEANED data
                for w in range(len(final_labels)):
                    # Filter by final score cutoff for CSV/Viz (e.g., 0.5)
                    if final_scores[w] > 0.5:
                        # Add to DataFrame
                        k = len(output_df) + 1
                        output_df.loc[k, 'label'] = final_labels[w]
                        output_df.loc[k, 'score'] = final_scores[w]
                        output_df.loc[k, 'x_0'] = final_boxes[w, 0]
                        output_df.loc[k, 'y_0'] = final_boxes[w, 1]
                        output_df.loc[k, 'x_1'] = final_boxes[w, 2]
                        output_df.loc[k, 'y_1'] = final_boxes[w, 3]
                        
                        # Add to Visualization List
                        all_predictions.append({
                            'label': final_labels[w],
                            'score': final_scores[w],
                            'box': final_boxes[w]
                        })
            # ----------------------------

            # --- NEW: Generate Stitched Visualization (After Loop) ---
            if len(all_predictions) > 0:
                print(f"Generating stitched visualization for {image_id}...")
                # Calculate scale factor to map Full-Res coords -> Downsampled Image
                scale_factor = 1.0 / gray_config['downsample_factor']
                
                for pred in all_predictions:
                    # Scale coordinates down to the visualization image size
                    gx0, gy0, gx1, gy1 = pred['box']
                    
                    sx0 = int(gx0 * scale_factor)
                    sy0 = int(gy0 * scale_factor)
                    sx1 = int(gx1 * scale_factor)
                    sy1 = int(gy1 * scale_factor)
                    
                    # Draw Rectangle (Red color in RGB is 255,0,0)
                    cv2.rectangle(visualization_image, (sx0, sy0), (sx1, sy1), (255, 0, 0), 2)
                    
                    # Draw Label
                    label_id = int(pred['label'])
                    class_name = class_dict.get(label_id, str(label_id))
                    short_name = class_name.split(' ')[0] # Shorten name
                    
                    text = f"{short_name}"
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Draw text background
                    cv2.rectangle(visualization_image, (sx0, sy0 - text_h - 4), (sx0 + text_w, sy0), (255, 0, 0), -1)
                    # Draw text (White)
                    cv2.putText(visualization_image, text, (sx0, sy0 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                # Save the stitched image
                save_path = os.path.join(VISUALIZATION_DIR, f"{image_id}_stitched.jpg")
                # Convert RGB back to BGR for OpenCV saving
                visualization_image_bgr = cv2.cvtColor(visualization_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, visualization_image_bgr)
                print(f"Saved stitched visualization to {save_path}")
            # ---------------------------------------------------------

            # --- RESTORED: Save Output CSV ---
            output_df['gray_area']=np.sum(gray_mask)
            output_df.to_csv(f'{OUTPUT_DIR}/{image_id}.csv',index=False)
            
            # --- RESTORED: Statistics Calculation ---
            try:
                # Pixel size in mm (already calculated above)
                # PIXEL_SIZE_MM = 0.0002827 # Removed hardcoded value
                
                # 1. Calculate Gray Matter Area in mm^2
                downsampled_area_pixels = np.sum(gray_mask)
                full_res_area_pixels = downsampled_area_pixels * (gray_config['downsample_factor'] ** 2)
                gray_area_mm2 = full_res_area_pixels * (PIXEL_SIZE_MM ** 2)
                
                # 2. Calculate Counts, Densities, and other Stats
                stats_data = {
                    'image_id': image_id,
                    'gray_area_mm2': gray_area_mm2
                }
                
                total_count = 0
                
                for class_name, label_id in target_dict.items():
                    if 'label' in output_df.columns:
                        class_df = output_df[output_df['label'] == label_id]
                        count = len(class_df)
                        
                        if count > 0:
                            mean_score = class_df['score'].mean()
                            widths = class_df['x_1'] - class_df['x_0']
                            heights = class_df['y_1'] - class_df['y_0']
                            areas_px = widths * heights
                            mean_area_mm2 = areas_px.mean() * (PIXEL_SIZE_MM ** 2)
                        else:
                            mean_score = 0.0
                            mean_area_mm2 = 0.0
                    else:
                        count = 0
                        mean_score = 0.0
                        mean_area_mm2 = 0.0
                        
                    density = count / gray_area_mm2 if gray_area_mm2 > 0 else 0.0
                    
                    stats_data[f'{class_name}_count'] = count
                    stats_data[f'{class_name}_density_mm2'] = density
                    stats_data[f'{class_name}_mean_score'] = mean_score
                    stats_data[f'{class_name}_mean_area_mm2'] = mean_area_mm2
                    
                    total_count += count
                
                stats_data['total_pathology_count'] = total_count
                stats_data['total_pathology_density_mm2'] = total_count / gray_area_mm2 if gray_area_mm2 > 0 else 0.0
                
                stats_df = pd.DataFrame([stats_data])
                stats_df.to_csv(f'{STATS_DIR}/{image_id}_stats.csv', index=False)
                
            except Exception as e:
                print(f"Error calculating statistics for {filename}: {e}")
            # -----------------------------------
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")