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

# Add staintools import
#import staintools

from inference_utils import calculate_mask

# --- Base path for all prediction runs ---
base_prediction_path = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/'

#where the models are stored
MODEL_DIR = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_gray_white/gray_white_segmentation/models'

# --- IMPORTANT: Set the path to your chosen template image for stain normalization ---
TEMPLATE_IMAGE_PATH = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/tiff/NA09-404_H02P23TB518P-229562_level-0.tiff' # <-- REPLACE WITH YOUR TEMPLATE IMAGE

config={
    'model_0_path':f'{MODEL_DIR}/model_level_2_ce_0',
    'model_1_path':f'{MODEL_DIR}/model_level_2_ce_1',
    'model_2_path':f'{MODEL_DIR}/model_level_2_ce_2',
    'model_3_path':f'{MODEL_DIR}/model_level_2_ce_3',
    'model_4_path':f'{MODEL_DIR}/model_level_2_ce_4',
    'image_size':520,
    'downsample_factor': 26.67, # Add this to control resizing
    
    # --- NEW: Configuration for Post-Processing ---
    'post_processing_method': 'weighted_argmax', # 'priority_threshold' or 'weighted_argmax'
    'white_matter_threshold': 0.2, # Threshold for white matter
    'gray_matter_threshold': 0.2,  # Threshold for gray matter
    'white_matter_weight': 1.5,    # Weight for weighted_argmax method
    
    # --- NEW: Noise Cleaning Step ---
    'clean_noise': True,                   # Set to True to enable this step
    'noise_clean_kernel_size': 21,          # Kernel size for cleaning. 3 or 5 is usually good.

    'fill_holes': False,           # Set to False to disable hole filling
    'hole_fill_method': 'defect_filling', # 'closing', 'convex_hull', or 'defect_filling'
    'hole_fill_kernel_size': 25,   # For 'closing' method
    'max_defect_depth': 50,        # For 'defect_filling' method (in pixels)
}

# --- Dynamically create the output folder name based on config ---
method = config.get('post_processing_method')
wm_weight = config.get('white_matter_weight')
gm_thresh = config.get('gray_matter_threshold')
wm_thresh = config.get('white_matter_threshold')

# Get hole filling info
fill_holes = config.get('fill_holes')
fill_method = config.get('hole_fill_method')
hole_kernel_size = config.get('hole_fill_kernel_size')
defect_depth = config.get('max_defect_depth')

# Get noise cleaning info
clean_noise = config.get('clean_noise')
noise_kernel_size = config.get('noise_clean_kernel_size')

# Construct the descriptive folder name
run_folder_name = f"prediction_{method}_wmw_{wm_weight}_gmt_{gm_thresh}_wmt_{wm_thresh}"

# Add noise cleaning info if enabled
if clean_noise:
    run_folder_name += f"_clean_{noise_kernel_size}"

# Add hole filling info if enabled
if fill_holes:
    run_folder_name += f"_fill_{fill_method}"
    if fill_method == 'closing':
        run_folder_name += f"_{hole_kernel_size}"
    elif fill_method == 'defect_filling':
        run_folder_name += f"_{defect_depth}"

# Define the unique base path for this entire run
bbm_home_mount_path = os.path.join(base_prediction_path, run_folder_name)
print(f"--- Output for this run will be saved to: {bbm_home_mount_path} ---")


# JOB_ID = str(uuid.uuid4())
# MAIN_SAVE_PATH = f'{bbm_home_mount_path}/{JOB_ID}'
# MASK_SAVE_PATH = f'{MAIN_SAVE_PATH}/prediction_masks'

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

# --- Stain Normalization Setup ---
# print("Setting up stain normalizer...")
# if not os.path.exists(TEMPLATE_IMAGE_PATH):
#     raise FileNotFoundError(f"Stain normalization template image not found at: {TEMPLATE_IMAGE_PATH}")

# # Read and prepare the template image
# target = staintools.read_image(TEMPLATE_IMAGE_PATH)
# # Ensure template is 3-channel RGB
# if target.shape[2] > 3:
#     target = target[:, :, :3]

# # Create and fit the normalizer
# normalizer = staintools.StainNormalizer(method='vahadane')
# normalizer.fit(target)
# print("Stain normalizer ready.")


def post_process_predictions(probs):
    """
    Applies advanced post-processing to fix model biases and artifacts.
    """
    method = config.get('post_processing_method', 'argmax')
    
    if method == 'priority_threshold':
        print("    -> Applying priority thresholding...")
        white_thresh = config.get('white_matter_threshold', 0.5)
        gray_thresh = config.get('gray_matter_threshold', 0.5)
        
        # Start with a mask of all background (class 0)
        final_preds = np.zeros(probs.shape[1:], dtype=np.uint8)
        
        # 1. White matter has priority.
        white_mask = probs[2] > white_thresh
        final_preds[white_mask] = 2
        
        # 2. Gray matter can only occupy pixels not already assigned to white matter.
        gray_mask = probs[1] > gray_thresh
        unassigned_mask = (final_preds == 0)
        final_preds[gray_mask & unassigned_mask] = 1

    elif method == 'weighted_argmax':
        print("    -> Applying weighted argmax...")
        weight = config.get('white_matter_weight', 1.2)
        weighted_probs = np.copy(probs)
        # Apply weight to the white matter probability channel
        weighted_probs[2, :, :] *= weight
        final_preds = np.argmax(weighted_probs, axis=0)
        
    else: # Default to original argmax method
        final_preds = np.argmax(probs, axis=0)

    # --- NEW: Optional Noise Cleaning Step ---
    if config.get('clean_noise', False):
        print("    -> Cleaning small false positives...")
        kernel_size = config.get('noise_clean_kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Create a binary mask of only the white matter predictions
        wm_mask = (final_preds == 2).astype(np.uint8)
        
        # Apply morphological opening to remove small, isolated pixels
        cleaned_wm_mask = cv2.morphologyEx(wm_mask, cv2.MORPH_OPEN, kernel)
        
        # Update the final predictions:
        # 1. Set all original white matter pixels to background (0)
        final_preds[final_preds == 2] = 0
        # 2. Add back only the cleaned white matter pixels
        final_preds[cleaned_wm_mask == 1] = 2

    # --- Hole Filling for Bubbles and Gaps ---
    if config.get('fill_holes', False):
        fill_method = config.get('hole_fill_method', 'closing')
        print(f"    -> Filling holes and gaps using '{fill_method}' method...")

        # Create a binary mask where 1 = any tissue, 0 = background
        tissue_mask = (final_preds > 0).astype(np.uint8)
        filled_tissue_mask = tissue_mask.copy() # Start with the original mask
        
        # Find contours of the tissue mask
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Assume the largest contour is the main tissue body
            largest_contour = max(contours, key=cv2.contourArea)

            if fill_method == 'defect_filling':
                max_depth = config.get('max_defect_depth', 50)
                # We need at least 4 points to compute convexity defects
                if len(largest_contour) > 3:
                    hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
                    defects = cv2.convexityDefects(largest_contour, hull_indices)
                    
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            # s, e, f are indices of start, end, and farthest points
                            # d is the approximate distance to the hull (depth)
                            
                            # The depth 'd' is in a fixed-point format, so we divide by 256
                            depth = d / 256.0
                            
                            if depth > max_depth:
                                start = tuple(largest_contour[s][0])
                                end = tuple(largest_contour[e][0])
                                far = tuple(largest_contour[f][0])
                                # Fill the triangular area of the deep defect
                                cv2.drawContours(filled_tissue_mask, [np.array([start, end, far])], 0, 1, -1)

            elif fill_method == 'convex_hull':
                hull = cv2.convexHull(largest_contour)
                cv2.drawContours(filled_tissue_mask, [hull], 0, 1, -1)

            elif fill_method == 'closing':
                kernel_size = config.get('hole_fill_kernel_size', 25)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                filled_tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find the newly filled pixels (holes and gaps)
        newly_filled_pixels = (filled_tissue_mask == 1) & (tissue_mask == 0)
        
        # Re-assign these pixels from background (0) to gray matter (1)
        final_preds[newly_filled_pixels] = 1

    return final_preds


def get_gray_white_mask(image):
    print("    --> Running inference with model 0...")
    probs_0 = calculate_mask(image, model_0, val_transform,step=2,n_channels=3,image_size=config['image_size'])
    print("    --> Running inference with model 1...")
    probs_1 = calculate_mask(image, model_1, val_transform)
    print("    --> Running inference with model 2...")
    probs_2 = calculate_mask(image, model_2, val_transform)
    print("    --> Running inference with model 3...")
    probs_3 = calculate_mask(image, model_3, val_transform)
    print("    --> Running inference with model 4...")
    probs_4 = calculate_mask(image, model_4, val_transform)

    final_probs = (probs_0+probs_1+probs_2+probs_3+probs_4)/5
    
    # --- Use the new post-processing function ---
    final_preds = post_process_predictions(final_probs)

    return final_preds, final_probs

def get_gray_white_mask_from_path(file_location, image_id, save_path):
    """Get the gray-white mask from a file path"""
    
    factor = config.get('downsample_factor', 1.0)

    print("    -> Reading full-resolution image into memory...")
    # Read the full image first. This is more robust than resizing on-the-fly.
    image = tifffile.imread(file_location)
    print("    -> Image loaded into memory.")

    # --- Resize the image to speed up processing ---
    if factor > 1:
        original_h, original_w = image.shape[:2]
        # Ensure the new dimensions are integers for OpenCV
        new_w = int(original_w / factor)
        new_h = int(original_h / factor)
        print(f"    -> Resizing image from {original_w}x{original_h} to {new_w}x{new_h}...")
        # Using INTER_AREA is good for down-sampling
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print("    -> Resizing complete.")

    # The image is loaded as RGB by tifffile, so we convert to BGR for OpenCV compatibility
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- Save the resized image ---
    resized_filename = os.path.join(save_path, f"{image_id}_resized_{factor}.png")
    cv2.imwrite(resized_filename, image)
    print(f"    -> Saved resized image to: {resized_filename}")

    # Convert from BGR (OpenCV default) to RGB for model processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    gray_white_mask, final_probs = get_gray_white_mask(image_rgb)
    # Return the original BGR image for overlay, the mask, and the probabilities
    return image, gray_white_mask, final_probs
        
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

def save_probability_heatmaps(probabilities, image_id, save_path):
    """
    Saves heatmaps of gray and white matter probabilities for debugging.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Get probability maps for gray matter (class 1) and white matter (class 2)
    gray_matter_prob = probabilities[1, :, :]
    white_matter_prob = probabilities[2, :, :]

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Probability Heatmaps for {image_id}', fontsize=16)

    # Plot Gray Matter Heatmap
    im1 = axes[0].imshow(gray_matter_prob, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Gray Matter Probability')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)

    # Plot White Matter Heatmap
    im2 = axes[1].imshow(white_matter_prob, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('White Matter Probability')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Save the figure
    heatmap_filename = os.path.join(save_path, f"{image_id}_heatmaps.png")
    plt.savefig(heatmap_filename, bbox_inches='tight', dpi=150)
    plt.close(fig) # Close the figure to free up memory
    print(f"    -> Saved probability heatmaps to: {heatmap_filename}")


def calculate_statistics(prediction_mask):
    """Calculates the percentage of gray and white matter."""
    gray_matter_pixels = np.sum(prediction_mask == 1)
    white_matter_pixels = np.sum(prediction_mask == 2)
    total_tissue_pixels = gray_matter_pixels + white_matter_pixels

    if total_tissue_pixels > 0:
        gray_percentage = (gray_matter_pixels / total_tissue_pixels) * 100
        white_percentage = (white_matter_pixels / total_tissue_pixels) * 100
    else:
        gray_percentage = 0
        white_percentage = 0
    
    return gray_percentage, white_percentage

def save_overlay_with_stats(original_image, prediction_mask, image_id, save_path):
    """Overlays the prediction mask on the original image, adds stats, and saves it."""
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # --- Create a color mask for the overlay ---
    # Blue for Gray Matter, Red for White Matter
    color_mask = np.zeros_like(original_image)
    color_mask[prediction_mask == 1] = [255, 0, 0]  # Blue
    color_mask[prediction_mask == 2] = [0, 0, 255]  # Red

    # --- Blend the original image with the color mask ---
    # The overlay is 70% original image, 30% color mask
    overlay_image = cv2.addWeighted(original_image, 0.7, color_mask, 0.3, 0)

    # --- Calculate percentages ---
    gray_percentage, white_percentage = calculate_statistics(prediction_mask)

    # --- Add text to the image ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255) # White
    thickness = 2
    
    # Position text at the top-left corner
    cv2.putText(overlay_image, f"Gray Matter: {gray_percentage:.1f}%", (50, 80), font, font_scale, font_color, thickness)
    cv2.putText(overlay_image, f"White Matter: {white_percentage:.1f}%", (50, 160), font, font_scale, font_color, thickness)

    # --- Save the final image ---
    output_filename = os.path.join(save_path, f"{image_id}_overlay.png")
    cv2.imwrite(output_filename, overlay_image)
    
    print(f"Saved overlay image to: {output_filename}")
    
    return output_filename

def process_single_file(file_location):
    """
    Processes a single TIFF file for gray/white matter segmentation.
    """
    try:
        image_id = os.path.splitext(os.path.basename(file_location))[0]

        # --- Check if already processed ---
        # Define save paths for this image
        MAIN_SAVE_PATH = os.path.join(bbm_home_mount_path, image_id)
        # if os.path.exists(MAIN_SAVE_PATH):
        #     print(f"Output directory for {image_id} already exists. Skipping.")
        #     return

        print(f"\n--- Processing: {file_location} ---")
        MASK_SAVE_PATH = os.path.join(MAIN_SAVE_PATH, 'prediction_masks')
        os.makedirs(MASK_SAVE_PATH, exist_ok=True)
        
        # Get the prediction mask and probabilities
        original_image, gray_white_mask, final_probs = get_gray_white_mask_from_path(file_location, image_id, MASK_SAVE_PATH)
        
        # --- Save probability map and heatmaps for debugging ---
        print("    -> Saving probability map and heatmaps...")
        probs_filename = os.path.join(MASK_SAVE_PATH, f"{image_id}_probabilities.npy")
        np.save(probs_filename, final_probs)
        print(f"    -> Saved raw probabilities to: {probs_filename}")
        save_probability_heatmaps(final_probs, image_id, MASK_SAVE_PATH)
        
        # Calculate statistics
        print("    -> Calculating statistics...")
        gray_percentage, white_percentage = calculate_statistics(gray_white_mask)
        
        # Save the overlay image
        print("    -> Saving overlay image with statistics...")
        save_overlay_with_stats(original_image, gray_white_mask, image_id, MASK_SAVE_PATH)
        
        # Store results for CSV
        results_data = {
            'image_name': os.path.basename(file_location),
            'gray_matter_percentage': gray_percentage,
            'white_matter_percentage': white_percentage
        }

        # --- Save result to a unique CSV file for this image ---
        results_df = pd.DataFrame([results_data])
        csv_path = os.path.join(MAIN_SAVE_PATH, f'{image_id}_result.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"--- Result for {image_id} saved to: {csv_path} ---")

    except Exception as e:
        print(f"Could not process image: {file_location}")
        print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run gray/white matter segmentation on a single TIFF file.')
    parser.add_argument('file_location', type=str, help='The full path to the TIFF file to process.')
    args = parser.parse_args()

    process_single_file(args.file_location)