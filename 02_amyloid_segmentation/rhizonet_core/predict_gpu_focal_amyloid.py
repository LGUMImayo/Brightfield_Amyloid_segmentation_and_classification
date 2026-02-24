"""
Script for running inference using the FOCAL LOSS trained U-Net model.
Imports Unet2D from unet2D_focal_amyloid.py
"""

import os
import sys
import glob
import argparse
import json
from tqdm import tqdm
from skimage import io, measure, morphology
import numpy as np
import torch
import cv2 

from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureType
from PIL import Image

# --- CRITICAL CHANGE: Force Import from local directory ---
# This ensures we get setup-specific changes from unet2D_focal_amyloid.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from unet2D_focal_amyloid import Unet2D, dynamic_scale
    from utils import MapImage, createBinaryAnnotation, extract_largest_component_bbox_image
except ImportError:
    # Fallback for relative imports if run as module
    from .unet2D_focal_amyloid import Unet2D, dynamic_scale 
    from .utils import MapImage, createBinaryAnnotation, extract_largest_component_bbox_image
# ----------------------------------------------------------

# Helper to re-use logic
def _parse_training_variables(argparse_args):
    args = vars(argparse_args)
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    args['pred_patch_size'] = tuple(args['pred_patch_size'])
    if args['gpus'] is None: args['gpus'] = -1 if torch.cuda.is_available() else 0
    return args

def transform_image(img_path):
    transform = Compose([EnsureType()])
    try:
        img = np.array(Image.open(img_path))
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, img_path

    if img.ndim == 2: img = np.expand_dims(img, axis=-1)
    if img.ndim == 3 and img.shape[-1] <= 4:
        img = np.transpose(img[..., :3] , (2, 0, 1)) 
    img = dynamic_scale(img)
    img = transform(img)
    return img, img_path

def clean_mask_v8(mask):
    """
    Removes specific bubble artifacts while preserving fragmented root segments.
    v8: 'Connect-and-Check' logic with relaxed thresholds.
    """
    # Ensure boolean
    binary = mask > 0
    cleaned = binary.copy()
    
    # --- STAGE 1: Arc/Broken Bubble Removal via Closing ---
    try:
        # disk(15) connects gaps up to ~30 pixels
        closed_mask = morphology.binary_closing(binary, morphology.disk(15))
        closed_labels = measure.label(closed_mask)
        closed_regions = measure.regionprops(closed_labels)
        
        bad_closed_labels = set()
        
        for props in closed_regions:
            is_hidden_bubble = False
            
            # A) Formed a Donut (Euler < 1)
            # Must be strictly round (Eccentricity < 0.60) and not huge (Area < 2500)
            if props.euler_number < 1:
                if props.eccentricity < 0.60 and props.area < 2500:
                    is_hidden_bubble = True
                    
            # B) Formed a Solid Circle
            # Must be very solid (> 0.90) and round (Eccentricity < 0.60)
            elif props.area > 100:
                if props.solidity > 0.90 and props.eccentricity < 0.60:
                    is_hidden_bubble = True
                    
            if is_hidden_bubble:
                bad_closed_labels.add(props.label)
        
        # Remove pixels falling within identified bubble regions
        if bad_closed_labels:
            bad_mask = np.isin(closed_labels, list(bad_closed_labels))
            cleaned[bad_mask] = 0
            
    except Exception as e:
        print(f"Warning: Morphological cleaning failed: {e}")
        # If it fails (e.g. memory), we continue to Stage 2 with original mask
    
    # --- STAGE 2: Individual Object Cleaning ---
    label_img = measure.label(cleaned)
    regions = measure.regionprops(label_img)
    
    for props in regions:
        is_artifact = False
        
        # 1. Micro-noise (Dust)
        if props.area < 5: 
            is_artifact = True
            
        # 2. Hollow Bubbles (Donuts)
        elif props.euler_number < 1:
            if props.eccentricity < 0.40 and props.area < 1000:
                is_artifact = True
        
        # 3. Solid Round Bubbles
        elif props.area > 100:
            if props.solidity > 0.90 and props.eccentricity < 0.45:
                is_artifact = True

        if is_artifact:
            cleaned[label_img == props.label] = 0
            
    return cleaned.astype(np.uint8) * 255

def predict_step(image_path, model, pred_patch_size):
    image, _ = transform_image(image_path)
    if image is None: return None, None

    cropped_image = extract_largest_component_bbox_image(image.unsqueeze(0), lab=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_cropped_image = torch.tensor(cropped_image).to(device)
    
    with torch.no_grad():
        logits = sliding_window_inference(inputs=tensor_cropped_image.float(), roi_size=pred_patch_size, sw_batch_size=4, predictor=model)
        
        # Calculate probabilities
        probs = torch.softmax(logits, dim=1)
        
        # --- AMYLOID STRATEGY 1: Confidence Thresholding ---
        # UPDATED: Lowered to 0.65 per user request
        confidence_threshold = 0.65 
        
        # Get the highest probability (max_probs) and the predicted class (preds)
        max_probs, preds = torch.max(probs, dim=1)
        
        # Create a boolean mask where confidence is too low
        low_confidence_mask = max_probs < confidence_threshold
        
        # Force the prediction to Background (0) for uncertain pixels
        preds[low_confidence_mask] = 0
        
        pred = preds.byte()
        # -------------------------------------------

    return pred, probs

def predict_model(args):
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load from the FOCAL implementation
    # weights_only=False is required for older checkpoints containing class definitions
    unet = Unet2D.load_from_checkpoint(args['model_path'], map_location=device, weights_only=False).to(device)
    unet.eval()

    # Determine processing mode (Single File vs Directory)
    if args.get('single_file'):
        files = [args['single_file']]
        output_base = os.path.dirname(args['single_file'])
    else:
        root = args.get('wsi_dir') or args['pred_data_dir']
        files = sorted(glob.glob(os.path.join(root, '*.tif')))
        output_base = root

        # --- Pipeline Path Fix ---
        # If no files found in root, check if they are in heatmap/seg_tiles
        if len(files) == 0:
            input_subdir = os.path.join(root, 'heatmap', 'seg_tiles')
            if os.path.isdir(input_subdir):
                print(f"No .tif files in root. Switching to {input_subdir}")
                files = sorted(glob.glob(os.path.join(input_subdir, '*.tif')))
                
                # If we found files, redirect output to heatmap/TAU_seg_tiles
                if len(files) > 0:
                    output_base = os.path.join(root, 'heatmap', 'TAU_seg_tiles')
                    os.makedirs(output_base, exist_ok=True)
                    print(f"Redirecting output to {output_base}")
        # -------------------------

    print(f"Running Focal Model Inference (Amyloid Clean) on {len(files)} files...")
    
    for file_path in tqdm(files):
        try:
            res = predict_step(file_path, unet, args['pred_patch_size'])
            if res[0] is None: continue
            pred, probs = res
            
            # Save raw probability of class 1 (Foreground)
            prob_map = probs[0, 1, :, :].cpu().numpy() # Shape (H, W)
            fname = os.path.basename(file_path).split('.')[0]
            np.save(os.path.join(output_base, f"{fname}_probs.npy"), prob_map)
            
            # Save Mask
            mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
            
            # --- Clean the mask using Morphological logic (v8) ---
            mask = clean_mask_v8(mask)
            # ---------------------------------------------------
            
            io.imsave(os.path.join(output_base, f"{fname}_mask.tif"), mask, check_contrast=False)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--single_file", type=str, help="Process specific file only")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--wsi_dir", type=str)
    args = parser.parse_args()
    args = _parse_training_variables(args)
    predict_model(args)

if __name__ == '__main__':
    main()
