import os
import sys
import argparse
import random
import numpy as np
import tifffile
import zarr
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION (Matches run_amyloid_thin_inference.sh) ---
TIFF_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/tiff/NA01-161_H02P23TB517P-193924_level-0.tiff"
MASK_ROOT = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/prediction_priority_threshold_wmw_1.5_gmt_0.2_wmt_0.2_clean_21_best_parameter"
MODEL_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_CNN/RO1_Amyloid_testing/log_new_data_with_edges/last.ckpt"
OUTPUT_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Ground_Truth_Prep"

# Add path to load the Unet model definition
sys.path.append("/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet")
# Also add the inner folder to path so 'unet2D' module can be found if checkingpoint used short import
sys.path.append("/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet")

# Defer import to avoid hang during startup check
# from rhizonet.unet2D import Unet2D

def load_gm_mask(mask_root, image_name):
    """
    Loads the Gray Matter probability mask for the given image.
    Follows the structure expected by the pipeline:
    root / image_id / prediction_masks / image_id_probabilities.npy
    """
    base_name = os.path.basename(image_name)
    image_id = os.path.splitext(base_name)[0]
    
    # Construct path
    search_path = os.path.join(mask_root, image_id, "prediction_masks", f"{image_id}_probabilities.npy")
    
    if os.path.exists(search_path):
        print(f"Loading GM mask from: {search_path}")
        data = np.load(search_path, mmap_mode='r')
        return data[1] # Return the GM channel (assuming channel 1 is GM)
    else:
        raise FileNotFoundError(f"Could not find mask for {image_id} at {search_path}")

def get_valid_cornes(tiff_path, mask_root, num_crops=5, crop_size=500):
    """
    Finds random top-left coordinates (y, x) for crops that are within the Gray Matter area.
    """
    # 1. Load GM Mask
    gm_mask = load_gm_mask(mask_root, tiff_path)
    mask_h, mask_w = gm_mask.shape
    
    # 2. Get Image Dimensions
    try:
        temp = tifffile.memmap(tiff_path)
        img_h, img_w = temp.shape[:2]
        del temp
    except Exception as e:
         print(f"Memmap failed, falling back to zarr wrapper: {e}")
         store = tifffile.imread(tiff_path, aszarr=True)
         z_grp = zarr.open(store, mode='r')
         z_img = z_grp['0'] if isinstance(z_grp, zarr.hierarchy.Group) else z_grp
         img_h, img_w = z_img.shape[:2]

    print(f"Image Dimensions: {img_w}x{img_h}")
    print(f"Mask Dimensions: {mask_w}x{mask_h}")

    # 3. Calculate Scaling (Mask to Image)
    scale_y = mask_h / img_h
    scale_x = mask_w / img_w
    
    valid_coords = []
    
    print(f"Scanning for {num_crops} valid {crop_size}x{crop_size} regions (GM Probability > 0.8)...")
    
    attempts = 0
    max_attempts = 2000
    
    while len(valid_coords) < num_crops and attempts < max_attempts:
        attempts += 1
        
        # Pick random top-left
        y = random.randint(0, img_h - crop_size)
        x = random.randint(0, img_w - crop_size)
        
        # Project to Mask Coordinates
        my = int(y * scale_y)
        mx = int(x * scale_x)
        mh = int(crop_size * scale_y)
        mw = int(crop_size * scale_x)
        
        # Boundary check on mask
        if my+mh >= mask_h or mx+mw >= mask_w:
            continue
            
        mask_patch = gm_mask[my:my+mh, mx:mx+mw]
        
        # Check if this area is Gray Matter
        # Criteria: Average probability > 0.8 to ensure we are well inside GM
        if np.mean(mask_patch) > 0.8:
            valid_coords.append((y, x))
            print(f"  [Match {len(valid_coords)}/{num_crops}] Found valid GM region at y={y}, x={x}")
            
    if len(valid_coords) < num_crops:
        print(f"Warning: Only found {len(valid_coords)} valid crops after {max_attempts} attempts.")
        
    return valid_coords, (img_h, img_w)

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Get Coordinates
    crops, (img_h, img_w) = get_valid_cornes(TIFF_PATH, MASK_ROOT, num_crops=5, crop_size=500)
    
    if not crops:
        print("No crops found. Check mask paths and image alignment.")
        return

    # 2. Open Image Access
    try:
        tiff_access = tifffile.memmap(TIFF_PATH)
    except:
        store = tifffile.imread(TIFF_PATH, aszarr=True)
        z_grp = zarr.open(store, mode='r')
        tiff_access = z_grp['0'] if isinstance(z_grp, zarr.hierarchy.Group) else z_grp

    # 3. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    from rhizonet.unet2D import Unet2D
    # Set weights_only=False to support legacy checkpoints with embedded classes
    model = Unet2D.load_from_checkpoint(MODEL_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # 4. Define Transform
    # Standard normalization used in this pipeline
    transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    print("Processing Tiles...")
    
    for i, (y, x) in enumerate(crops):
        crop_name = f"tile_{i:02d}_y{y}_x{x}"
        
        # A. Extract Raw Image Tile (500x500)
        img_patch = tiff_access[y:y+500, x:x+500]
        
        # Handle RGBA -> RGB if necessary
        if len(img_patch.shape) == 3 and img_patch.shape[2] == 4:
            img_patch = img_patch[:, :, :3]
            
        # Save Original Tile
        tif_save_path = os.path.join(OUTPUT_DIR, f"{crop_name}.tif")
        tifffile.imwrite(tif_save_path, img_patch)
        
        # B. Run Inference
        # Resize to 512x512 for smooth model processing 
        input_size = 512
        img_resized = cv2.resize(img_patch, (input_size, input_size))
        
        # Preprocess
        augmented = transform(image=img_resized)
        tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :] # Class 1 = Amyloid
            
            # --- Cleaned Pipeline Logic (Matches predict_amyloid_thin_slides.py) ---
            # 1. Initial Thresholding (> 0.95)
            preds_mask_mini = (probs > 0.95).float().cpu().numpy()[0]
            
        prob_map = probs.cpu().numpy()[0]
        
        # Resize back to 500x500 for perfect alignment with ground truth tile
        prob_map_original_size = cv2.resize(prob_map, (500, 500))
        
        # --- Apply Cleanup (Resize Mask -> Morph -> Filter) ---
        # 2. Resize Binary Mask to Final Size (Linear + Threshold for smooth edges)
        pred_mask_final = cv2.resize(preds_mask_mini, (500, 500), interpolation=cv2.INTER_LINEAR)
        pred_mask_final = (pred_mask_final > 0.5).astype(np.float32)
        
        # 3. Morphological Opening (5x5 Ellipse)
        p_mask_uint8 = (pred_mask_final * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        p_mask_uint8 = cv2.morphologyEx(p_mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # 4. Connected Components Filter (Size < 100)
        min_grain = 100
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(p_mask_uint8, connectivity=8)
        sizes = stats[:, cv2.CC_STAT_AREA]
        component_map = sizes[labels]
        
        p_mask_cleaned = np.zeros_like(p_mask_uint8)
        p_mask_cleaned[(component_map >= min_grain) & (labels != 0)] = 255

        # Save Raw Prediction (NPY)
        npy_save_path = os.path.join(OUTPUT_DIR, f"{crop_name}_prob.npy")
        np.save(npy_save_path, prob_map_original_size)
        
        # Save Raw Probability Viz
        viz_path = os.path.join(OUTPUT_DIR, f"{crop_name}_prob_viz.png")
        cv2.imwrite(viz_path, (prob_map_original_size * 255).astype(np.uint8))
        
        # Save Cleaned Prediction Viz (Matches Pipeline Output)
        cleaned_path = os.path.join(OUTPUT_DIR, f"{crop_name}_pred_cleaned.png")
        cv2.imwrite(cleaned_path, p_mask_cleaned)
        
        print(f"  Processed {crop_name}")
        print(f"    - Saved Tiff: {os.path.basename(tif_save_path)}")
        print(f"    - Saved Prob NPY: {os.path.basename(npy_save_path)}")
        print(f"    - Saved Cleaned Pred: {os.path.basename(cleaned_path)}")

    print(f"\nDone! Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
