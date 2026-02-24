import os
import argparse
import tifffile
import zarr
import numpy as np
import torch
import cv2
import albumentations as A # Ensure A is imported
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import sys
from skimage import io
import pandas as pd # Add pandas for CSV saving

# Add path to load the Unet model definition
sys.path.append("/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet")
from rhizonet.unet2D import Unet2D

# --- Resolution Settings ---
TRAIN_MICRONS_PER_PIXEL = 0.346 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_path", type=str, required=True, help="Path to raw Amyloid TIFF")
    parser.add_argument("--mask_root_dir", type=str, required=True, help="Root directory containing GM/WM mask subfolders")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to .ckpt")
    parser.add_argument("--resolution", type=float, default=0.2827, help="Microns per pixel of the input image")
    parser.add_argument("--gm_prob_path", type=str, default=None, help="Direct path to npy file (optional override)")
    parser.add_argument("--model_input_size", type=int, default=128, help="Input size for the model (default 128)")
    # NEW FLAG
    parser.add_argument("--gm_threshold", type=float, default=0.8, help="Minimum avg GM probability to keep tile (0.0-1.0)")
    parser.add_argument("--test_mode", action="store_true", help="Save tiles and overview images for debugging")
    # NEW FLAGS FOR CLEANING
    parser.add_argument("--blur_strength", type=int, default=5, help="Kernel size for pre-processing blur (odd number, e.g. 3, 5). 0 to disable.")
    parser.add_argument("--min_grain_size", type=int, default=50, help="Minimum size (in pixels) of particles to keep in post-processing.")
    return parser.parse_args()

class TiledSlideDataset(Dataset):
    def __init__(self, tiff_path, gm_prob_map, model_input_size=128, img_res=0.2827, train_res=0.346, gm_threshold=0.8, blur_strength=0): # Default to 0
        self.tiff_path = tiff_path
        self.tiff = None
        self.use_memmap = True
        self.blur_strength = blur_strength
        
        # Initial check to get dimensions and determine loading method
        try:
            # Try memmap to get shape
            temp = tifffile.memmap(tiff_path)
            self.h, self.w = temp.shape[:2]
            # No need to keep temp open, just dimensions
        except Exception:
             # Fallback for non-memmappable images (e.g. compressed tiles) using Zarr interface
             try:
                 store = tifffile.imread(tiff_path, aszarr=True)
                 z_grp = zarr.open(store, mode='r')
                 z_img = z_grp['0'] if isinstance(z_grp, zarr.hierarchy.Group) else z_grp
                 self.h, self.w = z_img.shape[:2]
                 self.use_memmap = False
             except Exception as e:
                 raise ValueError(f"Could not load TIFF dimensions: {e}")

        self.gm_prob_map = gm_prob_map
        self.model_input_size = model_input_size
        self.gm_threshold = gm_threshold
        
        # Calculate Extraction Size
        self.scale_factor = train_res / img_res
        self.extract_size = int(model_input_size * self.scale_factor)
        self.stride = self.extract_size
        
        # Calculate mask scaling
        self.mask_h, self.mask_w = self.gm_prob_map.shape
        self.mask_scale_y = self.mask_h / self.h
        self.mask_scale_x = self.mask_w / self.w
        
        self.coordinates = []
        
        print(f"Scanning slide for Gray Matter regions...")
        print(f"  - Image Size: {self.w}x{self.h}")
        print(f"  - Loading Method: {'Memmap' if self.use_memmap else 'Zarr'}")
        print(f"  - Extraction Patch Size: {self.extract_size} -> Model Input: {model_input_size}")
        print(f"  - GM Threshold: Avg > {self.gm_threshold} AND Peak > 0.3")
        
        for y in range(0, self.h, self.stride):
            for x in range(0, self.w, self.stride):
                h_extract = min(self.extract_size, self.h - y)
                w_extract = min(self.extract_size, self.w - x)
                
                y_m = int(y * self.mask_scale_y)
                x_m = int(x * self.mask_scale_x)
                h_m = int(h_extract * self.mask_scale_y)
                w_m = int(w_extract * self.mask_scale_x)
                
                if h_m < 1 or w_m < 1: continue

                mask_patch = self.gm_prob_map[y_m:y_m+h_m, x_m:x_m+w_m]
                
                # STRICTER FILTERING (Avg > 80%, Peak > 30%)
                if np.mean(mask_patch) > self.gm_threshold and np.max(mask_patch) > 0.3:
                    self.coordinates.append({
                        'y': y, 'x': x, 
                        'h': h_extract, 'w': w_extract
                    })

        self.transform = A.Compose([
            # PRE-PROCESSING: Gaussian Blur removed/optional
            A.GaussianBlur(blur_limit=(self.blur_strength, self.blur_strength), p=1.0) if self.blur_strength > 0 else A.NoOp(),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def _ensure_tiff_open(self):
        if self.tiff is None:
            if self.use_memmap:
                try:
                    self.tiff = tifffile.memmap(self.tiff_path)
                except Exception:
                    self.use_memmap = False # Fallback if fails at worker
                    self._ensure_tiff_open()
            else:
                store = tifffile.imread(self.tiff_path, aszarr=True)
                z_grp = zarr.open(store, mode='r')
                self.tiff = z_grp['0'] if isinstance(z_grp, zarr.hierarchy.Group) else z_grp

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        self._ensure_tiff_open()
        coord = self.coordinates[idx]
        y, x, h, w = coord['y'], coord['x'], coord['h'], coord['w']
        
        patch = self.tiff[y:y+h, x:x+w]
        
        # Handle 4-channel images (RGBA) -> RGB
        if len(patch.shape) == 3 and patch.shape[2] == 4:
            patch = patch[:, :, :3]
        
        # Pad if needed
        if h < self.extract_size or w < self.extract_size:
            if len(patch.shape) == 3:
                padded = np.zeros((self.extract_size, self.extract_size, patch.shape[2]), dtype=patch.dtype)
                padded[:h, :w, :] = patch
            else:
                padded = np.zeros((self.extract_size, self.extract_size), dtype=patch.dtype)
                padded[:h, :w] = patch
            patch = padded
            
        # Resize inputs to model size
        patch_resized = cv2.resize(patch, (self.model_input_size, self.model_input_size), interpolation=cv2.INTER_LINEAR)
        
        # Preprocess (now includes Blur)
        augmented = self.transform(image=patch_resized)
        
        # Return original resized patch for debugging visibility (numpy format)
        return augmented['image'], patch_resized, y, x, h, w

def load_gm_mask(mask_root, image_name, override_path=None):
    if override_path and os.path.exists(override_path):
        data = np.load(override_path, mmap_mode='r')
        return data[1] 

    # FAST FIX: Ensure we only use the filename, not the full path
    base_name = os.path.basename(image_name)
    image_id = os.path.splitext(base_name)[0]
    
    # Try exact folder match
    search_path = os.path.join(mask_root, image_id, "prediction_masks", f"{image_id}_probabilities.npy")
    
    if os.path.exists(search_path):
        data = np.load(search_path, mmap_mode='r')
        return data[1] 
    else:
        # Debugging aid
        print(f"DEBUG: Searched for mask at: {search_path}")
        raise FileNotFoundError(f"Could not find mask for {image_id}")

def save_debug_tile(save_dir, fname, y, x, image_np, mask_cleaned, prob_map_blur, image_blurred, prob_map_sharp=None):
    """
    Saves a 5-panel debug image:
    1. Original RGB (Sharp)
    2. Prediction on Sharp
    3. Blurred RGB (Actual Input)
    4. Prediction on Blurred (Actual Output)
    5. Final Cleaned Mask
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # 1. Original RGB
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original RGB\n(y={y}, x={x})")
    axes[0].axis('off')
    
    # 2. Prediction on Sharp (If available)
    if prob_map_sharp is not None:
        axes[1].imshow(prob_map_sharp, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Pred on SHARP\n(Shows Noise Sensitivity)")
    else:
        axes[1].text(0.5, 0.5, "N/A", ha='center')
    axes[1].axis('off')

    # 3. Blurred RGB
    axes[2].imshow(image_blurred)
    axes[2].set_title("Blurred Input\n(What Model Saw)")
    axes[2].axis('off')

    # 4. Prediction on Blurred
    axes[3].imshow(prob_map_blur, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title("Pred on BLURRED\n(Actual Prob Map)")
    axes[3].axis('off')

    # 5. Final Mask (Cleaned)
    axes[4].imshow(mask_cleaned, cmap='gray')
    axes[4].set_title("Final Cleaned Mask")
    axes[4].axis('off')

    out_path = os.path.join(save_dir, 'debug_tiles', fname, f"tile_{y}_{x}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    fname = os.path.basename(args.tiff_path).split('.')[0]
    final_out_path = os.path.join(args.output_dir, f"{fname}_amyloid_seg.tif")
    stats_out_path = os.path.join(args.output_dir, f"{fname}_stats.csv")
    
    # 1. Load GM Mask
    print(f"Loading Gray Matter Mask for {fname}...")
    try:
        # CORRECTED: Pass the basename 'fname' instead of full 'args.tiff_path' for clarity, 
        # though the fix inside load_gm_mask handles it either way now.
        gm_prob = load_gm_mask(args.mask_root_dir, fname, args.gm_prob_path)
    except Exception as e:
        print(f"Skipping {fname}: {e}")
        return

    # 2. Setup Dataset
    print(f"Initializing Dataset...")
    dataset = TiledSlideDataset(
        args.tiff_path, 
        gm_prob, 
        model_input_size=args.model_input_size, 
        img_res=args.resolution, 
        train_res=TRAIN_MICRONS_PER_PIXEL,
        gm_threshold=args.gm_threshold,
        blur_strength=args.blur_strength
    )

    # --- MOVED & CORRECTED: CALCULATE GM AREA ---
    # We must calculate this AFTER dataset init to access the scaling factors
    print("Calculating Gray Matter Area from mask...")
    raw_gm_pixel_count = np.sum(gm_prob > 0.5)
    
    # Scale the count up to Level 0 dimensions
    # Area_L0 = Area_Mask / (Scale_X * Scale_Y)
    # If mask is 1/10th size, Scale is 0.1. Divisor is 0.01. Area scales up by 100.
    correction_factor = 1.0 / (dataset.mask_scale_x * dataset.mask_scale_y)
    gm_pixel_count = raw_gm_pixel_count * correction_factor
    
    print(f"  - Raw Mask Pixels: {raw_gm_pixel_count}")
    print(f"  - Correction Factor: {correction_factor:.2f}")
    print(f"  - Adjusted Level-0 GM Pixels: {gm_pixel_count:.0f}")

    if len(dataset) == 0:
        print("No significant Gray Matter found.")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # 3. Load Model
    print(f"Loading Model from {args.model_path}...")
    model = Unet2D.load_from_checkpoint(args.model_path)
    model.to(device)
    model.eval()

    # 4. Prepare Output Storage
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    temp_zarr_path = os.path.join(args.output_dir, f"temp_{fname}.zarr")
    z_out = zarr.open(temp_zarr_path, mode='w', shape=(dataset.h, dataset.w), chunks=(4096, 4096), dtype='uint8', compressor=compressor)
    
    amyloid_pixel_count = 0 # Initialize counter

    # For rescaling functionality
    ds_factor = 0.1 # 10%
    new_h, new_w = int(dataset.h * ds_factor), int(dataset.w * ds_factor)
    
    # For 10% overview, we can't easily build it iteratively due to threading, 
    # but we can do a quick load + resize at the end using the Zarr and Original.
    
    # Define a transform for strict normalization WITHOUT blur (for the debug "sharp" inference)
    # Assuming standard ImageNet normalization used in training
    transform_sharp = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    print("Starting Inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Unpack (updated dataset return signature)
            images_tensor, images_np, ys, xs, hs, ws = batch
            
            images_tensor = images_tensor.to(device)
            
            logits = model(images_tensor)
            probs = torch.softmax(logits, dim=1)
            preds_mini = probs[:, 1, :, :] 
            
            # --- NEW: Get raw probability map ---
            preds_prob_np = preds_mini.cpu().numpy()
            
            # 1. Initial Thresholding
            preds_mask = (preds_mini > 0.95).float().cpu().numpy()
            
            # Convert images_np to numpy for saving
            images_np = images_np.numpy()
            
            for i in range(len(ys)):
                y, x = ys[i].item(), xs[i].item()
                h, w = hs[i].item(), ws[i].item() 
                
                # Get prediction
                p_mask = preds_mask[i]
                p_prob = preds_prob_np[i]

                # --- CHANGE START: Move Cleanup AFTER Resize ---
                
                # 1. Resize FIRST to final resolution
                # Use Linear + Threshold for smoother edges than Nearest
                p_mask_upscaled = cv2.resize(p_mask, (dataset.extract_size, dataset.extract_size), interpolation=cv2.INTER_LINEAR)
                p_mask_upscaled = (p_mask_upscaled > 0.5).astype(np.float32)

                # 2. Apply Cleanup on the FINAL scale mask
                if args.min_grain_size > 0:
                    p_mask_uint8 = (p_mask_upscaled * 255).astype(np.uint8)
                    
                    # --- RESTORED: Morphological Opening (Resulting in rounded blobs) ---
                    # Use (5, 5) kernel as used in V3
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    p_mask_uint8 = cv2.morphologyEx(p_mask_uint8, cv2.MORPH_OPEN, kernel)

                    # Connected Components Filter (Size based)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(p_mask_uint8, connectivity=8)
                    sizes = stats[:, cv2.CC_STAT_AREA]
                    
                    component_map = sizes[labels]
                    
                    p_mask_cleaned = np.zeros_like(p_mask_uint8)
                    p_mask_cleaned[(component_map >= args.min_grain_size) & (labels != 0)] = 255
                    
                    p_mask_upscaled = (p_mask_cleaned > 127).astype(np.float32)

                
                # SAVE DEBUG TILE if test mode
                if args.test_mode and i == 0: 
                    # 1. Generate the "Blurred" RGB image manually for viz (only if blur > 0)
                    if args.blur_strength > 0:
                        k_size = args.blur_strength
                        if k_size % 2 == 0: k_size += 1
                        debug_input_img = cv2.GaussianBlur(images_np[i], (k_size, k_size), 0)
                        input_title = "Blurred Input"
                    else:
                        debug_input_img = images_np[i]
                        input_title = "Original Input (No Blur)"

                    # 2. Generate the "Pre-Blur" (Sharp) Heatmap
                    # We run inference again on the raw image
                    sharp_tensor = transform_sharp(image=images_np[i])['image'].unsqueeze(0).to(device)
                    sharp_logits = model(sharp_tensor)
                    sharp_prob = torch.softmax(sharp_logits, dim=1)[:, 1, :, :].cpu().numpy()[0]
                    
                    # 3. Resize Cleaned Mask to match image size for visualization
                    viz_mask = cv2.resize(p_mask_upscaled, (images_np[i].shape[1], images_np[i].shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Save the 5-panel debug
                    save_debug_tile(
                        args.output_dir, fname, y, x, 
                        image_np=images_np[i],          # Original
                        mask_cleaned=viz_mask,          # Mask 
                        prob_map_blur=p_prob,           # Actual Model Output
                        image_blurred=debug_input_img,  # Actual Input
                        prob_map_sharp=sharp_prob       # Same as Actual Output if blur is 0
                    )

                valid_pred = p_mask_upscaled[:h, :w]
                
                # Write to Zarr
                z_out[y:y+h, x:x+w] = (valid_pred * 255).astype('uint8')
                
                # Accumulate stats
                amyloid_pixel_count += np.sum(valid_pred)
    
    print("Inference Complete.")
    
    # --- SAVE STATISTICS ---
    print("Calculating final statistics...")
    # Resolution squared (microns^2)
    pixel_area_um2 = args.resolution * args.resolution
    
    # Areas in mm^2 (1 mm^2 = 1,000,000 um^2)
    gm_area_mm2 = (gm_pixel_count * pixel_area_um2) / 1_000_000
    amyloid_area_mm2 = (amyloid_pixel_count * pixel_area_um2) / 1_000_000
    
    amyloid_density_percent = (amyloid_area_mm2 / gm_area_mm2 * 100) if gm_area_mm2 > 0 else 0
    
    stats_data = {
        'slide_id': [fname],
        'gm_pixel_count': [gm_pixel_count],
        'amyloid_pixel_count': [amyloid_pixel_count],
        'gm_area_mm2': [gm_area_mm2],
        'amyloid_area_mm2': [amyloid_area_mm2],
        'amyloid_density_percent': [amyloid_density_percent]
    }
    
    df = pd.DataFrame(stats_data)
    df.to_csv(stats_out_path, index=False)
    print(f"Stats saved to {stats_out_path}")
    print(f"  - GM Area: {gm_area_mm2:.2f} mm^2")
    print(f"  - Amyloid Area: {amyloid_area_mm2:.4f} mm^2")
    print(f"  - Density: {amyloid_density_percent:.4f}%")

    print("Saving BigTIFF...")

    # Cleanup memory before heavy save operation
    del model
    del dataloader
    del dataset
    if 'images_tensor' in locals(): del images_tensor
    if 'logits' in locals(): del logits
    if 'probs' in locals(): del probs
    if 'preds_mini' in locals(): del preds_mini
    if 'preds_mask' in locals(): del preds_mask
    import gc
    gc.collect()
    
    # Save Full Resolution BigTIFF
    try:
        tifffile.imwrite(
            final_out_path, 
            z_out, 
            bigtiff=True, 
            tile=(512, 512), 
            compression='zlib'
        )
    except Exception as e:
        print(f"Error saving BigTIFF: {e}")
        # Try a slower but more robust approach using a generator if OOM happens here
        # But usually tifffile handles zarr stores fine.
        raise e
    
    # --- 5. Generate 10% Overviews ---
    if args.test_mode:
        print("Generating 10% overview images...")
        
        # Resize logic: We can't load the full 60GB image into RAM.
        # We process in chunks or rely on zarr resizing if available, or just subsample.
        # Fast way: Read original TIFF at pyramid level 2 or 3 if available. 
        # Since it's a raw tiff, we might have to read stride.
        
        # Resizing the MASK (easy, read form zarr with step)
        step = int(1/ds_factor) # Step 10
        overview_mask = z_out[::step, ::step]
        cv2.imwrite(os.path.join(args.output_dir, f"{fname}_mask_overview_10pct.png"), overview_mask)
        
        # Resizing the ORIGINAL (read with step)
        # Note: tifffile.imread usually reads whole thing. Using memory mapping.
        try:
            try:
                memmap_tiff = tifffile.memmap(args.tiff_path)
            except:
                store = tifffile.imread(args.tiff_path, aszarr=True)
                z_grp = zarr.open(store, mode='r')
                memmap_tiff = z_grp['0'] if isinstance(z_grp, zarr.hierarchy.Group) else z_grp

            overview_img = memmap_tiff[::step, ::step]
            # Convert RGB <-> BGR for opencv if needed. Usually TIFF is RGB.
            if len(overview_img.shape) == 3 and overview_img.shape[2] == 3:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_RGB2BGR)
            elif len(overview_img.shape) == 3 and overview_img.shape[2] == 4:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_RGBA2BGR)

            cv2.imwrite(os.path.join(args.output_dir, f"{fname}_original_overview_10pct.png"), overview_img)
        except Exception as e:
            print(f"Could not create original image overview: {e}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_zarr_path)
    print(f"Saved: {final_out_path}")

if __name__ == "__main__":
    main()