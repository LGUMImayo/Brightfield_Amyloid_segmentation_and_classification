import os
import sys
import numpy as np
import cv2
import torch
import torchvision
import pandas as pd
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration ---
CASE_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline2/RO1_Amyloid/Cases/9117_22/9117_22_#5_MFG_Amyloid"
FILES_DIR = os.path.join(CASE_DIR, "9117_22_#5_MFG_Amyloid_files")

# Input Directories
SEG_TILES_DIR = os.path.join(FILES_DIR, "heatmap/seg_tiles") # RGB Tissue Tiles
MASK_TILES_DIR = os.path.join(FILES_DIR, "heatmap/TAU_seg_tiles") # Binary Mask Tiles
COORDS_FILE = os.path.join(FILES_DIR, "output/RES/tiles/tile_coordinates.npy")
RES10_PATH = os.path.join(FILES_DIR, "output/RES/res10_9117_22_#5_MFG_Amyloid.tiff")

# Output Directory
OUTPUT_BASE_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline2/RO1_Amyloid/Cases/9117_22/inference_test_output"
MASKED_TILES_OUT = os.path.join(OUTPUT_BASE_DIR, "masked_tiles")
VISUALS_OUT = os.path.join(OUTPUT_BASE_DIR, "visuals")
FINAL_OVERLAY_OUT = os.path.join(OUTPUT_BASE_DIR, "final_overlay.jpg")

for d in [OUTPUT_BASE_DIR, MASKED_TILES_OUT, VISUALS_OUT]:
    os.makedirs(d, exist_ok=True)

# Model Paths
MODEL_PATH = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_plaque_ai/s311590_plaque_ai/plaque_ai/models/plaque_ai_final.pt'

# Constants
TARGET_DICT = {'cgp': 1, 'ccp': 2, 'caa 1': 3, 'caa 2': 4}
CLASS_DICT = {v: k for k, v in TARGET_DICT.items()}
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3

# Resolution Configuration
# Model was trained on 0.0002827 mm/pixel
# Current images are 1/2890 = 0.0003460 mm/pixel
MODEL_PIXEL_SIZE = 0.0002827
CURRENT_PIXEL_SIZE = 1.0 / 2890.0 # ~0.0003460
SCALE_FACTOR = CURRENT_PIXEL_SIZE / MODEL_PIXEL_SIZE # ~1.22

# Inference Parameters
MODEL_INPUT_SIZE = 1024
# We need to extract a smaller patch so that when resized to 1024, it matches the model resolution
PATCH_SIZE = int(MODEL_INPUT_SIZE / SCALE_FACTOR) # ~836 pixels
STRIDE = int(PATCH_SIZE * 0.75) # 25% overlap (or whatever is desired)

print(f"Resolution Adjustment:")
print(f"  Model Resolution: {MODEL_PIXEL_SIZE:.7f} mm/px")
print(f"  Image Resolution: {CURRENT_PIXEL_SIZE:.7f} mm/px")
print(f"  Scale Factor: {SCALE_FACTOR:.4f}")
print(f"  Extraction Patch Size: {PATCH_SIZE} (will be resized to {MODEL_INPUT_SIZE})")
print(f"  Stride: {STRIDE}")

# --- Helper Functions ---

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(path, num_classes, device):
    model = get_model(num_classes)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model

def apply_mask_to_tile(tile_rgb, mask_binary):
    """
    Instead of masking with white, we normalize the image brightness 
    to match the training data distribution (Mean ~215).
    """
    # Target mean from training data analysis
    TARGET_MEAN = 215.0
    
    # Calculate current mean of the tile (ignoring completely black pixels if any)
    current_mean = np.mean(tile_rgb)
    
    # Avoid division by zero
    if current_mean < 1: 
        return tile_rgb
        
    # Calculate scaling factor
    # We dampen it slightly so we don't over-expose (e.g., 0.8 strength)
    scale_factor = (TARGET_MEAN / current_mean) * 0.8
    
    # Apply scaling
    tile_normalized = cv2.convertScaleAbs(tile_rgb, alpha=scale_factor, beta=10)
    
    return tile_normalized

def plot_prediction(img, boxes, labels, scores, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    
    for i, box in enumerate(boxes):
        if scores[i] < SCORE_THRESHOLD:
            continue
            
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        label = labels[i]
        class_name = CLASS_DICT.get(label, str(label))
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{class_name} {scores[i]:.2f}", color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

# --- Main Execution ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    print("Loading model...")
    model = load_model(MODEL_PATH, len(TARGET_DICT) + 1, device)
    
    # 2. Load Coordinates
    print("Loading tile coordinates...")
    coords = np.load(COORDS_FILE)
    
    # 3. Transform
    # Note: We resize to 1024x1024. Since input patch is smaller (~836), this performs the necessary upscaling.
    val_transform = A.Compose([
        A.Resize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, always_apply=True), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2(transpose_mask=True)
    ])

    global_boxes = []
    global_scores = []
    global_labels = []

    print(f"Processing {len(coords)} tiles...")
    
    # --- Main Processing Loop ---
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # Define tile names based on coordinates (assuming sequential naming)
    tile_names = [f"tile_{i:04d}.tif" for i in range(len(coords))]
    
    saved_tiles_count = 0
    
    print("Starting inference on tiles...")
    
    skipped_no_mask = 0
    skipped_low_amyloid = 0
    processed_count = 0
    
    for i, tile_name in enumerate(tqdm(tile_names)):
        tile_path = os.path.join(SEG_TILES_DIR, tile_name)
        
        # Construct mask filename: tile_XXXX.tif -> tile_XXXX_mask.tif
        mask_name = tile_name.replace(".tif", "_mask.tif")
        mask_path = os.path.join(MASK_TILES_DIR, mask_name)
        
        # 0. Check Mask for Amyloid Content (>1%)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                skipped_no_mask += 1
                continue
            
            # Calculate ratio of white pixels (amyloid)
            amyloid_ratio = np.count_nonzero(mask) / mask.size
            
            if amyloid_ratio <= 0.01:
                skipped_low_amyloid += 1
                continue
        else:
            # Debug: Print first missing mask path to check filename pattern
            if skipped_no_mask == 0:
                print(f"DEBUG: Mask not found at {mask_path}")
            skipped_no_mask += 1
            continue

        processed_count += 1
        # 1. Load Tile
        if not os.path.exists(tile_path):
            continue
            
        image = cv2.imread(tile_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Preprocess (Normalize)
        # We use the mask to find tissue area if needed, but here we just apply the normalization
        # The original code used a mask to calculate mean, here we might just use the image itself 
        # or assume the whole tile is relevant. 
        # Let's look at apply_mask_to_tile implementation if available or just use image.
        # For this script, let's assume we just pass the image.
        
        # Check if we have a corresponding mask for tissue segmentation to refine normalization?
        # The prompt implies we just want to visualize the "masked_tiles" output.
        
        # Apply normalization/masking
        masked_image = apply_mask_to_tile(image)
        
        # Save first 10 tiles for verification
        if saved_tiles_count < 10:
            # Calculate non-zero ratio to skip empty background tiles
            gray_temp = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
            non_zero_ratio = cv2.countNonZero(gray_temp) / gray_temp.size
            
            if non_zero_ratio > 0.5:
                save_path = os.path.join(MASKED_TILES_OUT, f"masked_{tile_name}")
                cv2.imwrite(save_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
                saved_tiles_count += 1
        # -------------------------------------

        h, w, _ = masked_image.shape
        
        tile_boxes = []
        tile_scores = []
        tile_labels = []
        
        for y in range(0, h, STRIDE):
            for x in range(0, w, STRIDE):
                # Calculate patch coordinates
                y_start = y
                x_start = x
                y_end = y + PATCH_SIZE
                x_end = x + PATCH_SIZE
                
                # Shift back if we go out of bounds to ensure full 1024x1024 patch
                if y_end > h:
                    y_end = h
                    y_start = max(0, h - PATCH_SIZE)
                if x_end > w:
                    x_end = w
                    x_start = max(0, w - PATCH_SIZE)
                
                # Extract patch
                patch = masked_image[y_start:y_end, x_start:x_end]
                
                # Ensure patch is correct size (handle edge cases)
                # If patch is smaller than PATCH_SIZE (at edges), we still resize to 1024.
                # This might distort aspect ratio slightly at very edges, but acceptable.
                # Ideally we pad, but resize is robust.

                # Inference
                transformed = val_transform(image=patch)
                img_tensor = transformed['image'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    prediction = model(img_tensor)[0]
                    
                p_boxes = prediction['boxes'].cpu().numpy()
                p_scores = prediction['scores'].cpu().numpy()
                p_labels = prediction['labels'].cpu().numpy()
                
                # Filter low confidence
                keep = p_scores > 0.05
                p_boxes = p_boxes[keep]
                p_scores = p_scores[keep]
                p_labels = p_labels[keep]
                
                if len(p_boxes) > 0:
                    # Scale boxes back to Patch Coordinate Space
                    # Model output is in 1024x1024 space.
                    # We need to map back to 'patch.shape' space.
                    # Note: patch.shape might be smaller than PATCH_SIZE at edges.
                    
                    scale_y = patch.shape[0] / MODEL_INPUT_SIZE
                    scale_x = patch.shape[1] / MODEL_INPUT_SIZE
                    
                    p_boxes[:, 0] *= scale_x
                    p_boxes[:, 1] *= scale_y
                    p_boxes[:, 2] *= scale_x
                    p_boxes[:, 3] *= scale_y

                    # Adjust coordinates to Tile Frame
                    p_boxes[:, 0] += x_start
                    p_boxes[:, 1] += y_start
                    p_boxes[:, 2] += x_start
                    p_boxes[:, 3] += y_start
                    
                    tile_boxes.append(p_boxes)
                    tile_scores.append(p_scores)
                    tile_labels.append(p_labels)
        
        # --- Local NMS (Merge detections from overlapping patches) ---
        if len(tile_boxes) > 0:
            t_boxes = np.concatenate(tile_boxes)
            t_scores = np.concatenate(tile_scores)
            t_labels = np.concatenate(tile_labels)
            
            keep_indices = nms(torch.tensor(t_boxes), torch.tensor(t_scores), IOU_THRESHOLD)
            
            t_boxes = t_boxes[keep_indices]
            t_scores = t_scores[keep_indices]
            t_labels = t_labels[keep_indices]
            
            # Save visualization for this tile (optional, maybe just first few)
            if i < 5:
                plot_prediction(masked_image, t_boxes, t_labels, t_scores, os.path.join(VISUALS_OUT, f"pred_{tile_name}.jpg"))
            
            # --- Convert to Global Coordinates ---
            # Coords: [y1, x1, y2, x2]
            y_offset, x_offset = coords[i][0], coords[i][1]
            
            t_boxes[:, 0] += x_offset
            t_boxes[:, 1] += y_offset
            t_boxes[:, 2] += x_offset
            t_boxes[:, 3] += y_offset
            
            global_boxes.append(t_boxes)
            global_scores.append(t_scores)
            global_labels.append(t_labels)

    print(f"\nSummary:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (No Mask/Read Error): {skipped_no_mask}")
    print(f"  Skipped (Low Amyloid <1%): {skipped_low_amyloid}")
    
    # --- Global NMS ---
    print("Running Global NMS...")
    if len(global_boxes) > 0:
        all_boxes = np.concatenate(global_boxes)
        all_scores = np.concatenate(global_scores)
        all_labels = np.concatenate(global_labels)
        
        # Convert to tensor for NMS
        keep_indices = nms(torch.tensor(all_boxes), torch.tensor(all_scores), IOU_THRESHOLD)
        
        final_boxes = all_boxes[keep_indices]
        final_scores = all_scores[keep_indices]
        final_labels = all_labels[keep_indices]
        
        # Filter final score
        final_keep = final_scores > SCORE_THRESHOLD
        final_boxes = final_boxes[final_keep]
        final_scores = final_scores[final_keep]
        final_labels = final_labels[final_keep]
        
        print(f"Found {len(final_boxes)} detections after NMS.")
        
        # --- Visualization on Res10 ---
        print("Generating Final Overlay...")
        with tifffile.TiffFile(RES10_PATH) as tif:
            res10_img = tif.pages[0].asarray()
        
        # Calculate scale factor
        max_y = coords[:, 2].max()
        max_x = coords[:, 3].max()
        
        scale_y_res10 = res10_img.shape[0] / max_y
        scale_x_res10 = res10_img.shape[1] / max_x
        
        print(f"Scale factors: Y={scale_y_res10}, X={scale_x_res10}")
        
        # Plot on Res10
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(res10_img)
        
        for k in range(len(final_boxes)):
            box = final_boxes[k]
            label = final_labels[k]
            score = final_scores[k]
            
            # Scale box to Res10
            x1 = box[0] * scale_x_res10
            y1 = box[1] * scale_y_res10
            x2 = box[2] * scale_x_res10
            y2 = box[3] * scale_y_res10
            w, h = x2 - x1, y2 - y1
            
            class_name = CLASS_DICT.get(label, str(label))
            
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
        plt.axis('off')
        plt.savefig(FINAL_OVERLAY_OUT, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved final overlay to {FINAL_OVERLAY_OUT}")
        
        # Save CSV
        df = pd.DataFrame({
            'x1': final_boxes[:, 0],
            'y1': final_boxes[:, 1],
            'x2': final_boxes[:, 2],
            'y2': final_boxes[:, 3],
            'score': final_scores,
            'label': final_labels,
            'class': [CLASS_DICT.get(l, str(l)) for l in final_labels]
        })
        csv_path = os.path.join(OUTPUT_BASE_DIR, "detections.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved detections to {csv_path}")
        
    else:
        print("No detections found.")

if __name__ == "__main__":
    main()
