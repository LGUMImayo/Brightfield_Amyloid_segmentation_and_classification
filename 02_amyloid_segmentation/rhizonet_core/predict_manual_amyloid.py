import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from skimage import io
from monai.data import list_data_collate

# --- CRITICAL FIX: NumPy 2.0 Compatibility ---
# Just in case legacy libraries hit this issue
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
# ---------------------------------------------

# --- CHANGE: Import from unet2D_focal_amyloid ---
try:
    from unet2D_focal_amyloid import Unet2D, PredDataset2D, ImageDataset
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from unet2D_focal_amyloid import Unet2D, PredDataset2D, ImageDataset

# Fix for PyTorch 2.6+ safe globals pickling error
if hasattr(torch.serialization, "add_safe_globals"):
    try:
        torch.serialization.add_safe_globals([ImageDataset, PredDataset2D])
        print("Registered ImageDataset and PredDataset2D as safe globals for pickling.")
    except Exception as e:
        print(f"Warning: Could not register safe globals: {e}")

def save_predictions(predictions, output_dir):
    """Helper to save images and probabilities"""
    mask_dir = os.path.join(output_dir, "masks")
    prob_dir = os.path.join(output_dir, "probs")
    vis_dir = os.path.join(output_dir, "probs_vis")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Saving predictions to {output_dir}...")
    cmap = plt.get_cmap('jet')
    
    for batch_res in predictions:
        if batch_res is None: continue
        probs_batch, preds_batch, filenames_batch = batch_res
        probs_batch = probs_batch.cpu().numpy()
        preds_batch = preds_batch.cpu().numpy()
        
        for i, filename in enumerate(filenames_batch):
            fname = os.path.basename(filename).split('.')[0]
            
            # Save Mask
            pred_mask = (preds_batch[i] * 255).astype(np.uint8)
            io.imsave(os.path.join(mask_dir, f"{fname}.png"), pred_mask, check_contrast=False)
            
            # Save Raw Probs
            np.save(os.path.join(prob_dir, f"{fname}.npy"), probs_batch[i])
            
            # Save Visualization
            num_classes = probs_batch[i].shape[0]
            for c in range(num_classes):
                prob_map = probs_batch[i][c]
                
                # --- CHANGED: Save with Colorbar ---
                vis_filename = f"{fname}_class_{c}_prob.png"
                save_path = os.path.join(vis_dir, vis_filename)

                # Create a figure with a single axis
                fig, ax = plt.subplots(figsize=(6, 5))
                # Display the heatmap
                im = ax.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
                ax.axis('off') # Hide axes ticks/labels

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Probability', rotation=270, labelpad=15)

                # Save the figure
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig) # Close to free memory
                # -----------------------------------

def main():
    parser = ArgumentParser(description="Run inference manually with the Amyloid-specific RhizoNet model.")
    parser.add_argument("--config_file", type=str, required=True, 
                        help="Path to the JSON configuration file used for training.")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to the .ckpt model file.")
    parser.add_argument("--pred_data_dir", type=str, required=True, 
                        help="Root directory containing the prediction data.")
    parser.add_argument("--output_dir", type=str, default="prediction_results", 
                        help="Directory to save the results.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")

    args = parser.parse_args()

    # 1. Parse Config
    print(f"Loading configuration from {args.config_file}...")
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    dataset_params = config['dataset_params']
    model_params = config['model_params']
    
    # Ensure tuples
    dataset_params['patch_size'] = tuple(dataset_params['patch_size'])
    model_params['pred_patch_size'] = tuple(model_params['pred_patch_size'])
    
    # 2. Setup Paths
    # Check if 'images' subdir exists in pred_data_dir, otherwise assume pred_data_dir is the images folder
    images_subdir = os.path.join(args.pred_data_dir, "images")
    if os.path.isdir(images_subdir):
        pred_img_path = images_subdir
    else:
        print(f"Note: 'images' subdirectory not found in {args.pred_data_dir}. Using root as image source.")
        pred_img_path = args.pred_data_dir
        
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Initialize Model
    # We instantiate the model shell using params, then load weights
    print(f"Loading custom AMYLOID model checkpoint from {args.ckpt_path}...")
    
    # Pass None for datasets as we are only predicting
    unet = Unet2D(None, None, **model_params)

    # 4. Prepare DataLoader
    print(f"Loading data from {pred_img_path}...")
    predict_dataset = PredDataset2D(pred_img_path, dataset_params)
    
    if len(predict_dataset) == 0:
        print("Error: No images found in the prediction directory.")
        return

    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=list_data_collate, 
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # 5. Run Prediction
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpus,
        strategy='auto',
        logger=False # Disable logging for simple inference
    )
    
    print("Starting prediction with Confidence Thresholding (0.85)...")
    preds = trainer.predict(unet, dataloaders=predict_loader, ckpt_path=args.ckpt_path)
    
    # 6. Save Results
    save_predictions(preds, args.output_dir)
    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()