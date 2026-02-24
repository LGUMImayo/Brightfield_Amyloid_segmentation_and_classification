"""
Script for training and evaluation a residual U-Net model for image segmentation.
The training process of RhizoNet includes logging, checkpointing and metrics evaluation.

Dependencies:
- PyTorch Lightning
- MONAI
- Scikit-image
- wandb

Usage:
    pip install rhizonet
    train_rhizonet --config_file ./setup_files/setup-train.json --gpus 2 --strategy "ddp" --accelerator "gpu"
"""

import os
import json
import csv
import numpy as np
from argparse import ArgumentParser
import torch
import glob
from tqdm import tqdm
import pytorch_lightning as pl
from skimage import io, color
from argparse import Namespace
from pathlib  import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set wandb to offline mode to prevent network errors on the cluster
os.environ["WANDB_MODE"] = "offline"

from monai.data import list_data_collate
from lightning.pytorch.loggers import WandbLogger
import wandb


# Import parent modules
try:
    from .utils import MapImage, createBinaryAnnotation, get_image_paths
    from .metrics import evaluate
    from .unet2D import Unet2D, ImageDataset, PredDataset2D
    from .simpleLogger import mySimpleLogger
except ImportError:
    from utils import MapImage, createBinaryAnnotation, get_image_paths
    from metrics import evaluate
    from unet2D import Unet2D, ImageDataset, PredDataset2D
    from simpleLogger import mySimpleLogger



def _parse_training_variables(argparse_args):
    """ 
    Parse and merge training variables from a JSON configuration file and command-line arguments.

    Args:
        argparse_args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        tuple: Updated arguments, dataset parameters, and model parameters. 
    """
    args = vars(argparse_args)

    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)

    dataset_args, model_args = args['dataset_params'], args['model_params']
    dataset_args['patch_size'] = tuple(dataset_args['patch_size'])  # tuple expected, not list
    model_args['pred_patch_size'] = tuple(model_args['pred_patch_size'])  # tuple expected, not list

    return args, dataset_args, model_args


def save_predictions(predictions, output_dir):
    """Helper to save images and probabilities"""
    mask_dir = os.path.join(output_dir, "masks")
    prob_dir = os.path.join(output_dir, "probs")
    vis_dir = os.path.join(output_dir, "probs_vis")
    
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Saving predictions to {output_dir}...")
    
    # Get colormap once
    cmap = plt.get_cmap('jet')
    
    for batch_res in predictions:
        # Unpack the tuple returned by predict_step
        probs_batch, preds_batch, filenames_batch = batch_res
        
        probs_batch = probs_batch.cpu().numpy()
        preds_batch = preds_batch.cpu().numpy()
        
        for i, filename in enumerate(filenames_batch):
            fname = os.path.basename(filename).split('.')[0]
            
            # Save Mask (.png)
            # Multiply by 255 to make binary mask visible (0=Black, 255=White)
            pred_mask = (preds_batch[i] * 255).astype(np.uint8)
            io.imsave(os.path.join(mask_dir, f"{fname}.png"), pred_mask, check_contrast=False)
            
            # Save Full Probability Map (.npy)
            np.save(os.path.join(prob_dir, f"{fname}.npy"), probs_batch[i])
            
            # Save Probability Visualization (.png) for each class
            num_classes = probs_batch[i].shape[0]
            for c in range(num_classes):
                prob_map = probs_batch[i][c] # Shape (H, W), values 0-1
                
                # Apply colormap: returns (H, W, 4) RGBA float 0-1
                heatmap = cmap(prob_map)
                
                # Convert to uint8
                heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                
                # Save
                vis_filename = f"{fname}_class_{c}_prob.png"
                io.imsave(os.path.join(vis_dir, vis_filename), heatmap_uint8, check_contrast=False)

def train_model(args):
    """
    Train and evaluate RhizoNet on a specified dataset.

    Args:
        args (Namespace): Command-line containing: 
            - config_file (str): Path to the JSON Configuration of the model.
            - gpus (int): Number of gpu nodes to use training.
            - strategy (str): Strategy to use for training (e.g., 'ddp', 'dp')
            - accelerator (str): cpu or gpu training
    
    Returns: None

    Notes:
        - The evaluation results are saved to the file specified by 'save_path' in the configuration file 
        - Training and validation metrics are also available in the WandDB project 'rhizonet'
        - Predictions associated to the full size images specified in 'pred_data_dir' are generated after training and saved in the 'save_path' directory.
        - Metrics (accuracy, precision, recall and IOU) are evaluated on full size images specified in 'pred_data_dir' and results are saved in a metrics.json files 

    Example::
        Run this script using the following command-line if 2 GPU nodes available: 
            python train.py --gpus 2 --strategy "ddp" --config_file "./setup_files/setup-train.json"
        
        Run this script using the following command-line if 1 GPU node available: 
            python train.py --gpus 1 --strategy "dp" --config_file "./setup_files/setup-train.json"
    """


    # Load image and label filepaths 
    args, dataset_params, model_params = _parse_training_variables(args)
    image_dir, label_dir, log_dir = model_params['image_dir'], model_params['label_dir'], model_params['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Save json parameters to log directory
    with open(os.path.join(log_dir, 'training_parameters.json'), 'w') as f:
        json.dump(args, f)

    # --- MODIFIED DATA PAIRING LOGIC ---
    print("Finding and pairing images and labels...")
    
    all_images = get_image_paths(image_dir)
    data_pairs = []
    
    for img_path in tqdm(all_images):
        # From: patch_..._img_..._crop-....tif
        # To:   patch_..._mask_img_..._crop-....png
        img_basename = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_basename)[0]
        
        label_name = img_name_no_ext.replace('_img_', '_mask_img_') + ".png"
        label_path = os.path.join(label_dir, label_name)
        
        # Only add the pair if the corresponding label exists and is valid
        if os.path.isfile(label_path):
            # Check if file is empty
            if os.path.getsize(label_path) == 0:
                print(f"Skipping empty label file: {label_path}")
                continue
            
            # Check if file can be opened (not corrupt)
            try:
                # Use PIL to verify image integrity without fully loading if possible, 
                # but io.imread is what failed, so let's test with that or just try-except block
                # Using PIL verify is faster
                from PIL import Image
                with Image.open(label_path) as img:
                    img.verify() 
                data_pairs.append((img_path, label_path))
            except Exception as e:
                print(f"Skipping corrupt label file: {label_path} - Error: {e}")

    print(f"Found {len(data_pairs)} matching image-label pairs.")

    # Shuffle the list of pairs
    np.random.shuffle(data_pairs)

    # Unzip back into separate, shuffled lists
    if not data_pairs:
        raise ValueError("No matching image-label pairs were found. Please check your file naming convention and paths.")
    images_shuffled, labels_shuffled = zip(*data_pairs)

    # Split data into training, validation and test sets
    train_len, val_len, test_len = np.cumsum(np.round(len(images_shuffled) * np.array(dataset_params['data_split'])).astype(int))

    train_images = list(images_shuffled[:train_len])
    train_labels = list(labels_shuffled[:train_len])
    val_images = list(images_shuffled[train_len:val_len])
    val_labels = list(labels_shuffled[train_len:val_len])
    test_images = list(images_shuffled[val_len:])
    test_labels = list(labels_shuffled[val_len:])
    # --- END MODIFIED LOGIC ---

    # --- DEBUG: Print a few pairs to verify matching ---
    print("\n--- Verifying Data Splits ---")
    print("Training Set Pairs (first 3):")
    for i in range(min(3, len(train_images))):
        print(f"  Image: {os.path.basename(train_images[i])}")
        print(f"  Label: {os.path.basename(train_labels[i])}\n")

    print("Validation Set Pairs (first 3):")
    for i in range(min(3, len(val_images))):
        print(f"  Image: {os.path.basename(val_images[i])}")
        print(f"  Label: {os.path.basename(val_labels[i])}\n")
    
    print("Test Set Pairs (first 3):")
    for i in range(min(3, len(test_images))):
        print(f"  Image: {os.path.basename(test_images[i])}")
        print(f"  Label: {os.path.basename(test_labels[i])}\n")
    print("--- End Verification ---\n")
    # --- END DEBUG ---
    
    # Create datasets
    train_dataset = ImageDataset(train_images, train_labels, dataset_params, training=True)
    val_dataset = ImageDataset(val_images, val_labels, dataset_params, )
    test_dataset = ImageDataset(test_images, test_labels, dataset_params, )

    # Initialize Lightning module
    unet = Unet2D(train_dataset, val_dataset, **model_params)

    # Set up logging and callbacks
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(log_model="all",
                               project="rhizonet",
                               id=os.getenv("WANDB_RUN_ID"),
                               save_dir=log_dir) # <-- ADD THIS LINE

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_dir,
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        every_n_epochs=1,
        save_weights_only=True,
        verbose=True,
        monitor="val_loss",
        mode='min')
        
    # Add a second callback to always save the latest model
    last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_dir,
        filename="last-checkpoint",
        save_top_k=1,
        save_last=True, # This is the key parameter
        verbose=True,
    )

    stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=1e-3,
                                                   patience=30,
                                                   verbose=True,
                                                   mode='min')

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=False)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        # precision="16-mixed", 
        # accumulate_grad_batches=2, # for batch size 2, add these 2 lines if memory issues
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, last_checkpoint_callback, stopping_callback],
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=wandb_logger,
        accelerator=args['accelerator'],
        devices=args['gpus'],
        strategy=args['strategy'],
        num_sanity_val_steps=0,
        max_epochs=model_params['nb_epochs']
    )

    # Train the model
    trainer.fit(unet)

    # --- Create Test DataLoader ---
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=model_params['batch_size'], 
        shuffle=False,
        collate_fn=list_data_collate, 
        num_workers=model_params['num_workers'],
        persistent_workers=model_params['num_workers'] > 0, 
        pin_memory=torch.cuda.is_available()
    )

    # Test the model (Best)
    # Explicitly pass the dataloader since Unet2D doesn't define test_dataloader()
    trainer.test(unet, dataloaders=test_loader)
    
    # --- PREPARE PREDICTION DATALOADER ---
    # Create a dataloader specifically for the prediction set
    pred_img_path = os.path.join(model_params['pred_data_dir'], "images")
    pred_lab_path = os.path.join(model_params['pred_data_dir'], "labels")
    
    predict_dataset = PredDataset2D(pred_img_path, dataset_params)
    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, 
        batch_size=1, # Process one by one for safety/simplicity in saving
        shuffle=False,
        collate_fn=list_data_collate, 
        num_workers=model_params["num_workers"],
        persistent_workers=True if model_params["num_workers"] > 0 else False,
        pin_memory=torch.cuda.is_available()
    )

    # --- 1. PREDICT WITH BEST MODEL ---
    print("Running prediction with BEST model...")
    # 'ckpt_path="best"' automatically loads the best checkpoint from ModelCheckpoint callback
    best_preds = trainer.predict(unet, dataloaders=predict_loader, ckpt_path="best")
    
    best_save_path = os.path.join(args['log_dir'], "predictions_best")
    save_predictions(best_preds, best_save_path)
    
    # Evaluate Best Model
    print("Evaluating BEST model metrics...")
    # Check if ground truth labels exist for evaluation
    if os.path.exists(pred_lab_path) and len(os.listdir(pred_lab_path)) > 0:
        try:
            if dataset_params['binary_preds']:
                evaluate(os.path.join(best_save_path, "masks"), pred_lab_path, best_save_path, task='binary', num_classes=2, frg_class=dataset_params['frg_class'])
            else:
                evaluate(os.path.join(best_save_path, "masks"), pred_lab_path, best_save_path, task='multiclass', num_classes=len(dataset_params['class_values']), frg_class=dataset_params['frg_class'])
        except Exception as e:
            print(f"Evaluation failed: {e}")
    else:
        print("No ground truth labels found for prediction set. Skipping evaluation.")
    
    # --- 2. PREDICT WITH LAST MODEL ---
    # The filename defined in ModelCheckpoint was "last-checkpoint"
    # PL adds .ckpt extension automatically
    last_ckpt_path = os.path.join(args['log_dir'], "last-checkpoint.ckpt")
    
    if os.path.exists(last_ckpt_path):
        print(f"Running prediction with LAST model ({last_ckpt_path})...")
        last_preds = trainer.predict(unet, dataloaders=predict_loader, ckpt_path=last_ckpt_path)
        
        last_save_path = os.path.join(args['log_dir'], "predictions_last")
        save_predictions(last_preds, last_save_path)
        
        # Evaluate Last Model
        print("Evaluating LAST model metrics...")
        if os.path.exists(pred_lab_path) and len(os.listdir(pred_lab_path)) > 0:
            try:
                if dataset_params['binary_preds']:
                    evaluate(os.path.join(last_save_path, "masks"), pred_lab_path, last_save_path, task='binary', num_classes=2, frg_class=dataset_params['frg_class'])
                else:
                    evaluate(os.path.join(last_save_path, "masks"), pred_lab_path, last_save_path, task='multiclass', num_classes=len(dataset_params['class_values']), frg_class=dataset_params['frg_class'])
            except Exception as e:
                print(f"Evaluation failed: {e}")
    else:
        print(f"Last checkpoint not found at {last_ckpt_path}")

def main():
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--config_file", type=str,
                        default="../data/setup_files/setup-train.json",
                        help="json file training data parameters")
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus to use")
    parser.add_argument("--strategy", type=str, default='ddp', help="pytorch strategy")
    parser.add_argument("--accelerator", type=str, default='gpu', help="cpu or gpu accelerator")

    args = parser.parse_args()

    train_model(args)

if __name__ == "__main__":
    main()

