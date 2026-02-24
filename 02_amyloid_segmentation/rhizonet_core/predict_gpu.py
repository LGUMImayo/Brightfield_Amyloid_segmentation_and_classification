"""
Script for running inference using a pre-trained residual U-Net model for image segmentation.
Optimized for GPU execution.

Usage:
    pip install rhizonet
    predict_rhizonet --config_file ./setup_files/setup-predict.json 
"""

import os
import glob
import argparse
import json
import re
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, util, color, filters, exposure, measure
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 # Import cv2 for fast morphology
from skimage.color import rgb2lab

from monai.data import ArrayDataset, create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    ScaleIntensityRange,
    EnsureType
)

from PIL import ImageDraw
import torchvision.transforms.functional as TF
from datetime import datetime

from typing import Tuple, List, Dict, Sequence, Union, Any
from collections.abc import Callable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Import parent modules 
try:
    from .utils import MapImage, createBinaryAnnotation, extract_largest_component_bbox_image
    from .unet2D import Unet2D, PredDataset2D, ImageDataset, tiff_reader, dynamic_scale
except ImportError:
    from utils import MapImage, createBinaryAnnotation, extract_largest_component_bbox_image
    from unet2D import Unet2D, PredDataset2D, ImageDataset, tiff_reader, dynamic_scale


def _parse_training_variables(argparse_args) -> Dict:
    """ 
    Parse and merge training variables from a JSON configuration file and command-line arguments.

    Args:
        argparse_args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        Dict: Updated arguments 
    """


    args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    args['pred_patch_size'] = tuple(args['pred_patch_size']) # tuple expected, not list
    if args['gpus'] is None:
        args['gpus'] = -1 if torch.cuda.is_available() else 0

    return args


def transform_image(img_path:str) -> Tuple[np.ndarray, str]:
    """
    Reads the filepath and returns the image in the correct shape for inference (C, H, W)

    Args:
        img_path (str): Filepath of the input image

    Returns:
        Tuple[np.ndarray, str]: Image in the correct shape, Filepath of the image 
    """
    transform = Compose(
        [
            EnsureType()
        ]
    )
    img = np.array(Image.open(img_path))# only 3 modalities are accepted in the channel dimension for now
    if img.ndim == 4 and img.shape[-1] < 4:  # If shape is (h, w, d, c) assuming there are maximum 4 channels or modalities 
        img = np.transpose(img[..., :3] , (3, 0, 1, 2))  # Move channel to the first position
        img = dynamic_scale(img)
    elif img.ndim == 3 and img.shape[-1] <= 4:  # If shape is (h, w, c)
        img = np.transpose(img[..., :3] , (2, 0, 1))  # Move channel to the first position
        img = dynamic_scale(img)
    elif img.ndim == 2: # if no batch dimension then create one
        img = np.expand_dims(img, axis=-1)
        img = dynamic_scale(img)
        img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}, channel dimension should be last and image should be either 2D or 3D")
        
    img = transform(img)
    return img, img_path


def pred_function(
        image : torch.Tensor, 
        model: Callable,
        pred_patch_size: Sequence[int]
        ) -> torch.Tensor:
    """
    Sliding window inference on `image` with `model`

    Args:
        image (torch.Tensor): input image to be processed
        model (Callable): given input tensor ``image`` in shape BCHW[D], the outputs of the function call ``model(input)`` should be a tensor.
        pred_patch_size (Sequence[int]): spatial window size for inference

    Returns:
        torch.Tensor: prediction tensor
    """
    
    return sliding_window_inference(inputs=image, roi_size=pred_patch_size, sw_batch_size=4, predictor=model)


def predict_step(
        image_path: str, 
        model: Callable,
        pred_patch_size: Sequence[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Call trained model and run inference on input image given by the filepath using monai's ``sliding_window_inference`` function. 

    Args:
        image_path (str): filepath of the input image to be processed
        model (Callable): given input tensor ``image`` in shape BCHW[D], the outputs of the function call ``model(input)`` should be a tensor.
        pred_patch_size (Sequence[int): spatial window size for inference

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - pred: prediction obtained by using argmax (computes maximum value along the class dimension)   
            - probs: raw probability map from the model after softmax.
    """
    image, _ = transform_image(image_path)
    cropped_image = extract_largest_component_bbox_image(image.unsqueeze(0), lab=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_cropped_image = torch.tensor(cropped_image).to(device)
    logits = pred_function(tensor_cropped_image.float(), model, pred_patch_size)
    
    # Get final prediction via argmax
    pred = torch.argmax(logits, dim=1).byte()
    
    # Get probabilities for saving
    probs = torch.softmax(logits, dim=1)

    return pred, probs


def remove_donut_and_noise(mask_uint8, min_size=50, opening_kernel=5):
    """
    Applies morphological opening to break thin rings (donuts) and noise,
    then filters out small fragments.
    """
    # 1. Morphological Opening (Erosion followed by Dilation)
    # This removes small noise AND breaks thin connections (like the walls of a bubble)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
    opened_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # 2. Connected Components to filter small stuff (and remains of broken donuts)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)
    
    # Create output mask
    new_mask = np.zeros_like(opened_mask)
    
    # Filter by Area
    for i in range(1, num_labels): # Skip background 0
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_size:
            # Optional: Check Solidity/Circularity here if needed later
            # For now, just keeping large solid objects
            new_mask[labels == i] = 255
            
    return new_mask

def get_prediction(
        file: str, 
        unet: Callable,
        pred_patch_size: Sequence[int], 
        save_path: str, 
        labels: Sequence[int],
        binary_preds: bool,
        frg_class: int):
    """
    Convert the prediction to a binary segmentation mask and saves the image in the ``save_path`` filepath specified in the configuration file.
    Also saves the raw probability map as a .npy file.
    """

    prediction, probs = predict_step(file, unet, pred_patch_size)
    prediction = prediction.squeeze(0) # Remove batch dimension
    probs = probs.squeeze(0) # Remove batch dimension

    # --- Save the raw probability map as a .npy file ---
    prob_filename = os.path.join(save_path, os.path.basename(file).split('.')[0] + "_probs.npy")
    np.save(prob_filename, probs.cpu().numpy())

    # --- Process and save the final segmentation mask ---
    prediction = MapImage(prediction, labels, reverse=True)
    pred = prediction.cpu().numpy().squeeze().astype(np.uint8)
    
    base_filename = os.path.basename(file).split('.')[0]

    final_mask = pred
    
    if binary_preds:
        # Create binary (0 and 1)
        binary_mask = createBinaryAnnotation(pred, frg_class=frg_class).astype(np.uint8)
        # Scale to 0-255 for CV2
        binary_mask_255 = binary_mask * 255
        
        # --- APPLY CLEANUP ---
        # INCREASED Kernel to 15 to ensure donut walls break.
        # min_size=100 removes the debris left after the ring breaks.
        final_mask = remove_donut_and_noise(binary_mask_255, min_size=100, opening_kernel=15)
    
    # Add "_mask" to the output filename to match HeatmapCreator# filepath: /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_gpu.py
    output_filename = os.path.join(save_path, base_filename + "_mask.tif")
    # Using threshold to ensure binary output if it wasn't already 255
    if binary_preds:
         io.imsave(output_filename, final_mask, check_contrast=False)
    else:
         io.imsave(output_filename, final_mask.astype(np.uint8), check_contrast=False)

def predict_model(args: Dict):
    """
    Compile all functions above to run inference on a list of images.
    This version processes one tile at a time to ensure stability.
    """
    root_search_dir = args.get('wsi_dir') or args['pred_data_dir']
    labels = args['labels']
    binary_preds = args['binary_preds']
    frg_class = args['frg_class']
    
    # --- Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = Unet2D.load_from_checkpoint(
        args['model_path'],
        map_location=device
    ).to(device)
    unet.eval()

    print(f"Searching for 'seg_tiles' directories within {root_search_dir}...")

    # --- Recursively find all 'seg_tiles' directories ---
    found_dir = False
    for dirpath, dirnames, filenames in os.walk(root_search_dir):
        if 'seg_tiles' in dirnames:
            found_dir = True
            seg_tiles_dir = os.path.join(dirpath, 'seg_tiles')
            
            output_dir = os.path.join(dirpath, 'TAU_seg_tiles')
            os.makedirs(output_dir, exist_ok=True)

            image_files = sorted(glob.glob(os.path.join(seg_tiles_dir, '*.tif')))
            if not image_files:
                # print("  -> No .tif files found, skipping.")
                continue
            
            # --- NEW: Check if output directory already has processed files ---
            # We check if the number of output masks matches the number of input images
            existing_masks = glob.glob(os.path.join(output_dir, '*_mask.tif'))
            
            # if len(existing_masks) >= len(image_files):
            #     print(f"\nSkipping {os.path.basename(dirpath)}: All {len(image_files)} tiles appear to be processed.")
            #     continue
            # -----------------------------------------------------------------

            print(f"\nFound input directory: {seg_tiles_dir}")
            print(f"Output will be saved to: {output_dir}")

            # --- Process all tiles one-by-one for stability ---
            with torch.no_grad(): # Disable gradient calculation for inference
                for file_path in tqdm(image_files, desc=f"Processing {os.path.basename(dirpath)}"):
                    # Check if this specific file is already done
                    base_name = os.path.basename(file_path).split('.')[0]
                    # expected_out = os.path.join(output_dir, base_name + "_mask.tif")
                    # if os.path.exists(expected_out):
                    #     continue

                    try:
                        get_prediction(
                            file=file_path,
                            unet=unet,
                            pred_patch_size=args['pred_patch_size'],
                            save_path=output_dir,
                            labels=labels,
                            binary_preds=binary_preds,
                            frg_class=frg_class
                        )
                    except Exception as e:
                        print(f"\nERROR: Failed to process tile {os.path.basename(file_path)}.")
                        print(f"  -> Reason: {e}")
                        continue
    
    if not found_dir:
        print(f"Warning: No 'seg_tiles' directory found within {root_search_dir}. Nothing to process.")

def main():

    parser = argparse.ArgumentParser(conflict_handler='resolve', description='Run inference using trained model')
    parser.add_argument("--config_file", type=str,
                        default="setup_files/setup-predict.json",
                        help="json file training data parameters")
    # --- NEW: Add argument for single WSI directory processing ---
    parser.add_argument("--wsi_dir", type=str, default=None,
                        help="Path to a single Whole Slide Image directory to process.")
    parser.add_argument("--gpus", type=int, default=None, help="how many gpus to use")
    args = parser.parse_args()
    args = _parse_training_variables(args)

    predict_model(args)

if __name__ == '__main__':
    main()
