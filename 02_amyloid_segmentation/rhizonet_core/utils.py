import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import os
from skimage.color import rgb2hsv
from skimage import exposure
from skimage import io, filters, measure
from scipy import ndimage as ndi
from typing import Union, List, Tuple, Sequence, Dict
from monai.transforms import MapLabelValued
import glob


def extract_largest_component_bbox_image(img: Union[np.ndarray, torch.Tensor], 
                                         lab: Union[np.ndarray, torch.Tensor] = None,) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Checks if an image is empty (all black). If it is not empty, it returns the
    original image for processing. This bypasses the original logic of finding
    and cropping to the largest component.

    Args:
        img (Union[numpy.ndarray, torch.Tensor]): 
            The input image. 
            - Can be a NumPy array or PyTorch tensor with shape (C, H, W) or (B, C, H, W) if a batch dimension is included.
            - Must not be None; if None, the function raises a ValueError.
        lab (Union[numpy.ndarray, torch.Tensor], optional): 
            The label or annotated image. Defaults to None.


    Returns:
        Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: 
            - If `lab` is None: Returns the original image as a NumPy array.
            - If `lab` is provided: Returns a tuple containing:
                - The original image (NumPy array).
                - The original label image (NumPy array).
    """

    if img is None:
        raise ValueError("Image cannot be None.")
    
    # The original function is too aggressive. Since tiles are pre-masked,
    # we just need to return the original image for prediction.
    # The model will handle all-black tiles efficiently.
    
    if lab is not None:
        return img, lab
    
    return img
        

def get_weights(
        labels: torch.Tensor, 
        classes: List[int], 
        device: str, 
        include_background=False,
        ) -> List[float]:
    """
    Computes the weights of each class in the batch of labeled images 

    Args:
        labels (torch.Tensor): Batch of labeled imagse with each pixel of an image equal to a class value. 
        classes (List[int]): List of the classes
        device (str): training device should be 'cuda' if training on GPU
        include_background (bool, optional): Boolean to include or not the background valued as 0 when calculating the weight of the classes in the images. Defaults to False.

    Returns:
        List[float]: List of the class weights (floats).
    """

    labels = labels.to(device)
    if not include_background:
        classes.remove(0)
    flat_labels = labels.view(-1)

    # FIX IF NOT ALL CLASSES ARE IN THE LABEL INPUT 
    n = len(classes)
    class_counts = torch.bincount(flat_labels, minlength=n)
    class_weights = torch.zeros(n, dtype=torch.float, device=device)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = 1 / class_counts[nonzero_mask]
    class_weights /= class_weights.sum()
    # print("class weights {}".format(class_weights))

    return class_weights


def MapImage(
        image: Union[np.ndarray, torch.Tensor], 
        original_values: List[int],
        reverse: bool
        ) -> Union[np.ndarray, torch.Tensor]:
    """
    Maps the current values of a given input image to the values given by the tuple (current values, new values).

    Args:
        image (Union[np.ndarray, torch.Tensor]): The input image to transform
        original_values (List[int]): List of original values to be mapped
        reverse (bool): True if mapping the other way around back to original values

    Raises:
        TypeError: If the input image is neither a numpy array or a torch tensor

    Returns:
        Union[np.ndarray, torch.Tensor]: the transformed input after mapping.
    
    Example::
        transformed_image = MapImage(image, [0, 85, 170])
        The values will be mapped to [0, 1, 2] with the first value of the original values specified being the background index. 
    """
    if isinstance(image, np.ndarray) :      
        data = image.copy()
    elif isinstance(image, torch.Tensor):
        data = image.detach()
    else:
        raise TypeError("Input must be a numpy.ndarray, torch.Tensor")
    
    target_values = list(range(len(original_values)))
    if not reverse:
        # Create the transform
        map_label_transform = MapLabelValued(["label"], original_values, target_values)
    else:
        # Create the transform
        map_label_transform = MapLabelValued(["label"], target_values, original_values)
        
    # Apply the transform
    mapped_label_image = map_label_transform({"label": data})["label"]

    return mapped_label_image


def elliptical_crop(img: np.ndarray, 
                    center_x: int, center_y: int, 
                    width: int, 
                    height: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops out an elliptical shape out of the input image and sets the rest as background 

    Args:
        img (np.ndarray): Input image
        center_x (int): Center x coordinate of the ellipse 
        center_y (int): Center y coordinate of the ellipse 
        width (int): Width of the wanted ellipse
        height (int): Height of the wanted ellipse

    Returns:
        Tuple[np.ndarray, np.ndarray]: cropped output image
    """

    image = Image.fromarray(img)
    image_width, image_height = image.size

    # Create an elliptical mask using PIL
    mask = Image.new('1', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), fill=1)

    # Convert the mask to a PyTorch tensor
    mask_tensor = TF.to_tensor(mask)

    # Apply the mask to the input image using element-wise multiplication
    cropped_image = TF.to_pil_image(torch.mul(TF.to_tensor(image), mask_tensor))

    return image, np.array(cropped_image)



def get_image_paths(directory):
    return glob.glob(os.path.join(directory, '*.*'))

def common_files(path_images, path_labels, prefix='', suffix=''):
    """
    Identifies common filenames between two directories, accounting for a prefix or suffix on the label files.

    Args:
        path_images (str): Path to the directory containing image files.
        path_labels (str): Path to the directory containing label files.
        prefix (str, optional): Prefix to remove from label filenames. Defaults to ''.
        suffix (str, optional): Suffix to remove from label filenames. Defaults to ''.

    Returns:
        list: A list of common base filenames.
    """
    f_imgs = [os.path.splitext(os.path.basename(f))[0] for f in get_image_paths(path_images)]
    f_labs = [os.path.splitext(os.path.basename(f))[0] for f in get_image_paths(path_labels)]
    
    processed_labs = []
    if suffix:
        # Remove suffix from the end of the label names
        processed_labs = [f[:-len(suffix)] if f.endswith(suffix) else f for f in f_labs]
    elif prefix:
        # Remove prefix from the start of the label names
        processed_labs = [f[len(prefix):] if f.startswith(prefix) else f for f in f_labs]
    else:
        # If no prefix or suffix, use the names as is
        processed_labs = f_labs

    unique_files = set(f_imgs).intersection(processed_labs)
    return list(unique_files)
    
def contrast_img(img: np.ndarray) -> np.ndarray:
    """
    Applies the Adaptive Equalization or histogram equalization contrast method. This method
    enhances the contrast of an image by adjusting the intensity values of pixels based on the 
    distribution of pixel intensities in the image's histogram. 

    Args:
        img (np.ndarray): input image

    Returns:
        np.ndarray: contrasted image
    """
    # HSV image
    hsv_img = rgb2hsv(img)  # 3 channels
    # select 1channel
    img = hsv_img[:, :, 0]
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img = exposure.equalize_hist(img)
    # Adaptive Equalization
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img


def createBinaryAnnotation(img: Union[np.ndarray, torch.Tensor],
                           frg_class: int) -> Union[np.ndarray, torch.Tensor]:
    """
    Creates a binary mask out of the prediction result with root as foreground and the rest as background

    Args:
        img (Union[np.ndarray, torch.Tensor]): Input image
        frg_class (int): value of the foreground class when creating binary segmentation masks

    Raises:
        TypeError: if the input is neither a numpy array or a torch tensor or if the foreground value is wrong. 

    Returns:
        Union[np.ndarray, torch.Tensor]: binary mask 
    """

    if isinstance(img, torch.Tensor):
        u = torch.unique(img)
        bkg = torch.zeros(img.shape)  # background
        if len(u) == 1:
            print("This prediction only contains background")
            return img
        else:
            try: 
                frg = (img == frg_class).int() * 255
            except: 
                print("Error in the foreground value")
    elif isinstance(img, np.ndarray):
        u = np.unique(img)
        bkg = np.zeros(img.shape)  # background
        if len(u) == 1:
            print("This prediction only contains background")
            return img
        else:
            try: 
                frg = (img == frg_class).astype(int) * 255
            except: 
                print("Error in the foreground value")
    else:
        raise TypeError("Input should be a PyTorch tensor or a NumPy array.")
    return bkg + frg

def get_biomass(binary_img: np.ndarray) -> int:
    """
    Calculate the biomass by counting the number of pixels equal to 1

    Args:
        binary_img (np.ndarray): input image as binary mask

    Returns:
        int: integer value corresponding to the pixel count or root biomass. 
    """
    roi = binary_img > 0
    nerror = 0
    binary_img = binary_img * roi
    biomass = np.unique(binary_img.flatten(), return_counts=True)
    try:
        nbiomass = biomass[1][1]
    except:
        nbiomass = 0
        nerror += 1
        print("Seg error in ")
    return nbiomass

