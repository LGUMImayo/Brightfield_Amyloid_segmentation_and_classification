
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import auc
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
import os
import random

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(SCRIPT_DIR, '../data/images_and_masks/')
IMG_DIR = os.path.join(MAIN_DIR, 'images')
MASK_DIR = os.path.join(MAIN_DIR, 'masks')
MODEL_PATH = os.path.join(SCRIPT_DIR, '../models/model_level_2_ce_0')
TEST_CSV = os.path.join(SCRIPT_DIR, '../data/test_gw_0.csv')
ROC_DATA_PATH = os.path.join(SCRIPT_DIR, '../evaluation_results/roc_data.npz')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../evaluation_results/')

target_dict = {
    'background': 0,
    'gray-matter': 1,
    'white-matter': 2
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Transform for inference (CenterCrop to display clean area)
val_transform = A.Compose([
    A.CenterCrop(520, 520),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True)
])

# Inverse transform for visualization
def inverse_normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

class HippAI_aperio_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, mask_dir, target_dict, transform=None):
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_dict = target_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.image_name.iloc[idx]
        blob_name = f"{self.img_dir}/{img_name}"
        img = Image.open(blob_name)
        img = np.array(img.convert('RGB'))
        
        blob_name = f"{self.mask_dir}/{img_name}"
        mask_np = np.array(Image.open(blob_name))
        mask = [mask_np]

        if self.transform is not None:
            transformed = self.transform(image=img, masks=mask)
            img = transformed["image"]
            masks = np.array(transformed['masks'])
            masks = torch.tensor(masks, dtype=torch.float32)

        return img, masks, img_name

def load_model(model_path):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def create_visualization_plot(model, dataset, roc_data_path, num_samples=3):
    # Load ROC Data
    if not os.path.exists(roc_data_path):
        print("ROC data not found. Please run generate_roc_pr_curve.py first.")
        return

    roc_data = np.load(roc_data_path)
    
    # Calculate AUCs
    roc_auc_gm = auc(roc_data['fpr_gm'], roc_data['tpr_gm'])
    roc_auc_wm = auc(roc_data['fpr_wm'], roc_data['tpr_wm'])
    
    # Approx PR AUC
    pr_auc_gm = auc(roc_data['recall_gm'], roc_data['precision_gm'])
    pr_auc_wm = auc(roc_data['recall_wm'], roc_data['precision_wm'])
    
    # 1. Setup Plot
    fig = plt.figure(figsize=(20, 6 + 4 * num_samples))
    gs = fig.add_gridspec(num_samples + 1, 5) # +1 for curves, 5 columns for images

    # 2. Plot Curves (Top Row)
    # ROC Curve spans first 2 columns
    ax_roc = fig.add_subplot(gs[0, 0:2])
    ax_roc.plot(roc_data['fpr_gm'], roc_data['tpr_gm'], label=f'Validation Group Gray Matter (AUC = {roc_auc_gm:.5f})', color='orange', linewidth=2)
    ax_roc.plot(roc_data['fpr_wm'], roc_data['tpr_wm'], label=f'Validation Group White Matter (AUC = {roc_auc_wm:.5f})', color='blue', linewidth=2)
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_title('ROC Curve')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # PR Curve spans next 2 columns
    ax_pr = fig.add_subplot(gs[0, 2:4])
    ax_pr.plot(roc_data['recall_gm'], roc_data['precision_gm'], label=f'Validation Group Gray Matter (AUC = {pr_auc_gm:.5f})', color='orange', linewidth=2)
    ax_pr.plot(roc_data['recall_wm'], roc_data['precision_wm'], label=f'Validation Group White Matter (AUC = {pr_auc_wm:.5f})', color='blue', linewidth=2)
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, alpha=0.3)

    # 3. Visualize Samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        inputs, masks, img_name = dataset[idx]
        inputs = inputs.unsqueeze(0).to(device) # (1, 3, H, W)
        
        with torch.no_grad():
            outputs = model(inputs)['out']
            probs = torch.softmax(outputs, dim=1) # (1, 3, H, W)
            preds = torch.argmax(probs, dim=1).cpu().numpy().squeeze() # (H, W)
            
            # Get heatmaps
            gm_heatmap = probs[0, 1, :, :].cpu().numpy()
            wm_heatmap = probs[0, 2, :, :].cpu().numpy()

        # Prepare Image for display
        img_disp = inverse_normalize(inputs[0]).permute(1, 2, 0).cpu().numpy()
        img_disp = np.clip(img_disp, 0, 1)
        
        # Prepare Mask for display
        mask_disp = masks.squeeze().numpy()

        row = i + 1
        
        # Col 0: Original Image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(img_disp)
        ax.set_title(f'Image: {img_name}')
        ax.axis('off')

        # Col 1: Ground Truth
        ax = fig.add_subplot(gs[row, 1])
        # Create RGB mask: Back=Black, GM=Orange, WM=Blue
        mask_rgb = np.zeros((*mask_disp.shape, 3))
        mask_rgb[mask_disp == 1] = [1, 0.5, 0] # Orange for GM
        mask_rgb[mask_disp == 2] = [0, 0, 1]   # Blue for WM
        ax.imshow(mask_rgb)
        ax.set_title('Ground Truth (GM=Org, WM=Blu)')
        ax.axis('off')

        # Col 2: GM Heatmap
        ax = fig.add_subplot(gs[row, 2])
        im = ax.imshow(gm_heatmap, cmap='jet', vmin=0, vmax=1)
        ax.set_title('GM Predicted Heatmap')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Col 3: WM Heatmap
        ax = fig.add_subplot(gs[row, 3])
        im = ax.imshow(wm_heatmap, cmap='jet', vmin=0, vmax=1)
        ax.set_title('WM Predicted Heatmap')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Col 4: Final Prediction (Argmax)
        ax = fig.add_subplot(gs[row, 4])
        pred_rgb = np.zeros((*preds.shape, 3))
        pred_rgb[preds == 1] = [1, 0.5, 0]
        pred_rgb[preds == 2] = [0, 0, 1]
        ax.imshow(pred_rgb)
        ax.set_title('Final Prediction')
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'visualization_with_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    df = pd.read_csv(TEST_CSV)
    dataset = HippAI_aperio_Dataset(df, IMG_DIR, MASK_DIR, target_dict, transform=val_transform)
    
    model = load_model(MODEL_PATH)
    
    create_visualization_plot(model, dataset, ROC_DATA_PATH)
