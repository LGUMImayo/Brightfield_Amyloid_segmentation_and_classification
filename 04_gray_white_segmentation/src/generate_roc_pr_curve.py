
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(SCRIPT_DIR, '../data/images_and_masks/') # Adjusted to be relative to script
IMG_DIR = os.path.join(MAIN_DIR, 'images')
MASK_DIR = os.path.join(MAIN_DIR, 'masks')
MODEL_PATH = os.path.join(SCRIPT_DIR, '../models/model_level_2_ce_0')
TEST_CSV = os.path.join(SCRIPT_DIR, '../data/test_gw_0.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../evaluation_results/')

os.makedirs(OUTPUT_DIR, exist_ok=True)

target_dict = {
    'background': 0,
    'gray-matter': 1,
    'white-matter': 2
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

val_transform = A.Compose([
    A.CenterCrop(520, 520),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True)
])

class HippAI_aperio_Dataset(Dataset):
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
        # Masks are likely indexed images
        mask_np = np.array(Image.open(blob_name))
        
        mask = [mask_np]

        # Dummy labels for albumentations (not really used in this transform setup but required by some versions if passed)
        # labels = torch.as_tensor([i+1 for i in range(len(self.target_dict))], dtype=torch.int64)

        if self.transform is not None:
            transformed = self.transform(
                image=img,
                masks=mask
            )
            img = transformed["image"]
            masks = np.array(transformed['masks'])
            masks = torch.tensor(masks, dtype=torch.float32)

        return img, masks

def load_model(model_path):
    # Initialize with aux_loss=True to match training structure (since Weights='DEFAULT' was used)
    # Default num_classes is 21 (COCO), which matches the unmodified aux_classifier.
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True) 
    
    # Modify the main classifier to have 3 classes as done in training
    model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def evaluate():
    print("Loading data...")
    df = pd.read_csv(TEST_CSV)
    # Check if image_name exists
    if 'image_name' not in df.columns:
         # Sometimes csv has index column as first
         if 'image_name' not in df.keys():
             print("Columns:", df.columns)
    
    dataset = HippAI_aperio_Dataset(df, IMG_DIR, MASK_DIR, target_dict, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    y_true_all = []
    y_score_gm_all = [] # Gray Matter
    y_score_wm_all = [] # White Matter

    print("Running inference...")
    with torch.no_grad():
        for inputs, masks in tqdm(dataloader):
            inputs = inputs.to(device)
            # masks shape: (B, 1, H, W)
            masks = masks.squeeze(1).long().cpu().numpy()
            
            outputs = model(inputs)['out'] # (B, 3, H, W)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Flatten and store
            # Ground truth: 0, 1, 2
            y_true_all.append(masks.flatten())
            
            # Probabilities for GM (class 1) and WM (class 2)
            y_score_gm_all.append(probs[:, 1, :, :].flatten())
            y_score_wm_all.append(probs[:, 2, :, :].flatten())

    y_true = np.concatenate(y_true_all)
    y_score_gm = np.concatenate(y_score_gm_all)
    y_score_wm = np.concatenate(y_score_wm_all)

    print("Computing curves...")
    
    # Calculate curves for Gray Matter (Class 1) vs Rest
    # Binarize labels for GM: 1 is positive, others negative
    y_true_gm = (y_true == 1).astype(int)
    fpr_gm, tpr_gm, _ = roc_curve(y_true_gm, y_score_gm)
    roc_auc_gm = auc(fpr_gm, tpr_gm)
    
    precision_gm, recall_gm, _ = precision_recall_curve(y_true_gm, y_score_gm)
    pr_auc_gm = average_precision_score(y_true_gm, y_score_gm)

    # Calculate curves for White Matter (Class 2) vs Rest
    # Binarize labels for WM: 2 is positive, others negative
    y_true_wm = (y_true == 2).astype(int)
    fpr_wm, tpr_wm, _ = roc_curve(y_true_wm, y_score_wm)
    roc_auc_wm = auc(fpr_wm, tpr_wm)

    precision_wm, recall_wm, _ = precision_recall_curve(y_true_wm, y_score_wm)
    pr_auc_wm = average_precision_score(y_true_wm, y_score_wm)

    print(f"Gray Matter ROC AUC: {roc_auc_gm}, PR AUC: {pr_auc_gm}")
    print(f"White Matter ROC AUC: {roc_auc_wm}, PR AUC: {pr_auc_wm}")

    # Plotting
    plt.figure(figsize=(12, 5))
    
    # ROC Plot
    plt.subplot(1, 2, 1)
    plt.plot(fpr_gm, tpr_gm, label=f'Gray Matter (AUC = {roc_auc_gm:.2f})')
    plt.plot(fpr_wm, tpr_wm, label=f'White Matter (AUC = {roc_auc_wm:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # PR Plot
    plt.subplot(1, 2, 2)
    plt.plot(recall_gm, precision_gm, label=f'Gray Matter (AUC = {pr_auc_gm:.2f})')
    plt.plot(recall_wm, precision_wm, label=f'White Matter (AUC = {pr_auc_wm:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    save_path = os.path.join(OUTPUT_DIR, 'roc_pr_curves.png')
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")

    # Save data for further analysis if needed
    np.savez(os.path.join(OUTPUT_DIR, 'roc_data.npz'), 
             fpr_gm=fpr_gm, tpr_gm=tpr_gm, 
             fpr_wm=fpr_wm, tpr_wm=tpr_wm,
             precision_gm=precision_gm, recall_gm=recall_gm,
             precision_wm=precision_wm, recall_wm=recall_wm)

if __name__ == "__main__":
    evaluate()
