
import os
import torch
import pandas as pd
import numpy as np
import ast
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import cv2

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(SCRIPT_DIR, '../data/test_df.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, '../models/plaque_ai_final.pt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../evaluation_results/')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'plaque_metrics.npz')

target_dict = {
    'cgp': 1,
    'ccp': 2,
    'caa 1': 3,
    'caa 2': 4
}
colors = {
    1: 'orange', # cgp
    2: 'blue',   # ccp
    3: 'green',  # caa 1
    4: 'red'     # caa 2
}
class_names = {v: k for k, v in target_dict.items()}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def find_image_paths(data_root):
    image_paths = {}
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(('.jpg', '.png', '.tif', '.tiff')):
                image_paths[file] = os.path.join(root, file)
    return image_paths

DATA_DIR_REL = os.path.join(SCRIPT_DIR, '../data')
IMAGE_MAP = find_image_paths(DATA_DIR_REL)

class PlaqueTestDataset(Dataset):
    def __init__(self, df, target_dict, image_map, transform=None):
        self.df = df
        self.target_dict = target_dict
        self.image_map = image_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_name']
        if not img_name.endswith('.jpg'): 
             if img_name + '.jpg' in self.image_map:
                 img_name = img_name + '.jpg'
             elif img_name + '.png' in self.image_map:
                 img_name = img_name + '.png'

        if img_name in self.image_map:
            img_path = self.image_map[img_name]
        else:
            img = torch.zeros((3, 520, 520), dtype=torch.float32)
            target = {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}
            return img, target, img_name

        img_pil = Image.open(img_path).convert('RGB')
        
        # Parse labels and boxes
        row_labels = self.df.iloc[idx]['labels']
        row_bboxes = self.df.iloc[idx]['bboxes']
        
        try:
            labels_list = ast.literal_eval(row_labels)
        except:
            labels_list = []
            
        try:
            bboxes_list = ast.literal_eval(row_bboxes)
        except:
            bboxes_list = []

        valid_boxes = []
        valid_labels = []
        
        for lbl, box in zip(labels_list, bboxes_list):
            if lbl in self.target_dict:
                valid_labels.append(self.target_dict[lbl])
                valid_boxes.append(box)
        
        valid_boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
        valid_labels = torch.as_tensor(valid_labels, dtype=torch.int64)

        target = {}
        target["boxes"] = valid_boxes
        target["labels"] = valid_labels
        
        img_tensor = torchvision.transforms.functional.to_tensor(img_pil)

        return img_tensor, target, img_name

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def create_heatmap(shape, boxes, scores):
    # shape: (H, W)
    heatmap = np.zeros(shape, dtype=np.float32)
    for i in range(len(boxes)):
        box = boxes[i].int().numpy()
        score = scores[i].item()
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(shape[1], box[2]), min(shape[0], box[3])
        heatmap[y1:y2, x1:x2] += score # accumulate confidence
        
    # Smoothen
    if len(boxes) > 0:
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        # Normalize
        heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap

def visualize():
    if not os.path.exists(METRICS_PATH):
        print("Metrics file not found. Run evaluation first.")
        return

    metrics = np.load(METRICS_PATH)
    
    df = pd.read_csv(TEST_CSV)
    dataset = PlaqueTestDataset(df, target_dict, IMAGE_MAP)
    
    model = get_model(len(target_dict) + 1)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    num_samples = 3
    indices = random.sample(range(len(dataset)), num_samples)

    # Figure Layout
    fig = plt.figure(figsize=(24, 6 + 4 * num_samples))
    gs = fig.add_gridspec(num_samples + 1, 4) # 4 columns: Image, GT, Heatmap, Pred Boxes

    # 1. Plot Curves
    ax_pr = fig.add_subplot(gs[0, :2])
    color_list = ['orange', 'blue', 'green', 'red']
    
    for idx, (class_name, class_id) in enumerate(target_dict.items()):
        recall_key = f'{class_name}_recall'
        precision_key = f'{class_name}_precision'
        ap_key = f'{class_name}_ap'
        
        if recall_key in metrics:
            recall = metrics[recall_key]
            precision = metrics[precision_key]
            ap = metrics[ap_key]
            
            ax_pr.plot(recall, precision, label=f'Validation Group {class_name} (AP = {ap:.5f})', 
                       color=color_list[idx % 4], linewidth=2)

    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()
    ax_pr.grid(True, alpha=0.3)
    
    # 2. Visualize Samples
    prediction_threshold = 0.5
    
    for i, idx in enumerate(indices):
        img_tensor, target, img_name = dataset[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        H, W, _ = img_np.shape
        
        # Run inference
        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]
            
        pred_boxes = output['boxes'].cpu()
        pred_scores = output['scores'].cpu()
        pred_labels = output['labels'].cpu()
        
        # Filter low score
        keep = pred_scores > prediction_threshold
        pred_boxes_filt = pred_boxes[keep]
        pred_labels_filt = pred_labels[keep]
        
        row = i + 1
        
        # Col 0: Original Image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(img_np)
        ax.set_title(f'Image: {img_name}')
        ax.axis('off')

        # Col 1: Ground Truth
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(img_np)
        ax.set_title('Ground Truth')
        ax.axis('off')
        # Add GT boxes
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        for k in range(len(gt_boxes)):
            box = gt_boxes[k].numpy()
            label = gt_labels[k].item()
            c = colors.get(label, 'white')
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                     linewidth=2, edgecolor=c, facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], class_names[label], color=c, fontsize=8, backgroundcolor='white')

        # Col 2: Heatmap (Aggregate confidence)
        # Create heatmap from all predictions (weighted by score)
        hm = create_heatmap((H, W), pred_boxes, pred_scores)
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(img_np)
        im = ax.imshow(hm, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax.set_title('Prediction Confidence Heatmap')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Col 3: Prediction Boxes
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(img_np)
        ax.set_title(f'Predictions (Score > {prediction_threshold})')
        ax.axis('off')
        for k in range(len(pred_boxes_filt)):
            box = pred_boxes_filt[k].numpy()
            label = pred_labels_filt[k].item()
            c = colors.get(label, 'white')
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                     linewidth=2, edgecolor=c, facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'{class_names.get(label,"?"):} {pred_scores[keep][k]:.2f}', color=c, fontsize=8, backgroundcolor='white')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'plaque_visualization_full.png')
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize()
