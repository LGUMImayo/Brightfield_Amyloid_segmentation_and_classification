
import os
import torch
import pandas as pd
import numpy as np
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, '../../data/') # Adjust based on where scripts are run
TEST_CSV = os.path.join(SCRIPT_DIR, '../data/test_df.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, '../models/plaque_ai_es.pt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../evaluation_results/')

os.makedirs(OUTPUT_DIR, exist_ok=True)

target_dict = {
    'cgp': 1,
    'ccp': 2,
    'caa 1': 3,
    'caa 2': 4
}
# Inverse mapping for display
label_map = {v: k for k, v in target_dict.items()}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def find_image_paths(data_root):
    image_paths = {}
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(('.jpg', '.png', '.tif', '.tiff')):
                image_paths[file] = os.path.join(root, file)
    return image_paths

# Construct image mapping once
print("Scanning for images...")
# The data folder is actually at ../../data relative to src?
# The structure is s311590_plaque_ai/s311590_plaque_ai/plaque_ai/data/train_X
# And script is in s311590_plaque_ai/s311590_plaque_ai/plaque_ai/src
# So ../data is correct.
DATA_DIR_REL = os.path.join(SCRIPT_DIR, '../data')
IMAGE_MAP = find_image_paths(DATA_DIR_REL)
print(f"Found {len(IMAGE_MAP)} images.")

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
             # Try appending .jpg if missing, based on observation of files
             if img_name + '.jpg' in self.image_map:
                 img_name = img_name + '.jpg'
             elif img_name + '.png' in self.image_map:
                 img_name = img_name + '.png'

        if img_name in self.image_map:
            img_path = self.image_map[img_name]
        else:
            # Fallback/Error
            print(f"Warning: Image {img_name} not found.")
            # Return a dummy image
            img = torch.zeros((3, 520, 520), dtype=torch.float32)
            target = {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}
            return img, target, img_name

        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        
        # Parse labels and boxes
        row_labels = self.df.iloc[idx]['labels']
        row_bboxes = self.df.iloc[idx]['bboxes']
        
        # Assuming they are strings representation of list
        try:
            labels_list = ast.literal_eval(row_labels)
        except:
            labels_list = []
            
        try:
            bboxes_list = ast.literal_eval(row_bboxes)
        except:
            bboxes_list = []

        # Filter 'no label' or 'none'
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
        
        # Transform to tensor
        img_tensor = torchvision.transforms.functional.to_tensor(img_pil)

        return img_tensor, target, img_name

def get_model(num_classes):
    # Load Faster R-CNN with ResNet50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def calculate_iou(box1, box2):
    # shape (N, 4) and (M, 4) -> (N, M)
    return torchvision.ops.box_iou(box1, box2)

def evaluate():
    print("Loading test data...")
    df = pd.read_csv(TEST_CSV)
    dataset = PlaqueTestDataset(df, target_dict, IMAGE_MAP)
    #: batch_size=1 because images might have different sizes, 
    # unless we resize. FasterRCNN handles different sizes.
    # But collation needs to be handled.
    def collate_fn(batch):
        return tuple(zip(*batch))
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Loading model...")
    model = get_model(len(target_dict) + 1) # +1 for background
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    print("Running inference...")
    with torch.no_grad():
        for images, targets, img_names in tqdm(dataloader):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for i in range(len(images)):
                pred = outputs[i]
                target = targets[i]
                
                # Move to cpu
                pred_boxes = pred['boxes'].cpu()
                pred_scores = pred['scores'].cpu()
                pred_labels = pred['labels'].cpu()
                
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                all_predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
                all_targets.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })

    print("Calculating metrics...")
    
    # We will accumulate scores and true/false positives for each class
    # For ROC/PR curves
    
    results = {c: {'scores': [], 'tp': [], 'fp': [], 'num_gt': 0} for c in target_dict.values()}

    iou_threshold = 0.5

    for i in range(len(all_predictions)):
        pred = all_predictions[i]
        gt = all_targets[i]
        
        for class_id in results.keys():
            # Get preds and gts for this class
            p_inds = (pred['labels'] == class_id)
            p_boxes = pred['boxes'][p_inds]
            p_scores = pred['scores'][p_inds]
            
            g_inds = (gt['labels'] == class_id)
            g_boxes = gt['boxes'][g_inds]
            
            results[class_id]['num_gt'] += len(g_boxes)
            
            if len(p_boxes) == 0:
                continue
                
            # Sort predictions by score descending
            sorted_inds = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[sorted_inds]
            p_scores = p_scores[sorted_inds]
            
            # Match predictions to GT
            # Calculate IoU matrix
            if len(g_boxes) > 0:
                iou_matrix = torchvision.ops.box_iou(p_boxes, g_boxes)
                # For each prediction, find max IoU match
                matched_gt = set()
                
                for j in range(len(p_boxes)):
                    score = p_scores[j].item()
                    best_iou, best_gt_idx = torch.max(iou_matrix[j], dim=0)
                    
                    if best_iou >= iou_threshold and best_gt_idx.item() not in matched_gt:
                        results[class_id]['tp'].append(1)
                        results[class_id]['fp'].append(0)
                        results[class_id]['scores'].append(score)
                        matched_gt.add(best_gt_idx.item())
                    else:
                        results[class_id]['tp'].append(0)
                        results[class_id]['fp'].append(1)
                        results[class_id]['scores'].append(score)
            else:
                # All predictions are FP
                for j in range(len(p_boxes)):
                    results[class_id]['tp'].append(0)
                    results[class_id]['fp'].append(1)
                    results[class_id]['scores'].append(p_scores[j].item())

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['orange', 'blue', 'green', 'red']
    
    # Store aggregated data for saving
    save_data = {}

    for idx, (class_name, class_id) in enumerate(target_dict.items()):
        res = results[class_id]
        if res['num_gt'] == 0 and len(res['scores']) == 0:
            print(f"No samples for class {class_name}")
            continue

        scores = np.array(res['scores'])
        tp = np.array(res['tp'])
        fp = np.array(res['fp'])
        
        # Sort by score
        if len(scores) > 0:
            sorted_indices = np.argsort(-scores)
            tp = tp[sorted_indices]
            fp = fp[sorted_indices]
            scores = scores[sorted_indices] # thresholds
            
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            recall = cum_tp / res['num_gt'] if res['num_gt'] > 0 else np.zeros_like(cum_tp)
            precision = cum_tp / (cum_tp + cum_fp + 1e-8)
            
            # ROC
            # TPR = Recall
            # FPR = FP / N (N is total negatives... hard to define in detection without window counting)
            # Standard in detection is Precision-Recall curve. 
            # However, user asked for ROC.
            # For ROC in detection, we often use FPR vs TPR where FPR is usually defined relative 
            # to total possible detections or just FP count.
            # But technically standard ROC requires True Negatives, which is undefined for object detection (infinite background).
            # Often people use "False Positives Per Image" instead of FPR.
            # Or they treat every proposal as a sample.
            # Given the previous request was segmentation where pixels are defined, this is tricky.
            # We will plot Precision-Recall which is the standard.
            # And for ROC, maybe skip or use FP count? 
            # I will assume PR is the main goal, but plot "ROC-like" if needed or explain.
            # Actually, let's plot Precision-Recall mainly.
            # If user insists on ROC, we can assume a large number of negatives (all potential boxes),
            # but that's arbitrary.
            # BUT, let's try to do what we can. 
            
            ap = auc(recall, precision)
            
            axes[1].plot(recall, precision, label=f'{class_name} (AP = {ap:.5f})', color=colors[idx % len(colors)], linewidth=2)
            
            save_data[f'{class_name}_recall'] = recall
            save_data[f'{class_name}_precision'] = precision
            save_data[f'{class_name}_ap'] = ap
            
    axes[1].set_title('Precision-Recall Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Since ROC is ill-defined for detection without a negative set, I'll stick to PR.
    # But to satisfy "same stat... graph ... as well", I'll create a dummy ROC subplot 
    # or just use PR and explain.
    # Actually, for the "ROC" part, a common proxy is TPR vs FPPI (False Positives Per Image).
    
    # Let's plot FROC (Free-Response ROC) or just TPR vs cumulative FP.
    # Axes[0] -> TPR vs FP
    for idx, (class_name, class_id) in enumerate(target_dict.items()):
         res = results[class_id]
         if len(res['scores']) > 0 and res['num_gt'] > 0:
             scores = np.array(res['scores'])
             tp = np.array(res['tp'])
             fp = np.array(res['fp'])
             sorted_indices = np.argsort(-scores)
             tp = tp[sorted_indices]
             fp = fp[sorted_indices]
             cum_tp = np.cumsum(tp)
             cum_fp = np.cumsum(fp)
             tpr = cum_tp / res['num_gt']
             
             axes[0].plot(cum_fp, tpr, label=f'{class_name}', color=colors[idx % len(colors)], linewidth=2)
    
    axes[0].set_title('TPR vs False Positives Count')
    axes[0].set_xlabel('False Positives')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'plaque_pr_curves.png')
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")
    
    # Save data
    np.savez(os.path.join(OUTPUT_DIR, 'plaque_metrics.npz'), **save_data)
    
    # Print Stats
    for k in save_data:
        if k.endswith('_ap'):
            print(f"{k}: {save_data[k]}")

if __name__ == "__main__":
    evaluate()
