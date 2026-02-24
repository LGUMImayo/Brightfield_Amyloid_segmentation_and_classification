
import numpy as np
from sklearn.metrics import auc
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '../evaluation_results/roc_data.npz')

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        exit(1)

    data = np.load(DATA_PATH)
    
    fpr_gm = data['fpr_gm']
    tpr_gm = data['tpr_gm']
    roc_auc_gm = auc(fpr_gm, tpr_gm)
    
    # PR AUC is Area Under Precision-Recall Curve
    # Note: sklearn's average_precision_score implementation is slightly different from 
    # auc(recall, precision) but usually preferred. However, we only saved points.
    # We can approximate with auc(recall, precision).
    precision_gm = data['precision_gm']
    recall_gm = data['recall_gm']
    pr_auc_gm = auc(recall_gm, precision_gm)
    
    fpr_wm = data['fpr_wm']
    tpr_wm = data['tpr_wm']
    roc_auc_wm = auc(fpr_wm, tpr_wm)
    
    precision_wm = data['precision_wm']
    recall_wm = data['recall_wm']
    pr_auc_wm = auc(recall_wm, precision_wm)
    
    print(f"Gray Matter ROC AUC: {roc_auc_gm}")
    print(f"Gray Matter PR AUC: {pr_auc_gm}")
    print(f"White Matter ROC AUC: {roc_auc_wm}")
    print(f"White Matter PR AUC: {pr_auc_wm}")
