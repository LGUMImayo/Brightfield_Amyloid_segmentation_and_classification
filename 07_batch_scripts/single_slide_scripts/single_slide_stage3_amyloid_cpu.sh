#!/bin/bash
#SBATCH --job-name=amyloid3_cpu_%a
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_cpu_%a_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_cpu_%a_%j.err
#SBATCH --partition=lg-n64-256g
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00

# ============================================================================
# Single Slide Stage 3: Heatmap + Cleanup - Amyloid (CPU VERSION)
# ============================================================================

echo "============================================================"
echo "STAGE 3: Single Slide Heatmap + Cleanup - Amyloid (CPU)"
echo "Slide: $SLIDE_NAME"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate gdaltest
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau:$PYTHONPATH"

HEATMAP_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt"
FILES_DIR="${SLIDE_DIR}/${SLIDE_NAME}_files"
FAILED_LOG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/failed_amyloid_stage3.txt"
SUCCESS_LOG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/success_amyloid_stage3.txt"

# Check if Amyloid_seg_tiles exists
if [ ! -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ]; then
    echo "No Amyloid_seg_tiles found, nothing to do"
    echo "$SLIDE_NAME" >> "$FAILED_LOG"
    exit 1
fi

echo "Files dir: $FILES_DIR"

# Step 1: Run Pipeline 2 (Heatmap creation)
echo "Running Pipeline 2 (Heatmap)..."
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline/run_pipeline_part2.py \
    "$FILES_DIR" "$HEATMAP_CONFIG" "Amyloid" 2>&1

# Check if heatmap created
if [ -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii" ]; then
    echo "Heatmap created successfully"
    
    # Verify heatmap has actual intensity values (not empty)
    echo "Verifying heatmap intensity..."
    NONZERO_CHECK=$(python -c "
import nibabel as nib
import numpy as np
import sys
try:
    img = nib.load('$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii')
    data = img.get_fdata()
    nonzero_count = np.count_nonzero(data)
    total_voxels = data.size
    nonzero_pct = 100.0 * nonzero_count / total_voxels if total_voxels > 0 else 0
    max_val = np.max(data)
    print(f'{nonzero_pct:.2f}% non-zero, max={max_val:.1f}')
    sys.exit(0 if nonzero_count > 0 else 1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>&1)
    
    if [ $? -eq 0 ]; then
        echo "Heatmap intensity OK: $NONZERO_CHECK"
    else
        echo "FAILED - Heatmap is empty (all zeros): $NONZERO_CHECK"
        echo "Keeping intermediate files for debugging"
        echo "$SLIDE_NAME" >> "$FAILED_LOG"
        exit 1
    fi
    
    # Step 2: Cleanup intermediate files
    echo "Cleaning up intermediate files..."
    
    rm -rf "$FILES_DIR/heatmap/output/tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/mask/tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/seg_tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/Amyloid_seg_tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/color_map_0.1/" 2>/dev/null
    rm -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1.npy" 2>/dev/null
    
    REMAINING=$(du -sh "$FILES_DIR" 2>/dev/null | cut -f1)
    echo "Cleanup complete. Remaining size: $REMAINING"
    echo "SUCCESS"
    echo "$SLIDE_NAME" >> "$SUCCESS_LOG"
    exit 0
else
    echo "FAILED - Heatmap not created"
    echo "$SLIDE_NAME" >> "$FAILED_LOG"
    exit 1
fi
