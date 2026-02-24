#!/bin/bash
#SBATCH --job-name=amyloid3_%a
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_%a_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_%a_%j.err
#SBATCH --partition=gpu-n24-170g-4x-a100-40g
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00

# ============================================================================
# Single Slide Stage 3: Heatmap + Cleanup - Amyloid
# Usage: sbatch --export=SLIDE_NAME=<name>,SLIDE_DIR=<path> single_slide_stage3_amyloid.sh
# ============================================================================

echo "============================================================"
echo "STAGE 3: Single Slide Heatmap + Cleanup - Amyloid"
echo "Slide: $SLIDE_NAME"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet
export WANDB_MODE=disabled
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/usr/local/biotools/cuda/12.1/lib64
export PATH=/usr/local/biotools/cuda/12.1/bin:${PATH}

HEATMAP_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-pipeline2_Amyloid_efficientnet_v2.json"
FILES_DIR="${SLIDE_DIR}/${SLIDE_NAME}_files"

# Check if Amyloid_seg_tiles exists
if [ ! -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ]; then
    echo "No Amyloid_seg_tiles found, nothing to do"
    exit 1
fi

TILE_COUNT=$(find "$FILES_DIR/heatmap/Amyloid_seg_tiles" -name "*.png" 2>/dev/null | wc -l)
echo "Files dir: $FILES_DIR"
echo "Amyloid_seg_tiles count: $TILE_COUNT"

if [ "$TILE_COUNT" -eq 0 ]; then
    echo "No tiles in Amyloid_seg_tiles, skipping"
    exit 1
fi

# Step 1: Run Pipeline 2 (Heatmap creation)
echo "Running Pipeline 2 (Heatmap)..."
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline/run_pipeline_part2.py \
    "$FILES_DIR" "$HEATMAP_CONFIG" 2>&1

# Check if heatmap created
if [ -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii" ]; then
    echo "Heatmap created successfully"
    
    # Step 2: Cleanup intermediate files
    echo "Cleaning up intermediate files..."
    
    # Remove large intermediate directories (only tiles/ subfolder)
    rm -rf "$FILES_DIR/heatmap/output/tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/mask/tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/seg_tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/Amyloid_seg_tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/TAU_seg_tiles/" 2>/dev/null
    rm -rf "$FILES_DIR/heatmap/color_map_0.1/" 2>/dev/null
    
    # Remove numpy heatmap (keep only nii)
    rm -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1.npy" 2>/dev/null
    
    # Show remaining size
    REMAINING=$(du -sh "$FILES_DIR" 2>/dev/null | cut -f1)
    echo "Cleanup complete. Remaining size: $REMAINING"
    echo "SUCCESS"
    exit 0
else
    echo "FAILED - Heatmap not created"
    exit 1
fi
