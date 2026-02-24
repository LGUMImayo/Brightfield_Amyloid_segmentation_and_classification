#!/bin/bash
#SBATCH --job-name=amyloid2_cpu_%a
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_slide_cpu_%a_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_slide_cpu_%a_%j.err
#SBATCH --partition=lg-n64-256g
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# ============================================================================
# Single Slide Stage 2: EfficientNet Prediction - Amyloid (CPU VERSION)
# Usage: sbatch --export=SLIDE_NAME=<name>,SLIDE_DIR=<path> single_slide_stage2_amyloid_cpu.sh
# ============================================================================

echo "============================================================"
echo "STAGE 2: Single Slide EfficientNet Prediction - Amyloid (CPU)"
echo "Slide: $SLIDE_NAME"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet
export WANDB_MODE=disabled

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16

PREDICTION_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-predict_Amyloid_efficientnet_v2.json"
FILES_DIR="${SLIDE_DIR}/${SLIDE_NAME}_files"
MARKER_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/logs/markers"
mkdir -p "$MARKER_DIR"

echo "CPU Info:"
lscpu | grep -E "Model name|CPU\(s\)|Thread"
echo ""
echo "Files dir: $FILES_DIR"

# Step 1: Delete old Amyloid_seg_tiles
if [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ]; then
    TILE_COUNT=$(find "$FILES_DIR/heatmap/Amyloid_seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
    echo "Deleting old Amyloid_seg_tiles ($TILE_COUNT tiles)..."
    rm -rf "$FILES_DIR/heatmap/Amyloid_seg_tiles"
fi

# Clean up old heatmaps
rm -rf "$FILES_DIR/heatmap/hm_map_0.1" 2>/dev/null
rm -rf "$FILES_DIR/heatmap/color_map_0.1" 2>/dev/null

# Step 2: Run EfficientNet prediction (will use CPU)
echo "Running EfficientNet prediction on CPU..."
START_TIME=$(date +%s)

python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_iron_effnet.py \
    --config_file "$PREDICTION_CONFIG" --wsi_dir "$FILES_DIR" \
    --output_folder Amyloid_seg_tiles 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))

echo ""
echo "Elapsed time: ${MINUTES} minutes (${ELAPSED} seconds)"

# Check result
if [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ] && [ "$(ls -A $FILES_DIR/heatmap/Amyloid_seg_tiles 2>/dev/null | head -1)" ]; then
    NEW_COUNT=$(find "$FILES_DIR/heatmap/Amyloid_seg_tiles" -name "*_mask.tif" 2>/dev/null | wc -l)
    echo "SUCCESS - New tiles: $NEW_COUNT"
    echo "Performance: ${MINUTES} minutes for ${NEW_COUNT} tiles"
    # Create success marker for stage 3
    echo "SUCCESS" > "${MARKER_DIR}/${SLIDE_NAME}_stage2.done"
    
    # Submit Stage 3 job with dependency
    echo "Submitting Stage 3 (heatmap) job..."
    STAGE3_JOB=$(sbatch --dependency=afterok:$SLURM_JOB_ID \
        --export=SLIDE_DIR="$SLIDE_DIR",SLIDE_NAME="$SLIDE_NAME" \
        /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts/single_slide_stage3_amyloid_cpu.sh | awk '{print $4}')
    echo "Stage 3 job submitted: $STAGE3_JOB (will run after this job completes)"
    
    exit 0
else
    echo "FAILED - No output tiles created"
    echo "FAILED" > "${MARKER_DIR}/${SLIDE_NAME}_stage2.done"
    exit 1
fi
