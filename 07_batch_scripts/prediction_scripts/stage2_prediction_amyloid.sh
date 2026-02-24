#!/bin/bash
#SBATCH --job-name=amyloid_stage2
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage2_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage2_%j.err
#SBATCH --partition=gpu-n12-85g-1x-a100-40g
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# ============================================================================
# Stage 2: EfficientNet Prediction - GPU Node
# ============================================================================

STAIN="Amyloid"
STAIN_PATTERN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
PREDICTION_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-predict_Amyloid_efficientnet_v2.json"

# Failure log file
LOG_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/logs"
mkdir -p "$LOG_DIR"
FAILED_LOG="${LOG_DIR}/failed_stage2_${STAIN}_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================================"
echo "STAGE 2: EfficientNet Prediction - ${STAIN}"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Failed slides will be logged to: $FAILED_LOG"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet
export WANDB_MODE=disabled
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/usr/local/biotools/cuda/12.1/lib64
export PATH=/usr/local/biotools/cuda/12.1/bin:${PATH}

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

TOTAL=0
PROCESSED=0
SKIPPED=0
FAILED=0

# Count slides with seg_tiles (ready for prediction)
for CASE_DIR in "$BASE_DIR"/*/; do
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN_PATTERN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}${SLIDE_NAME}_files"
        [ -d "$FILES_DIR/heatmap/seg_tiles" ] && TOTAL=$((TOTAL + 1))
    done
done

echo "Total slides ready for prediction: $TOTAL"
echo ""

# Process each slide
for CASE_DIR in "$BASE_DIR"/*/; do
    CASE_NAME=$(basename "$CASE_DIR")
    
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN_PATTERN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}${SLIDE_NAME}_files"
        
        # Skip if no seg_tiles
        [ -d "$FILES_DIR/heatmap/seg_tiles" ] || continue
        
        echo "----------------------------------------"
        echo "[$((PROCESSED + SKIPPED + FAILED + 1))/$TOTAL] $SLIDE_NAME"
        
        # Check if already has prediction output
        if [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ] && [ "$(ls -A $FILES_DIR/heatmap/Amyloid_seg_tiles 2>/dev/null | head -1)" ]; then
            echo "  -> Already has Amyloid_seg_tiles, skipping"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        # Remove old predictions (including incorrectly named TAU_seg_tiles)
        rm -rf "$FILES_DIR/heatmap/Amyloid_seg_tiles" 2>/dev/null
        rm -rf "$FILES_DIR/heatmap/TAU_seg_tiles" 2>/dev/null
        rm -rf "$FILES_DIR/heatmap/hm_map_0.1" 2>/dev/null
        rm -rf "$FILES_DIR/heatmap/color_map_0.1" 2>/dev/null
        
        # Run Prediction
        echo "  -> Running EfficientNet prediction..."
        python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_iron_effnet.py \
            --config_file "$PREDICTION_CONFIG" --wsi_dir "$FILES_DIR" 2>&1 | tail -5
        
        if [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ] && [ "$(ls -A $FILES_DIR/heatmap/Amyloid_seg_tiles 2>/dev/null | head -1)" ]; then
            echo "  -> SUCCESS"
            PROCESSED=$((PROCESSED + 1))
        else
            echo "  -> FAILED"
            echo "$SLIDE_DIR" >> "$FAILED_LOG"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo ""
echo "============================================================"
echo "STAGE 2 COMPLETE"
echo "Processed: $PROCESSED | Skipped: $SKIPPED | Failed: $FAILED"
if [ $FAILED -gt 0 ]; then
    echo "Failed slides logged to: $FAILED_LOG"
fi
echo "End: $(date)"
echo "============================================================"

# Don't exit with error - let pipeline continue, failures are logged
exit 0
