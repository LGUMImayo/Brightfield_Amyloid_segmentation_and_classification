#!/bin/bash
#SBATCH --job-name=amyloid_stage1
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage1_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage1_%j.err
#SBATCH --partition=huge-n128-512g
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00

# ============================================================================
# Stage 1: Pipeline 1 (Tiling & Masks) - High Memory CPU Node
# ============================================================================

STAIN="Amyloid"
STAIN_PATTERN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
PIPELINE_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-amyloid/pipeline_config.txt"

# Failure log file
LOG_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/logs"
mkdir -p "$LOG_DIR"
FAILED_LOG="${LOG_DIR}/failed_stage1_${STAIN}_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================================"
echo "STAGE 1: Pipeline 1 (Tiling & Masks) - ${STAIN}"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Failed slides will be logged to: $FAILED_LOG"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate gdaltest
export PYTHONPATH=$PYTHONPATH:/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-amyloid

TOTAL=0
PROCESSED=0
SKIPPED=0
FAILED=0

# Find all slides
for CASE_DIR in "$BASE_DIR"/*/; do
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN_PATTERN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        TOTAL=$((TOTAL + 1))
    done
done

echo "Total slides to process: $TOTAL"
echo ""

# Process each slide
for CASE_DIR in "$BASE_DIR"/*/; do
    CASE_NAME=$(basename "$CASE_DIR")
    
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN_PATTERN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}${SLIDE_NAME}_files"
        
        echo "----------------------------------------"
        echo "[$((PROCESSED + SKIPPED + FAILED + 1))/$TOTAL] $SLIDE_NAME"
        
        # Check if already done
        if [ -d "$FILES_DIR/heatmap/seg_tiles" ] && [ "$(ls -A $FILES_DIR/heatmap/seg_tiles 2>/dev/null | head -1)" ]; then
            echo "  -> Already has seg_tiles, skipping"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        # Run Pipeline 1
        echo "  -> Running Pipeline 1..."
        bash /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-amyloid/scripts/run_pipeline_full.sh \
            "$SLIDE_DIR" "$PIPELINE_CONFIG" > /dev/null 2>&1
        
        if [ -d "$FILES_DIR/heatmap/seg_tiles" ] && [ "$(ls -A $FILES_DIR/heatmap/seg_tiles 2>/dev/null | head -1)" ]; then
            echo "  -> SUCCESS"
            PROCESSED=$((PROCESSED + 1))
        else
            echo "  -> FAILED (no seg_tiles created)"
            echo "$SLIDE_DIR" >> "$FAILED_LOG"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo ""
echo "============================================================"
echo "STAGE 1 COMPLETE"
echo "Processed: $PROCESSED | Skipped: $SKIPPED | Failed: $FAILED"
if [ $FAILED -gt 0 ]; then
    echo "Failed slides logged to: $FAILED_LOG"
fi
echo "End: $(date)"
echo "============================================================"

# Don't exit with error - let pipeline continue, failures are logged
exit 0
