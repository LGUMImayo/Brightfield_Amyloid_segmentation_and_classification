#!/bin/bash
#SBATCH --job-name=amyloid1_launcher
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage1_launcher_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage1_launcher_%j.err
#SBATCH --partition=lg-n64-256g
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

# ============================================================================
# Amyloid Stage 1 Launcher - Submits parallel Stage 1 (tiling) jobs
# ============================================================================

STAIN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
BATCH_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts"

echo "============================================================"
echo "Amyloid Stage 1 Launcher - Submitting parallel tiling jobs"
echo "Start: $(date)"
echo "============================================================"

SUBMITTED=0
SKIPPED=0

# Find all Amyloid slides
for CASE_DIR in "$BASE_DIR"/*/; do
    [ -d "$CASE_DIR" ] || continue
    
    for SLIDE_DIR in "$CASE_DIR"/*Amyloid*/; do
        [ -d "$SLIDE_DIR" ] || continue
        
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}/${SLIDE_NAME}_files"
        
        # Check if seg_tiles already exists
        if [ -d "$FILES_DIR/heatmap/seg_tiles" ]; then
            TILE_COUNT=$(find "$FILES_DIR/heatmap/seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
            if [ "$TILE_COUNT" -gt 0 ]; then
                echo "[$SLIDE_NAME] Skipping: Already has seg_tiles ($TILE_COUNT tiles)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
        
        # Submit Stage 1 job
        JOB_ID=$(sbatch --parsable \
            --export=SLIDE_NAME="$SLIDE_NAME",SLIDE_DIR="$SLIDE_DIR" \
            "${BATCH_DIR}/single_slide_stage1_amyloid.sh")
        
        echo "[$SLIDE_NAME] Submitted Stage 1: $JOB_ID"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "============================================================"
echo "Launcher Complete - $(date)"
echo "============================================================"
echo "Submitted:  $SUBMITTED Stage 1 jobs"
echo "Skipped:    $SKIPPED slides (already tiled)"
echo "============================================================"
