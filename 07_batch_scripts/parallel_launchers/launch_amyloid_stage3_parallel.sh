#!/bin/bash
#SBATCH --job-name=amyloid3_launcher
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_launcher_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_launcher_%j.err
#SBATCH --partition=gpu-n24-170g-4x-a100-40g
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

# ============================================================================
# Amyloid Stage 3 Launcher - Submits parallel Stage 3 jobs for all slides
# This job runs after Stage 2 completes and submits individual Stage 3 jobs
# ============================================================================

STAIN="Amyloid"
STAIN_PATTERN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
BATCH_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts"

echo "============================================================"
echo "Amyloid Stage 3 Launcher - Submitting parallel jobs"
echo "Start: $(date)"
echo "============================================================"

SUBMITTED=0
SKIPPED=0

# Find all slides with Amyloid_seg_tiles and submit Stage 3 jobs
for CASE_DIR in "$BASE_DIR"/*/; do
    [ -d "$CASE_DIR" ] || continue
    CASE_NAME=$(basename "$CASE_DIR")
    
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN_PATTERN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}${SLIDE_NAME}_files"
        
        # Check if Amyloid_seg_tiles exists with files
        if [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ]; then
            TILE_COUNT=$(find "$FILES_DIR/heatmap/Amyloid_seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
            
            if [ "$TILE_COUNT" -gt 0 ]; then
                # Submit Stage 3 job (re-run even if heatmap exists)
                JOB_ID=$(sbatch --parsable \
                    --export=SLIDE_NAME="$SLIDE_NAME",SLIDE_DIR="$SLIDE_DIR" \
                    "${BATCH_DIR}/single_slide_stage3_amyloid.sh")
                
                echo "[$SLIDE_NAME] Submitted Stage 3: $JOB_ID (tiles: $TILE_COUNT)"
                SUBMITTED=$((SUBMITTED + 1))
            fi
        fi
    done
done

echo ""
echo "============================================================"
echo "Amyloid Stage 3 Launcher Complete"
echo "Submitted: $SUBMITTED jobs"
echo "Skipped (already done): $SKIPPED"
echo "End: $(date)"
echo "============================================================"
