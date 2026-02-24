#!/bin/bash
#SBATCH --job-name=amyloid2_launcher_cpu
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage2_launcher_cpu_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage2_launcher_cpu_%j.err
#SBATCH --partition=lg-n64-256g
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

# ============================================================================
# Amyloid Stage 2 Launcher (CPU) - Submits parallel Stage 2 prediction jobs
# ============================================================================

STAIN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
BATCH_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts"

echo "============================================================"
echo "Amyloid Stage 2 Launcher (CPU) - Submitting parallel jobs"
echo "Start: $(date)"
echo "============================================================"

SUBMITTED=0
SKIPPED=0
NO_TILES=0

# Find all slides with seg_tiles (Stage 1 complete)
for CASE_DIR in "$BASE_DIR"/*/; do
    [ -d "$CASE_DIR" ] || continue
    
    for SLIDE_DIR in "$CASE_DIR"/*Amyloid*/; do
        [ -d "$SLIDE_DIR" ] || continue
        
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}/${SLIDE_NAME}_files"
        
        # Check if seg_tiles exists (Stage 1 done)
        if [ -d "$FILES_DIR/heatmap/seg_tiles" ]; then
            TILE_COUNT=$(find "$FILES_DIR/heatmap/seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
            
            if [ "$TILE_COUNT" -gt 0 ]; then
                # Check if Amyloid_seg_tiles already exists
                if [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ]; then
                    MASK_COUNT=$(find "$FILES_DIR/heatmap/Amyloid_seg_tiles" -name "*_mask.tif" 2>/dev/null | wc -l)
                    if [ "$MASK_COUNT" -ge "$TILE_COUNT" ]; then
                        echo "[$SLIDE_NAME] Skipping: Already has Amyloid_seg_tiles ($MASK_COUNT masks)"
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi
                fi
                
                # Submit Stage 2 job (CPU version)
                JOB_ID=$(sbatch --parsable \
                    --export=SLIDE_NAME="$SLIDE_NAME",SLIDE_DIR="$SLIDE_DIR" \
                    "${BATCH_DIR}/single_slide_stage2_amyloid_cpu.sh")
                
                echo "[$SLIDE_NAME] Submitted Stage 2 (CPU): $JOB_ID (tiles: $TILE_COUNT)"
                SUBMITTED=$((SUBMITTED + 1))
            else
                NO_TILES=$((NO_TILES + 1))
            fi
        fi
    done
done

echo ""
echo "============================================================"
echo "Launcher Complete - $(date)"
echo "============================================================"
echo "Submitted:  $SUBMITTED Stage 2 jobs"
echo "Skipped:    $SKIPPED slides (already processed)"
echo "No tiles:   $NO_TILES slides"
echo "============================================================"
