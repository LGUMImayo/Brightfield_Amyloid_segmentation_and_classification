#!/bin/bash
#SBATCH --job-name=amyloid3_launcher_cpu
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_launcher_cpu_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_launcher_cpu_%j.err
#SBATCH --partition=lg-n64-256g
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

# ============================================================================
# Amyloid Stage 3 Launcher (CPU) - Submits parallel Stage 3 jobs
# ============================================================================

STAIN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
BATCH_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts"

echo "============================================================"
echo "Amyloid Stage 3 Launcher (CPU) - Submitting parallel jobs"
echo "Start: $(date)"
echo "============================================================"

SUBMITTED=0
NO_TILES=0

for CASE_DIR in "$BASE_DIR"/*/; do
    [ -d "$CASE_DIR" ] || continue
    
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}${SLIDE_NAME}_files"
        
        if [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ]; then
            TILE_COUNT=$(find "$FILES_DIR/heatmap/Amyloid_seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
            
            if [ "$TILE_COUNT" -gt 0 ]; then
                JOB_ID=$(sbatch --parsable \
                    --export=SLIDE_NAME="$SLIDE_NAME",SLIDE_DIR="$SLIDE_DIR" \
                    "${BATCH_DIR}/single_slide_stage3_amyloid_cpu.sh")
                
                echo "[$SLIDE_NAME] Submitted Stage 3 (CPU): $JOB_ID (tiles: $TILE_COUNT)"
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
echo "Submitted:  $SUBMITTED Stage 3 jobs"
echo "No tiles:   $NO_TILES slides"
echo "============================================================"
