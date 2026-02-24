#!/bin/bash
#SBATCH --job-name=amyloid1_%a
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage1_%a_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage1_%a_%j.err
#SBATCH --partition=huge-n128-512g
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# ============================================================================
# Single Slide Stage 1: Pipeline 1 (Tiling & Masks) - Amyloid
# Usage: sbatch --export=SLIDE_NAME=<name>,SLIDE_DIR=<path> single_slide_stage1_amyloid.sh
# ============================================================================

echo "============================================================"
echo "STAGE 1: Pipeline 1 (Tiling & Masks) - Amyloid"
echo "Slide: $SLIDE_NAME"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate gdaltest
export PYTHONPATH="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau:$PYTHONPATH"

PIPELINE_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt"
FILES_DIR="${SLIDE_DIR}/${SLIDE_NAME}_files"

echo "Slide directory: $SLIDE_DIR"
echo "Files directory: $FILES_DIR"
echo ""

# Check if already done
if [ -d "$FILES_DIR/heatmap/seg_tiles" ]; then
    TILE_COUNT=$(find "$FILES_DIR/heatmap/seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
    if [ "$TILE_COUNT" -gt 0 ]; then
        echo "Already has seg_tiles ($TILE_COUNT tiles), skipping"
        exit 0
    fi
fi

# Check for partial outputs - if output and mask tiles exist, just create seg_tiles
if [ -d "$FILES_DIR/output/RES/tiles" ] && [ -d "$FILES_DIR/mask/final_mask/tiles" ]; then
    OUTPUT_COUNT=$(find "$FILES_DIR/output/RES/tiles" -name "*.tif" 2>/dev/null | wc -l)
    MASK_COUNT=$(find "$FILES_DIR/mask/final_mask/tiles" -name "*.tif" 2>/dev/null | wc -l)
    
    if [ "$OUTPUT_COUNT" -gt 0 ] && [ "$MASK_COUNT" -gt 0 ]; then
        echo "Found partial outputs - output tiles: $OUTPUT_COUNT, mask tiles: $MASK_COUNT"
        echo "Creating seg_tiles from existing outputs (skipping re-tiling)..."
        
        mkdir -p "$FILES_DIR/heatmap/seg_tiles"
        
        # Use pipeline script to create seg_tiles from existing tiles
        bash /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/scripts/run_pipeline_full.sh \
            "$SLIDE_DIR" "$PIPELINE_CONFIG" 2>&1
        
        # Verify
        SEG_COUNT=$(find "$FILES_DIR/heatmap/seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
        if [ "$SEG_COUNT" -gt 0 ]; then
            echo "SUCCESS - Created $SEG_COUNT seg_tiles from existing outputs"
            exit 0
        fi
        echo "Note: Pipeline detected existing tiles and created seg_tiles"
    fi
fi

# Run full Pipeline 1 (tiling and masking)
echo "Running full Pipeline 1..."
START_TIME=$(date +%s)

bash /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/scripts/run_pipeline_full.sh \
    "$SLIDE_DIR" "$PIPELINE_CONFIG" 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))

echo ""
echo "Elapsed time: ${MINUTES} minutes (${ELAPSED} seconds)"

# Check result
if [ -d "$FILES_DIR/heatmap/seg_tiles" ]; then
    TILE_COUNT=$(find "$FILES_DIR/heatmap/seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
    if [ "$TILE_COUNT" -gt 0 ]; then
        echo "SUCCESS - Created $TILE_COUNT tiles"
        
        # Submit Stage 2 job with dependency
        echo "Submitting Stage 2 (prediction) job..."
        STAGE2_JOB=$(sbatch --dependency=afterok:$SLURM_JOB_ID \
            --export=SLIDE_DIR="$SLIDE_DIR",SLIDE_NAME="$SLIDE_NAME" \
            /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts/single_slide_stage2_amyloid_cpu.sh | awk '{print $4}')
        echo "Stage 2 job submitted: $STAGE2_JOB (will run after this job completes)"
        
        exit 0
    fi
fi

echo "FAILED - No seg_tiles created"
exit 1
