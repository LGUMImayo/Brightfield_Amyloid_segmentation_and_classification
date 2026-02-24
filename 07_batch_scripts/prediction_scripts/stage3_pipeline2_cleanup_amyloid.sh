#!/bin/bash
#SBATCH --job-name=amyloid_stage3
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_stage3_%j.err
#SBATCH --partition=gpu-n12-85g-1x-a100-40g
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# ============================================================================
# Stage 3: Pipeline 2 (Heatmap) + Cleanup
# ============================================================================

STAIN="Amyloid"
STAIN_PATTERN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
PIPELINE_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-amyloid/pipeline_config.txt"

# Failure log file
LOG_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/logs"
mkdir -p "$LOG_DIR"
FAILED_LOG="${LOG_DIR}/failed_stage3_${STAIN}_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================================"
echo "STAGE 3: Pipeline 2 + Cleanup - ${STAIN}"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Failed slides will be logged to: $FAILED_LOG"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate gdaltest
export PYTHONPATH=$PYTHONPATH:/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-amyloid

TOTAL=0
PROCESSED=0
SKIPPED=0
FAILED=0

# Count slides with Amyloid_seg_tiles (ready for pipeline2)
for CASE_DIR in "$BASE_DIR"/*/; do
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN_PATTERN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}${SLIDE_NAME}_files"
        [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ] && TOTAL=$((TOTAL + 1))
    done
done

echo "Total slides ready for Pipeline 2: $TOTAL"
echo ""

# Process each slide
for CASE_DIR in "$BASE_DIR"/*/; do
    CASE_NAME=$(basename "$CASE_DIR")
    
    for SLIDE_DIR in "$CASE_DIR"/*"$STAIN_PATTERN"*/; do
        [ -d "$SLIDE_DIR" ] || continue
        
        SLIDE_NAME=$(basename "$SLIDE_DIR")
        FILES_DIR="${SLIDE_DIR}${SLIDE_NAME}_files"
        
        # Skip if no Amyloid_seg_tiles
        [ -d "$FILES_DIR/heatmap/Amyloid_seg_tiles" ] || continue
        
        echo "----------------------------------------"
        echo "[$((PROCESSED + SKIPPED + FAILED + 1))/$TOTAL] $SLIDE_NAME"
        
        # Check if already has final heatmap
        if [ -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii" ]; then
            # Check if it's not empty (more than 1MB)
            SIZE=$(stat -c%s "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii" 2>/dev/null || echo 0)
            if [ "$SIZE" -gt 1000000 ]; then
                echo "  -> Already has valid heatmap, skipping"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
        
        # Remove old heatmap
        rm -rf "$FILES_DIR/heatmap/hm_map_0.1" 2>/dev/null
        
        # Run Pipeline 2
        echo "  -> Running Pipeline 2..."
        python /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-amyloid/pipeline/run_pipeline_part2.py \
            "$FILES_DIR" "$PIPELINE_CONFIG" > /dev/null 2>&1
        
        if [ -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii" ]; then
            echo "  -> Pipeline 2 SUCCESS"
            
            # Cleanup intermediate files
            echo "  -> Cleaning up intermediate files..."
            BEFORE=$(du -sh "$FILES_DIR" 2>/dev/null | cut -f1)
            
            rm -rf "$FILES_DIR/output/tiles" 2>/dev/null
            rm -rf "$FILES_DIR/mask/tiles" 2>/dev/null
            rm -rf "$FILES_DIR/heatmap/seg_tiles" 2>/dev/null
            rm -rf "$FILES_DIR/heatmap/Amyloid_seg_tiles" 2>/dev/null
            # Remove large intermediate heatmap files (keep only res10.nii)
            rm -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1.npy" 2>/dev/null
            rm -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.npy" 2>/dev/null
            rm -rf "$FILES_DIR/heatmap/color_map_0.1" 2>/dev/null
            
            AFTER=$(du -sh "$FILES_DIR" 2>/dev/null | cut -f1)
            echo "  -> Cleaned: $BEFORE -> $AFTER"
            
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
echo "STAGE 3 COMPLETE"
echo "Processed: $PROCESSED | Skipped: $SKIPPED | Failed: $FAILED"
if [ $FAILED -gt 0 ]; then
    echo "Failed slides logged to: $FAILED_LOG"
fi
echo "End: $(date)"
echo "============================================================"

# Summary
echo ""
echo "Final storage check:"
du -sh "$BASE_DIR" 2>/dev/null

# Summary of all failure logs for this stain
echo ""
echo "All failure logs for ${STAIN}:"
ls -la ${LOG_DIR}/failed_*_${STAIN}_*.txt 2>/dev/null || echo "No failure logs found"
