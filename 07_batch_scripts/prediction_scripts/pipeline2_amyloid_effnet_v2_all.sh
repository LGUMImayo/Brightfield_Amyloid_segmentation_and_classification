#!/bin/bash
#SBATCH --job-name=amyloid_pipe2_all
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_effnet_v2_pipeline2_%A_%a.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_effnet_v2_pipeline2_%A_%a.err
#SBATCH --partition=huge-n128-512g
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30-00:00:00

# ============================================================================
# Pipeline 2: Stitching and Heatmap Creation for Amyloid Slides
# ============================================================================
# Run this AFTER prediction_amyloid_effnet_v2_all.sh completes
# This script stitches segmentation results and creates heatmaps
# ============================================================================

WSI_DIR_ROOT="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP"
DIR_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_pipeline2_dir_list.txt"
FAILED_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_pipeline2_failed_slides.txt"
SUCCESS_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_pipeline2_success_slides.txt"
CONFIG_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt"

# This block runs only when you first execute the script
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "============================================================"
    echo "Pipeline 2 (Stitching & Heatmap) for Amyloid - Job Preparation"
    echo "============================================================"
    echo "Searching for Amyloid slide directories in: $WSI_DIR_ROOT"
    echo ""
    
    # Find all slide directories with predictions from EfficientNet model
    echo "Generating directory list..."
    
    # Find _files directories that contain TAU_seg_tiles (predictions)
    # Search in both Pipeline and Pipeline2 for RO1_Amyloid
    find "$WSI_DIR_ROOT/Pipeline/RO1_Amyloid" "$WSI_DIR_ROOT/Pipeline2/RO1_Amyloid" \
        -type d -name "*_files" 2>/dev/null | \
        while read dir; do
            # Only include if it has TAU_seg_tiles (predictions exist)
            if [ -d "$dir/heatmap/TAU_seg_tiles" ]; then
                echo "$dir"
            fi
        done | sort -u > "$DIR_LIST_FILE"
    
    NUM_DIRS=$(wc -l < "$DIR_LIST_FILE")
    echo "Found $NUM_DIRS slide directories for pipeline 2."
    echo ""
    
    # Initialize tracking files
    echo "# Failed slides - Pipeline 2 Amyloid" > "$FAILED_LIST_FILE"
    echo "# Generated: $(date)" >> "$FAILED_LIST_FILE"
    echo "" >> "$FAILED_LIST_FILE"
    
    echo "# Successful slides - Pipeline 2 Amyloid" > "$SUCCESS_LIST_FILE"
    echo "# Generated: $(date)" >> "$SUCCESS_LIST_FILE"
    echo "" >> "$SUCCESS_LIST_FILE"

    if [ "$NUM_DIRS" -gt 0 ]; then
        echo "Submitting job array for 1 to $NUM_DIRS tasks."
        echo ""
        echo "After completion, check:"
        echo "  - Failed slides: $FAILED_LIST_FILE"
        echo "  - Success slides: $SUCCESS_LIST_FILE"
        echo ""
        sbatch --array=1-$NUM_DIRS "$0"
    else
        echo "No directories found for pipeline 2."
        echo "Make sure predictions have been run (TAU_seg_tiles folder exists)."
    fi
    exit 0
fi

# --- This part is executed by each SLURM array task ---

echo "============================================================"
echo "Pipeline 2 Amyloid - Task $SLURM_ARRAY_TASK_ID"
echo "============================================================"

DIR_TO_PROCESS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR_LIST_FILE")
SLIDE_NAME=$(basename "$DIR_TO_PROCESS")

echo "Processing: $SLIDE_NAME"
echo "Path: $DIR_TO_PROCESS"
echo ""

# Initialize environment
source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate gdaltest
export PYTHONPATH=$PYTHONPATH:/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau

PIPELINE_SUCCESS=0

if [ -d "$DIR_TO_PROCESS" ]; then
    echo "Starting pipeline 2 at $(date)"
    
    python /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline/run_pipeline_part2.py \
        "$DIR_TO_PROCESS" "$CONFIG_FILE"
    
    PIPE_EXIT_CODE=$?
    
    if [ $PIPE_EXIT_CODE -eq 0 ]; then
        echo "Pipeline 2 completed successfully at $(date)"
        PIPELINE_SUCCESS=1
    else
        echo "ERROR: Pipeline 2 failed with exit code $PIPE_EXIT_CODE"
    fi
else
    echo "ERROR: Directory '$DIR_TO_PROCESS' not found!"
fi

# Log result
if [ $PIPELINE_SUCCESS -eq 1 ]; then
    echo "$DIR_TO_PROCESS" >> "$SUCCESS_LIST_FILE"
    echo "✓ Logged to success list"
else
    echo "$DIR_TO_PROCESS | Task $SLURM_ARRAY_TASK_ID | $(date)" >> "$FAILED_LIST_FILE"
    echo "✗ Logged to failed list"
fi

echo ""
echo "============================================================"
echo "Pipeline 2 Amyloid - Task $SLURM_ARRAY_TASK_ID Complete"
echo "============================================================"
