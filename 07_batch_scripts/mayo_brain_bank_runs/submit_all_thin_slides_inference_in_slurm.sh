#!/bin/bash
#SBATCH --job-name=amy_thin_batch
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amy_thin_batch_%A_%a.out
#SBATCH --partition=gpu-n16-60g-1x-tesla-t4
#SBATCH --gres=gpu:1
#SBATCH --mem=59G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

# --- Configuration ---
TIFF_DIR_ROOT="/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/tiff"
DIR_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_thin_tiff_list.txt"

# Params for the python script
MASK_ROOT_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/prediction_priority_threshold_wmw_1.5_gmt_0.2_wmt_0.2_clean_21_best_parameter"
OUTPUT_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/amyloid_predictions_final_v1"
MODEL_PATH="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_CNN/RO1_Amyloid_testing/log_new_data_with_edges/last.ckpt"

# --- Master Logic (Job Submission) ---
# This block runs only when you first execute the script (not in the array tasks)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "--- Preparing for job submission ---"
    
    # 1. Create the list of TIFF files to process
    echo "Generating TIFF list from $TIFF_DIR_ROOT..."
    find "$TIFF_DIR_ROOT" -name "*.tiff" | sort > "$DIR_LIST_FILE"
    
    # 2. Count the number of files
    NUM_FILES=$(wc -l < "$DIR_LIST_FILE")
    echo "Found $NUM_FILES TIFF slides ready for prediction."

    # 3. Submit this script as a job array
    if [ "$NUM_FILES" -gt 0 ]; then
        echo "Submitting job array for 1 to $NUM_FILES tasks."
        # Submit self as array job
        sbatch --array=1-$NUM_FILES "$0"
    else
        echo "No .tiff files found in $TIFF_DIR_ROOT."
    fi
    exit 0
fi

# --- Worker Logic (SLURM Array Task) ---

echo "--- Starting SLURM Array Task $SLURM_ARRAY_TASK_ID ---"

# 1. Get the specific TIFF file for this task from the list
TARGET_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR_LIST_FILE")

if [ ! -f "$TARGET_FILE" ]; then
    echo "ERROR: File '$TARGET_FILE' not found for task $SLURM_ARRAY_TASK_ID."
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID assigned to file: $TARGET_FILE"

# 2. Initialize conda and set environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

# 3. Run the Python script
# Note: --test_mode is REMOVED for production run to save space/time unless you really want debug tiles for all slides
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_amyloid_thin_slides.py \
    --tiff_path "$TARGET_FILE" \
    --mask_root_dir "$MASK_ROOT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --resolution 0.2827 \
    --model_input_size 128 \
    --gm_threshold 0.9 \
    --blur_strength 0 \
    --min_grain_size 100

echo "--- SLURM Array Task $SLURM_ARRAY_TASK_ID Complete ---"