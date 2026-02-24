#!/bin/bash
#SBATCH --job-name=gw_inference_array
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/gw_inference_%A_%a.out
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/gw_inference_%A_%a.err
#SBATCH --partition=med-n16-64g    # Use a more standard partition for smaller jobs
#SBATCH --mem=60G                 # Memory per job task
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per job task
#SBATCH --time=04:00:00           # Set a reasonable time limit per file

# --- Script Configuration ---
TIFF_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/H&E_Braak_0_Slides/TIFF"
FILE_LIST="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/tiff_file_list.txt"

# This block runs only when you execute the script manually
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "--- Preparing for job submission ---"
    
    # 1. Create the list of TIFF files to process
    echo "Generating file list..."
    find "$TIFF_DIR" -type f \( -iname "*.tif" -o -iname "*.tiff" \) > "$FILE_LIST"
    
    # 2. Count the number of files
    NUM_FILES=$(wc -l < "$FILE_LIST")
    echo "Found $NUM_FILES files to process."

    # 3. Submit this script as a job array
    if [ "$NUM_FILES" -gt 0 ]; then
        echo "Submitting job array for 1 to $NUM_FILES tasks."
        sbatch --array=1-$NUM_FILES "$0"
    else
        echo "No TIFF files found to process. Exiting."
    fi
    exit 0
fi

# --- This part of the script is executed by each SLURM array task ---

echo "--- Starting SLURM Array Task $SLURM_ARRAY_TASK_ID ---"

# 1. Get the specific file path for this task from the list
FILE_TO_PROCESS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$FILE_LIST")

echo "Task $SLURM_ARRAY_TASK_ID assigned to file: $FILE_TO_PROCESS"

# 2. Setup Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate matt_code
export PYTHONPATH=$PYTHONPATH:/fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_gray_white/gray_white_segmentation/src

# 3. Run the Python script on the single file
if [ -f "$FILE_TO_PROCESS" ]; then
    python /fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_gray_white/gray_white_segmentation/src/inference_test.py "$FILE_TO_PROCESS"
else
    echo "ERROR: File '$FILE_TO_PROCESS' not found for task $SLURM_ARRAY_TASK_ID."
fi

echo "--- SLURM Array Task $SLURM_ARRAY_TASK_ID Complete ---"