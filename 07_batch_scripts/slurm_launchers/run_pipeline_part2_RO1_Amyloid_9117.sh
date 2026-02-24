#!/bin/bash
#SBATCH --job-name=pipeline2_amyloid_9117
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/pipeline2_amyloid_9117_%A_%a.out
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/pipeline2_amyloid_9117_%A_%a.err
#SBATCH --partition=huge-n128-512g
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=30-00:00:00

# Set the target directory to loop through
TARGET_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline2/RO1_Amyloid/Cases/9117_22"
DIR_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/pipeline2_amyloid_9117_dir_list.txt"

# This block runs only when you first execute the script
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "--- Preparing for job submission ---"
    
    # 1. Create the list of slide directories to process.
    echo "Generating directory list..."
    
    find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -not -path "*_files*" > "$DIR_LIST_FILE"
    
    # 2. Count the number of directories
    NUM_DIRS=$(wc -l < "$DIR_LIST_FILE")
    echo "Found $NUM_DIRS directories to process."

    # 3. Submit this script as a job array
    if [ "$NUM_DIRS" -gt 0 ]; then
        echo "Submitting job array for 1 to $NUM_DIRS tasks."
        sbatch --array=1-$NUM_DIRS "$0"
    else
        echo "No directories found to process in $TARGET_DIR. Exiting."
    fi
    exit 0
fi

# --- This part of the script is executed by each SLURM array task ---

echo "--- Starting SLURM Array Task $SLURM_ARRAY_TASK_ID ---"

# 1. Get the specific directory path for this task from the list
DIR_TO_PROCESS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR_LIST_FILE")

echo "Task $SLURM_ARRAY_TASK_ID assigned to directory: $DIR_TO_PROCESS"

if [ -d "$DIR_TO_PROCESS" ]; then
    echo "Processing directory: $DIR_TO_PROCESS"
    # Run the pipeline script on the single directory
    bash /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/scripts/run_pipeline_full2.sh "$DIR_TO_PROCESS" '/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt'
else
    echo "ERROR: Directory '$DIR_TO_PROCESS' not found for task $SLURM_ARRAY_TASK_ID."
fi

echo "--- SLURM Array Task $SLURM_ARRAY_TASK_ID Complete ---"
