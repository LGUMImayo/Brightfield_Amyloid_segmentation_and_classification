#!/bin/bash
#SBATCH --job-name=amyloid_pred_focal_clean
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/rhizonet_prediction_amyloid_focal_clean_%A_%a.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/rhizonet_prediction_amyloid_focal_clean_%A_%a.err
#SBATCH --partition=gpu-n64-240g-4x-tesla-t4
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=99:00:00

# --- Script Configuration ---
# Target the entire stain folder in Pipeline
WSI_DIR_ROOT="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline/RO1_Amyloid"
DIR_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_dir_list_pipeline_all_focal_clean.txt"

# This block runs only when you first execute the script (not in the array tasks)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "--- Preparing for job submission ---"
    
    # 1. Create the list of slide directories to process.
    # We look for directories that contain a 'heatmap' subfolder.
    echo "Generating directory list from $WSI_DIR_ROOT..."
    find "$WSI_DIR_ROOT" -type d -name "heatmap" | sed 's|/heatmap$||' > "$DIR_LIST_FILE"
    
    # 2. Count the number of directories
    NUM_DIRS=$(wc -l < "$DIR_LIST_FILE")
    echo "Found $NUM_DIRS slide directories ready for prediction (containing 'heatmap')."

    # 3. Submit this script as a job array
    if [ "$NUM_DIRS" -gt 0 ]; then
        echo "Submitting job array for 1 to $NUM_DIRS tasks."
        # Submit self as array job
        sbatch --array=1-$NUM_DIRS "$0"
    else
        echo "No 'heatmap' directories found in $WSI_DIR_ROOT."
        echo "This usually means the 'TileMasker' stage of the pipeline hasn't finished yet."
    fi
    exit 0
fi

# --- This part of the script is executed by each SLURM array task ---

echo "--- Starting SLURM Array Task $SLURM_ARRAY_TASK_ID ---"

# 1. Get the specific directory path for this task from the list
DIR_TO_PROCESS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR_LIST_FILE")

echo "Task $SLURM_ARRAY_TASK_ID assigned to directory: $DIR_TO_PROCESS"

# 2. Initialize conda and set environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet
export WANDB_MODE=disabled

SYSTEM_CUDA_PATH=/usr/local/biotools/cuda/12.1
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${SYSTEM_CUDA_PATH}/lib64
export PATH=${SYSTEM_CUDA_PATH}/bin:${PATH}

# 3. Run the Python script on the single directory
if [ -d "$DIR_TO_PROCESS" ]; then
    echo "--- Environment ---"
    nvidia-smi
    echo "-------------------"
    echo "Using Strategy 1: Confidence Thresholding + Cleanup (0.65)"
    
    python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_gpu_focal_amyloid.py \
        --config_file /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-predict_Amyloid_focal_amyloid_pipeline.json \
        --wsi_dir "$DIR_TO_PROCESS"
else
    echo "ERROR: Directory '$DIR_TO_PROCESS' not found for task $SLURM_ARRAY_TASK_ID."
fi

echo "--- SLURM Array Task $SLURM_ARRAY_TASK_ID Complete ---"
