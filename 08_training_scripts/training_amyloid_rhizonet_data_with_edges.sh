#!/bin/bash
#SBATCH --job-name=train_amy_edges
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/train_amy_edges_%j.log
#SBATCH --partition=gpu-n12-85g-1x-a100-40g
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=15-00:00:00

# 1. Define the path to the system's CUDA installation
# We will build our paths from this instead of using 'module load'
SYSTEM_CUDA_PATH=/usr/local/biotools/cuda/12.1

# 2. Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

export WANDB_API_KEY="cee28daf562db3ae9590a61bd1cb12f1a08f36fa"
export WANDB_RUN_ID=$SLURM_JOB_ID
# Force wandb to create its directory inside the rhizonet project folder
export WANDB_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/wandb"

# 3. Manually and explicitly construct the LD_LIBRARY_PATH
# This ensures both conda's libraries and the system's CUDA libraries are found.
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${SYSTEM_CUDA_PATH}/lib64

# 4. Add CUDA binaries to the main PATH as well
export PATH=${SYSTEM_CUDA_PATH}/bin:${PATH}

# 5. (DEBUG) Print the environment variables to the log file
echo "--- ENVIRONMENT ---"
echo "Which Python: $(which python)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "PATH: ${PATH}"
nvidia-smi
echo "-------------------"
# 6. Run the training script
# The --gpus flag is deprecated, use --devices instead for modern PyTorch Lightning
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/train.py --config_file /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-train_Amyloid_new_data_with_edges.json --gpus 1 --strategy ddp --accelerator gpu

# 7. Automatically sync the completed offline run to the cloud
echo "Training finished. Syncing run to Weights & Biases..."

# The offline run directory is named with the format: offline-run-YYYYMMDD_HHMMSS-RUN_ID
# We find the directory associated with the current SLURM job ID.
RUN_DIR=$(find /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/wandb/ -type d -name "*-${SLURM_JOB_ID}")

if [ -n "$RUN_DIR" ]; then
    WANDB_INSECURE_DISABLE_SSL=true wandb sync "$RUN_DIR"
    echo "Sync complete."
else
    echo "Error: Could not find the wandb run directory for job ID ${SLURM_JOB_ID}."
fi
