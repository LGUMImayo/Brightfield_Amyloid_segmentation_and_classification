#!/bin/bash
#SBATCH --job-name=gen_weights_train_amyloid_effnet
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/gen_weights_train_amyloid_effnet_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/gen_weights_train_amyloid_effnet_%j.err
#SBATCH --partition=gpu-n12-85g-1x-a100-40g
#SBATCH --gres=gpu:1
#SBATCH --time=15-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

echo "============================================================"
echo "Amyloid EfficientNet-B4 Training Pipeline"
echo "============================================================"
echo "Start time: $(date)"

# 1. Define the path to the system's CUDA installation
SYSTEM_CUDA_PATH=/usr/local/biotools/cuda/12.1

# 2. Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

# 3. Manually and explicitly construct the LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${SYSTEM_CUDA_PATH}/lib64

# 4. Add CUDA binaries to the main PATH as well
export PATH=${SYSTEM_CUDA_PATH}/bin:${PATH}

# --- CONTROL DEBUG MODE HERE ---
export RHIZONET_DEBUG="False"

echo ""
echo "--- STEP 1: GENERATE WEIGHT MAPS ---"
echo "Generating weight maps for Amyloid patches..."
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/generate_attention_maps.py \
    --data_dir /fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_CNN/RO1_Amyloid_testing/prepare_test

echo "Weight map generation complete."
echo ""

echo "--- STEP 2: TRAIN EFFICIENTNET-B4 ---"
nvidia-smi
echo "Debug Mode: $RHIZONET_DEBUG"
echo "-------------------"

# Run training with EfficientNet-B4 + Focal/Tversky loss
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/train_iron.py \
    --config_file /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-train_Amyloid_efficientnet_v2.json \
    --gpus 1 --strategy ddp --accelerator gpu

echo "============================================================"
echo "Training completed at $(date)"
echo "============================================================"
