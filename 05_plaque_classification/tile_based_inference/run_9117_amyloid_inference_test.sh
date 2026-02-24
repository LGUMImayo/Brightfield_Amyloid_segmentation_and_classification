#!/bin/bash
#SBATCH --job-name=amyloid_tiles_9117
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_tiles_9117_%j.out
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amyloid_tiles_9117_%j.err
#SBATCH --partition=med-n16-64g
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

echo "--- Starting Amyloid Inference on Tiles ---"

# Setup Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate matt_code

# Run Script
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/script/Plaque_classification_with_amyloid_segmentation_in_rhizonet/run_inference_on_tiles_9117.py

echo "--- Job Complete ---"