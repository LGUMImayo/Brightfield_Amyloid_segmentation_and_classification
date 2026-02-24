#!/bin/bash
#SBATCH --job-name=gen_atten_maps_amyloid
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/gen_atten_maps_amyloid_%j.out
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/gen_atten_maps_amyloid_%j.err
#SBATCH --partition=med-n16-64g
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

# Activate environment
source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

# Define the data directory
DATA_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_CNN/RO1_Amyloid_testing/prepare_test"

echo "Starting Attention Map Generation for Amyloid..."
echo "Target Directory: $DATA_DIR"

# Run the script
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/generate_attention_maps.py --data_dir "$DATA_DIR"

echo "Done."
