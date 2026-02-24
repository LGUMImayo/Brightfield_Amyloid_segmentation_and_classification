#!/bin/bash
#SBATCH --job-name=prep_amy_cleanup
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/prep_amy_cleanup_%j.out
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/prep_amy_cleanup_%j.err
#SBATCH --partition=med-n16-64g
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Activate environment
source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

# Cleanup
echo "Cleaning up Amyloid directories..."
rm -rf /fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_CNN/RO1_Amyloid_testing/pred_test
rm -rf /fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_CNN/RO1_Amyloid_testing/prepare_test

# Run patch preparation
echo "Running Amyloid patch preparation..."
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/prepare_patches.py --config_file /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-prepare_Amyloid_data.json
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/prepare_patches.py --config_file /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-prepare_Amyloid_new_data_with_edges.json
