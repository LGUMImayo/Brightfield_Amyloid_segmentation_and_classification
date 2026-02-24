#!/bin/bash
#SBATCH --job-name=combine_stats
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/combine_stats_%j.out
#SBATCH --partition=med-n16-64g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

# Run the python script for Gray/White Matter Stats
echo "Running combine_gray_white_stats.py..."
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/combine_gray_white_stats.py

# Run the python script for Amyloid Classification Stats
echo "Running combine_classification_stats.py..."
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/combine_amyloid_final.py
