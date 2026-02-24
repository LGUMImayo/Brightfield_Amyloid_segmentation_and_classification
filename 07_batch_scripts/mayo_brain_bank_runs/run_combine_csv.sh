#!/bin/bash
#SBATCH --job-name=combine_csvs
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/combine_csvs_%j.out
#SBATCH --partition=med-n16-64g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

# Run the python script
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/amyloid_predictions_final_v1/combine_results.py