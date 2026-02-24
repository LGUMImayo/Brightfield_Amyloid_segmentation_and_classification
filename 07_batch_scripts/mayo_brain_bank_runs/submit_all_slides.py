import os
import subprocess
import glob

# --- CONFIGURATION ---
TIFF_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/tiff"
OUTPUT_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/amyloid_predictions_final_v1"
MASK_ROOT_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/prediction_priority_threshold_wmw_1.5_gmt_0.2_wmt_0.2_clean_21_best_parameter"
MODEL_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_CNN/RO1_Amyloid_testing/log_new_data_with_edges/last.ckpt"
SCRIPT_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_amyloid_thin_slides.py"

# Parameters (Same as your best v4 run)
PARAMS = {
    "resolution": 0.2827,
    "model_input_size": 128,
    "gm_threshold": 0.9,
    "blur_strength": 0,
    "min_grain_size": 100
}

def submit_job(tiff_file):
    fname = os.path.basename(tiff_file)
    job_name = f"amy_{fname[:10]}"
    
    # Construct the python command
    cmd = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/amy_{fname}_%j.out
#SBATCH --partition=gpu-n16-60g-1x-tesla-t4
#SBATCH --gres=gpu:1
#SBATCH --mem=59G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

python {SCRIPT_PATH} \\
    --tiff_path "{tiff_file}" \\
    --mask_root_dir "{MASK_ROOT_DIR}" \\
    --output_dir "{OUTPUT_DIR}" \\
    --model_path "{MODEL_PATH}" \\
    --resolution {PARAMS['resolution']} \\
    --model_input_size {PARAMS['model_input_size']} \\
    --gm_threshold {PARAMS['gm_threshold']} \\
    --blur_strength {PARAMS['blur_strength']} \\
    --min_grain_size {PARAMS['min_grain_size']}
"""
    
    # Write temporary sbatch file
    temp_script = f"temp_submit_{fname}.sh"
    with open(temp_script, "w") as f:
        f.write(cmd)
        
    print(f"Submitting {fname}...")
    subprocess.run(["sbatch", temp_script])
    os.remove(temp_script)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tiff_files = glob.glob(os.path.join(TIFF_DIR, "*.tiff"))
    print(f"Found {len(tiff_files)} slides to process.")
    
    for tiff in tiff_files:
        # Check if already processed (optional, check for csv)
        fname = os.path.basename(tiff).split('.')[0]
        expected_csv = os.path.join(OUTPUT_DIR, f"{fname}_stats.csv")
        
        if os.path.exists(expected_csv):
            print(f"Skipping {fname} (Already done)")
            continue
            
        submit_job(tiff)

if __name__ == "__main__":
    main()