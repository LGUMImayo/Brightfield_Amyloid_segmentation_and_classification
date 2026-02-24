#!/bin/bash
#SBATCH --job-name=cnn_pipeline_1_all
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/tf_pipeline_1_RO1_all_%j.out
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/tf_pipeline_1_RO1_all_%j.err
#SBATCH --partition=huge-n128-512g
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=30-00:00:00

# Set the target directory to loop through (all RO1_GCP subfolders)
TARGET_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP"
PROCESSED_CSV="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts/processed_slides_RO1_all.csv"
echo "Slide_Path,Slide_Name" > "$PROCESSED_CSV"

# Counter for processed slides
processed_count=0
skipped_count=0

# Find all directories that contain a "_files" subdirectory (indicating a slide directory)
# Then check if background_mask_full_res.tif exists in the final_mask folder
find "$TARGET_DIR" -type d -name "*_files" 2>/dev/null | while IFS= read -r files_dir; do
  # Get the parent directory (the slide directory)
  slide_dir=$(dirname "$files_dir")
  slide_name=$(basename "$slide_dir")
  
  # Verify this is a proper slide directory (slide_name_files pattern)
  expected_files_dir="${slide_dir}/${slide_name}_files"
  if [ "$files_dir" != "$expected_files_dir" ]; then
    continue
  fi
  
  # Check for the background_mask_full_res.tif file in the final_mask folder
  mask_file="${files_dir}/mask/final_mask/background_mask_full_res.tif"
  
  if [ ! -f "$mask_file" ]; then
    echo "Processing directory (Missing background_mask_full_res.tif): $slide_dir"
    echo "  - Missing: $mask_file"
    
    echo "$slide_dir,$slide_name" >> "$PROCESSED_CSV"
    
    # Run the pipeline for this slide
    bash /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/scripts/run_pipeline_full.sh "$slide_dir" '/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt'
    
    ((processed_count++))
  else
    echo "Skipping directory (background_mask_full_res.tif exists): $slide_dir"
    ((skipped_count++))
  fi
done

echo "Process complete."
echo "Processed: $processed_count slides"
echo "Skipped: $skipped_count slides"
