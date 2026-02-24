import pandas as pd
import glob
import os

# Define the base directory to search
SEARCH_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/prediction_priority_threshold_wmw_1.5_gmt_0.2_wmt_0.2_clean_21_best_parameter"
OUTPUT_FILE = os.path.join(SEARCH_DIR, "Amyloid_gray_white_stat.csv")

def main():
    print(f"Searching for result CSVs in {SEARCH_DIR}...")
    # Recursively find all files ending in _result.csv
    # The user example: .../subfolder/AF..._result.csv
    csv_files = glob.glob(os.path.join(SEARCH_DIR, "**", "*_result.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} files.")
    
    if not csv_files:
        print("No CSV files found.")
        return

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Ensure we have the image_name column. 
            # In the sample provided, it had: image_name,gray_matter_percentage,white_matter_percentage
            # We accept whatever columns are there, assuming they are consistent.
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by image_name if available
        if 'image_name' in final_df.columns:
            final_df = final_df.sort_values('image_name')
        
        # Save combined CSV
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Combined stats saved to: {OUTPUT_FILE}")
        print(final_df.head())
    else:
        print("No valid dataframes to combine.")

if __name__ == "__main__":
    main()
