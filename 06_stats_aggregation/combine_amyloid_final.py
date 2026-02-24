import pandas as pd
import glob
import os

# Define the source directory
SOURCE_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_statistics"
OUTPUT_FILE = os.path.join(SOURCE_DIR, "Amyloid_classification_final.csv")

def main():
    print(f"Searching for stats CSVs in {SOURCE_DIR}...")
    # Glob for files ending in _stats.csv
    csv_files = glob.glob(os.path.join(SOURCE_DIR, "*_stats.csv"))
    
    # Filter out the output file itself if it exists and matches the pattern
    csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(OUTPUT_FILE)]
    
    print(f"Found {len(csv_files)} files.")
    
    if not csv_files:
        print("No CSV files found.")
        return

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        # Concatenate all dataframes
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by image_id if available
        if 'image_id' in final_df.columns:
            final_df = final_df.sort_values('image_id')
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Combined statistics saved to: {OUTPUT_FILE}")
        print(f"Final shape: {final_df.shape}")
        print(final_df.head())
    else:
        print("No valid dataframes to combine.")

if __name__ == "__main__":
    main()
