import pandas as pd
import os
import glob
from tqdm import tqdm

# --- Configuration ---
STATS_DIR = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_statistics'
MASTER_CSV_PATH = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_neuropath/Amyloid_neuropath.csv'
OUTPUT_CSV_PATH = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_neuropath/Amyloid_neuropath_stat.csv'

def main():
    print(f"Reading master CSV: {MASTER_CSV_PATH}")
    try:
        master_df = pd.read_csv(MASTER_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Master CSV not found at {MASTER_CSV_PATH}")
        return

    # Ensure Slide ID is string for matching
    if 'Slide ID' in master_df.columns:
        master_df['Slide ID'] = master_df['Slide ID'].astype(str).str.strip()
    else:
        print("Error: 'Slide ID' column not found in master CSV.")
        return

    print(f"Found {len(master_df)} rows in master CSV.")

    # Get list of all stat CSVs
    stat_files = glob.glob(os.path.join(STATS_DIR, '*_stats.csv'))
    print(f"Found {len(stat_files)} statistic files in {STATS_DIR}")

    # Create a list to hold all the stats dataframes
    all_stats = []

    for file_path in tqdm(stat_files, desc="Reading stat files"):
        try:
            df = pd.read_csv(file_path)
            
            # Extract Slide ID from image_id
            # Format: {SlideID}_level-0_stats.csv or inside the csv image_id column: {SlideID}_level-0
            if 'image_id' in df.columns and not df.empty:
                image_id = str(df.iloc[0]['image_id'])
                
                # Logic: Name before '_level-0' is the Slide ID
                if '_level-0' in image_id:
                    slide_id = image_id.split('_level-0')[0]
                else:
                    # Fallback if naming convention differs slightly
                    slide_id = image_id
                
                df['Slide ID'] = slide_id
                all_stats.append(df)
            else:
                print(f"Warning: 'image_id' not found or empty in {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not all_stats:
        print("No valid statistic data found to merge.")
        return

    # Concatenate all individual stats into one big dataframe
    combined_stats_df = pd.concat(all_stats, ignore_index=True)
    
    # Drop the 'image_id' column from stats if you don't want it in the final merge, 
    # or keep it. Let's keep it but ensure we don't have duplicate columns during merge.
    
    print("Merging data...")
    # Merge master_df with combined_stats_df on 'Slide ID'
    # how='left' keeps all rows from the master neuropath sheet, adding stats where matches are found.
    merged_df = pd.merge(master_df, combined_stats_df, on='Slide ID', how='left')

    print(f"Saving merged result to: {OUTPUT_CSV_PATH}")
    merged_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()