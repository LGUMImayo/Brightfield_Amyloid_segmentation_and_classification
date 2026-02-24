import pandas as pd
import os

# Define file paths
BASE_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic"
FILE_CLASS = os.path.join(BASE_DIR, "Amyloid_classification_stats.csv")
FILE_SEG = os.path.join(BASE_DIR, "Amyloid_full_segmentation_stats.csv")
FILE_GW = os.path.join(BASE_DIR, "Amyloid_gray_white_stat.csv")
FILE_PATH = os.path.join(BASE_DIR, "Amyloid_pathology.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Amyloid_combined_analysis.csv")

def normalize_id(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    
    # Remove common file extensions
    extensions = ['.tiff', '.svs', '.ndpi', '.csv']
    for ext in extensions:
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
            
    # Remove _level-0 suffix
    if s.endswith('_level-0'):
        s = s[:-8]
        
    return s

def main():
    print("Loading dataframes...")
    
    # 1. Classification Stats
    # Key: image_id
    try:
        df_class = pd.read_csv(FILE_CLASS)
        print(f"Loaded Classification Stats: {df_class.shape}")
        if 'image_id' in df_class.columns:
            df_class['slide_id'] = df_class['image_id'].apply(normalize_id)
            # drop original key if you want, or keep it. Let's keep distinct columns to avoid overwriting or confusion if they differ slightly
            # But we must drop the one we rename to avoid duplicate columns after merge if we rely on 'slide_id'
        else:
            print("Warning: 'image_id' not found in Classification Stats")
    except Exception as e:
        print(f"Error loading {FILE_CLASS}: {e}")
        df_class = pd.DataFrame()

    # 2. Full Segmentation Stats
    # Key: slide_id
    try:
        df_seg = pd.read_csv(FILE_SEG)
        print(f"Loaded Segmentation Stats: {df_seg.shape}")
        if 'slide_id' in df_seg.columns:
            df_seg['slide_id'] = df_seg['slide_id'].apply(normalize_id)
        else:
            print("Warning: 'slide_id' not found in Segmentation Stats")
    except Exception as e:
        print(f"Error loading {FILE_SEG}: {e}")
        df_seg = pd.DataFrame()

    # 3. Gray/White Stats
    # Key: image_name
    try:
        df_gw = pd.read_csv(FILE_GW)
        print(f"Loaded Gray/White Stats: {df_gw.shape}")
        if 'image_name' in df_gw.columns:
            df_gw['slide_id'] = df_gw['image_name'].apply(normalize_id)
        else:
            print("Warning: 'image_name' not found in Gray/White Stats")
    except Exception as e:
        print(f"Error loading {FILE_GW}: {e}")
        df_gw = pd.DataFrame()

    # 4. Pathology
    # Key: Slide_ID
    try:
        df_path = pd.read_csv(FILE_PATH)
        print(f"Loaded Pathology: {df_path.shape}")
        if 'Slide_ID' in df_path.columns:
            df_path['slide_id'] = df_path['Slide_ID'].apply(normalize_id)
        else:
            print("Warning: 'Slide_ID' not found in Pathology")
    except Exception as e:
        print(f"Error loading {FILE_PATH}: {e}")
        df_path = pd.DataFrame()

    print("Merging dataframes...")
    
    # Start merging. We use outer join to keep all records.
    # Base could be any, let's start with df_class
    
    # List of DFs to merge
    dfs_to_merge = [
        (df_class, 'Classification'),
        (df_seg, 'Segmentation'),
        (df_gw, 'GrayWhite'),
        (df_path, 'Pathology')
    ]
    
    final_df = None
    
    for df, name in dfs_to_merge:
        if df.empty:
            print(f"Skipping empty dataframe: {name}")
            continue
            
        if 'slide_id' not in df.columns:
            print(f"Skipping {name} (no slide_id column)")
            continue
            
        if final_df is None:
            final_df = df
        else:
            # Merge
            # Suffixes might be needed if column names collide
            final_df = pd.merge(final_df, df, on='slide_id', how='outer', suffixes=('', f'_{name}'))

    if final_df is not None:
        # Calculate wm_area_mm2 using gm_area_mm2 and gray_matter_percentage
        print("Calculating wm_area_mm2...")
        if 'gm_area_mm2' in final_df.columns and 'gray_matter_percentage' in final_df.columns:
            # Formula: Total Area = gm_area / (gm_percent/100)
            # wm_area = Total Area - gm_area
            # implies wm_area = gm_area * (100/gm_percent - 1)
            final_df['wm_area_mm2'] = final_df.apply(
                lambda row: row['gm_area_mm2'] * ((100 / row['gray_matter_percentage']) - 1)
                if pd.notnull(row['gm_area_mm2']) and pd.notnull(row['gray_matter_percentage']) and row['gray_matter_percentage'] != 0
                else None, axis=1
            )
        else:
            print("Warning: gm_area_mm2 or gray_matter_percentage not found, skipping wm_area_mm2 calculation")

        # Move slide_id to front
        cols = ['slide_id'] + [c for c in final_df.columns if c != 'slide_id']
        final_df = final_df[cols]
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully created combined analysis file: {OUTPUT_FILE}")
        print(f"Final shape: {final_df.shape}")
        print(final_df.head())
    else:
        print("No dataframes could be merged.")

if __name__ == "__main__":
    main()
