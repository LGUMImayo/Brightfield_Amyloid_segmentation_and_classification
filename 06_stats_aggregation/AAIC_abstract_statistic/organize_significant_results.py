import pandas as pd
import os
import shutil

# Define paths
BASE_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic"
INPUT_CSV = os.path.join(BASE_DIR, "AD_correlations.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "AD_correlations_significant.csv")
SOURCE_PLOTS_DIR = os.path.join(BASE_DIR, "AD_correlation_plots")
DEST_PLOTS_DIR = os.path.join(BASE_DIR, "AD_correlation_plots_significant")

def clean_filename(s):
    """Sanitize string for filename - Must match logic in graph_ad_correlations.py"""
    return "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in s).strip().replace(" ", "_")

def main():
    # Clear old outputs
    print("Clearing old output files...")
    if os.path.exists(OUTPUT_CSV):
        try:
            os.remove(OUTPUT_CSV)
            print(f"Removed old output CSV: {OUTPUT_CSV}")
        except Exception as e:
            print(f"Error removing {OUTPUT_CSV}: {e}")
            
    if os.path.exists(DEST_PLOTS_DIR):
        try:
            shutil.rmtree(DEST_PLOTS_DIR)
            print(f"Removed old output directory: {DEST_PLOTS_DIR}")
        except Exception as e:
            print(f"Error removing {DEST_PLOTS_DIR}: {e}")

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Filter for significant results (P < 0.05)
    sig_df = df[df['P_Value'] < 0.05].copy()
    
    print(f"Found {len(sig_df)} significant correlations.")
    
    if sig_df.empty:
        print("No significant results found.")
        return

    # Save significant CSV
    sig_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Significant results saved to {OUTPUT_CSV}")

    # Organizer plots
    print(f"Organizing plots into {DEST_PLOTS_DIR}...")
    os.makedirs(DEST_PLOTS_DIR, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    for _, row in sig_df.iterrows():
        group = row['Group']
        target = row['Target']
        metric = row['Metric']
        
        # Determine subdirectory
        if group == "All Regions":
            subdir = "All_Regions"
        elif str(group).startswith("Region: "):
            region = group.replace("Region: ", "")
            subdir = clean_filename(region)
        else:
            print(f"Skipping unknown group format: {group}")
            continue
            
        # Determine filename
        filename = f"{clean_filename(target)}_vs_{clean_filename(metric)}.png"
        
        # Source path
        src_path = os.path.join(SOURCE_PLOTS_DIR, subdir, filename)
        
        # Dest path (preserve structure)
        dest_subdir = os.path.join(DEST_PLOTS_DIR, subdir)
        os.makedirs(dest_subdir, exist_ok=True)
        dest_path = os.path.join(dest_subdir, filename)
        
        # Copy
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            print(f"Warning: Plot not found: {src_path}")
            missing_count += 1
            
    print(f"Finished.")
    print(f"Plots copied: {copied_count}")
    print(f"Plots missing: {missing_count}")

if __name__ == "__main__":
    main()
