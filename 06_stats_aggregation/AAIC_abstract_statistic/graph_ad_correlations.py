import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Define input/output paths
BASE_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic"
DATA_FILE = os.path.join(BASE_DIR, "Amyloid_combined_analysis.csv")
CORRELATION_FILE = os.path.join(BASE_DIR, "AD_correlations.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "AD_correlation_plots")

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGETS = ['Braak stage', 'Thal phase']
METRICS = [
    'gray_area_mm2', 
    'cgp_count', 'cgp_density_mm2', 
    'ccp_count', 'ccp_density_mm2', 
    'caa 1_count', 'caa 1_density_mm2', 
    'caa 2_count', 'caa 2_density_mm2', 
    'total_pathology_count', 'total_pathology_density_mm2', 
    'gm_pixel_count', 'amyloid_pixel_count', 
    'gm_area_mm2', 'amyloid_area_mm2', 
    'amyloid_density_percent', 
    'gray_matter_percentage', 'white_matter_percentage', 
    'wm_area_mm2'
]

def clean_data(df):
    """
    Cleans the dataframe before processing:
    1. Converts targets and metrics to numeric (coercing errors to NaN).
    2. Drops empty columns if any.
    """
    print("Running clearing function...")
    
    # Clean Targets
    for t in TARGETS:
        if t in df.columns:
            # Check how many non-numeric before
            non_numeric = pd.to_numeric(df[t], errors='coerce').isna().sum()
            total = len(df)
            # Convert
            df[t] = pd.to_numeric(df[t], errors='coerce')
            if non_numeric > 0:
                print(f"  - {t}: {non_numeric}/{total} values are NaN or non-numeric")

    # Clean Metrics
    for metric in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
    return df

def clean_filename(s):
    """Sanitize string for filename"""
    return "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in s).strip().replace(" ", "_")

def main():
    print(f"Reading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    # Filter for AD cases (as done in the analysis script)
    is_ad = df['ClinicalDx'].astype(str).str.contains('AD', na=False)
    is_braak0 = pd.to_numeric(df['Braak stage'], errors='coerce') == 0
    ad_df = df[is_ad | is_braak0].copy()
    print(f"Filtered AD cases (ClinicalDx) + Braak 0: {len(ad_df)}")
    
    # Run clearing function
    ad_df = clean_data(ad_df)

    # Read the correlations results to identify what to plot
    # The user asked to graph "each set", which might mean all calculated ones, 
    # but let's prioritize the ones that were calculated (present in the results CSV).
    # Since plotting all non-significant ones might be excessive, we can plot all but maybe grouping them.
    # Or, we can plot the significant ones in a 'Significant' folder and others in 'All'.
    # For now, let's plot ALL calculated pairs found in the summary CSV.
    
    if not os.path.exists(CORRELATION_FILE):
        print(f"Correlation file {CORRELATION_FILE} not found. Run analysis script first.")
        return

    res_df = pd.read_csv(CORRELATION_FILE)
    
    # Iterate through the results dataframe rows
    print(f"Generating plots for {len(res_df)} correlation pairs...")
    
    for idx, row in res_df.iterrows():
        group = row['Group']
        target = row['Target']
        metric = row['Metric']
        pval = row['P_Value']
        r = row['Spearman_R']
        
        # Prepare the data subset
        if group == "All Regions":
            plot_df = ad_df.copy()
            subdir = "All_Regions"
        elif group.startswith("Region: "):
            region = group.replace("Region: ", "")
            plot_df = ad_df[ad_df['Region'] == region].copy()
            subdir = clean_filename(region)
        else:
            continue
            
        # Ensure numeric (Should already be done by clean_data, but kept for safety/subsetting)
        # Drop NaNs for plotting
        plot_df = plot_df[[metric, target]].dropna()
        
        if plot_df.empty:
            continue

        # Create sub-directory for the region
        region_dir = os.path.join(PLOTS_DIR, subdir)
        os.makedirs(region_dir, exist_ok=True)
        
        # Plot
        plt.figure(figsize=(8, 6))
        
        # Determine if significant for title color
        title_color = 'red' if pval < 0.05 else 'black'
        
        # Scatter plot with regression line
        # Using jitter for categorical target (Staging is ordinal integers)
        # to make points visible if they overlap
        sns.regplot(x=target, y=metric, data=plot_df, x_jitter=0.1, 
                    scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
        
        plt.title(f"{group}\n{metric} vs {target}\nR={r:.3f}, p={pval:.4f}", color=title_color)
        plt.xlabel(target)
        plt.ylabel(metric)
        plt.tight_layout()
        
        # Save filename
        # e.g., Braak_stage_vs_gray_area_mm2.png
        filename = f"{clean_filename(target)}_vs_{clean_filename(metric)}.png"
        save_path = os.path.join(region_dir, filename)
        
        plt.savefig(save_path)
        plt.close()
        
        if (idx + 1) % 50 == 0:
            print(f"Generated {idx + 1} plots...")

    print(f"Finished generating plots in {PLOTS_DIR}")

if __name__ == "__main__":
    main()
