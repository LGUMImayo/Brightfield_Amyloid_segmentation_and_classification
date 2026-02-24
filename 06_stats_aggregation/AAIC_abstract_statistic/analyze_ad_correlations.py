import pandas as pd
from scipy import stats
import os
import numpy as np

FILE_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis.csv"
OUTPUT_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "AD_correlations.csv")

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

TARGETS = ['Braak stage', 'Thal phase']

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

def calculate_correlations(df, group_name="All Regions"):
    results = []
    
    for metric in METRICS:
        if metric not in df.columns:
            continue
            
        # Drop NaNs for the pair
        for target in TARGETS:
            subset = df[['slide_id', metric, target]].dropna()
            
            # Get list of slide IDs used
            included_slides = ";".join(subset['slide_id'].astype(str).tolist())
            
            if len(subset) < 3: # Need at least a few points for correlation
                corr, p_val = np.nan, np.nan
                n = len(subset)
            else:
                corr, p_val = stats.spearmanr(subset[metric], subset[target])
                n = len(subset)
            
            results.append({
                'Group': group_name,
                'N': n,
                'Metric': metric,
                'Target': target,
                'Spearman_R': corr,
                'P_Value': p_val,
                'Slide_IDs': included_slides
            })
            
    return results

def main():
    print(f"Reading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    # Filter for AD cases (ClinicalDx) or Braak Stage 0
    is_ad = df['ClinicalDx'].str.contains('AD', na=False)
    is_braak0 = pd.to_numeric(df['Braak stage'], errors='coerce') == 0
    ad_df = df[is_ad | is_braak0].copy()
    print(f"Filtered AD cases (ClinicalDx) + Braak 0: {len(ad_df)}")
    
    # Save filtered AD cases to CSV
    FILTERED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Amyloid_combined_analysis_AD_cases.csv")
    ad_df.to_csv(FILTERED_OUTPUT_FILE, index=False)
    print(f"Saved filtered AD cases to {FILTERED_OUTPUT_FILE}")
    
    # Run clearing function
    ad_df = clean_data(ad_df)
    
    all_results = []
    
    # 1. Global Analysis
    print("Calculating global correlations...")
    all_results.extend(calculate_correlations(ad_df, "All Regions"))
    
    # 2. Per-Region Analysis
    print("Calculating per-region correlations...")
    regions = ad_df['Region'].dropna().unique()
    for region in regions:
        region_df = ad_df[ad_df['Region'] == region]
        if len(region_df) >= 5: # Threshold to run analysis
            all_results.extend(calculate_correlations(region_df, f"Region: {region}"))
        else:
            print(f"Skipping region '{region}' (N={len(region_df)})")
            
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by Group, Target, and P-Value
    if not results_df.empty:
        results_df = results_df.sort_values(['Group', 'Target', 'P_Value'])
        
        # Save
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Results saved to {OUTPUT_FILE}")
        
        # Print significant findings (p < 0.05)
        print("\nSignificant Findings (P < 0.05):")
        sig_df = results_df[results_df['P_Value'] < 0.05]
        if not sig_df.empty:
            print(sig_df[['Group', 'Target', 'Metric', 'Spearman_R', 'P_Value', 'N']].to_string(index=False))
        else:
            print("No significant correlations found.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
