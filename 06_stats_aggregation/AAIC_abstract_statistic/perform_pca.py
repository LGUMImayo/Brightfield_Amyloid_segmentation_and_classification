import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis_with_groups.csv"
SIGNIFICANT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/AD_correlations_significant.csv" # New input
OUTPUT_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/PCA_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC = 'amyloid_density_percent' # Main metric for PCA

def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for Clinical AD cases only
    print("Filtering for ClinicalDx containing 'AD'...")
    df = df[df['ClinicalDx'].str.contains('AD', na=False, case=False)]
    print(f"Data restricted to {df['NPID'].nunique()} patients with Clinical AD.")

    # 1. Get Significant Regions
    print(f"Reading significant regions from {SIGNIFICANT_FILE}...")
    sig_df = pd.read_csv(SIGNIFICANT_FILE)
    
    # CLEANING STEP: Remove "Region: " prefix from the 'Group' column
    # Also ignore "All Regions" as it's not a specific anatomical area
    significant_regions = []
    for r in sig_df['Group'].unique():
        if r == "All Regions":
            continue
        # Strip the prefix if it exists
        clean_r = r.replace("Region: ", "").strip()
        significant_regions.append(clean_r)
        
    print(f"Found {len(significant_regions)} significant regions: {significant_regions}")

    # 2. Pivot using RAW 'Region' (not groups) to match the significant list
    # Note: We must use the original 'Region' column, not 'Anatomical_Group' or 'Thal_Region_Group'
    print("Pivoting data table using raw 'Region'...")
    pivot_df = df.pivot_table(index='NPID', 
                              columns='Region', 
                              values=METRIC, 
                              aggfunc='mean')
    
    # 3. Filter Columns to keep only significant ones
    # Intersection of pivoted columns and significant regions
    cols_to_keep = [col for col in pivot_df.columns if col in significant_regions]
    
    if not cols_to_keep:
        print("Error: No overlapping regions found between data and significant list.")
        return

    print(f"Restricting PCA to {len(cols_to_keep)} regions: {cols_to_keep}")
    pivot_df = pivot_df[cols_to_keep]

    print(f"Initial shape after pivot: {pivot_df.shape}")
    
    # Filter for columns that are present in enough patients
    # If a region group is rare, drop it
    # If a patient is missing many regions, drop them
    
    # Let's keep columns with at least 30% data
    pivot_df = pivot_df.dropna(axis=1, thresh=len(pivot_df)*0.3)
    
    # Impute missing values with column mean to retain more patients
    # (Amyloid data is often missing not at random, but mean imputation is a start for PCA visualization)
    print("Imputing missing data with column means...")
    dropped_df = pivot_df.fillna(pivot_df.mean())
    
    print(f"Shape for PCA: {dropped_df.shape}")
    
    X = dropped_df.values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_scaled)
    
    # Create Result DataFrame
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['NPID'] = dropped_df.index
    
    # Merge back metadata (Braak stage, Thal phase, ClinicalDx)
    # We take the first entry for each NPID from original df
    meta_df = df[['NPID', 'Braak stage', 'Thal phase', 'ClinicalDx']].drop_duplicates('NPID')
    final_df = pd.merge(pca_df, meta_df, on='NPID', how='left')
    
    # Explained Variance
    expl_var = pca.explained_variance_ratio_
    print(f"Explained Variance: PC1={expl_var[0]:.2f}, PC2={expl_var[1]:.2f}")
    
    # Plotting
    plot_pca(final_df, 'Braak stage', expl_var)
    plot_pca(final_df, 'Thal phase', expl_var)
    plot_pca(final_df, 'ClinicalDx', expl_var)
    
    print("PCA complete.")

def plot_pca(df, color_col, expl_var):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='PC1', y='PC2', hue=color_col, data=df, palette='viridis', style=color_col, s=100)
    plt.title(f'PCA of Amyloid Distribution (Color: {color_col})\nPC1: {expl_var[0]:.2%}, PC2: {expl_var[1]:.2%}')
    
    filename = os.path.join(OUTPUT_DIR, f"PCA_{color_col.replace(' ', '_')}.png")
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

if __name__ == "__main__":
    main()
