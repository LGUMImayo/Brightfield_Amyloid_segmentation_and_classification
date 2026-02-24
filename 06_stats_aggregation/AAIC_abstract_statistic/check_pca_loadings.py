import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Define paths
INPUT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis_with_groups.csv"
SIGNIFICANT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/AD_correlations_significant.csv"
METRIC = 'amyloid_density_percent'

def main():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    df = df[df['ClinicalDx'].str.contains('AD', na=False, case=False)]
    
    # 2. Load Significant Regions and Clean Names
    print("Loading significant regions...")
    sig_df = pd.read_csv(SIGNIFICANT_FILE)
    significant_regions = []
    for r in sig_df['Group'].unique():
        if r == "All Regions": continue
        significant_regions.append(r.replace("Region: ", "").strip())
        
    # 3. Pivot Data
    pivot_df = df.pivot_table(index='NPID', columns='Region', values=METRIC, aggfunc='mean')
    
    # 4. Filter for only Significant Regions
    cols_to_keep = [c for c in pivot_df.columns if c in significant_regions]
    pivot_df = pivot_df[cols_to_keep]
    pivot_df = pivot_df.fillna(pivot_df.mean()) # Fill gaps
    
    # 5. Run PCA
    print(f"\nRunning PCA on {len(cols_to_keep)} regions: {cols_to_keep}")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    
    # 6. Print Loadings for PC1
    print("\n--- EVIDENCE: PC1 Feature Weights (Loadings) ---")
    print("Positive value means: Higher density moves dot to RIGHT")
    print("Negative value means: Higher density moves dot to LEFT")
    
    loadings = pd.DataFrame(pca.components_.T, index=pivot_df.columns, columns=['PC1', 'PC2'])
    print(loadings['PC1'].sort_values(ascending=False))

if __name__ == "__main__":
    main()