import pandas as pd
import os

# NOTE: We switch back to the FULL Combined Analysis file to ensure we get Braak 0 cases,
# because the previous 'Amyloid_combined_analysis_AD_cases.csv' might have already excluded them.
INPUT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis.csv"
OUTPUT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/AD_Demographics_Summary.csv"

def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter: Keep if ClinicalDx contains "AD" OR Braak Stage is 0
    print("Filtering: Clinical AD cases OR Braak Stage 0 (Controls)...")
    is_ad = df['ClinicalDx'].astype(str).str.contains('AD', na=False, case=False)
    is_braak0 = pd.to_numeric(df['Braak stage'], errors='coerce') == 0
    
    df = df[is_ad | is_braak0].copy()
    
    # Define Cohort Label
    df['Cohort'] = df.apply(lambda x: 'Control (Braak 0)' if pd.to_numeric(x['Braak stage'], errors='coerce') == 0 else 'Clinical AD', axis=1)

    # Aggregate Regions per patient before deduplication
    print("Aggregating regions per patient...")
    # Join unique regions for each NPID
    region_map = df.groupby('NPID')['Region'].apply(lambda x: "; ".join(sorted(x.dropna().unique()))).to_dict()

    # Define columns to keep for final table
    # Added PathDx as requested
    demo_cols = ['NPID', 'Cohort', 'Age', 'Sex', 'Race', 'ClinicalDx', 'PathDx', 'Braak stage', 'Thal phase']
    
    # Drop duplicates to get one row per patient
    patient_df = df.drop_duplicates(subset=['NPID'])
    patient_df = patient_df[demo_cols].copy()
    
    # Add the aggregated Regions column
    patient_df['Available_Regions'] = patient_df['NPID'].map(region_map)
    
    # Ensure numeric Age
    patient_df['Age'] = pd.to_numeric(patient_df['Age'], errors='coerce')
    
    print(f"\n--- Demographics Summary (N={len(patient_df)}) ---")
    print("\nCohort Counts:")
    print(patient_df['Cohort'].value_counts())
    
    # Sex
    print("\nSex Distribution:")
    print(patient_df['Sex'].value_counts())
    
    # Age Summary
    mean_age = patient_df['Age'].mean()
    std_age = patient_df['Age'].std()
    min_age = patient_df['Age'].min()
    max_age = patient_df['Age'].max()
    print(f"\nAge: {mean_age:.1f} ± {std_age:.1f} (Range: {min_age}-{max_age})")
    
    # PathDx Breakdown (Since ClinicalDx is filtered for AD)
    print("\nPath Diagnosis Distribution:")
    print(patient_df['PathDx'].value_counts().head(10))
    
    # Save table
    patient_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDemographic table saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
