import pandas as pd
import os

INPUT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Braak_Valid_Cases/Amyloid_combined_analysis_Braak_cases.csv"
OUTPUT_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "AD_case_demographics_all.csv")

def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Select demographic columns
    demo_cols = ['NPID', 'Age', 'Sex', 'Race', 'ClinicalDx', 'PathDx', 'Braak stage', 'Thal phase']
    
    # Check if columns exist
    missing_cols = [c for c in demo_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return

    # Drop duplicates to get unique cases
    # We assume demographic info is consistent for an NPID across multiple slides
    unique_cases = df[demo_cols].drop_duplicates(subset='NPID').copy()
    
    # Sort by NPID
    unique_cases = unique_cases.sort_values('NPID')
    
    print(f"\nFound {len(unique_cases)} unique cases.")
    
    # Save the per-case demographic file
    unique_cases.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved unique case demographics to {OUTPUT_FILE}")
    
    # --- Generate Summary Statistics ---
    print("\n" + "="*40)
    print("      DEMOGRAPHIC SUMMARY (N={})".format(len(unique_cases)))
    print("="*40)
    
    # 1. Age
    # Convert to numeric, errors='coerce' to handle potential non-numeric data
    age_numeric = pd.to_numeric(unique_cases['Age'], errors='coerce')
    mean_age = age_numeric.mean()
    std_age = age_numeric.std()
    min_age = age_numeric.min()
    max_age = age_numeric.max()
    print(f"\nAge (years):")
    print(f"  Mean (SD): {mean_age:.1f} ({std_age:.1f})")
    print(f"  Range:     {min_age:.1f} - {max_age:.1f}")
    
    # 2. Sex
    print("\nSex:")
    print(unique_cases['Sex'].value_counts().to_string())
    
    # 3. Race
    print("\nRace:")
    print(unique_cases['Race'].value_counts().to_string())
    
    # 4. ClinicalDx
    print("\nClinical Diagnosis:")
    print(unique_cases['ClinicalDx'].value_counts().to_string())
    
    # 5. PathDx
    # PathDx can be long, maybe truncate or just list counts
    print("\nPathological Diagnosis:")
    print(unique_cases['PathDx'].value_counts().head(10).to_string()) # Limit to top 10 if many
    if len(unique_cases['PathDx'].unique()) > 10:
        print("  ... (showing top 10)")
        
    # 6. Braak Stage
    print("\nBraak Stage:")
    print(unique_cases['Braak stage'].value_counts().sort_index().to_string())
    
    # 7. Thal Phase
    print("\nThal Phase:")
    print(unique_cases['Thal phase'].value_counts().sort_index().to_string())
    
    print("="*40)

if __name__ == "__main__":
    main()
