import pandas as pd

FILE_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis.csv"

def inspect_columns():
    df = pd.read_csv(FILE_PATH)
    
    print("Unique ClinicalDx:")
    print(df['ClinicalDx'].unique())
    
    print("\nUnique Braak stage:")
    print(df['Braak stage'].unique())
    
    print("\nUnique Thal phase:")
    print(df['Thal phase'].unique())
    
    print("\nUnique Region:")
    print(df['Region'].unique())
    
    # Check for AD cases count
    ad_cases = df[df['ClinicalDx'].str.contains('AD', na=False, case=False)]
    print(f"\nNumber of AD cases found: {len(ad_cases)}")

if __name__ == "__main__":
    inspect_columns()
