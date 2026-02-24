import pandas as pd
import os

FILE_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis.csv"

def main():
    try:
        df = pd.read_csv(FILE_PATH)
        regions = df['Region'].dropna().unique()
        print("Unique Regions Found:")
        for r in sorted(regions):
            print(f"- {r}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
