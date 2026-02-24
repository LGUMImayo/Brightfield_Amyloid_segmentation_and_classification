import pandas as pd
import os

FILE_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis.csv"

def get_sets(df, col, filter_str):
    # Ensure string type for reliable contains check
    subset = df[df[col].astype(str).str.contains(filter_str, na=False)]
    slides = set(subset['slide_id'].dropna().unique())
    patients = set(subset['NPID'].dropna().unique())
    return slides, patients

def print_set_stats(name_a, set_a, name_b, set_b, item_name):
    intersection = set_a.intersection(set_b)
    only_a = set_a - set_b
    only_b = set_b - set_a
    
    print(f"\n--- {item_name} Overlap Analysis ---")
    print(f"Total {name_a}: {len(set_a)}")
    print(f"Total {name_b}: {len(set_b)}")
    print(f"Intersection (Both): {len(intersection)}")
    print(f"Only in {name_a}: {len(only_a)}")
    print(f"Only in {name_b}: {len(only_b)}")
    
    return intersection, only_a, only_b

def main():
    if not os.path.exists(FILE_PATH):
        print("Data file not found.")
        return
        
    print(f"Reading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    # 1. Clinical AD
    # Note: Using case-sensitive 'AD' as per previous scripts, but ensure standard matching
    clin_slides, clin_patients = get_sets(df, 'ClinicalDx', 'AD')
    
    # 2. Pathological AD
    path_slides, path_patients = get_sets(df, 'PathDx', 'AD')
    
    # Analyze Slides
    intersect_slides, only_clin_slides, only_path_slides = print_set_stats("ClinicalDx AD", clin_slides, "PathDx AD", path_slides, "Slides")
    
    # Analyze Patients
    intersect_pts, only_clin_pts, only_path_pts = print_set_stats("ClinicalDx AD", clin_patients, "PathDx AD", path_patients, "NPIDs (Patients)")
    
    print("\n" + "="*50)
    print("PATIENT DETAILS (NPIDs)")
    print("="*50)
    
    print(f"\nPatients Clinical only ({len(only_clin_pts)}):")
    print(", ".join(sorted(list(only_clin_pts))))
    
    print(f"\nPatients PathDx only ({len(only_path_pts)}):")
    print(", ".join(sorted(list(only_path_pts))))

    print(f"\nPatients in Both ({len(intersect_pts)}):")
    print(", ".join(sorted(list(intersect_pts))))

if __name__ == "__main__":
    main()
