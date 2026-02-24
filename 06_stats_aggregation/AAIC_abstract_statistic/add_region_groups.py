import pandas as pd
import os

FILE_PATH = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis.csv"
OUTPUT_FILE = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/AAIC_abstract_statistic/Amyloid_combined_analysis_with_groups.csv"

# Define Mappings based on neuropathological staging
# Note: This is an approximation. Adjust as necessary.

BRAAK_REGION_MAPPING = {
    # Braak I-II (Transentorhinal/Entorhinal/Hippocampal)
    'Posterior Hippocampus': 'Braak_I_II',
    'Anterior hippocampus': 'Braak_I_II',
    
    # Braak III-IV (Limbic)
    'Subthalamic nucleus': 'Braak_III_IV',
    'Basal forebrain': 'Braak_III_IV', # Often affected early but part of subcortical
    'Cingulate cortex and Superior Frontal': 'Braak_V_VI', # Cingulate is III/IV, Superior Frontal is V/VI. Splitting hard. Assignment to later for now.
    
    # Braak V-VI (Isocortical)
    'Medial frontal': 'Braak_V_VI', 
    'Superior temporal': 'Braak_V_VI',
    'Inferior parietal': 'Braak_V_VI',
    'Visual cortex': 'Braak_V_VI',
    'Motor cortex': 'Braak_V_VI', 
    
    # Other/Unclassified for Braak Staging purposes in this context
    'Caudate/Putamen/Nucleus Accumbens': 'Subcortical',
    'Midbrain': 'Brainstem',
    'Pons': 'Brainstem',
    'Medulla': 'Brainstem',
    'Cerebellum': 'Cerebellum',
    'Cerebellar vermis': 'Cerebellum'
}

THAL_REGION_MAPPING = {
    # Phase 1: Neocortex
    'Medial frontal': 'Thal_1',
    'Superior temporal': 'Thal_1',
    'Inferior parietal': 'Thal_1',
    'Visual cortex': 'Thal_1',
    'Motor cortex': 'Thal_1',
    
    # Phase 2: Allocortex
    'Posterior Hippocampus': 'Thal_2',
    'Anterior hippocampus': 'Thal_2',
    'Cingulate cortex and Superior Frontal': 'Thal_1_2_Mix', # Treating as Neocortical mostly for Superior Frontal
    
    # Phase 3: Subcortical
    'Basal forebrain': 'Thal_3',
    'Caudate/Putamen/Nucleus Accumbens': 'Thal_3', 
    'Subthalamic nucleus': 'Thal_3',
    
    # Phase 4: Brainstem
    'Midbrain': 'Thal_4',
    'Pons': 'Thal_4',
    'Medulla': 'Thal_4',
    
    # Phase 5: Cerebellum
    'Cerebellum': 'Thal_5',
    'Cerebellar vermis': 'Thal_5'
}

GENERAL_REGION_MAPPING = {
    # Neocortex
    'Medial frontal': 'Neocortex',
    'Superior temporal': 'Neocortex',
    'Inferior parietal': 'Neocortex',
    'Visual cortex': 'Neocortex',
    'Motor cortex': 'Neocortex',
    'Cingulate cortex and Superior Frontal': 'Neocortex',
    
    # Hippocampus (Allocortex)
    'Posterior Hippocampus': 'Hippocampus',
    'Anterior hippocampus': 'Hippocampus',
    
    # Subcortical
    'Basal forebrain': 'Subcortical',
    'Caudate/Putamen/Nucleus Accumbens': 'Subcortical',
    'Subthalamic nucleus': 'Subcortical',
    
    # Brainstem
    'Midbrain': 'Brainstem',
    'Pons': 'Brainstem',
    'Medulla': 'Brainstem',
    
    # Cerebellum
    'Cerebellum': 'Cerebellum',
    'Cerebellar vermis': 'Cerebellum'
}

def main():
    print(f"Reading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    # Map Regions
    print("Mapping Braak Region Groups...")
    df['Braak_Region_Group'] = df['Region'].map(BRAAK_REGION_MAPPING)
    
    print("Mapping Thal Region Groups...")
    df['Thal_Region_Group'] = df['Region'].map(THAL_REGION_MAPPING)

    print("Mapping General Anatomical Groups...")
    df['Anatomical_Group'] = df['Region'].map(GENERAL_REGION_MAPPING)
    
    # Check for unmapped
    unmapped_braak = df[df['Braak_Region_Group'].isna()]['Region'].unique()
    if len(unmapped_braak) > 0:
        print(f"Warning: The following regions were not mapped for Braak: {unmapped_braak}")
        
    unmapped_thal = df[df['Thal_Region_Group'].isna()]['Region'].unique()
    if len(unmapped_thal) > 0:
        print(f"Warning: The following regions were not mapped for Thal: {unmapped_thal}")
        
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved grouped data to {OUTPUT_FILE}")
    
    # Display counts
    print("\nBraak Group Counts:")
    print(df['Braak_Region_Group'].value_counts())
    
    print("\nThal Group Counts:")
    print(df['Thal_Region_Group'].value_counts())

if __name__ == "__main__":
    main()
