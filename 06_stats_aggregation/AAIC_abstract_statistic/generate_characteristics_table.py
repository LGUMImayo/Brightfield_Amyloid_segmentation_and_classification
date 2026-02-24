import pandas as pd
import numpy as np
from scipy import stats

# Paths
demo_summ_path = 'Amyloid_Slides/AAIC_abstract_statistic/AD_Demographics_Summary.csv'
pathology_path = 'Amyloid_Slides/AAIC_abstract_statistic/Amyloid_pathology.csv'
output_path = 'Amyloid_Slides/AAIC_abstract_statistic/Participant_Characteristics_Table.csv'

# Load Data
print(f"Loading {demo_summ_path}...")
df_demo = pd.read_csv(demo_summ_path)

print(f"Loading {pathology_path}...")
# Pathology file might have encoding issues or bad lines, strictly reading header first could help but let's try standard
try:
    df_path = pd.read_csv(pathology_path)
except Exception as e:
    print(f"Error reading pathology file: {e}")
    # Try with on_bad_lines='skip' if needed
    df_path = pd.read_csv(pathology_path, on_bad_lines='skip')

# De-duplicate pathology file to get unique NPID attributes (taking first occurrence)
# Columns of interest from pathology that might not be in demo or useful to add
# FHx: Family History
# TDP-43: TDP-43 pathology presence
# LBD type: Lewy Body Disease type
extra_cols = ['NPID', 'FHx', 'TDP-43', 'LBD type'] 
existing_cols = [c for c in extra_cols if c in df_path.columns]
df_path_unique = df_path[existing_cols].drop_duplicates(subset=['NPID'])

# Merge
df_merged = pd.merge(df_demo, df_path_unique, on='NPID', how='left')

# Define Groups
group1_name = "Clinical AD"
group2_name = "Control (Braak 0)"

g1 = df_merged[df_merged['Cohort'] == group1_name]
g2 = df_merged[df_merged['Cohort'] == group2_name]

print(f"Group 1 ({group1_name}): {len(g1)}")
print(f"Group 2 ({group2_name}): {len(g2)}")

rows = []

def get_pval_str(p):
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

# 1. N
rows.append({
    'Characteristic': 'N',
    group1_name: str(len(g1)),
    group2_name: str(len(g2)),
    'Test Statistic': '-',
    'p value': '-'
})

# 2. Age (Mean/SD) - T-test
v = 'Age'
m1, s1 = g1[v].mean(), g1[v].std()
m2, s2 = g2[v].mean(), g2[v].std()
t_stat, p_val = stats.ttest_ind(g1[v].dropna(), g2[v].dropna(), equal_var=False)

rows.append({
    'Characteristic': 'Age (years), Mean (SD)',
    group1_name: f"{m1:.1f} ({s1:.1f})",
    group2_name: f"{m2:.1f} ({s2:.1f})",
    'Test Statistic': f"t={t_stat:.3f}",
    'p value': get_pval_str(p_val)
})

# 3. Sex (M/F) - Chi-square
v = 'Sex'
vc1 = g1[v].value_counts()
vc2 = g2[v].value_counts()
m1_cnt = vc1.get('M', 0)
f1_cnt = vc1.get('F', 0)
m2_cnt = vc2.get('M', 0)
f2_cnt = vc2.get('F', 0)

contingency = [[m1_cnt, m2_cnt], [f1_cnt, f2_cnt]] # Columns: AD, Ctrl. Rows: M, F
# Note: scipy expects [[r1c1, r1c2], [r2c1, r2c2]] 
# Row 1 (M): AD_M, Ctrl_M. Row 2 (F): AD_F, Ctrl_F
cont_table = [[m1_cnt, m2_cnt], [f1_cnt, f2_cnt]] 

chi2, p_val, dof, ex = stats.chi2_contingency(cont_table)

rows.append({
    'Characteristic': 'Sex (M/F)',
    group1_name: f"{m1_cnt}/{f1_cnt}",
    group2_name: f"{m2_cnt}/{f2_cnt}",
    'Test Statistic': f"chi2={chi2:.3f}",
    'p value': get_pval_str(p_val)
})

# 4. Braak Stage (Mean/SD for display, Mann-Whitney U for test)
v = 'Braak stage'
# Should be numeric.
m1, s1 = g1[v].mean(), g1[v].std()
m2, s2 = g2[v].mean(), g2[v].std()
u_stat, p_val = stats.mannwhitneyu(g1[v].dropna(), g2[v].dropna())

rows.append({
    'Characteristic': 'Braak stage, Mean (SD)',
    group1_name: f"{m1:.1f} ({s1:.1f})",
    group2_name: f"{m2:.1f} ({s2:.1f})",
    'Test Statistic': f"U={u_stat:.1f}",
    'p value': get_pval_str(p_val)
})

# Braak Stage Breakdown (0-6)
# Ensure v is numeric integer for comparison if possible, though mean calculation worked so it is numeric.
# We will display counts for 0-6.
for i in range(7):
    # Using np.isclose in case of slight float issues, or just equality if int
    c1 = g1[v].apply(lambda x: 1 if x == i else 0).sum()
    c2 = g2[v].apply(lambda x: 1 if x == i else 0).sum()
    
    # Only add row if at least one subject has this stage (optional, but standard tables usually show all or used ones)
    # Let's show all 0-6 as standard AD staging
    rows.append({
        'Characteristic': f'  Braak Stage {i} (%)',
        group1_name: f"{c1} ({c1/len(g1)*100:.1f}%)",
        group2_name: f"{c2} ({c2/len(g2)*100:.1f}%)",
        'Test Statistic': '-',
        'p value': '-'
    })

# 5. Thal Phase
v = 'Thal phase'
m1, s1 = g1[v].mean(), g1[v].std()
m2, s2 = g2[v].mean(), g2[v].std()
u_stat, p_val = stats.mannwhitneyu(g1[v].dropna(), g2[v].dropna())

rows.append({
    'Characteristic': 'Thal phase, Mean (SD)',
    group1_name: f"{m1:.1f} ({s1:.1f})",
    group2_name: f"{m2:.1f} ({s2:.1f})",
    'Test Statistic': f"U={u_stat:.1f}",
    'p value': get_pval_str(p_val)
})

# Thal Phase Breakdown (0-5)
for i in range(6):
    c1 = g1[v].apply(lambda x: 1 if x == i else 0).sum()
    c2 = g2[v].apply(lambda x: 1 if x == i else 0).sum()
    
    rows.append({
        'Characteristic': f'  Thal Phase {i} (%)',
        group1_name: f"{c1} ({c1/len(g1)*100:.1f}%)",
        group2_name: f"{c2} ({c2/len(g2)*100:.1f}%)",
        'Test Statistic': '-',
        'p value': '-'
    })

# 6. Race (Caucasian %)
v = 'Race'
def count_cauc(series):
    return series.astype(str).str.contains('Caucasian', case=False).sum()

y1 = count_cauc(g1[v])
n1 = len(g1) - y1
y2 = count_cauc(g2[v])
n2 = len(g2) - y2

# Check if any non-caucasian exists to decide if we run stats (if 100% vs 100%, chi2 fails or is 1.0)
rows.append({
    'Characteristic': 'Race (Caucasian %)',
    group1_name: f"{y1} ({y1/len(g1)*100:.1f}%)",
    group2_name: f"{y2} ({y2/len(g2)*100:.1f}%)",
    'Test Statistic': '-', # Skip test if homogenous or effectively so
    'p value': '-'
})

# 7. Family History (FHx) % Yes
v = 'FHx'
# Normalize 'Yes' vs others
# Check unique values
# Assuming 'Yes' and 'No' or nan
if v in df_merged.columns:
    def count_yes(series):
        return series.astype(str).str.contains('Yes', case=False).sum()
    
    y1 = count_yes(g1[v])
    n1 = len(g1) - y1 # Treat unknown as No? Or exclude? Let's use total N
    # Actually table usually shows % of known, or % of total.
    # Let's show % of Total (assuming NAs are No/Unknown)
    
    y2 = count_yes(g2[v])
    n2 = len(g2) - y2
    
    # Chi-square
    cont_table = [[y1, y2], [n1, n2]]
    chi2, p_val, _, _ = stats.chi2_contingency(cont_table)
    
    pct1 = (y1/len(g1))*100
    pct2 = (y2/len(g2))*100
    
    rows.append({
        'Characteristic': 'Family History (Yes %)',
        group1_name: f"{y1} ({pct1:.1f}%)",
        group2_name: f"{y2} ({pct2:.1f}%)",
        'Test Statistic': f"chi2={chi2:.3f}",
        'p value': get_pval_str(p_val)
    })

# 8. TDP-43 Pathology (Presence > 0)
v = 'TDP-43'
if v in df_merged.columns:
    # Assuming numeric 0, 1, 2, 3 etc. or strings "0", "1"...
    # Safely convert to numeric, coerce errors (strings) to NaN then fill?
    # Or just check if string is '0' or nan.
    def is_tdp_present(x):
        try:
            val = float(x)
            return val > 0
        except:
            return str(x).lower() not in ['nan', 'none', '0', '']
            
    y1 = g1[v].apply(is_tdp_present).sum()
    y2 = g2[v].apply(is_tdp_present).sum()
    n1, n2 = len(g1), len(g2)
    
    cont_table = [[y1, y2], [n1-y1, n2-y2]]
    chi2, p_val, _, _ = stats.chi2_contingency(cont_table)
    
    rows.append({
        'Characteristic': 'TDP-43 Pathology (%)',
        group1_name: f"{y1} ({y1/n1*100:.1f}%)",
        group2_name: f"{y2} ({y2/n2*100:.1f}%)",
        'Test Statistic': f"chi2={chi2:.3f}",
        'p value': get_pval_str(p_val)
    })

# 9. LBD Pathology (Presence)
v = 'LBD type'
if v in df_merged.columns:
    # Anything not "0", "None", NaN is present
    def is_lbd_present(x):
        s = str(x).lower()
        if s in ['nan', 'none', '0', '']: return False
        try:
             if float(x) == 0: return False
        except: pass
        return True
    
    y1 = g1[v].apply(is_lbd_present).sum()
    y2 = g2[v].apply(is_lbd_present).sum()
    n1, n2 = len(g1), len(g2)
    
    cont_table = [[y1, y2], [n1-y1, n2-y2]]
    chi2, p_val, _, _ = stats.chi2_contingency(cont_table)
    
    rows.append({
        'Characteristic': 'Lewy Body Pathology (%)',
        group1_name: f"{y1} ({y1/n1*100:.1f}%)",
        group2_name: f"{y2} ({y2/n2*100:.1f}%)",
        'Test Statistic': f"chi2={chi2:.3f}",
        'p value': get_pval_str(p_val)
    })

# Create DataFrame
result_df = pd.DataFrame(rows)
print("\nGenerated Table:")
print(result_df.to_string(index=False))

result_df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
