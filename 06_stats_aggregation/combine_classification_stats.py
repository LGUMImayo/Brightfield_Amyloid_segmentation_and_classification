import pandas as pd
import glob
import os

# Define the directory
SEARCH_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Amyloid_Slides/Amyloid_prediction"
OUTPUT_FILE = os.path.join(SEARCH_DIR, "Amyloid_classification_stat.csv")

def main():
    print(f"Searching for CSVs in {SEARCH_DIR}...")
    # Flat search as per the file listing observed
    csv_files = glob.glob(os.path.join(SEARCH_DIR, "*.csv"))
    
    # Filter out the output file itself if it exists to avoid recursion loop if re-run
    csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(OUTPUT_FILE)]
    
    print(f"Found {len(csv_files)} files.")
    
    if not csv_files:
        print("No CSV files found.")
        return

    summary_rows = []

    for f in csv_files:
        try:
            # Read CSV
            df = pd.read_csv(f)
            
            # Extract basic info
            image_name = os.path.basename(f).replace('.csv', '')
            
            # Get Gray Area (assuming constant for the slide, take first value)
            if 'gray_area' in df.columns and not df.empty:
                gray_area = df['gray_area'].iloc[0]
            else:
                gray_area = 0 # Or None
                
            # Count labels
            if 'label' in df.columns:
                # Value counts gives a Series with index=label, value=count
                label_counts = df['label'].value_counts().to_dict()
            else:
                label_counts = {}
            
            # Construct row
            row = {
                'image_name': image_name,
                'gray_area': gray_area
            }
            
            # Add label counts to row, prefixing with 'label_' for clarity
            for label, count in label_counts.items():
                row[f'label_{label}_count'] = count
                
            summary_rows.append(row)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if summary_rows:
        # Create DataFrame
        final_df = pd.DataFrame(summary_rows)
        
        # Fill NaNs with 0 for counts (if a label didn't appear in a file)
        # Identify count columns
        count_cols = [c for c in final_df.columns if 'count' in c]
        final_df[count_cols] = final_df[count_cols].fillna(0)
        
        # Sort by image_name
        final_df = final_df.sort_values('image_name')
        
        # Save
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Combined classification stats saved to: {OUTPUT_FILE}")
        print(final_df.head())
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    main()
