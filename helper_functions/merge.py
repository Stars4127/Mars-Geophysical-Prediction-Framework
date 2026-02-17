import os
import pandas as pd

# ============================
# Configuration
# ============================
DATA_FOLDER = r"backend\models\datasets\source_csvs"   # Folder containing CSV files
OUTPUT_PATH = r"backend\models\datasets\8ppd_features_full.csv"
MERGE_KEYS = ["Lat", "Lon"]        # adjust to your actual column names
MERGE_HOW = "outer"                # 'outer' keeps all rows, 'inner' keeps only common rows

# ============================
# Find all CSV files
# ============================
csv_files = [os.path.join(DATA_FOLDER, f) 
             for f in os.listdir(DATA_FOLDER) 
             if f.lower().endswith(".csv")]

if not csv_files:
    raise ValueError(f"No CSV files found in {DATA_FOLDER}")

print(f"Found {len(csv_files)} CSV files to merge.")

# ============================
# Load first CSV as base
# ============================
merged = pd.read_csv(csv_files[0])
print(f"Loaded {csv_files[0]} with {len(merged)} rows.")

# ============================
# Merge remaining CSVs
# ============================
for file_path in csv_files[1:]:
    print(f"Merging {file_path}...")
    df = pd.read_csv(file_path)
    merged = pd.merge(merged, df, on=MERGE_KEYS, how=MERGE_HOW)

# ============================
# Handle missing values
# ============================
# Fill numeric NaNs with 0
numeric_cols = merged.select_dtypes(include='number').columns
merged[numeric_cols] = merged[numeric_cols].fillna(0)

# Fill non-numeric NaNs with empty string
non_numeric_cols = merged.select_dtypes(exclude='number').columns
merged[non_numeric_cols] = merged[non_numeric_cols].fillna("")

# ============================
# Save merged CSV
# ============================
merged.to_csv(OUTPUT_PATH, index=False)
print(f" All CSVs merged into {OUTPUT_PATH}, total rows: {len(merged)}")