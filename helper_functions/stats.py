import pandas as pd
import numpy as np

# --- CONFIG ---
INPUT_CSV = r"backend\models\datasets\source_csvs\core_set_2ppd_tiff_extrapolated.csv"
OUTPUT_STATS = r"helper_functions\stats.csv"

# --- LOAD DATASET ---
df = pd.read_csv(INPUT_CSV)

# Identify variable columns (exclude row/col/Lon/Lat)
exclude_cols = {"row", "col", "Lon", "Lat"}
var_cols = [c for c in df.columns if c not in exclude_cols]

stats_list = []

for col in var_cols:
    data = df[col].astype(np.float32)
    
    mean_val = data.mean()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    
    # Number of unique values
    num_unique = data.nunique(dropna=True)
    
    # Percentage of missing values
    percent_missing = 100 * data.isna().sum() / len(data)
    
    # Artificial bands detection: repeated values in blocks
    # Heuristic: check if any value occurs more than 10% of pixels
    counts = data.value_counts(dropna=True)
    most_common_frac = counts.iloc[0] / len(data) if len(counts) > 0 else 0
    artificial_bands = most_common_frac > 0.1
    
    # Lat/Lon min/max for pixels with valid data
    valid_idx = data.notna()
    lon_min = df.loc[valid_idx, "Lon"].min()
    lon_max = df.loc[valid_idx, "Lon"].max()
    lat_min = df.loc[valid_idx, "Lat"].min()
    lat_max = df.loc[valid_idx, "Lat"].max()
    
    stats_list.append({
        "variable": col,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "num_unique": num_unique,
        "percent_missing": percent_missing,
        "artificial_bands": artificial_bands,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max
    })

# --- SAVE STATS CSV ---
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(OUTPUT_STATS, index=False)
print(f" Saved dataset stats to {OUTPUT_STATS}")