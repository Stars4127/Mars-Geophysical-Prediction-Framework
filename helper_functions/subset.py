import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# =============================
# CONFIGURATION
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "backend", "models", "datasets", "8ppd_extrapolated_targets.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "backend", "models", "datasets", "8ppd_extrapolated_targets_subset.csv")

CHUNK_SIZE = 1_000_000  
SCALE = 27.0 
PERCENT_PER_CELL = 0.02  # Target ~1M rows for 50M source

# Sampling grid
LAT_BINS = 180
LON_BINS = 360
lat_edges = np.linspace(-90, 90, LAT_BINS + 1)
lon_edges = np.linspace(-180, 180, LON_BINS + 1)

def main():
    if not os.path.exists(INPUT_PATH):
        print(f" Input file not found: {INPUT_PATH}")
        return

    print(f" Refactoring subset with coordinate transformation...")
    
    first_chunk = True
    total_rows = 0
    total_sampled = 0
    
    # Track global range for verification
    global_min_lat, global_max_lat = 999, -999
    global_min_lon, global_max_lon = 999, -999

    reader = pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)
    
    for chunk in tqdm(reader, desc="Processing", unit="chunk"):
        # 1. Transform: 0-4860 -> -90/90, 0-9720 -> -180/180
        chunk['Lat'] = (chunk['Lat'] / SCALE) - 90
        chunk['Lon'] = (chunk['Lon'] / SCALE) - 180
        
        # Strict Capping to ensure valid global bounds
        chunk['Lat'] = chunk['Lat'].clip(-90.0, 90.0)
        chunk['Lon'] = chunk['Lon'].clip(-180.0, 180.0)
        
        # 2. Assign Grid for stratified sampling
        chunk['lat_bin'] = np.digitize(chunk['Lat'], lat_edges) - 1
        chunk['lon_bin'] = np.digitize(chunk['Lon'], lon_edges) - 1
        chunk['lat_bin'] = chunk['lat_bin'].clip(0, LAT_BINS - 1)
        chunk['lon_bin'] = chunk['lon_bin'].clip(0, LON_BINS - 1)
        chunk['grid_cell'] = chunk['lat_bin'] * LON_BINS + chunk['lon_bin']
        
        # 3. Sample
        def safe_sample(group):
            n = max(1, int(len(group) * PERCENT_PER_CELL))
            return group.sample(n=min(len(group), n), random_state=42)

        sampled_chunk = chunk.groupby('grid_cell', group_keys=False).apply(safe_sample, include_groups=False)
        
        # 4. Update Global Bounds
        global_min_lat = min(global_min_lat, sampled_chunk['Lat'].min())
        global_max_lat = max(global_max_lat, sampled_chunk['Lat'].max())
        global_min_lon = min(global_min_lon, sampled_chunk['Lon'].min())
        global_max_lon = max(global_max_lon, sampled_chunk['Lon'].max())

        # 5. Cleanup & Save
        sampled_chunk = sampled_chunk.sort_values(['Lat', 'Lon'])
        cols_to_drop = [c for c in ['lat_bin', 'lon_bin', 'grid_cell'] if c in sampled_chunk.columns]
        sampled_chunk = sampled_chunk.drop(columns=cols_to_drop)
        
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        sampled_chunk.to_csv(OUTPUT_PATH, mode=mode, index=False, header=header)
        
        total_rows += len(chunk)
        total_sampled += len(sampled_chunk)
        first_chunk = False

    print(f"\n Done!")
    print(f" Processed: {total_rows:,} rows")
    print(f" Sampled: {total_sampled:,} rows")
    print(f" Full Range Lat: {global_min_lat:.2f} to {global_max_lat:.2f}")
    print(f" Full Range Lon: {global_min_lon:.2f} to {global_max_lon:.2f}")
    print(f" {OUTPUT_PATH}")

if __name__ == "__main__":
    main()