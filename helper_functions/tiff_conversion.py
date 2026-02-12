import os
import rasterio
import numpy as np
import pandas as pd

# Folder containing all JMARS TIFFs (one per variable)
TIFF_FOLDER = r"helper_functions\tiffs"
OUTPUT_CSV = r"backend\models\datasets\source_csvs\mars_combined.csv"

# Collect all TIFF file paths
tiff_files = [os.path.join(TIFF_FOLDER, f) for f in os.listdir(TIFF_FOLDER) if f.lower().endswith(".tif")]
if not tiff_files:
    raise ValueError("No TIFF files found in folder.")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

all_dfs = []

for tiff_path in tiff_files:
    var_name = os.path.splitext(os.path.basename(tiff_path))[0]  # use filename as variable name
    print(f"Processing {var_name}...")

    with rasterio.open(tiff_path) as src:
        # Read the raster data
        data = src.read(1)  # first band
        data = data.astype(float)
        data[data == src.nodata] = np.nan  # replace nodata with NaN

        # Get coordinates
        rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        xs = np.array(xs)
        ys = np.array(ys)

        # Flatten everything
        df = pd.DataFrame({
            "Lon": xs.flatten(),
            "Lat": ys.flatten(),
            var_name: data.flatten()
        })

        all_dfs.append(df)

# Merge all variables on Lat/Lon
from functools import reduce
merged_df = reduce(lambda left, right: pd.merge(left, right, on=["Lat", "Lon"], how="outer"), all_dfs)

# Save to CSV
merged_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved merged CSV to {OUTPUT_CSV}")