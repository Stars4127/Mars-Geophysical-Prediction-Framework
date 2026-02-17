import os
import rasterio
import numpy as np
import pandas as pd
from functools import reduce
from scipy.ndimage import generic_filter

# --- CONFIG ---
TIFF_FOLDER = r"helper_functions\tiffs"
OUTPUT_CSV = r"backend\models\datasets\core_set_2ppd_tiff_extrapolated.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# --- COLLECT TIFF FILES ---
tiff_files = [os.path.join(TIFF_FOLDER, f) for f in os.listdir(TIFF_FOLDER) if f.lower().endswith(".tif")]
if not tiff_files:
    raise ValueError("No TIFF files found in folder.")

all_dfs = []
common_shape = None
common_transform = None

for tiff_path in tiff_files:
    var_name = os.path.splitext(os.path.basename(tiff_path))[0]
    print(f"Processing {var_name}...")

    with rasterio.open(tiff_path) as src:
        data = src.read(1).astype(np.float32)

        # --- Handle nodata robustly ---
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        # Catch extreme planetary nodata values
        data[data <= -1e30] = np.nan
        # Optional: treat zeros as missing if appropriate
        data[data == 0] = np.nan

        # --- Resample to common shape if needed ---
        if common_shape is None:
            common_shape = data.shape
            common_transform = src.transform
        else:
            if data.shape != common_shape:
                from rasterio.enums import Resampling
                data = src.read(
                    1,
                    out_shape=common_shape,
                    resampling=Resampling.bilinear
                ).astype(np.float32)

        # --- Interpolate missing pixels with 3x3 neighbor mean ---
        # Only applies where NaN exists
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            data = generic_filter(data, np.nanmean, size=3, mode='constant', cval=np.nan)

        # --- Create dataframe with row/col ---
        rows, cols = np.indices(common_shape)
        df = pd.DataFrame({
            "row": rows.flatten(),
            "col": cols.flatten(),
            var_name: data.flatten()
        })
        all_dfs.append(df)

# --- Merge all rasters on grid indices ---
merged_df = reduce(lambda l, r: pd.merge(l, r, on=["row", "col"], how="outer"), all_dfs)

# --- Compute Lat/Lon from row/col once ---
xs, ys = rasterio.transform.xy(common_transform, merged_df["row"], merged_df["col"])
merged_df["Lon"] = xs
merged_df["Lat"] = ys

# --- Save CSV ---
merged_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved merged CSV to {OUTPUT_CSV}")