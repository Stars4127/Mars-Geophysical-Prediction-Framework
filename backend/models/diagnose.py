import pandas as pd
import numpy as np

def diagnose_dataset(csv_path, target_column, features, lat_col='Latitude (N)', aspect_col='MOLA 128ppd Aspect'):
    """
    Diagnoses dataset issues for ML pipelines.
    
    Reports:
    - Missing features / target
    - NaN counts
    - Target stats (min, max, mean, std)
    - Latitude min/max by train/test split suggestion
    - Aspect stats
    - Outliers
    """
    df = pd.read_csv(csv_path)
    print(f" Loaded CSV: {csv_path} | {len(df)} rows")

    # Check target
    if target_column not in df.columns:
        print(f" Target column '{target_column}' missing!")
    else:
        y = pd.to_numeric(df[target_column], errors='coerce')
        print(f" Target '{target_column}' stats:")
        print(f"   - Missing: {y.isna().sum()} / {len(y)} ({y.isna().mean()*100:.2f}%)")
        print(f"   - Min: {y.min():.3f} | Max: {y.max():.3f} | Mean: {y.mean():.3f} | Std: {y.std():.3f}")

    # Check features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f" Missing features: {missing_features}")
    else:
        print(f" All {len(features)} features found")

    # Feature NaNs
    for f in features:
        if f in df.columns:
            nan_pct = df[f].isna().mean() * 100
            if nan_pct > 0:
                print(f" Feature '{f}' has {nan_pct:.2f}% missing values")

    # Latitude & Aspect
    for col in [lat_col, aspect_col]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            print(f" {col} stats -> min: {vals.min():.2f}, max: {vals.max():.2f}, mean: {vals.mean():.2f}, NaNs: {vals.isna().sum()}")

    # Detect extreme outliers in target (3 std dev from mean)
    if target_column in df.columns:
        outlier_threshold = 3 * y.std()
        outliers = ((y - y.mean()).abs() > outlier_threshold).sum()
        print(f" Target outliers (>3σ): {outliers} points ({outliers/len(y)*100:.2f}%)")

    # Quick spatial split diagnostic
    if lat_col in df.columns:
        df['lat_bin'] = pd.cut(df[lat_col], bins=10)
        bins = df['lat_bin'].unique()
        print(f" Latitude bins for spatial split: {bins}")

    print(" Dataset diagnosis complete\n")
    return df

# =========================
# Usage example
# =========================
if __name__ == "__main__":
    df = diagnose_dataset(
        csv_path="backend/models/datasets/core_set_2ppd.csv",
        target_column='Yearly Mars Surface Temperature Variation (C)',
        features=[
            "Latitude (N)",
            "MOLA 128ppd Elevation",
            "MOLA 128ppd Slope",
            "MOLA 128ppd Aspect",
            "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
            "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "MOLA Roughness 0.6km Baseline - Numeric"
        ]
    )