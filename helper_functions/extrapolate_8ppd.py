import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# ── Resolve repo root so common_fn and models are importable ──────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from models.model import ConfigurableANN
from common_fn.data import load_and_preprocess_data   # for engineer_features logic

# =============================
# Configuration
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENSEMBLE_SIZE = 5
CORE_2PPD_PATH = os.path.join(ROOT, "backend/datasets/core_set_2ppd.csv")
FEATURES_8PPD_PATH = os.path.join(ROOT, "backend/datasets/8ppd_features_full.csv")
OUTPUT_PATH = os.path.join(ROOT, "backend/datasets/8ppd_extrapolated_targets.csv")

# ── Per-model metadata (features must match training config exactly) ──────────
MODELS_INFO = [
    {
        "name": "Water",
        "dir": os.path.join(ROOT, "backend/neural_nets/water_predictor"),
        "target": "Mars Odyssey Neutron Spectrometer % WEH",
        "features": [
            'MOLA 128ppd Elevation',
            'OMEGA Est. Lambert Albedo 1080nm Ody et al 2012',
            'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
            'TES Basalt Abundance - Numeric',
            'Latitude (N)',
            'Mars Crustal Thickness (GSFC)'
        ]
    },
    {
        "name": "Dust",
        "dir": os.path.join(ROOT, "backend/neural_nets/dust_predictor"),
        "target": "OMEGA Ferric/Dust 860nm ratio Ody et al 2012",
        "features": [
            "MOLA 128ppd Elevation",
            "MOLA 128ppd Slope",
            "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "MOLA 128ppd Aspect",
            "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012"
        ]
    },
    {
        "name": "Surface_Temp",
        "dir": os.path.join(ROOT, "backend/neural_nets/temp_predictor"),
        "target": "Yearly Average Mars Surface Temperature (C)",
        "features": [
            "Latitude (N)",
            "MOLA 128ppd Elevation",
            "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
            "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "MOLA Roughness 0.6km Baseline - Numeric",
            "MOLA Roughness 2.4km Baseline - Numeric"
        ]
    },
    {
        "name": "Temp_Variation",
        "dir": os.path.join(ROOT, "backend/neural_nets/temp_variation_predictor"),
        "target": "Yearly Mars Surface Temperature Variation (C)",
        "features": [
            "Latitude (N)",
            "MOLA 128ppd Elevation",
            "MOLA 128ppd Slope",
            "MOLA 128ppd Aspect",
            "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
            "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "MOLA Roughness 0.6km Baseline - Numeric"
        ]
    },
    {
        "name": "Gravity",
        "dir": os.path.join(ROOT, "backend/neural_nets/gravity_predictor"),
        "target": "Mars Bouguer Gravity Anomaly",
        "features": [
            'MOLA 128ppd Elevation',
            'MOLA 128ppd Slope',
            'MOLA Roughness 2.4km Baseline - Numeric',
            'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
            'Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
            'TES Basalt Abundance - Numeric',
            'TES Rock Abundance (Nowicki and Christensen 2007) - Numeric',
            'Latitude (N)'
        ]
    },
    {
        "name": "TI_Variance",
        "dir": os.path.join(ROOT, "backend/neural_nets/ti_variance_predictor"),
        "target": "TI_Variance",
        "features": [
            "Latitude (N)",
            "MOLA 128ppd Elevation",
            "MOLA 128ppd Slope",
            "MOLA 128ppd Aspect",
            "MOLA Roughness 0.6km Baseline - Numeric",
            "MOLA Roughness 2.4km Baseline - Numeric",
            "TES Basalt Abundance - Numeric",
            "TES Andesite Abundance - Numeric",
            "TES Rock Abundance (Nowicki and Christensen 2007) - Numeric",
            "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
            "Mars Crustal Thickness (GSFC)"
        ]
    }
]

# Column rename map: 8ppd_features_full.csv headers → training header names
COLUMN_RENAME_MAP = {
    "Lon": "Longitude (E)",
    "Lat": "Latitude (N)",
    "mola_128ppd_aspect": "MOLA 128ppd Aspect",
    "mola_128ppd_slope": "MOLA 128ppd Slope",
    "MOLA_128ppd_topo": "MOLA 128ppd Elevation",
    "mola_roughness_0.6km_numeric": "MOLA Roughness 0.6km Baseline - Numeric",
    "mola_roughness_2.4km_numeric": "MOLA Roughness 2.4km Baseline - Numeric",
    "omega_albedo_r1080": "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
    "tes_dayside_ti_putzig_2007": "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "tes_nightside_ti_putzig_2007": "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "grav_ggmro_120_crust": "Mars Crustal Thickness (GSFC)",
    "TES_Andesite_numeric": "TES Andesite Abundance - Numeric",
    "TES_Basalt_numeric": "TES Basalt Abundance - Numeric",
    "tes_rock_abundance_numeric": "TES Rock Abundance (Nowicki and Christensen 2007) - Numeric"
}


def engineer_features(df: pd.DataFrame, features_list: list) -> tuple[pd.DataFrame, list]:
    """
    Mirrors common_fn/data.py: replaces Latitude and Aspect raw values with
    sin/cos encodings so the feature vector matches what the models were trained on.
    """
    out_features = list(features_list)
    lat_col = 'Latitude (N)'
    asp_col = 'MOLA 128ppd Aspect'

    if lat_col in df.columns:
        df['latitude_sin'] = np.sin(np.radians(df[lat_col]))
        df['latitude_cos'] = np.cos(np.radians(df[lat_col]))
        if lat_col in out_features:
            out_features = [f for f in out_features if f != lat_col]
            out_features.extend(['latitude_sin', 'latitude_cos'])

    if asp_col in df.columns:
        df['aspect_sin'] = np.sin(np.radians(df[asp_col]))
        df['aspect_cos'] = np.cos(np.radians(df[asp_col]))
        if asp_col in out_features:
            out_features = [f for f in out_features if f != asp_col]
            out_features.extend(['aspect_sin', 'aspect_cos'])

    return df, out_features


def build_model(params: dict, input_dim: int) -> ConfigurableANN:
    """Constructs a ConfigurableANN matching the saved hyperparameters."""
    return ConfigurableANN(
        input_dim=input_dim,
        layers=params["layers"],
        units=params["units"],
        activation=params["activation"],
        dropout=params.get("dropout", 0.0)
    )


def main():
    print(" Starting 8ppd Extrapolation...")
    print(f" Device: {DEVICE}")

    # ── Phase 1: Fit scalers on 2ppd data & load ensemble weights ────────────
    print(f"\n Loading 2ppd core set from {CORE_2PPD_PATH}...")
    df_2ppd = pd.read_csv(CORE_2PPD_PATH, na_values=['null', 'Null', 'NULL'])

    for info in MODELS_INFO:
        print(f"  Preparing [{info['name']}]...")

        params_path = os.path.join(info['dir'], "best_params.json")
        with open(params_path) as f:
            params = json.load(f)
        info['params'] = params

        # Derive TI_Variance target if needed
        temp_df = df_2ppd.copy()
        if info['name'] == "TI_Variance":
            day_col  = 'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
            night_col = 'Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
            temp_df[info['target']] = (
                pd.to_numeric(temp_df[day_col], errors='coerce') -
                pd.to_numeric(temp_df[night_col], errors='coerce')
            ).abs()

        # Keep only the rows where all required features + target exist
        needed = list(set(info['features'] + [info['target']]))
        needed = [c for c in needed if c in temp_df.columns]
        temp_df = temp_df[needed].dropna()

        # Apply the same feature engineering used during training
        temp_df, eng_feats = engineer_features(temp_df, info['features'])
        info['engineered_features'] = eng_feats

        # Load RobustScaler from trained artifact, or fit and cache if not yet saved
        import joblib
        scaler_path = os.path.join(info['dir'], "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"    Loaded RobustScaler from {scaler_path}")
        else:
            print(f"    Scaler not found at {scaler_path}. Fitting on 2ppd data and saving...")
            scaler = RobustScaler()
            scaler.fit(temp_df[eng_feats].values)
            joblib.dump(scaler, scaler_path)
        info['scaler'] = scaler

        # Load ensemble members using ConfigurableANN (supports gelu/elu/relu/tanh)
        input_dim = len(eng_feats)
        info['models'] = []
        for i in range(ENSEMBLE_SIZE):
            model_path = os.path.join(info['dir'], f"model_{i}.pt")
            model = build_model(params, input_dim).to(DEVICE)
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            info['models'].append(model)

        print(f"    activation={params['activation']}, layers={params['layers']}, "
              f"units={params['units']}, input_dim={input_dim}")

    # ── Phase 2: Run inference on 8ppd feature grid in chunks ────────────────
    print(f"\n Reading 8ppd features from {FEATURES_8PPD_PATH}...")
    chunk_size = 500_000
    reader = pd.read_csv(FEATURES_8PPD_PATH, chunksize=chunk_size)
    first_chunk = True

    for chunk in tqdm(reader, desc="Extrapolating chunks"):
        chunk = chunk.rename(columns=COLUMN_RENAME_MAP)

        # Coerce all feature columns to numeric, fill residual NaNs with median
        all_feature_cols = set(f for info in MODELS_INFO for f in info['features'])
        for col in all_feature_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                if chunk[col].isna().any():
                    median = chunk[col].median()
                    chunk[col] = chunk[col].fillna(median if not np.isnan(median) else 0.0)

        results_chunk = pd.DataFrame({
            'Lat': chunk['Latitude (N)'],
            'Lon': chunk['Longitude (E)']
        })

        for info in MODELS_INFO:
            proc_chunk, eng_feats = engineer_features(chunk.copy(), info['features'])
            X_scaled = info['scaler'].transform(proc_chunk[eng_feats].values)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

            ensemble_preds = []
            with torch.no_grad():
                for model in info['models']:
                    p = model(X_tensor).cpu().numpy().flatten()
                    ensemble_preds.append(p)

            preds_stack = np.stack(ensemble_preds)
            results_chunk[info['name']]                    = preds_stack.mean(axis=0)
            results_chunk[f"{info['name']}_Uncertainty"]   = preds_stack.std(axis=0)

        mode = 'w' if first_chunk else 'a'
        results_chunk.to_csv(OUTPUT_PATH, mode=mode, header=first_chunk, index=False)
        first_chunk = False

    print(f"\n Extrapolation complete. Output saved to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
