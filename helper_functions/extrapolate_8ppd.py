import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# =============================
# Helper: Generic MLP Model
# =============================
class GenericANN(nn.Module):
    def __init__(self, input_dim, layers, units, activation):
        super().__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        net = []
        prev_dim = input_dim
        for _ in range(layers):
            net.append(nn.Linear(prev_dim, units))
            net.append(act_fn)
            prev_dim = units
        net.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

# =============================
# Configuration
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENSEMBLE_SIZE = 5
CORE_2PPD_PATH = "backend/models/datasets/core_set_2ppd.csv"
FEATURES_8PPD_PATH = "backend/models/datasets/8ppd_features_full.csv"
OUTPUT_PATH = "backend/models/datasets/8ppd_extrapolated_targets.csv"

# Models Metadata
# Note: Feature names here must match exactly what was in the 2ppd training set.
# Mapping from 8ppd_features_full.csv will handle translation.
MODELS_INFO = [
    {
        "name": "Water",
        "dir": "backend/models/neural_nets/water_predictor",
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
        "dir": "backend/models/neural_nets/dust_predictor",
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
        "dir": "backend/models/neural_nets/temp_predictor",
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
        "dir": "backend/models/neural_nets/temp_variation_predictor",
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
        "dir": "backend/models/neural_nets/gravity_predictor",
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
        "dir": "backend/models/neural_nets/ti_variance_predictor",
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

def engineer_features(df, features_list):
    """Encodes Latitude and Aspect if present in features_list."""
    out_features = features_list.copy()
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

def main():
    print("🚀 Starting 8ppd Extrapolation (Direct Native Mode)...")
    
    # 1. Load 2ppd data ONLY for fitting scalers
    print("📖 Loading 2ppd data for scaler fitting...")
    df_2ppd = pd.read_csv(CORE_2PPD_PATH, na_values=['null', 'Null', 'NULL'])
    
    # Prepare Scalers and Models
    for info in MODELS_INFO:
        print(f"🛠  Preparing {info['name']} model...")
        
        # Load params
        params_path = os.path.join(info['dir'], "best_params.json")
        with open(params_path) as f:
            info['params'] = json.load(f)
            
        # Fit Scaler on 2ppd data
        temp_df = df_2ppd.copy()
        
        if info['name'] == "TI_Variance":
             day_ti_col = 'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
             night_ti_col = 'Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
             temp_df[info['target']] = (pd.to_numeric(temp_df[day_ti_col], errors='coerce') - 
                                         pd.to_numeric(temp_df[night_ti_col], errors='coerce')).abs()
        
        # Select features needed
        needed_cols = list(set(info['features'] + [info['target']]))
        needed_cols = [c for c in needed_cols if c in temp_df.columns]
        temp_df = temp_df[needed_cols].dropna()
        
        # Fit scaler
        temp_df, engineered_features = engineer_features(temp_df, info['features'])
        scaler = RobustScaler()
        scaler.fit(temp_df[engineered_features].values)
        info['scaler'] = scaler
        info['engineered_features'] = engineered_features
        
        # Load ensemble members
        info['models'] = []
        for i in range(ENSEMBLE_SIZE):
            model = GenericANN(
                input_dim=len(engineered_features),
                layers=info['params']['layers'],
                units=info['params']['units'],
                activation=info['params']['activation']
            ).to(DEVICE)
            model_path = os.path.join(info['dir'], f"model_{i}.pt")
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            
            # Standardize state dict keys (handle mix of 'net.' and 'model.' prefixes)
            new_state_dict = {k.replace("net.", "model."): v for k, v in state_dict.items()}
            
            model.load_state_dict(new_state_dict)
            model.eval()
            info['models'].append(model)

    # 2. Process 8ppd data in chunks (No more lookup needed!)
    print(f"📖 Reading 8ppd features from {FEATURES_8PPD_PATH}...")
    chunk_size = 500000
    reader = pd.read_csv(FEATURES_8PPD_PATH, chunksize=chunk_size)
    
    first_chunk = True
    
    # Column mapping to translate 8ppd_features_full.csv to Training Header Names
    column_rename_map = {
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

    for chunk in tqdm(reader, desc="Processing chunks"):
        # Rename columns to match 2ppd expected headers
        chunk = chunk.rename(columns=column_rename_map)
        
        # Clean up: Ensure all needed columns are numeric and handle NaNs
        # We fill NaNs with column medians for robustness, so gaps in one sensor 
        # (like Albedo) don't break the prediction for other sensors.
        for info in MODELS_INFO:
            for f in info['features']:
                if f in chunk.columns:
                    chunk[f] = pd.to_numeric(chunk[f], errors='coerce')
                    # Fill NaNs with median if still present
                    if chunk[f].isna().any():
                        chunk[f] = chunk[f].fillna(chunk[f].median() if not chunk[f].isna().all() else 0)
        
        # Prepare results chunk
        results_chunk = pd.DataFrame()
        results_chunk['Lat'] = chunk['Latitude (N)']
        results_chunk['Lon'] = chunk['Longitude (E)']
        
        # Predict for each model
        for info in MODELS_INFO:
            # Preprocess chunk for this model
            proc_chunk, eng_feats = engineer_features(chunk.copy(), info['features'])
            X_raw = proc_chunk[eng_feats].values
            X_scaled = info['scaler'].transform(X_raw)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
            
            # Ensemble inference
            ensemble_preds = []
            with torch.no_grad():
                for model in info['models']:
                    p = model(X_tensor).cpu().numpy().flatten()
                    ensemble_preds.append(p)
            
            preds_stack = np.stack(ensemble_preds)
            mean_pred = preds_stack.mean(0)
            std_pred = preds_stack.std(0)
            
            # Add to results
            results_chunk[f"{info['name']}"] = mean_pred
            results_chunk[f"{info['name']}_Uncertainty"] = std_pred
            
        # Write to CSV
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        results_chunk.to_csv(OUTPUT_PATH, mode=mode, header=header, index=False)
        first_chunk = False

    print(f"✅ Extrapolation complete! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
