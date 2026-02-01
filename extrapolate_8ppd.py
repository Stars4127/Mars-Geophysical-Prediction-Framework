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
FEATURES_8PPD_PATH = "backend/models/datasets/8ppd_core_features_for_mapping.csv"
OUTPUT_PATH = "backend/models/datasets/8ppd_extrapolated_targets.csv"

# Models Metadata
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
        "dir": "backend/models/neural_nets/temp_predictor", # Matches directory in file list
        "target": "Yearly Average Mars Surface Temperature (C)",
        "features": [
            "Latitude (N)",
            "MOLA 128ppd Elevation",
            "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
            "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "MOLA Roughness 0.6km Baseline - Numeric"
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

# Features missing from 8ppd set that must be pulled from 2ppd
MISSING_FEATURES = [
    "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
    "TES Basalt Abundance - Numeric",
    "TES Andesite Abundance - Numeric"
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
    print("🚀 Starting 8ppd Extrapolation...")
    
    # 1. Load 2ppd data for scalers and missing feature lookup
    print("📖 Loading 2ppd data...")
    df_2ppd = pd.read_csv(CORE_2PPD_PATH, na_values=['null', 'Null', 'NULL'])
    
    # Create lookup for missing features
    # Rounding to 0.5 resolution (2ppd)
    df_2ppd['lat_round'] = (df_2ppd['Latitude (N)'] * 2).round() / 2
    df_2ppd['lon_round'] = (df_2ppd['Longitude (E)'] * 2).round() / 2
    
    # Handle TI_Variance target if needed for scaler training (though target doesn't affect feature scaler)
    # Actually we just need features for the scalers.
    
    # Prepare Scalers and Models
    for info in MODELS_INFO:
        print(f"🛠  Preparing {info['name']} model...")
        
        # Load params
        params_path = os.path.join(info['dir'], "best_params.json")
        with open(params_path) as f:
            info['params'] = json.load(f)
            
        # Fit Scaler on 2ppd data
        # We need to replicate the preprocessing of the original scripts
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
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.eval()
            info['models'].append(model)

    # 2. Process 8ppd data in chunks
    print(f"📖 Reading 8ppd features from {FEATURES_8PPD_PATH}...")
    chunk_size = 500000
    reader = pd.read_csv(FEATURES_8PPD_PATH, chunksize=chunk_size)
    
    first_chunk = True
    
    # Prepare 2ppd lookup table for missing features
    lookup_df = df_2ppd.groupby(['lat_round', 'lon_round'])[MISSING_FEATURES].mean().reset_index()

    for chunk in tqdm(reader, desc="Processing chunks"):
        # Fill missing features by merging with lookup
        chunk['lat_round'] = (chunk['Latitude (N)'] * 2).round() / 2
        chunk['lon_round'] = (chunk['Longitude (E)'] * 2).round() / 2
        
        chunk = pd.merge(chunk, lookup_df, on=['lat_round', 'lon_round'], how='left')
        
        # For any still missing (NaN after merge), fill with global 2ppd mean
        for col in MISSING_FEATURES:
            if chunk[col].isna().any():
                chunk[col] = chunk[col].fillna(df_2ppd[col].mean())

        results_chunk = chunk[['OID', 'Latitude (N)', 'Longitude (E)']].copy()
        
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
            results_chunk[f"{info['target']}"] = mean_pred
            results_chunk[f"{info['target']}_Uncertainty"] = std_pred
            
        # Write to CSV
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        results_chunk.to_csv(OUTPUT_PATH, mode=mode, header=header, index=False)
        first_chunk = False

    print(f"✅ Extrapolation complete! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
