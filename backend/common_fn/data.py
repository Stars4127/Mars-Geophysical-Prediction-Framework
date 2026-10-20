import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

class DatasetSplits:
    def __init__(self, train_loader, x_val_tensor, y_val_tensor, x_test_tensor, y_test_raw, y_train, y_val, scaler, features):
        self.train_loader = train_loader
        self.x_val_tensor = x_val_tensor
        self.y_val_tensor = y_val_tensor
        self.x_test_tensor = x_test_tensor
        self.y_test_raw = y_test_raw
        self.y_train = y_train
        self.y_val = y_val
        self.scaler = scaler
        self.features = features

def load_and_preprocess_data(
    csv_path: str,
    target_column: str,
    features: list,
    batch_size: int = 1024,
    seed: int = 42,
    device: str = "cpu",
    derived_target_fn = None,
    save_scaler_path: str = None
) -> DatasetSplits:
    """
    Loads dataset, handles optional derived target, encodes periodic features,
    performs spatial latitude splitting, fits robust scaler, and constructs PyTorch DataLoaders/Tensors.
    Optionally saves the fitted RobustScaler to save_scaler_path using joblib.
    """
    df = pd.read_csv(csv_path, na_values=['null', 'Null', 'NULL'])
    feature_list = list(features)

    # 1. Custom or built-in target derivation if target column is not present
    if derived_target_fn is not None:
        df = derived_target_fn(df, target_column)
    elif target_column == 'TI_Variance' and target_column not in df.columns:
        day_ti_col = 'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
        night_ti_col = 'Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
        df[day_ti_col] = pd.to_numeric(df[day_ti_col], errors='coerce')
        df[night_ti_col] = pd.to_numeric(df[night_ti_col], errors='coerce')
        df[target_column] = (df[day_ti_col] - df[night_ti_col]).abs()

        # 3-sigma clipping
        mean_y, std_y = df[target_column].mean(), df[target_column].std()
        lower, upper = mean_y - 3 * std_y, mean_y + 3 * std_y
        df[target_column] = df[target_column].clip(lower, upper)
        print(f" Target '{target_column}' derived and clipped to ±3σ range: [{lower:.2f}, {upper:.2f}]")

    lat_col = 'Latitude (N)'
    asp_col = 'MOLA 128ppd Aspect'

    # 2. Periodic feature engineering
    if lat_col in df.columns:
        df['latitude_sin'] = np.sin(np.radians(df[lat_col]))
        df['latitude_cos'] = np.cos(np.radians(df[lat_col]))
        if lat_col in feature_list:
            feature_list = [f for f in feature_list if f != lat_col]
            feature_list.extend(['latitude_sin', 'latitude_cos'])

    if asp_col in df.columns:
        df['aspect_sin'] = np.sin(np.radians(df[asp_col]))
        df['aspect_cos'] = np.cos(np.radians(df[asp_col]))
        if asp_col in feature_list:
            feature_list = [f for f in feature_list if f != asp_col]
            feature_list.extend(['aspect_sin', 'aspect_cos'])

    # 3. Filter required columns
    needed_cols = list(set(feature_list + [target_column, lat_col]))
    needed_cols = [c for c in needed_cols if c in df.columns]
    df = df[needed_cols]

    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    original_len = len(df)
    df = df.dropna()
    print(f" Dropped {original_len - len(df)} rows. Final size: {len(df)}")

    if len(df) == 0:
        raise ValueError("Dataset empty after preprocessing.")

    # 4. Spatial Splitting by Latitude Bands
    df["lat_bin"] = pd.cut(df[lat_col], bins=10)
    df_train_val, df_test = train_test_split(df, test_size=0.10, stratify=df["lat_bin"], random_state=seed)
    df_train, df_val = train_test_split(df_train_val, test_size=0.20, stratify=df_train_val["lat_bin"], random_state=seed)

    print(f"Spatial Split -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    X_train = df_train[feature_list].values
    y_train = df_train[target_column].values
    X_val = df_val[feature_list].values
    y_val = df_val[target_column].values
    X_test = df_test[feature_list].values
    y_test_raw = df_test[target_column].values

    if save_scaler_path and os.path.exists(save_scaler_path):
        print(f" Loading RobustScaler from {save_scaler_path}")
        scaler = joblib.load(save_scaler_path)
        x_train_scaled = scaler.transform(X_train)
    else:
        scaler = RobustScaler()
        x_train_scaled = scaler.fit_transform(X_train)
        if save_scaler_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_scaler_path)), exist_ok=True)
            joblib.dump(scaler, save_scaler_path)
            print(f" Saved RobustScaler to {save_scaler_path}")

    x_val_scaled = scaler.transform(X_val)
    x_test_scaled = scaler.transform(X_test)

    # 5. DataLoaders and Tensors
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    return DatasetSplits(
        train_loader=train_loader,
        x_val_tensor=x_val_tensor,
        y_val_tensor=y_val_tensor,
        x_test_tensor=x_test_tensor,
        y_test_raw=y_test_raw,
        y_train=y_train,
        y_val=y_val,
        scaler=scaler,
        features=feature_list
    )
