import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import optuna

# =============================
# Global config
# =============================
SEED = 42
BATCH_SIZE = 1024
ENSEMBLE_SIZE = 5
MODEL_DIR = "backend/models/neural_nets/temp_predictor"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
MODEL_PATH = f"{MODEL_DIR}/model.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
os.makedirs(MODEL_DIR, exist_ok=True)
print(f" Using device: {device}")


# =============================
# MLP model
# =============================
class TempANN(nn.Module):
    def __init__(self, input_dim, layers, units, activation, dropout=0.0):
        super().__init__()
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            act_fn = nn.ReLU()
            
        net = []
        prev_dim = input_dim
        for _ in range(layers):
            net.append(nn.Linear(prev_dim, units))
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            net.append(act_fn)
            prev_dim = units
        net.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)


# =============================
# Predictor
# =============================
class TempPredictor:
    def __init__(self, path, target_column, features, test_size=0.2):
        self.path = path
        self.target_column = target_column
        self.features = features
        self.test_size = test_size
        self.scaler = RobustScaler()
        self.best_params = None

    # ----------------------------
    # Preprocess
    # ----------------------------
    def preprocess(self, val_fraction=0.2):
        df = pd.read_csv(self.path)
        # Select only relevant columns BEFORE dropping NAs
        # This prevents irrelevant empty columns from killing the dataset
        # Select subset: We need original features + target + Latitude (for splitting)
        lat_col = 'Latitude (N)'
        asp_col = 'MOLA 128ppd Aspect'
        
        # Engineering: Encoding Latitude and Aspect
        if lat_col in df.columns:
            df['latitude_sin'] = np.sin(np.radians(df[lat_col]))
            df['latitude_cos'] = np.cos(np.radians(df[lat_col]))
            if lat_col in self.features:
                self.features = [f for f in self.features if f != lat_col]
                self.features.extend(['latitude_sin', 'latitude_cos'])

        if asp_col in df.columns:
            df['aspect_sin'] = np.sin(np.radians(df[asp_col]))
            df['aspect_cos'] = np.cos(np.radians(df[asp_col]))
            if asp_col in self.features:
                self.features = [f for f in self.features if f != asp_col]
                self.features.extend(['aspect_sin', 'aspect_cos'])

        # Now filter df to needed columns (features, target_column, and lat_col for splitting)
        needed_cols = list(set(self.features + [self.target_column, lat_col]))
        needed_cols = [c for c in needed_cols if c in df.columns]
        df = df[needed_cols]

        # Coerce to numeric
        for col in needed_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        original_len = len(df)
        df = df.dropna()
        print(f" Dropped {original_len - len(df)} rows containing NaNs. Remaining: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("Dataset is empty after preprocessing! Check your features for 100% NaN columns.")

        # --- Spatial Splitting by Latitude Bands ---
        df["lat_bin"] = pd.cut(df["Latitude (N)"], bins=10)
        
        # We need stratified split
        df_train_val, df_test = train_test_split(df, test_size=0.10, stratify=df["lat_bin"], random_state=SEED)
        df_train, df_val = train_test_split(df_train_val, test_size=0.20, stratify=df_train_val["lat_bin"], random_state=SEED)
        
        print(f"Spatial Split -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        X_train = df_train[self.features].values
        self.y_train = df_train[self.target_column].values
        X_val = df_val[self.features].values
        self.y_val = df_val[self.target_column].values
        X_test = df_test[self.features].values
        y_test = df_test[self.target_column].values
        
        self.y_test_raw = y_test.copy()

        # self.y_train = np.log1p(y_train)
        # self.y_val = np.log1p(y_val)
        # self.y_test = np.log1p(y_test)

        self.x_train = self.scaler.fit_transform(X_train)
        self.x_val = self.scaler.transform(X_val)
        self.x_test = self.scaler.transform(X_test)

        # Dataloaders
        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(self.x_train, dtype=torch.float32),
                torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        self.x_val_tensor = torch.tensor(self.x_val, dtype=torch.float32).to(device)
        self.y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1).to(device)
        self.x_test_tensor = torch.tensor(self.x_test, dtype=torch.float32).to(device)

        print(" Preprocessing complete")

    # ----------------------------
    # Training loop
    # ----------------------------
    def _train(self, model, lr, max_epochs=250, patience=30, trial_idx=None, weight_decay=0.0):
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_ctr = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(self.x_val_tensor).cpu().numpy().flatten()
                val_r2 = r2_score(self.y_val, val_preds)
                val_loss = criterion(model(self.x_val_tensor), self.y_val_tensor).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1

            # ----- Simple epoch logging -----
            if trial_idx is not None:
                print(f"[Trial {trial_idx}]")
            print(f"Epoch {epoch + 1}/{max_epochs} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f} | Val R²: {val_r2:.4f}")

            if patience_ctr >= patience:
                print(f"⏹ Early stopping at epoch {epoch + 1}")
                break

        if best_state is None:
            best_state = model.state_dict()
        model.load_state_dict(best_state)
        return model

    # ----------------------------
    # Hyperparameter tuning (with trial logging)
    # ----------------------------
    def tune(self, n_trials=30, max_epochs=30):
        def objective(trial):
            layers = trial.suggest_int("layers", 1, 6)
            units = trial.suggest_int("units", 128, 256, step=16)
            activation = trial.suggest_categorical("activation", ["gelu", "elu"])
            lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
            use_dropout = trial.suggest_categorical("use_dropout", [True, False])
            dropout = trial.suggest_float("dropout", 0.1, 0.2) if use_dropout else 0.0

            model = TempANN(len(self.features), layers, units, activation, dropout=dropout).to(device)
            self._train(model, lr=lr, max_epochs=max_epochs, trial_idx=trial.number, weight_decay=weight_decay)

            # Evaluate on validation
            model.eval()
            with torch.no_grad():
                preds = model(self.x_val_tensor).cpu().numpy().flatten()
            r2 = r2_score(self.y_val, preds)
            return -r2  # maximize r2

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params

        # Save best params
        with open(PARAMS_PATH, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(" Best params saved:", self.best_params)

    # ----------------------------
    # Train best model (Ensemble)
    # ----------------------------
    def train_best(self, max_epochs=300):
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH) as f:
                params = json.load(f)
        else:
            raise RuntimeError("No saved params. Run tune() first.")

        lr = params["lr"]
        layers = params["layers"]
        units = params["units"]
        activation = params["activation"]
        weight_decay = params.get("weight_decay", 0.0)
        dropout = params.get("dropout", 0.0)

        print(f" Training Ensemble of {ENSEMBLE_SIZE} models...")
        # Ensure MODEL_DIR exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        for i in range(ENSEMBLE_SIZE):
            print(f"\n--- Training Member {i+1}/{ENSEMBLE_SIZE} ---")
            torch.manual_seed(SEED + i)
            
            model = TempANN(len(self.features), layers, units, activation, dropout=dropout)
            model = self._train(model, lr, max_epochs=max_epochs, weight_decay=weight_decay)
            
            save_path = f"{MODEL_DIR}/model_{i}.pt"
            torch.save(model.state_dict(), save_path)
            print(f" Member {i+1} saved to {save_path}")

        print(f" Ensemble training complete in {MODEL_DIR}")
        # self.model is set to the last trained model, but for ensemble,
        # individual models are loaded during evaluation.
        self.model = model 

    # ----------------------------
    # Load and evaluate (Ensemble)
    # ----------------------------
    def load_and_eval(self):
        if not os.path.exists(PARAMS_PATH):
            raise FileNotFoundError("No best params found. Run tune() first.")
        
        with open(PARAMS_PATH) as f:
            params = json.load(f)

        all_preds = []
        print(f" Loading Ensemble of {ENSEMBLE_SIZE} models from {MODEL_DIR}...")

        for i in range(ENSEMBLE_SIZE):
            dropout = params.get("dropout", 0.0)
            model = TempANN(len(self.features), params["layers"], params["units"], params["activation"], dropout=dropout).to(device)
            save_path = f"{MODEL_DIR}/model_{i}.pt"
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"Ensemble member {i} not found at {save_path}. Run train_best() first.")
            
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            model.eval()
            
            with torch.no_grad():
                p = model(self.x_test_tensor).cpu().numpy().flatten()
                all_preds.append(p)

        # Stack predictions
        preds_stack = np.stack(all_preds)
        
        # Calculate ensemble stats
        mean_pred = preds_stack.mean(0)
        uncertainty = preds_stack.std(0)

        r2 = r2_score(self.y_test_raw, mean_pred)
        mae = mean_absolute_error(self.y_test_raw, mean_pred)

        print(f" Ensemble Test R²: {r2:.4f} | MAE: {mae:.4f}")
        print(f" Mean Uncertainty (StdDev): {uncertainty.mean():.4f}")
        
        return mean_pred, uncertainty


# =============================
# Usage
# =============================
if __name__ == "__main__":
    predictor = TempPredictor(
        path="backend/models/datasets/core_set_2ppd.csv",
        target_column='Yearly Average Mars Surface Temperature (C)',
        features=[
            "Latitude (N)",
            "MOLA 128ppd Elevation",
            "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
            "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
            "MOLA Roughness 0.6km Baseline - Numeric",
            "MOLA Roughness 2.4km Baseline - Numeric"
        ]
    )
    predictor.preprocess()
    predictor.tune(n_trials=30, max_epochs=30)
    predictor.train_best(max_epochs=500)
    predictor.load_and_eval()

# Metrics
#  Dropped 5995 rows containing NaNs. Remaining: 261400
# Spatial Split -> Train: 183319, Val: 51896, Test: 26185
#  Preprocessing complete
#  Loading Ensemble of 5 models from backend/models/neural_nets/temp_predictor...
#  Ensemble Test R²: 0.7822 | MAE: 2.3193
#  Mean Uncertainty (StdDev): 1.6222