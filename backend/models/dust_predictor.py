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
MODEL_DIR = "backend/models/neural_nets/dust_predictor"
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
class DustANN(nn.Module):
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
# Predictor
# =============================
class DustPredictor:
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
        df = pd.read_csv(self.path, na_values=['null', 'Null', 'NULL'])
        
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
        print(f" Dropped {original_len - len(df)} rows. Final size: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("Dataset empty after preprocessing.")

        # --- Spatial Splitting by Latitude Bands ---
        df["lat_bin"] = pd.cut(df["Latitude (N)"], bins=10)
        unique_bins = sorted(df["lat_bin"].unique().tolist())
        
        import random as rnd
        rnd.seed(SEED)
        rnd.shuffle(unique_bins)
        
        train_bins = unique_bins[:7]
        val_bins = unique_bins[7:9]
        test_bins = unique_bins[9:]
        
        df_train = df[df["lat_bin"].isin(train_bins)]
        df_val = df[df["lat_bin"].isin(val_bins)]
        df_test = df[df["lat_bin"].isin(test_bins)]
        
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
    def _train(self, model, lr, max_epochs=250, patience=15, trial_idx=None):
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
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
    def tune(self, n_trials=25, max_epochs=10):
        def objective(trial):
            layers = trial.suggest_int("layers", 1, 4)
            units = trial.suggest_int("units", 16, 128, step=16)
            activation = trial.suggest_categorical("activation", ["relu", "tanh"])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            model = DustANN(len(self.features), layers, units, activation).to(device)
            self._train(model, lr=lr, max_epochs=max_epochs, trial_idx=trial.number)

            model.eval()
            with torch.no_grad():
                preds = model(self.x_val_tensor).cpu().numpy().flatten()
            r2 = r2_score(self.y_val, preds)
            return -r2  

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
        if self.best_params is None:
            if os.path.exists(PARAMS_PATH):
                with open(PARAMS_PATH) as f:
                    self.best_params = json.load(f)
            else:
                raise RuntimeError("No saved params found. Run tune() first.")

        lr = self.best_params["lr"]
        layers = self.best_params["layers"]
        units = self.best_params["units"]
        activation = self.best_params["activation"]

        print(f" Training Ensemble of {ENSEMBLE_SIZE} models...")
        for i in range(ENSEMBLE_SIZE):
            print(f"\n--- Training Member {i+1}/{ENSEMBLE_SIZE} ---")
            torch.manual_seed(SEED + i)
            
            model = DustANN(len(self.features), layers, units, activation)
            model = self._train(model, lr, max_epochs=max_epochs)
            
            save_path = f"{MODEL_DIR}/model_{i}.pt"
            torch.save(model.state_dict(), save_path)
            print(f" Member {i+1} saved to {save_path}")

        print(f" Ensemble training complete in {MODEL_DIR}")
        self.model = model

    # ----------------------------
    # Load and evaluate (Ensemble)
    # ----------------------------
    def load_and_eval(self):
        # Check if any model exists, using the first ensemble member's path as a proxy
        if not os.path.exists(f"{MODEL_DIR}/model_0.pt"):
            raise FileNotFoundError(f"No trained ensemble models found in {MODEL_DIR}. Run train_best() first.")

        with open(PARAMS_PATH) as f:
            params = json.load(f)

        all_preds = []
        print(f" Loading Ensemble of {ENSEMBLE_SIZE} models from {MODEL_DIR}...")

        for i in range(ENSEMBLE_SIZE):
            model = DustANN(len(self.features), params["layers"], params["units"], params["activation"]).to(device)
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
    predictor = DustPredictor(
        path= "backend/models/datasets/core_set_2ppd.csv",
        target_column='OMEGA Ferric/Dust 860nm ratio Ody et al 2012',
        features=[
        "MOLA 128ppd Elevation",
        "MOLA 128ppd Slope",
        "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
        "MOLA 128ppd Aspect",
        "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012"
    ])
    predictor.preprocess()
    # predictor.tune(n_trials=25, max_epochs=10)
    # predictor.train_best()
    predictor.load_and_eval()

# Metrics
#  Dropped 24402 rows. Final size: 242993
# Spatial Split -> Train: 167582, Val: 50919, Test: 24492
#  Preprocessing complete
#  Loading Ensemble of 5 models from backend/models/neural_nets/dust_predictor...
#  Ensemble Test R²: 0.5678 | MAE: 0.0061
#  Mean Uncertainty (StdDev): 0.0045