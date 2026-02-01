import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import optuna

# =============================
# Global config
# =============================
SEED = 42
BATCH_SIZE = 1024
ENSEMBLE_SIZE = 5
MODEL_DIR = "backend/models/neural_nets/ti_variance_predictor"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"🔥 Using device: {device}")


# =============================
# MLP model
# =============================
class ThermalInertiaVarianceANN(nn.Module):
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
class ThermalInertiaVariancePredictor:
    def __init__(self, path, features):
        self.path = path
        # Target will be derived during preprocessing
        self.target_column = 'TI_Variance'
        self.original_features = features.copy()
        self.features = features.copy()
        self.scaler = RobustScaler()
        self.best_params = None

    # ----------------------------
    # Preprocess
    # ----------------------------
    def preprocess(self):
        df = pd.read_csv(self.path)
        
        # Derived Target: Diurnal Thermal Inertia Variance
        day_ti_col = 'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
        night_ti_col = 'Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'
        
        # Coerce TI columns to numeric before subtraction
        df[day_ti_col] = pd.to_numeric(df[day_ti_col], errors='coerce')
        df[night_ti_col] = pd.to_numeric(df[night_ti_col], errors='coerce')
        
        # Derivation (Absolute difference as variance proxy for diurnal swing)
        df[self.target_column] = (df[day_ti_col] - df[night_ti_col]).abs()
        
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

        # Filter df to needed columns
        needed_cols = list(set(self.features + [self.target_column, lat_col]))
        needed_cols = [c for c in needed_cols if c in df.columns]
        df = df[needed_cols]

        # Coerce all to numeric
        for col in needed_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        original_len = len(df)
        df = df.dropna()
        print(f"📉 Dropped {original_len - len(df)} rows containing NaNs. Remaining: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("Dataset is empty after preprocessing! Check feature alignment.")

        # --- Spatial Splitting by Latitude Bands ---
        df["lat_bin"] = pd.cut(df[lat_col], bins=10)
        unique_bins = sorted(df["lat_bin"].unique().tolist())
        
        # Deterministic shuffle for splitting
        rng = random.Random(SEED)
        rng.shuffle(unique_bins)
        
        train_bins = unique_bins[:7]
        val_bins = unique_bins[7:9]
        test_bins = unique_bins[9:]
        
        df_train = df[df["lat_bin"].isin(train_bins)]
        df_val = df[df["lat_bin"].isin(val_bins)]
        df_test = df[df["lat_bin"].isin(test_bins)]
        
        print(f"Spatial Split -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        self.x_train_raw = df_train[self.features].values
        self.y_train = df_train[self.target_column].values
        self.x_val_raw = df_val[self.features].values
        self.y_val = df_val[self.target_column].values
        self.x_test_raw = df_test[self.features].values
        self.y_test = df_test[self.target_column].values
        
        self.y_test_raw = self.y_test.copy()

        self.x_train = self.scaler.fit_transform(self.x_train_raw)
        self.x_val = self.scaler.transform(self.x_val_raw)
        self.x_test = self.scaler.transform(self.x_test_raw)

        # Dataloaders
        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(self.x_train, dtype=torch.float32),
                torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
            ),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.x_val_tensor = torch.tensor(self.x_val, dtype=torch.float32).to(device)
        self.y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1).to(device)
        self.x_test_tensor = torch.tensor(self.x_test, dtype=torch.float32).to(device)

        print("✅ Preprocessing complete. Derived Target 'TI_Variance' created.")

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

            # Simple logging
            prefix = f"[Trial {trial_idx}] " if trial_idx is not None else ""
            print(f"{prefix}Epoch {epoch + 1}/{max_epochs} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f} | Val R²: {val_r2:.4f}")

            if patience_ctr >= patience:
                print(f"⏹ Early stopping at epoch {epoch + 1}")
                break

        if best_state is None:
            best_state = model.state_dict()
        model.load_state_dict(best_state)
        return model

    # ----------------------------
    # Hyperparameter tuning
    # ----------------------------
    def tune(self, n_trials=25, max_epochs=15):
        def objective(trial):
            layers = trial.suggest_int("layers", 1, 4)
            units = trial.suggest_int("units", 16, 128, step=16)
            activation = trial.suggest_categorical("activation", ["relu", "tanh"])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            model = ThermalInertiaVarianceANN(len(self.features), layers, units, activation).to(device)
            self._train(model, lr=lr, max_epochs=max_epochs, trial_idx=trial.number)

            model.eval()
            with torch.no_grad():
                preds = model(self.x_val_tensor).cpu().numpy().flatten()
            r2 = r2_score(self.y_val, preds)
            return r2

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params

        with open(PARAMS_PATH, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print("✅ Best params saved:", self.best_params)

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

        print(f"🚀 Training Ensemble of {ENSEMBLE_SIZE} models...")
        for i in range(ENSEMBLE_SIZE):
            print(f"\n--- Training Member {i+1}/{ENSEMBLE_SIZE} ---")
            torch.manual_seed(SEED + i)
            model = ThermalInertiaVarianceANN(len(self.features), layers, units, activation)
            model = self._train(model, lr, max_epochs=max_epochs)
            
            save_path = f"{MODEL_DIR}/model_{i}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Member {i+1} saved to {save_path}")

        print(f"💾 Ensemble training complete in {MODEL_DIR}")

    # ----------------------------
    # Load and evaluate (Ensemble)
    # ----------------------------
    def load_and_eval(self):
        if not os.path.exists(PARAMS_PATH):
            raise FileNotFoundError("No best params found.")
        
        with open(PARAMS_PATH) as f:
            params = json.load(f)

        all_preds = []
        for i in range(ENSEMBLE_SIZE):
            model = ThermalInertiaVarianceANN(len(self.features), params["layers"], params["units"], params["activation"]).to(device)
            save_path = f"{MODEL_DIR}/model_{i}.pt"
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                p = model(self.x_test_tensor).cpu().numpy().flatten()
                all_preds.append(p)

        preds_stack = np.stack(all_preds)
        mean_pred = preds_stack.mean(0)
        uncertainty = preds_stack.std(0)

        r2 = r2_score(self.y_test_raw, mean_pred)
        mae = mean_absolute_error(self.y_test_raw, mean_pred)

        print(f"📊 Ensemble Test R²: {r2:.4f} | MAE: {mae:.4f}")
        print(f"💡 Mean Uncertainty (StdDev): {uncertainty.mean():.4f}")
        
        return mean_pred, uncertainty


# =============================
# Usage
# =============================
if __name__ == "__main__":
    predictor = ThermalInertiaVariancePredictor(
        path="backend/models/datasets/core_set_2ppd.csv",
        features=[
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
    )
    predictor.preprocess()
    predictor.tune(n_trials=25, max_epochs=15)
    predictor.train_best(max_epochs=300)
    predictor.load_and_eval()
