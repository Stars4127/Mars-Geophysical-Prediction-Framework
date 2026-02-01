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
MODEL_DIR = "backend/models/neural_nets/water_predictor"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
MODEL_PATH = f"{MODEL_DIR}/model.pt" # Base path

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🔥 Using device: {device}")

# =============================
# Model
# =============================
class WaterANN(nn.Module):
    def __init__(self, input_dim, layers, units, activation):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.Tanh()

        net = []
        dim = input_dim
        for _ in range(layers):
            net += [nn.Linear(dim, units), act]
            dim = units
        net.append(nn.Linear(dim, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

# =============================
# Predictor
# =============================
class WaterPredictor:
    def __init__(self, csv_path, target, features):
        self.csv_path = csv_path
        self.target = target
        self.features = features
        self.scaler = RobustScaler()

        self.best_params = None
        os.makedirs(MODEL_DIR, exist_ok=True)

    # -------------------------
    # Preprocess
    # -------------------------
    def preprocess(self):
        df = pd.read_csv(self.csv_path, na_values=['null', 'Null', 'NULL'])
        
        # Select subset: We need original features + target
        lat_col = 'Latitude (N)'
        asp_col = 'MOLA 128ppd Aspect'
        
        # Standardize target reference
        self.target_column = self.target
        
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
            
        print(f"Original size: {len(df)}")
        original_len = len(df)
        df = df.dropna()
        print(f"📉 Dropped {original_len - len(df)} rows containing NaNs. Remaining: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("Dataset is empty after preprocessing!")

        # --- Spatial Splitting by Latitude Bands ---
        df["lat_bin"] = pd.cut(df["Latitude (N)"], bins=10)
        unique_bins = sorted(df["lat_bin"].unique().tolist())
        
        # Shuffle bins deterministicly
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

        self.x_train_raw = df_train[self.features].values
        self.y_train = df_train[self.target].values
        self.x_val_raw = df_val[self.features].values
        self.y_val = df_val[self.target].values
        self.x_test_raw = df_test[self.features].values
        y_test = df_test[self.target].values
        
        self.y_test_raw = y_test.copy()

        # No log-transform for target; keep original values

        X_train = self.scaler.fit_transform(self.x_train_raw)
        X_val = self.scaler.transform(self.x_val_raw)
        X_test = self.scaler.transform(self.x_test_raw)

        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1).to(device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

        print("✅ Preprocessing complete")

    # -------------------------
    # Train loop
    # -------------------------
    def _train(self, model, lr, max_epochs=200, patience=15, trial_idx=None):
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        patience_ctr = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            losses = []

            for xb, yb in self.train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())

            train_loss = np.mean(losses)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_preds = model(self.X_val).cpu().numpy().flatten()
                y_true = self.y_val.cpu().numpy().flatten()
                val_loss = loss_fn(model(self.X_val), self.y_val).item()
                val_r2 = r2_score(y_true, val_preds)

            if trial_idx is not None:
                print(f"[Trial {trial_idx}]")
            print(f"Epoch {epoch + 1}/{max_epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val R²: {val_r2:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= patience:
                print(f"Stopping early at epoch {epoch + 1} due to patience limit.")
                break

        model.load_state_dict(best_state)
        return model

    # -------------------------
    # Tuning (optional)
    # -------------------------
    def tune(self, n_trials=20, max_epochs=10):
        def objective(trial):
            model_params = {
                "layers": trial.suggest_int("layers", 1, 4),
                "units": trial.suggest_int("units", 16, 128, step=16),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            }

            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            model = WaterANN(
                input_dim=len(self.features),
                **model_params
            ).to(device)

            self._train(model, lr, max_epochs=max_epochs)

            model.eval()
            with torch.no_grad():
                preds = model(self.X_val).cpu().numpy().flatten()
                y_true = self.y_val.cpu().numpy().flatten()

            val_r2 = r2_score(y_true, preds)
            return -val_r2

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params

        with open(PARAMS_PATH, "w") as f:
            json.dump(self.best_params, f, indent=2)

        print("🏆 Best params saved:", self.best_params)

    # -------------------------
    # Train best
    # -------------------------
    def train_best(self, max_epochs=300):
        if not os.path.exists(PARAMS_PATH):
            raise RuntimeError("❌ No saved params. Run tune() first.")

        with open(PARAMS_PATH) as f:
            self.best_params = json.load(f)

        model_params = {
            "layers": self.best_params["layers"],
            "units": self.best_params["units"],
            "activation": self.best_params["activation"]
        }
        lr = self.best_params["lr"]

        print(f"🚀 Training Ensemble of {ENSEMBLE_SIZE} models...")
        for i in range(ENSEMBLE_SIZE):
            print(f"\n--- Training Member {i+1}/{ENSEMBLE_SIZE} ---")
            # Diverge seed for initialization variety
            torch.manual_seed(SEED + i)
            
            model = WaterANN(len(self.features), **model_params)
            model = self._train(model, lr, max_epochs=max_epochs)
            
            save_path = f"{MODEL_DIR}/model_{i}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Member {i+1} saved to {save_path}")

        print(f"💾 Ensemble training complete in {MODEL_DIR}")

    # -------------------------
    # Load + Evaluate
    # -------------------------
    # -------------------------
    # Load + Evaluate (Ensemble)
    # -------------------------
    def load_and_eval(self):
        with open(PARAMS_PATH) as f:
            params = json.load(f)

        model_params = {
            "layers": params["layers"],
            "units": params["units"],
            "activation": params["activation"]
        }

        all_preds = []
        print(f"🔍 Loading Ensemble of {ENSEMBLE_SIZE} models from {MODEL_DIR}...")
        
        for i in range(ENSEMBLE_SIZE):
            model = WaterANN(len(self.features), **model_params).to(device)
            save_path = f"{MODEL_DIR}/model_{i}.pt"
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            model.eval()
            
            with torch.no_grad():
                p = model(self.X_test).cpu().numpy().flatten()
                all_preds.append(p)

        # Stack predictions: [EnsembleSize, NumSamples]
        preds_stack = np.stack(all_preds)
        
        # Calculate ensemble stats
        mean_pred = preds_stack.mean(0)
        uncertainty = preds_stack.std(0)

        r2 = r2_score(self.y_test_raw, mean_pred)
        mae = mean_absolute_error(self.y_test_raw, mean_pred)

        print(f"📊 Ensemble Test R²: {r2:.4f} | MAE: {mae:.4f}")
        print(f"💡 Mean Uncertainty (StdDev): {uncertainty.mean():.4f}")
        
        return mean_pred, uncertainty


def main():
    predictor = WaterPredictor(
        csv_path="backend/models/datasets/core_set_2ppd.csv",
        target="Mars Odyssey Neutron Spectrometer % WEH",
        features=[
            'MOLA 128ppd Elevation',
            'OMEGA Est. Lambert Albedo 1080nm Ody et al 2012',
            'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
            'TES Basalt Abundance - Numeric',
            'Latitude (N)',
            'Mars Crustal Thickness (GSFC)'
        ]
    )

    predictor.preprocess()
    predictor.tune(n_trials=25, max_epochs=10)
    predictor.train_best()
    predictor.load_and_eval()

if __name__ == "__main__":
    main()
