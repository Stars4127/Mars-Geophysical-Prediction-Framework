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
BATCH_SIZE = 4096
MODEL_DIR = "neural_nets/temp_predictor"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
MODEL_PATH = f"{MODEL_DIR}/best_model.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"🔥 Using device: {device}")


# =============================
# MLP model
# =============================
class TempANN(nn.Module):
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
        df = pd.read_csv(self.path).dropna()
        X = df[self.features].values
        y = df[self.target_column].values

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=SEED
        )
        X_train, X_val, self.y_train, self.y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_fraction, random_state=SEED
        )

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

        print("✅ Preprocessing complete")

    # ----------------------------
    # Training loop
    # ----------------------------
    def _train(self, model, lr, max_epochs=250, patience=7, trial_idx=None):
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
                val_loss = criterion(model(self.x_val_tensor), self.y_val_tensor).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1

            # ----- Simple epoch logging -----
            if trial_idx is not None:
                print(f"[Trial {trial_idx}] Epoch {epoch + 1}/{max_epochs} | "
                      f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

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

            model = TempANN(len(self.features), layers, units, activation).to(device)
            self._train(model, lr=lr, max_epochs=max_epochs, trial_idx=trial.number)

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
        print("✅ Best params saved:", self.best_params)

    # ----------------------------
    # Train best model
    # ----------------------------
    def train_best(self, max_epochs=250):
        if self.best_params is None:
            if os.path.exists(PARAMS_PATH):
                with open(PARAMS_PATH) as f:
                    self.best_params = json.load(f)
            else:
                raise RuntimeError("No best params found. Run tune() first.")

        p = self.best_params
        model = TempANN(len(self.features), p["layers"], p["units"], p["activation"])
        model = self._train(model, lr=p["lr"], max_epochs=max_epochs)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Best model saved to {MODEL_PATH}")
        self.model = model

    # ----------------------------
    # Load + evaluate
    # ----------------------------
    def load_and_eval(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("No trained model found. Run train_best() first.")

        if self.model is None:
            p = self.best_params or json.load(open(PARAMS_PATH))
            model = TempANN(len(self.features), p["layers"], p["units"], p["activation"])
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            self.model = model

        with torch.no_grad():
            preds = self.model(self.x_test_tensor).cpu().numpy().flatten()
        r2 = r2_score(self.y_test_raw, preds)
        mae = mean_absolute_error(self.y_test_raw, preds)
        print(f"📊 Test R²: {r2:.4f} | MAE: {mae:.4f}")
        return preds


# =============================
# Usage
# =============================
if __name__ == "__main__":
    predictor = TempPredictor(
        path=r"datasets/cleaned_surface_temp.csv",
        target_column='Yearly Average Mars Surface Temperature (C)',
        features=[
            'Latitude (N)',
            'MOLA 128ppd Elevation',
            'OMEGA Est. Lambert Albedo 1080nm',
            'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
            'Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
            'OMEGA Ferric/Dust 860nm ratio',
            'MOLA Roughness 0.6km Baseline - Numeric',
            'Avg_CRISM_CO2_Frost_Prob'
        ]
    )

    predictor.preprocess()
    # predictor.tune(n_trials=10, max_epochs=10)
    # predictor.train_best()
    predictor.load_and_eval()
