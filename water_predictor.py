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
MODEL_DIR = "neural_nets/water_predictor"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
MODEL_PATH = f"{MODEL_DIR}/best_model.pt"

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
        df = pd.read_csv(self.csv_path).dropna()

        X = df[self.features].values
        y = df[self.target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED
        )

        self.y_test_raw = y_test.copy()

        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
        y_test = np.log1p(y_test)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

        print("✅ Preprocessing complete")

    # -------------------------
    # Train loop
    # -------------------------
    def _train(self, model, lr, max_epochs=200, patience=30):
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

            print(f"Epoch {epoch + 1}/{max_epochs} | "
                  f"Train MSE: {train_loss:.4f} | "
                  f"Val MSE: {val_loss:.4f} | "
                  f"Val R²: {val_r2:.4f}")

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
    def tune(self, n_trials=20):
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

            self._train(model, lr, max_epochs=50)

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
    def train_best(self):
        if not os.path.exists(PARAMS_PATH):
            raise RuntimeError("❌ No saved params. Run tune() first.")

        with open(PARAMS_PATH) as f:
            self.best_params = json.load(f)

        # Separate model params and optimizer params
        model_params = {
            "layers": self.best_params["layers"],
            "units": self.best_params["units"],
            "activation": self.best_params["activation"]
        }
        lr = self.best_params["lr"]

        # Initialize model
        model = WaterANN(len(self.features), **model_params)

        # Train with the learning rate
        model = self._train(model, lr ,max_epochs=300)

        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Best model saved")

    # -------------------------
    # Load + Evaluate
    # -------------------------
    # -------------------------
    # Load + Evaluate
    # -------------------------
    def load_and_eval(self):
        with open(PARAMS_PATH) as f:
            params = json.load(f)

        # Only pass model parameters to WaterANN
        model_params = {
            "layers": params["layers"],
            "units": params["units"],
            "activation": params["activation"]
        }

        model = WaterANN(len(self.features), **model_params).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        preds = model(self.X_test).cpu().detach().numpy().flatten()
        preds = np.expm1(preds)

        r2 = r2_score(self.y_test_raw, preds)
        mae = mean_absolute_error(self.y_test_raw, preds)

        print(f"📊 Test R²: {r2:.4f} | MAE: {mae:.4f}")
        return preds


def main():
    predictor = WaterPredictor(
        csv_path="datasets/water_WEH_%.csv",
        target="Mars Odyssey Neutron Spectrometer % WEH",
        features=[
            'MOLA 128ppd Elevation',
            'OMEGA Est. Lambert Albedo 1080nm',
            'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
            'TES Basalt Abundance - Numeric',
            'Yearly Average Mars Surface Temperature',
            'Latitude (N)',
            'OMEGA Band depth at 2000 nm',
            'Crustal Thickness (km)'
        ]
    )

    predictor.preprocess()
    # predictor.tune(n_trials=25)
    # predictor.train_best()
    predictor.load_and_eval()

if __name__ == "__main__":
    main()

