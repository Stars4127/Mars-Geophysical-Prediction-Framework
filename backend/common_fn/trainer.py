import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_single_model(
    model: nn.Module,
    train_loader,
    x_val_tensor,
    y_val_tensor,
    y_val_np,
    lr: float,
    max_epochs: int = 250,
    patience: int = 30,
    trial_idx: int = None,
    weight_decay: float = 0.0,
    device: str = "cpu"
) -> nn.Module:
    """
    Trains a single PyTorch neural network model with early stopping.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_preds = model(x_val_tensor).cpu().numpy().flatten()
            val_r2 = r2_score(y_val_np, val_preds)
            val_loss = criterion(model(x_val_tensor), y_val_tensor).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1

        if trial_idx is not None:
            print(f"[Trial {trial_idx}] Epoch {epoch + 1}/{max_epochs} | Train MSE: {epoch_loss:.8e} | Val MSE: {val_loss:.8e} | Val R²: {val_r2:.4f}")
        else:
            if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch + 1}/{max_epochs} | Train MSE: {epoch_loss:.8e} | Val MSE: {val_loss:.8e} | Val R²: {val_r2:.4f}")

        if patience_ctr >= patience:
            print(f" Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model


def train_ensemble(
    model_factory_fn,
    data_splits,
    model_dir: str,
    ensemble_size: int = 5,
    max_epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
    device: str = "cpu"
):
    """
    Trains an ensemble of model instances with different seeds and saves checkpoints.
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f" Training Ensemble of {ensemble_size} models...")

    for i in range(ensemble_size):
        print(f"\n--- Training Member {i+1}/{ensemble_size} ---")
        torch.manual_seed(seed + i)
        
        model = model_factory_fn()
        model = train_single_model(
            model=model,
            train_loader=data_splits.train_loader,
            x_val_tensor=data_splits.x_val_tensor,
            y_val_tensor=data_splits.y_val_tensor,
            y_val_np=data_splits.y_val,
            lr=lr,
            max_epochs=max_epochs,
            weight_decay=weight_decay,
            device=device
        )
        
        save_path = os.path.join(model_dir, f"model_{i}.pt")
        torch.save(model.state_dict(), save_path)
        print(f" Member {i+1} saved to {save_path}")

    print(f" Ensemble training complete in {model_dir}")


def evaluate_ensemble(
    model_factory_fn,
    data_splits,
    model_dir: str,
    ensemble_size: int = 5,
    device: str = "cpu"
):
    """
    Loads saved ensemble checkpoints, runs inference on test split, and computes statistics.
    """
    all_preds = []
    print(f" Loading Ensemble of {ensemble_size} models from {model_dir}...")

    for i in range(ensemble_size):
        save_path = os.path.join(model_dir, f"model_{i}.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Ensemble member {i} not found at {save_path}.")

        model = model_factory_fn().to(device)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            p = model(data_splits.x_test_tensor).cpu().numpy().flatten()
            all_preds.append(p)

    preds_stack = np.stack(all_preds)
    mean_pred = preds_stack.mean(0)
    uncertainty = preds_stack.std(0)

    y_test = data_splits.y_test_raw
    r2 = r2_score(y_test, mean_pred)
    mae = mean_absolute_error(y_test, mean_pred)
    rmse = np.sqrt(mean_squared_error(y_test, mean_pred))

    print(f" Ensemble Test R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f" Mean Uncertainty (StdDev): {uncertainty.mean():.4f}")

    return mean_pred, uncertainty, r2, rmse, mae
