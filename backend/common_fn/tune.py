import os
import json
import torch
import optuna
from sklearn.metrics import r2_score
from common_fn.trainer import train_single_model

def tune_hyperparameters(
    model_builder_fn,
    data_splits,
    params_save_path: str,
    n_trials: int = 30,
    max_epochs: int = 30,
    device: str = "cpu"
) -> dict:
    """
    Executes Optuna hyperparameter optimization study and saves the best parameter set.
    """
    input_dim = len(data_splits.features)

    def objective(trial):
        layers = trial.suggest_int("layers", 1, 6)
        units = trial.suggest_int("units", 128, 256, step=16)
        activation = trial.suggest_categorical("activation", ["gelu", "elu"])
        lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
        use_dropout = trial.suggest_categorical("use_dropout", [True, False])
        dropout = trial.suggest_float("dropout", 0.1, 0.2) if use_dropout else 0.0

        model = model_builder_fn(
            input_dim=input_dim,
            layers=layers,
            units=units,
            activation=activation,
            dropout=dropout
        )

        model = train_single_model(
            model=model,
            train_loader=data_splits.train_loader,
            x_val_tensor=data_splits.x_val_tensor,
            y_val_tensor=data_splits.y_val_tensor,
            y_val_np=data_splits.y_val,
            lr=lr,
            max_epochs=max_epochs,
            trial_idx=trial.number,
            weight_decay=weight_decay,
            device=device
        )

        model.eval()
        with torch.no_grad():
            preds = model(data_splits.x_val_tensor).cpu().numpy().flatten()
        r2 = r2_score(data_splits.y_val, preds)
        return -r2

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    os.makedirs(os.path.dirname(params_save_path), exist_ok=True)
    with open(params_save_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(" Best hyper-parameters saved to:", params_save_path)
    print(" Best params:", best_params)

    return best_params
