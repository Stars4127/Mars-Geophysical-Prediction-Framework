import sys
import os
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config
from common_fn.data import load_and_preprocess_data
from common_fn.trainer import train_ensemble, evaluate_ensemble
from common_fn.tune import tune_hyperparameters
from models.model import ConfigurableANN

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def get_model_builder(layers, units, activation, dropout, input_dim):
    def builder():
        return ConfigurableANN(
            input_dim=input_dim,
            layers=layers,
            units=units,
            activation=activation,
            dropout=dropout
        )
    return builder

def main():
    print(f" Loading data for {config.MODEL_NAME} on device: {device}")
    data_splits = load_and_preprocess_data(
        csv_path=config.DATASET_PATH,
        target_column=config.TARGET_COLUMN,
        features=config.FEATURES,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        device=device,
        save_scaler_path=getattr(config, "SCALER_PATH", None)
    )

    if config.TUNE:
        print("\n--- Phase 1: Hyperparameter Tuning ---")
        def tune_builder(input_dim, layers, units, activation, dropout):
            return ConfigurableANN(input_dim=input_dim, layers=layers, units=units, activation=activation, dropout=dropout)

        tune_hyperparameters(
            model_builder_fn=tune_builder,
            data_splits=data_splits,
            params_save_path=config.PARAMS_PATH,
            n_trials=getattr(config, "N_TRIALS", 30),
            max_epochs=getattr(config, "TUNE_EPOCHS", 30),
            device=device
        )

    if config.TRAIN:
        print("\n--- Phase 2: Ensemble Training ---")
        if not os.path.exists(config.PARAMS_PATH):
            raise FileNotFoundError(f"Best params file not found at {config.PARAMS_PATH}. Set TUNE = True in config.py first.")

        with open(config.PARAMS_PATH) as f:
            best_params = json.load(f)

        model_factory = get_model_builder(
            layers=best_params["layers"],
            units=best_params["units"],
            activation=best_params["activation"],
            dropout=best_params.get("dropout", 0.0),
            input_dim=len(data_splits.features)
        )

        train_ensemble(
            model_factory_fn=model_factory,
            data_splits=data_splits,
            model_dir=config.MODEL_DIR,
            ensemble_size=config.ENSEMBLE_SIZE,
            max_epochs=getattr(config, "TRAIN_EPOCHS", 500),
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 0.0),
            seed=config.SEED,
            device=device
        )

    if config.EVAL:
        print("\n--- Phase 3: Ensemble Evaluation ---")
        if not os.path.exists(config.PARAMS_PATH):
            raise FileNotFoundError(f"Best params file not found at {config.PARAMS_PATH}.")

        with open(config.PARAMS_PATH) as f:
            best_params = json.load(f)

        model_factory = get_model_builder(
            layers=best_params["layers"],
            units=best_params["units"],
            activation=best_params["activation"],
            dropout=best_params.get("dropout", 0.0),
            input_dim=len(data_splits.features)
        )

        evaluate_ensemble(
            model_factory_fn=model_factory,
            data_splits=data_splits,
            model_dir=config.MODEL_DIR,
            ensemble_size=config.ENSEMBLE_SIZE,
            device=device
        )

if __name__ == "__main__":
    main()
