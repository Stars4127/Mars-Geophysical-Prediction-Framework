import os
import json
import numpy as np
import traceback
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_fn.data import load_and_preprocess_data
from common_fn.trainer import evaluate_ensemble
from models.model import ConfigurableANN

import run.dust_predictor.config as dust_cfg
import run.gravity_predictor.config as gravity_cfg
import run.temp_predictor.config as temp_cfg
import run.temp_variation_predictor.config as temp_var_cfg
import run.ti_variance_predictor.config as ti_var_cfg
import run.water_predictor.config as water_cfg

METRICS_DIR = "backend/metrics"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def evaluate_and_log(model_cfg, model_name):
    print(f"\n======================================")
    print(f" Evaluating {model_name}...")
    print(f"======================================")
    
    try:
        data_splits = load_and_preprocess_data(
            csv_path=model_cfg.DATASET_PATH,
            target_column=model_cfg.TARGET_COLUMN,
            features=model_cfg.FEATURES,
            batch_size=model_cfg.BATCH_SIZE,
            seed=model_cfg.SEED,
            device=device
        )

        with open(model_cfg.PARAMS_PATH) as f:
            best_params = json.load(f)

        def model_factory():
            return ConfigurableANN(
                input_dim=len(data_splits.features),
                layers=best_params["layers"],
                units=best_params["units"],
                activation=best_params["activation"],
                dropout=best_params.get("dropout", 0.0)
            )

        mean_pred, uncertainty, r2, rmse, mae = evaluate_ensemble(
            model_factory_fn=model_factory,
            data_splits=data_splits,
            model_dir=model_cfg.MODEL_DIR,
            ensemble_size=model_cfg.ENSEMBLE_SIZE,
            device=device
        )
        
        metrics = {
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Mean_Uncertainty": float(np.mean(uncertainty))
        }
        
        os.makedirs(METRICS_DIR, exist_ok=True)
        out_path = os.path.join(METRICS_DIR, f"{model_name}.json")
        
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f" Successfully logged {model_name} metrics to {out_path}.")
        
    except Exception as e:
        print(f" Failed to evaluate {model_name}: {e}")
        traceback.print_exc()

def main():
    if not os.path.exists(METRICS_DIR):
        os.makedirs(METRICS_DIR)
        
    models_to_run = [
        ("dust_predictor", dust_cfg),
        ("gravity_predictor", gravity_cfg),
        ("surface_temp", temp_cfg),
        ("temp_variation_predictor", temp_var_cfg),
        ("ti_variance_predictor", ti_var_cfg),
        ("water_predictor", water_cfg)
    ]

    for model_name, cfg in models_to_run:
        evaluate_and_log(cfg, model_name)

if __name__ == "__main__":
    main()
