import os

SEED = 42
BATCH_SIZE = 1024
ENSEMBLE_SIZE = 5

# Execution flags
TUNE = False
TRAIN = True
EVAL = True

N_TRIALS = 50
TUNE_EPOCHS = 30
TRAIN_EPOCHS = 500

MODEL_NAME = "temp_variation_predictor"
DATASET_PATH = "backend/datasets/core_set_2ppd.csv"
TARGET_COLUMN = "Yearly Mars Surface Temperature Variation (C)"
MODEL_DIR = f"backend/neural_nets/{MODEL_NAME}"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"

FEATURES = [
    "Latitude (N)",
    "MOLA 128ppd Elevation",
    "MOLA 128ppd Slope",
    "MOLA 128ppd Aspect",
    "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
    "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "MOLA Roughness 0.6km Baseline - Numeric"
]
