import os

SEED = 42
BATCH_SIZE = 1024
ENSEMBLE_SIZE = 5

# Execution flags
TUNE = False
TRAIN = True
EVAL = True

N_TRIALS = 30
TUNE_EPOCHS = 30
TRAIN_EPOCHS = 500

MODEL_NAME = "temp_predictor"
DATASET_PATH = "backend/datasets/core_set_2ppd.csv"
TARGET_COLUMN = "Yearly Average Mars Surface Temperature (C)"
MODEL_DIR = f"backend/neural_nets/{MODEL_NAME}"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"

FEATURES = [
    "Latitude (N)",
    "MOLA 128ppd Elevation",
    "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
    "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "MOLA Roughness 0.6km Baseline - Numeric",
    "MOLA Roughness 2.4km Baseline - Numeric"
]
