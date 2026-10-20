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

MODEL_NAME = "gravity_predictor"
DATASET_PATH = "backend/datasets/core_set_2ppd.csv"
TARGET_COLUMN = "Mars Bouguer Gravity Anomaly"
MODEL_DIR = f"backend/neural_nets/{MODEL_NAME}"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"

FEATURES = [
    "MOLA 128ppd Elevation",
    "MOLA 128ppd Slope",
    "MOLA Roughness 2.4km Baseline - Numeric",
    "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
    "TES Basalt Abundance - Numeric",
    "TES Rock Abundance (Nowicki and Christensen 2007) - Numeric",
    "Latitude (N)"
]
