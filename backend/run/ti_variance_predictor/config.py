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

MODEL_NAME = "ti_variance_predictor"
DATASET_PATH = "backend/datasets/core_set_2ppd.csv"
TARGET_COLUMN = "TI_Variance"
MODEL_DIR = f"backend/neural_nets/{MODEL_NAME}"
PARAMS_PATH = f"{MODEL_DIR}/best_params.json"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"

FEATURES = [
    "Latitude (N)",
    "MOLA 128ppd Elevation",
    "MOLA 128ppd Slope",
    "MOLA 128ppd Aspect",
    "MOLA Roughness 0.6km Baseline - Numeric",
    "MOLA Roughness 2.4km Baseline - Numeric",
    "TES Basalt Abundance - Numeric",
    "TES Andesite Abundance - Numeric",
    "TES Rock Abundance (Nowicki and Christensen 2007) - Numeric",
    "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
    "Mars Crustal Thickness (GSFC)"
]
