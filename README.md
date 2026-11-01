# Martian Geophysical Prediction Framework (MGPF)

![Mars Banner](https://img.shields.io/badge/Mars-Exploration-red?style=for-the-badge&logo=nasa)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-Machine%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)

## 📖 Abstract

Selecting sites for planetary exploration is a difficult task. Changing regions, outdated planetary scans, the vast scope of the Martian surface, and low‑resolution geophysical data complicate decision‑making. Traditional methods, which often rely on rover comparisons or low-quality remote sensing, are slow, resource-intensive, and frequently produce inaccurate predictions of geophysical qualities like water concentration.

**MSPF** provides a scalable, machine learning-oriented approach to make initial site selection more efficient. By utilizing low-level surface qualities (topography, mineral densities, and thermal properties), this system predicts critical geophysical parameters at higher resolutions. Using a pipeline of feed-forward MLP neural networks, MSPF systematically evaluates candidate landing sites, reducing the need for exhaustive scanning and reliance on sparse data.

---

## 🔬 Methodology & Architecture

The core of the MSPF framework relies on a robust engineering and machine learning pipeline designed for stability and scientific accuracy:

- **ML Architecture**: Implements multiple feed-forward Multi-Layer Perceptron (MLP) neural networks using **PyTorch**.
- **Data Engineering**: Features are scaled using `RobustScaler` to ensure convergence and stability. A rigorous spatial train-test split is used to prevent data leakage and ensure global generalizability.
- **Hyperparameter Optimization**: Models undergo automated tuning through **Bayesian Optimization** (using Optuna’s TPE model) with R² score as the primary objective.
- **Validation**: Quantitative results are validated against historical Martian landing sites and verified against 8ppd extrapolated ground-truth proxies.
- **Performance**: Preliminary results indicate that key properties can be predicted with an **Average R² ≈ 0.93** (excluding variance models), providing a high-confidence baseline for mission planning.

---

## 🚀 Key Features

- **Multi-Model Predictive Analysis**: Six specialized predictors for:
  - **Water Enrichment (WEH%)**: Geophysical hydrogen estimation.
  - **Dust Coverage**: Global surface dust distribution.
  - **Surface Temperatures**: Average and extreme thermal mapping.
  - **Gravity Anomalies**: Mapping gravitational fluctuations.
  - **TI Variance**: Analyzing diurnal thermal inertia stability.
  - **Temp Variation**: Tracking seasonal surface temperature swings.
- **Dynamic Heatmapping**: A real-time rendering engine generates a "Landing Goodness Score" based on user-defined weights for scientific priority and safety risk.
- **Interactive Map Interface**: A premium glassmorphism UI featuring live coordinate tracking, point-specific surveys, and cached NASA Viking MDIM 2.1 global mosaics.

---

## 🛠️ Technical Stack

- **Backend**: Python 3.10+, FastAPI, PyTorch, NumPy, Pandas, Scikit-learn, Optuna, Uvicorn.
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism), Vanilla JavaScript, Leaflet.js.
- **Infrastructure**: Parquet for high-performance lazy loading, intelligent tile caching, and multi-threaded rendering logic.

---

## 📂 Architecture Overview

```text
nova_scope_2/
├── backend/
│   ├── API/                   # FastAPI server, dynamic rendering engine, and NASA tile proxy
│   │   ├── engine.py          # Spatial indexing (cKDTree), tile generation, & goodness scoring
│   │   └── main.py            # REST endpoints (/tile, /goodness, /point, /heatmap, /meta)
│   ├── common_fn/             # Shared ML pipeline components
│   │   ├── data.py            # Preprocessing, periodic feature encoding, spatial splits, & RobustScaler
│   │   ├── trainer.py         # Single model training with early stopping & ensemble trainer/evaluator
│   │   └── tune.py            # Optuna Bayesian hyperparameter optimization
│   ├── datasets/              # Datasets managed via Git LFS
│   │   ├── core_set_2ppd.csv  # 2ppd Martian feature baseline dataset
│   │   └── 8ppd_extrapolated_targets.parquet # 8ppd multi-model extrapolation targets
│   ├── metrics/               # Evaluation metrics per model (R², RMSE, MAE, Uncertainty)
│   ├── models/
│   │   └── model.py           # Unified PyTorch ConfigurableANN (MLP) architecture
│   ├── neural_nets/           # Trained ensemble weights (model_0..4.pt), best_params.json, scaler.joblib
│   │   ├── dust_predictor/
│   │   ├── gravity_predictor/
│   │   ├── temp_predictor/
│   │   ├── temp_variation_predictor/
│   │   ├── ti_variance_predictor/
│   │   └── water_predictor/
│   ├── run/                   # Per-predictor execution scripts and configurations
│   │   └── <model_name>/      # config.py and main.py for tuning, training, and evaluation
│   ├── diagnose.py            # Feature distribution and dataset validation utility
│   └── log_metrics.py         # Batch evaluation and metrics generation across all models
├── frontend/                  # Modern Glassmorphism web interface
│   ├── index.html             # UI layout and Leaflet map container
│   ├── app.js                 # Dynamic heatmaps, coordinate tracking, & point surveys
│   └── style.css              # Glassmorphic visual design system
├── helper_functions/          # Extrapolation and statistical processing utilities
├── tile_cache/                # Cached NASA MDIM 2.1 planetary tiles
├── CODE_MANIFEST.md           # Index of authored and acquired components
└── requirements.txt           # Python dependencies
```

---

## 📊 Model Performance Metrics

Each predictor uses a 5-model deep ensemble with Bayesian-optimized hyperparameters and spatial train/val/test splits (stratified by 10 latitude bands):

| Predictor | Target Property | R² | RMSE | MAE | Mean Uncertainty (σ) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Water Predictor** | Mars Odyssey Neutron Spectrometer % WEH | **0.969** | 4.060 | 1.737 | 1.121 |
| **Surface Temp** | Surface Temperature (TES / Thermal) | **0.998** | 0.994 | 0.607 | 0.289 |
| **Temp Variation** | Seasonal Temperature Swing | **0.981** | 2.373 | 1.431 | 0.526 |
| **Gravity Predictor** | Bouguer Gravity Anomaly | **0.910** | 55.332 | 37.458 | 17.865 |
| **Dust Predictor** | Global Dust Coverage Index | **0.845** | 0.009 | 0.006 | 0.002 |
| **TI Variance** | Diurnal Thermal Inertia Difference (ΔTI) | **0.342** | 24.935 | 15.502 | 3.317 |

> **Note**: Primary predictors achieve an **Average R² ≈ 0.93** (excluding complex diurnal variance modeling).

---

## 📥 Installation & Setup

### 1. Prerequisites: Git LFS
This repository uses **Git Large File Storage (Git LFS)** for datasets and neural network weights. Ensure Git LFS is installed prior to cloning:
```bash
git lfs install
```

### 2. Clone the Repository
```bash
git clone https://github.com/Stars4127/Mars-Geophysical-Prediction-Framework.git
cd Mars-Geophysical-Prediction-Framework
git lfs pull
```

### 3. Set up the Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Launch the API Server
```bash
python backend/API/main.py
```
The FastAPI server will start on `http://127.0.0.1:8000`.

### 5. Open the Frontend
Open `frontend/index.html` directly in any modern web browser or serve it via a local static server.

---

## 🏋️ Training & Tuning Models

To tune, train, or evaluate any of the six individual neural network predictors:

1. Navigate to the model's configuration: `backend/run/<model_name>/config.py`.
2. Toggle flags as desired (`TUNE = True`, `TRAIN = True`, `EVAL = True`).
3. Run the model pipeline:
   ```bash
   python backend/run/<model_name>/main.py
   ```
4. To re-evaluate all models and update performance logs:
   ```bash
   python backend/log_metrics.py
   ```

---

## 🛰️ Usage

1. **Adjust Weights**: Use the "Mission Parameters" sidebar to define mission criteria (e.g., prioritize Water enrichment or minimize slope/dust hazards).
2. **Dynamic Heatmaps**: View dynamically generated heatmaps rendered server-side via Gaussian convolution and mapped to NASA Viking MDIM 2.1 base tiles.
3. **Point Surveys**: Click anywhere on Mars to trigger a sub-millisecond nearest-neighbor search (`cKDTree`) retrieving full predictions and ensemble uncertainties for that coordinate.

