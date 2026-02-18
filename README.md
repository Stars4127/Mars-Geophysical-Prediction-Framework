# Martian Subsurface Prediction Framework (MSPF)

![Mars Banner](https://img.shields.io/badge/Mars-Exploration-red?style=for-the-badge&logo=nasa)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-Machine%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)

## 📖 Abstract

Selecting sites for planetary exploration is a difficult task. Changing regions, outdated planetary scans, the vast scope of the Martian surface, and low‑resolution subsurface data complicate decision‑making. Traditional methods, which often rely on rover comparisons or low-quality remote sensing, are slow, resource-intensive, and frequently produce inaccurate predictions of subsurface qualities like water concentration.

**MSPF** provides a scalable, machine learning-oriented approach to make initial site selection more efficient. By utilizing low-level surface qualities (topography, mineral densities, and thermal properties), this system predicts critical subsurface parameters at higher resolutions. Using a pipeline of feed-forward MLP neural networks, MSPF systematically evaluates candidate landing sites, reducing the need for exhaustive scanning and reliance on sparse data.

---

## 🔬 Methodology & Architecture

The core of the MSPF framework relies on a robust engineering and machine learning pipeline designed for stability and scientific accuracy:

- **ML Architecture**: Implements multiple feed-forward Multi-Layer Perceptron (MLP) neural networks using **PyTorch**.
- **Data Engineering**: Features are scaled using `RobustScaler` to ensure convergence and stability. A rigorous spatial train-test split is used to prevent data leakage and ensure global generalizability.
- **Hyperparameter Optimization**: Models undergo automated tuning through **Bayesian Optimization** (using Optuna’s TPE model) with R² score as the primary objective.
- **Validation**: Quantitative results are validated against historical Martian landing sites and verified against 8ppd extrapolated ground-truth proxies.
- **Performance**: Preliminary results indicate that key properties can be predicted with an **Average R² ≈ 0.66** (excluding variance models), providing a high-confidence baseline for mission planning.

---

## 🚀 Key Features

- **Multi-Model Predictive Analysis**: Six specialized predictors for:
  - **Water Enrichment (WEH%)**: Subsurface hydrogen estimation.
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
├── backend/            # Core logic and ML engine
│   ├── API/            # FastAPI server, rendering engine, and tile proxy
│   ├── models/         # Trained PyTorch predictors and hyperparameter configs
│   └── data/           # Optimized Parquet datasets
├── frontend/           # Web-based interactive interface
│   ├── app.js          # Map logic and Leaflet implementation
│   └── style.css       # Premium UI design
├── helper_functions/   # Data processing and optimization utilities
├── logging/            # System and model performance logs
└── CODE_MANIFEST.md    # Index of authored and acquired code
```

---

## 📥 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Stars4127/nova_scope_2.git
   cd nova_scope_2
   ```

2. **Set up the Python environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the API Server**:
   ```bash
   python backend/API/main.py
   ```

4. **Run the Frontend**:
   Simply open `frontend/index.html` in a modern web browser.

---

## 🛰️ Usage

1. **Adjust Weights**: Use the "Mission Parameters" sidebar to define priorities (e.g., maximize Water, maximize Slope safety).
2. **Recalculate**: Click "Recalculate Heatmap" to update the global suitability model.
3. **Explore**: Navigate the map to identify high-scoring (green) scientific targets.
4. **Inspect**: Click any location for a site survey with predicted values from all six neural networks.
