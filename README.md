# Martian Landing Site Selection Intelligence (MSPF)

![Mars Banner](https://img.shields.io/badge/Mars-Exploration-red?style=for-the-badge&logo=nasa)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![ML](https://img.shields.io/badge/ML-Decision%20Support-orange?style=for-the-badge&logo=scikit-learn)

MSPF is a high-performance, AI-assisted decision-support system designed to evaluate and identify optimal landing zones on Mars. By synthesizing multiple environmental and geological factors, NovaScope 2 provides scientists and mission planners with an interactive, data-driven platform for Martian surface exploration.

---

## 🚀 Key Features

- **Multi-Model Predictive Analysis**: Integrates six specialized machine learning models to predict critical surface parameters:
  - **Water Enrichment (WEH%)**: Estimating subsurface hydrogen.
  - **Dust Coverage**: Quantifying surface dust distribution.
  - **Surface Temperatures**: Predicting average and extreme thermal conditions.
  - **Gravity Anomalies**: Mapping gravitational variations.
  - **TI Variance**: Analyzing Thermal Inertia stability.
  - **Temp Variation**: Yearly surface temperature fluctuations.
- **Dynamic Heatmapping**: A real-time rendering engine that generates a "Landing Goodness Score" based on user-defined weight parameters for scientific priority and safety risk.
- **High-Resolution Global Data**: Operates on a 8ppd (pixels per degree) extrapolated dataset, providing granular details across the entire Martian globe.
- **Interactive Map Interface**: A premium, glassmorphism-styled UI powered by Leaflet.js, featuring:
  - **Live Coordinate Tracking**: Real-time Lat/Lon display on hover.
  - **Point-Specific Surveys**: Detailed data popups for any clicked location.
  - **NASA Tile Integration**: Proxied and cached high-resolution Viking MDIM 2.1 global color mosaic.
- **Performance Optimized**: Built for scale using Parquet lazy loading, chunked computation for 50M+ data points, and intelligent tile caching.

---

## 🛠️ Technical Stack

- **Backend**: Python 3.10+, FastAPI, NumPy, Pandas, Scikit-learn, Uvicorn.
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism), Vanilla JavaScript, Leaflet.js.
- **Data Engineering**: Parquet for efficient storage, custom downcasting for memory optimization, and spatial splitting for robust model training.
- **API**: RESTful endpoints for dynamic tile rendering and point-specific data retrieval.

---

## 📂 Architecture Overview

```text
nova_scope_2/
├── backend/            # Core logic and ML engine
│   ├── API/            # FastAPI server, rendering engine, and tile proxy
│   ├── models/         # Trained ML predictors (Water, Dust, Temp, etc.)
│   └── data/           # Optimized datasets (Parquet/CSV)
├── frontend/           # Web-based interactive interface
│   ├── index.html      # Main application entry point
│   ├── app.js          # Map logic and API interaction
│   └── style.css       # Premium UI design
├── helper_functions/   # Data processing and optimization utilities
├── logging/            # System and model performance logs
└── tile_cache/         # Local cache for NASA planetary tiles
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
   Simply open `frontend/index.html` in a modern web browser, or serve it via a local development server.

---

## 🛰️ Usage

1. **Adjust Weights**: Use the "Mission Parameters" sidebar to define what factors matter most (e.g., maximize Water, minimize Dust).
2. **Recalculate**: Click "Recalculate Heatmap" to update the global suitability map.
3. **Explore**: Navigate the map to find high-scoring (green) zones.
4. **Inspect**: Click on any location to view a comprehensive site survey with predicted values from all ML models.

---

## 📜 Project Goals

The MSPF framework aims to bridge the gap between raw planetary data and mission-critical decisions by:
1. **Ensuring Safety**: Quantifying hazards like extreme slopes and temperature swings.
2. **Maximizing Science**: Identifying areas with high resource potential (WEH) and geological diversity.
3. **Accelerating Planning**: Reducing the time from data collection to site selection through AI-assisted visualization.

---
