# MSPF: Code Manifest & Documentation

---

## 📑 Index of Core Components

| Component | Responsibility | Location | Status |
| :--- | :--- | :--- | :--- |
| **API Server** | FastAPI entry point, tile proxy, and REST endpoints. | `backend/API/main.py` | **Written** |
| **Rendering Engine** | High-performance spatial analysis and tile generation. | `backend/API/engine.py` | **Written** |
| **Common ML Fn** | Shared data loading, training pipeline, and Optuna tuning. | `backend/common_fn/` | **Written** |
| **ML Model Construct** | Singular configurable PyTorch neural network construction. | `backend/models/model.py` | **Written** |
| **Model Run Scripts** | Per-model configs (`config.py`) and entrypoints (`main.py`). | `backend/run/<model>/` | **Written** |
| **Interactive Map** | Premium Glassmorphism UI and Leaflet integration. | `frontend/` | **Written** |
| **Data Utilities** | Parquet lazy-loading and 8ppd extrapolation logic. | `helper_functions/` | **Written** |
| **Diagnostics** | Feature signal analysis and statistical validation. | `backend/diagnose.py` | **Written** |
| **Tile Cache** | Local persistence of NASA MDIM 2.1 planetary tiles. | `tile_cache/` | **Written** |

---

## 🛠️ Custom Developed Code

The following modules were custom-built and engineered to meet the project's performance and design requirements:

### 1. Spatial Rendering Engine (`backend/API/engine.py`)
- **Parallel Processing**: Uses chunked computation (2M points/batch) to handle 50M+ data points without memory exhaustion.
- **Normalization Engine**: Extracts min/max statistics directly from Parquet metadata for global absolute scoring.
- **Spatial Indexing**: Implements `cKDTree` for sub-millisecond nearest-neighbor lookups (Point Surveys).
- **Dynamic Heatmapping**: Custom Gaussian-blurred tile rendering logic with `RdYlGn` colormaps.

### 2. Standardized ML Pipeline (`backend/common_fn/`, `backend/models/model.py`, `backend/run/`)
- **Universal Architecture**: Configurable `ConfigurableANN` architecture in `backend/models/model.py`.
- **Modular Utilities**: Shared `data.py` (dataloading & spatial splits), `trainer.py` (ensemble training), and `tune.py` (Optuna tuning) in `backend/common_fn/`.
- **Model Run Scripts**: Dedicated execution directories in `backend/run/<model_name>/` containing `config.py` and `main.py`.

### 3. Premium Frontend (`frontend/app.js`, `style.css`)
- **Glassmorphism UI**: A modern, translucent interface with interactive sidebars and micro-animations.
- **Leaflet Integration**: Custom tile layers for both NASA imagery and the dynamic heatmap.
- **Live Diagnostics**: Real-time Lat/Lon tracking and point-specific data hydration from the API.

---

## 📦 Acquired & External Components

The following components were sourced from external repositories, libraries, or provided as base datasets:

### 1. Primary Datasets
- **Core Set**: `core_set_2ppd.csv` (Base collection of Martian planetary features).
- **8ppd Feature Data**: '8ppd_features_full.csv' (8ppd feature data).
- **Planetary Sources**:
    - **MOLA**: NASA Mars Orbiter Laser Altimeter (Elevation, Slope, Roughness).
    - **OMEGA/TES**: Mineralogical and thermal datasets (Albedo, Dust, Thermal Inertia).
    - **GRS**: Gamma Ray Spectrometer (Hydrogen/Water proxies).
### 2. External Libraries (Attributions)
- **FastAPI/Uvicorn**: High-performance Python web framework.
- **Leaflet.js**: Open-source JavaScript library for interactive maps.
- **PyTorch/Scikit-Learn**: Machine learning and statistical modeling.
- **Pandas/PyArrow**: Data manipulation and high-performance Parquet I/O.
- **Matplotlib/PIL**: Image generation and color mapping for tiles.

### 3. NASA WMTS Proxy
- **NASA JPL**: Tiles are proxied from the `Mars_Viking_MDIM21_ClrMosaic` catalog via the NASA planetary API.

---

## 🔍 Main Code Entry Points

- **Backend API**: Run `python backend/API/main.py` (Starts server on port 8000).
- **Frontend UI**: Open `frontend/index.html` (Connects to the backend via REST).
- **Model Training**: Run individual scripts in `backend/models/` to retrain specific parameters.

---

## 📜 Attributions

- **Data Credits**: All planetary data provided by NASA, JPL, and the USGS Astrogeology Science Center.
- **Engineering**: Development, refactoring, and performance optimization performed by the project author.
- **Core Methodology**: Based on the *Martian Landing Site Selection Framework*.
