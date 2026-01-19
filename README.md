# Nova Scope 2: Mars Landing Zone ML Project

## Project Overview
This project aims to create an AI-assisted landing site evaluation system for Mars. The system generates a **Landing Goodness Score** for any latitude-longitude location, helping users (e.g., scientists or mission planners) identify preliminary landing zones that are safe, scientifically valuable, and operationally feasible.

The final product will be deployed as a web application or interactive map, showing heatmaps for overall suitability based on multiple underlying factors.

### Data Paradigm
The project relies on a structured Mars dataset as the base for all models, following these principles:
*   **Time-independent features**: All inputs are static or long-term averages.
*   **Resolution Blocks**: Features are grouped by native resolution (e.g., 2 PPD) to avoid mismatched granularity.
*   **Independent Models**: Specific models predict hard-to-measure quantities (e.g., WEH, dust) which feed into the final score.

## Key Project Goals
1.  **Safety Assessment**: Quantify hazards such as slopes, surface roughness, and extreme temperatures.
2.  **Resource Analysis**: Estimate subsurface water content (WEH), mineral composition, and dust coverage.
3.  **Scientific Opportunity**: Identify locations with unique geological features, seasonal variations, or mineral diversity.
4.  **Operational Feasibility**: Determine survivability factors like temperature extremes and frost likelihood.

## Current Tasks
- [ ] Update `dust_predictor.py` to use `core_set_2ppd.csv`
- [ ] Update `water_predictor.py` to use `core_set_2ppd.csv`
- [ ] Work on the Landing Goodness Score logic
- [ ] Start frontend / web app
- [ ] Figure out if we need a thermal inertia predictor
- [ ] Add more models as needed
