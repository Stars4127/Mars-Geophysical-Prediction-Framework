import os
import json
import numpy as np
import traceback
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from dust_predictor import DustPredictor
from gravity_predictor import GravityPredictor
from surface_temp import TempPredictor
from temp_variation_predictor import TempVariationPredictor
from ti_variance_predictor import ThermalInertiaVariancePredictor
from water_predictor import WaterPredictor

METRICS_DIR = "backend/models/metrics"

def evaluate_and_log(predictor, model_name):
    print(f"\n======================================")
    print(f" Evaluating {model_name}...")
    print(f"======================================")
    
    try:
        predictor.preprocess()
        outputs = predictor.load_and_eval()
        
        # Safely extract mean_pred whether it returned 2 or 5 items
        mean_pred = outputs[0]
        uncertainty = outputs[1]
        y_true = predictor.y_test_raw
        
        r2 = float(r2_score(y_true, mean_pred))
        mae = float(mean_absolute_error(y_true, mean_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, mean_pred)))
        mean_uncertainty = float(np.mean(uncertainty))
        
        metrics = {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Mean_Uncertainty": mean_uncertainty
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
        ("dust_predictor", DustPredictor(
            path= "backend/models/datasets/core_set_2ppd.csv",
            target_column='OMEGA Ferric/Dust 860nm ratio Ody et al 2012',
            features=[
                "MOLA 128ppd Elevation",
                "MOLA 128ppd Slope",
                "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
                "MOLA 128ppd Aspect",
                "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012"
            ]
        )),
        ("gravity_predictor", GravityPredictor(
            csv_path="backend/models/datasets/core_set_2ppd.csv",
            target="Mars Bouguer Gravity Anomaly",
            features=[
                'MOLA 128ppd Elevation',
                'MOLA 128ppd Slope',
                'MOLA Roughness 2.4km Baseline - Numeric',
                'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
                'Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
                'TES Basalt Abundance - Numeric',
                'TES Rock Abundance (Nowicki and Christensen 2007) - Numeric',
                'Latitude (N)'
            ]
        )),
        ("surface_temp", TempPredictor(
            path="backend/models/datasets/core_set_2ppd.csv",
            target_column='Yearly Average Mars Surface Temperature (C)',
            features=[
                "Latitude (N)",
                "MOLA 128ppd Elevation",
                "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
                "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
                "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
                "MOLA Roughness 0.6km Baseline - Numeric",
                "MOLA Roughness 2.4km Baseline - Numeric"
            ]
        )),
        ("temp_variation_predictor", TempVariationPredictor(
            path="backend/models/datasets/core_set_2ppd.csv",
            target_column='Yearly Mars Surface Temperature Variation (C)',
            features=[
                "Latitude (N)",
                "MOLA 128ppd Elevation",
                "MOLA 128ppd Slope",
                "MOLA 128ppd Aspect",
                "OMEGA Est. Lambert Albedo 1080nm Ody et al 2012",
                "Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
                "Nightside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)",
                "MOLA Roughness 0.6km Baseline - Numeric"
            ]
        )),
        ("ti_variance_predictor", ThermalInertiaVariancePredictor(
            path="backend/models/datasets/core_set_2ppd.csv",
            features=[
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
        )),
        ("water_predictor", WaterPredictor(
            csv_path="backend/models/datasets/core_set_2ppd.csv",
            target="Mars Odyssey Neutron Spectrometer % WEH",
            features=[
                'MOLA 128ppd Elevation',
                'OMEGA Est. Lambert Albedo 1080nm Ody et al 2012',
                'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
                'TES Basalt Abundance - Numeric',
                'Latitude (N)',
                'Mars Crustal Thickness (GSFC)'
            ]
        ))
    ]

    for name, predictor_ins in models_to_run:
        evaluate_and_log(predictor_ins, name)

if __name__ == "__main__":
    main()
