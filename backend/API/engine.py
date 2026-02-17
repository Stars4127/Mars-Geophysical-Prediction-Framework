import pandas as pd
import numpy as np
import os
import threading
from scipy.spatial import cKDTree
import pyarrow.parquet as pq

DATA_PATH = "backend/models/datasets/8ppd_extrapolated_targets.parquet"

SCALE = 27.0

def raw_to_geo(lat_raw, lon_raw):
    """Single source of truth for coordinate conversion."""
    lat = (lat_raw / SCALE) - 90
    lon = (lon_raw / SCALE) - 180
    return lat, lon

class MSPF:
    def __init__(self):
        self.df = None
        self.tree = None
        self.normalized_data = {}
        self.render_lock = threading.Lock()
        self.last_global_stats = {
            "weights": None,
            "p10": 0.0,
            "p90": 1.0
        }
        self.load_data()

    def load_data(self):
        if not os.path.exists(DATA_PATH):
            print(f" Data file not found at {DATA_PATH}")
            return

        print(f" Lazy-loading 8ppd index from Parquet...")
        self.df = pd.read_parquet(DATA_PATH, columns=['Lat', 'Lon'])
        self.df['Lat'], self.df['Lon'] = raw_to_geo(self.df['Lat'].values, self.df['Lon'].values)
        self.df['Lon'] = ((self.df['Lon'] + 180) % 360) - 180

        print(" Extracting normalization stats from Parquet Metadata...")
        features = ['Water', 'Dust', 'Surface_Temp', 'Temp_Variation', 'Gravity', 'TI_Variance']
        all_cols = features + [f + "_Uncertainty" for f in features]
        
        pf = pq.ParquetFile(DATA_PATH)
        metadata = pf.metadata
        for col_name in all_cols:
            col_idx = -1
            for i in range(pf.num_row_groups):
                rg = pf.metadata.row_group(i)
                for j in range(rg.num_columns):
                    if rg.column(j).path_in_schema == col_name:
                        stats = rg.column(j).statistics
                        if stats:
                            current = self.normalized_data.get(col_name, {'min': float('inf'), 'max': float('-inf')})
                            self.normalized_data[col_name] = {
                                'min': min(current['min'], stats.min),
                                'max': max(current['max'], stats.max)
                            }
                        break

        print(" Building Spatial Index (cKDTree)...")
        self.tree = cKDTree(self.df[['Lat', 'Lon']].values)
        print(" engine ready.")

    def get_nearest_point(self, lat, lon):
        dist, idx = self.tree.query([lat, lon])
        pt = self.df.iloc[idx]
        
        raw_lat = (pt['Lat'] + 90) * SCALE
        raw_lon = (pt['Lon'] + 180) * SCALE
        
        filters = [('Lat', '==', raw_lat), ('Lon', '==', raw_lon)]
        table = pq.read_table(DATA_PATH, filters=filters)
        
        if table.num_rows == 0:
            return pt.to_dict()
            
        row = table.to_pandas().iloc[0].to_dict()
        return {k: (0 if pd.isna(v) or np.isinf(v) else v) for k, v in row.items()}

    def calculate_goodness(self, weights):
        """
        weights: dict of {feature_name: weight_value}
        e.g. {"Water": 1.0, "Surface_Temp": 0.5, "Uncertainty_Penalty": 0.8}
        """
        if self.df is None:
            print(" Engine: DataFrame is missing!")
            return []

        try:
            n_rows = len(self.df)
            goodness = np.zeros(n_rows, dtype=np.float32)
            total_weight = 0
            total_weight = 0
            penalty_weight = float(weights.get("Uncertainty_Penalty", 0.5))

            features_to_load = [f for f, w in weights.items() if f in self.normalized_data and float(w) != 0]
            if not features_to_load:
                return []
                
            cols = []
            for f in features_to_load:
                cols.append(f)
                if f"{f}_Uncertainty" in self.normalized_data:
                    cols.append(f"{f}_Uncertainty")

            parquet_file = pq.ParquetFile(DATA_PATH)
            
            row_idx = 0
            for batch in parquet_file.iter_batches(batch_size=2_000_000, columns=cols):
                chunk = batch.to_pandas().astype('float32')
                batch_len = len(chunk)
                
                for feat in features_to_load:
                    weight = float(weights[feat])
                    if feat not in chunk.columns: continue
                    
                    stats = self.normalized_data[feat]
                    val = (chunk[feat].values - stats['min']) / (stats['max'] - stats['min'] + 1e-6)
                    
                    if feat == "Water":
                        score = np.tanh(val * 1.8)
                    elif feat == "Dust":
                        score = 1.0 - np.power(val, 1.5)
                    elif feat == "Surface_Temp":
                        diff = val - 0.65
                        score = np.where(diff < 0, np.exp(-(diff**2) / 0.06), np.exp(-(diff**2) / 0.18))
                    elif feat in ["Temp_Variation", "TI_Variance"]:
                        score = 1.0 - np.clip(val * 1.8, 0, 1)
                    elif feat == "Gravity":
                        score = np.exp(-((val - 0.5)**2) / 0.05)
                    else:
                        score = val

                    unc_feat = f"{feat}_Uncertainty"
                    if unc_feat in chunk.columns:
                        u_stats = self.normalized_data[unc_feat]
                        u_val = (chunk[unc_feat].values - u_stats['min']) / (u_stats['max'] - u_stats['min'] + 1e-6)
                        confidence = np.exp(-u_val * penalty_weight)
                        score = score * confidence

                    goodness[row_idx:row_idx+batch_len] += score * weight
                
                row_idx += batch_len
                if total_weight == 0: # Initialize total_weight on first relevant feat
                     for f in features_to_load: total_weight += abs(float(weights[f]))

            if total_weight > 0:
                sample_goodness = goodness[::500] if len(goodness) > 1000 else goodness
                p5, p95 = np.percentile(sample_goodness, [5, 95])
                
                self.last_global_stats = {
                    "weights": weights.copy(),
                    "p5": p5,
                    "p95": p95
                }
                goodness = 100 * (goodness - p5) / (p95 - p5 + 1e-6)
                goodness = np.clip(goodness, 0, 100)

            subset = self.df.sample(frac=0.002, random_state=42)

            lons = subset['Lon'].values
            adjusted_lons = ((lons + 180) % 360) - 180

            lats = np.nan_to_num(subset['Lat'].values)
            adjusted_lons = np.nan_to_num(adjusted_lons)
            
            final_scores = np.nan_to_num(goodness[subset.index])

            results = []
            for i in range(len(subset)):
                results.append({
                    "Lat": round(float(lats[i]), 4),
                    "Lon": round(float(adjusted_lons[i]), 4),
                    "score": round(float(final_scores[i]), 2)
                })

            print(f" Success: Calculated {len(results)} points.")
            return results

        except Exception as e:
            print(f" Backend Error in calculate_goodness: {e}")
            raise e


    def get_tile_bounds(self, z, x, y):
        num_tiles_x = 2 * (2 ** z)
        num_tiles_y = 1 * (2 ** z)
        
        deg_per_tile_x = 360.0 / num_tiles_x
        deg_per_tile_y = 180.0 / num_tiles_y
        
        min_lon = -180.0 + (x * deg_per_tile_x)
        max_lon = min_lon + deg_per_tile_x
        
        max_lat = 90.0 - (y * deg_per_tile_y)
        min_lat = max_lat - deg_per_tile_y
        
        return min_lon, max_lon, min_lat, max_lat

    def render_tile(self, z, x, y, weights):
        with self.render_lock: # Limit concurrency to avoid memory spikes
            import matplotlib
            matplotlib.use('Agg') # Non-interactive backend
            import matplotlib.pyplot as plt
            from PIL import Image
            from scipy.ndimage import gaussian_filter
            import io

            if self.last_global_stats['weights'] != weights:
                print(" Tile Engine: Global stats outdated for requested weights. Synchronizing...")
                self.calculate_goodness(weights)
            
            global_stats = self.last_global_stats
            
            if 'p5' not in global_stats or 'p95' not in global_stats:
                print(" Tile Engine: Missing global stats. Computing now...")
                self.calculate_goodness(weights)
                global_stats = self.last_global_stats

        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(z, x, y)
        
        sigma = min(2.0, 1.0 + z * 0.25) 
        pad = int(4 * sigma)
        
        deg_per_px_x = (max_lon - min_lon) / 256.0
        deg_per_px_y = (max_lat - min_lat) / 256.0
        
        q_min_lon = min_lon - (pad * deg_per_px_x)
        q_max_lon = max_lon + (pad * deg_per_px_x)
        q_min_lat = min_lat - (pad * deg_per_px_y)
        q_max_lat = max_lat + (pad * deg_per_px_y)
        
        raw_min_lat = (q_min_lat + 90) * SCALE
        raw_max_lat = (q_max_lat + 90) * SCALE
        raw_min_lon = (q_min_lon + 180) * SCALE
        raw_max_lon = (q_max_lon + 180) * SCALE
        
        filters = [
            ('Lon', '>=', raw_min_lon), ('Lon', '<', raw_max_lon),
            ('Lat', '>=', raw_min_lat), ('Lat', '<', raw_max_lat)
        ]
        
        # Only load columns needed for this specific weight set
        needed_cols = ['Lat', 'Lon']
        for f, w in weights.items():
            if float(w) != 0 and f in self.normalized_data:
                needed_cols.append(f)
                if f"{f}_Uncertainty" in self.normalized_data:
                    needed_cols.append(f"{f}_Uncertainty")
        
        table = pq.read_table(DATA_PATH, columns=needed_cols, filters=filters)
        subset = table.to_pandas().astype('float32')
        
        subset['Lat'], subset['Lon'] = raw_to_geo(subset['Lat'].values, subset['Lon'].values)
        subset['Lon'] = ((subset['Lon'] + 180) % 360) - 180
        
        if subset.empty:
            img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        n_subset = len(subset)
        goodness = np.zeros(n_subset, dtype=np.float32)
        total_weight = 0
        penalty_weight = float(weights.get("Uncertainty_Penalty", 0.5))

        for feat, weight_val in weights.items():
            weight = float(weight_val)
            if weight == 0 or feat not in subset.columns: continue
            
            feat_stats = self.normalized_data[feat]
            val = (subset[feat].values - feat_stats['min']) / (feat_stats['max'] - feat_stats['min'] + 1e-6)
            
            if feat == "Water":
                score = np.tanh(val * 1.8)
            elif feat == "Dust":
                score = 1.0 - np.power(val, 1.5)
            elif feat == "Surface_Temp":
                diff = val - 0.65
                score = np.where(diff < 0, np.exp(-(diff**2) / 0.06), np.exp(-(diff**2) / 0.18))
            elif feat in ["Temp_Variation", "TI_Variance"]:
                score = 1.0 - np.clip(val * 1.8, 0, 1)
            elif feat == "Gravity":
                score = np.exp(-((val - 0.5)**2) / 0.05)
            else:
                score = val

            # ===== UNCERTAINTY SCALER =====
            unc_feat = f"{feat}_Uncertainty"
            if unc_feat in subset.columns:
                u_stats = self.normalized_data[unc_feat]
                u_val = (subset[unc_feat].values - u_stats['min']) / (u_stats['max'] - u_stats['min'] + 1e-6)
                confidence = np.exp(-u_val * penalty_weight)
                score = score * confidence

            goodness += score * weight
            total_weight += abs(weight)

        # Global Absolute Normalization (Using p5/p95)
        goodness = (goodness - global_stats['p5']) / (global_stats['p95'] - global_stats['p5'] + 1e-6)
        goodness = np.clip(goodness, 0, 1)

        grid_dim = 256 + 2 * pad
        grid_sum = np.zeros((grid_dim, grid_dim))
        grid_count = np.zeros((grid_dim, grid_dim))
        
        pixel_x = ((subset['Lon'].values - q_min_lon) / (q_max_lon - q_min_lon) * grid_dim)
        pixel_y = ((q_max_lat - subset['Lat'].values) / (q_max_lat - q_min_lat) * grid_dim)
        
        pixel_x = np.round(pixel_x).astype(int)
        pixel_y = np.round(pixel_y).astype(int)
        
        pixel_x = np.clip(pixel_x, 0, grid_dim - 1)
        pixel_y = np.clip(pixel_y, 0, grid_dim - 1)

        np.add.at(grid_sum, (pixel_y, pixel_x), goodness)
        np.add.at(grid_count, (pixel_y, pixel_x), 1)

        sum_smoothed = gaussian_filter(grid_sum, sigma=sigma)
        count_smoothed = gaussian_filter(grid_count, sigma=sigma)

        with np.errstate(divide='ignore', invalid='ignore'):
            final_grid = sum_smoothed / (count_smoothed + 1e-6)
            final_grid = np.nan_to_num(final_grid)
            final_grid = np.clip(final_grid, 0, 1)

        # Crop back to 256x256
        final_grid = final_grid[pad:pad+256, pad:pad+256]
        final_count = count_smoothed[pad:pad+256, pad:pad+256]

        alpha_mult = 4.0
        mask_threshold = 0.005
        
        mask_visibility = (final_count > mask_threshold)
        cmap = plt.get_cmap('RdYlGn')
        rgba_img = cmap(final_grid)

        alpha_channel = np.clip(final_count * alpha_mult, 0.3, 0.9)
        alpha_channel[~mask_visibility] = 0
        rgba_img[..., 3] = alpha_channel

        rgba_uint8 = (rgba_img * 255).astype(np.uint8)
        img = Image.fromarray(rgba_uint8)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

# Singleton instance
engine = MSPF()
