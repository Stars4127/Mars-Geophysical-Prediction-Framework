import pandas as pd
import numpy as np
import os
import threading
from scipy.spatial import cKDTree
import pyarrow.parquet as pq

DATA_PATH = "backend/models/datasets/8ppd_extrapolated_targets.parquet"

class NovaEngine:
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
            print(f"❌ Data file not found at {DATA_PATH}")
            return

        print(f"📖 Lazy-loading 8ppd index from Parquet...")
        # Only load Lat/Lon to save RAM (~400MB for 50M points)
        self.df = pd.read_parquet(DATA_PATH, columns=['Lat', 'Lon'])
        self.df['Lat'] = self.df['Lat'].astype('float32')
        self.df['Lon'] = self.df['Lon'].astype('float32')

        # Normalize Longitude once
        self.df['Lon'] = ((self.df['Lon'] + 180) % 360) - 180

        print("📊 Extracting normalization stats from Parquet Metadata...")
        features = ['Water', 'Dust', 'Surface_Temp', 'Temp_Variation', 'Gravity', 'TI_Variance']
        all_cols = features + [f + "_Uncertainty" for f in features]
        
        # Use Parquet metadata for nearly instant min/max extraction
        pf = pq.ParquetFile(DATA_PATH)
        metadata = pf.metadata
        for col_name in all_cols:
            col_idx = -1
            # Find column index by name in metadata
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

        print("🌲 Building Spatial Index (cKDTree)...")
        self.tree = cKDTree(self.df[['Lat', 'Lon']].values)
        print("✅ engine ready.")

    def get_nearest_point(self, lat, lon):
        dist, idx = self.tree.query([lat, lon])
        # Convert to dict and replace NaNs/Infs with 0 for JSON compliance
        row = self.df.iloc[idx].copy()
        row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
        return row.to_dict()

    def calculate_goodness(self, weights):
        """
        weights: dict of {feature_name: weight_value}
        e.g. {"Water": 1.0, "Surface_Temp": 0.5, "Uncertainty_Penalty": 0.8}
        """
        if self.df is None:
            print("⚠️ Engine: DataFrame is missing!")
            return []

        try:
            n_rows = len(self.df)
            goodness = np.zeros(n_rows, dtype=np.float32)
            total_weight = 0
            penalty_weight = float(weights.get("Uncertainty_Penalty", 0.5))

            # Stream features in batches to compute global score
            features_to_load = [f for f, w in weights.items() if f in self.normalized_data and float(w) != 0]
            if not features_to_load:
                return []
                
            cols = []
            for f in features_to_load:
                cols.append(f)
                if f"{f}_Uncertainty" in self.normalized_data:
                    cols.append(f"{f}_Uncertainty")

            # Load only needed features in one pass (or chunks if needed, but 1M rows * 10 cols fits)
            # For 50M rows, we chunk it.
            parquet_file = pq.ParquetFile(DATA_PATH)
            
            row_idx = 0
            for batch in parquet_file.iter_batches(batch_size=2_000_000, columns=cols):
                chunk = batch.to_pandas().astype('float32')
                batch_len = len(chunk)
                
                for feat in features_to_load:
                    weight = float(weights[feat])
                    if feat not in chunk.columns: continue
                    
                    # Normalize on the fly
                    stats = self.normalized_data[feat]
                    val = (chunk[feat].values - stats['min']) / (stats['max'] - stats['min'] + 1e-6)
                    
                    # ===== SCIENTIFIC SCALING =====
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
                    if unc_feat in chunk.columns:
                        u_stats = self.normalized_data[unc_feat]
                        u_val = (chunk[unc_feat].values - u_stats['min']) / (u_stats['max'] - u_stats['min'] + 1e-6)
                        confidence = np.exp(-u_val * penalty_weight)
                        score = score * confidence

                    goodness[row_idx:row_idx+batch_len] += score * weight
                
                row_idx += batch_len
                if total_weight == 0: # Initialize total_weight on first relevant feat
                     for f in features_to_load: total_weight += abs(float(weights[f]))

            # ===== GLOBAL NORMALIZATION =====
            if total_weight > 0:
                p10, p90 = np.percentile(goodness, [10, 90])
                self.last_global_stats = {
                    "weights": weights.copy(),
                    "p10": p10,
                    "p90": p90
                }
                goodness = 100 * (goodness - p10) / (p90 - p10 + 1e-6)
                goodness = np.clip(goodness, 0, 100)

            # ===== PREPARE LEAFLET OUTPUT =====
            step = 500
            subset = self.df.iloc[::step]

            lons = subset['Lon'].values
            adjusted_lons = ((lons + 180) % 360) - 180

            lats = np.nan_to_num(subset['Lat'].values)
            adjusted_lons = np.nan_to_num(adjusted_lons)
            final_scores = np.nan_to_num(goodness[::step])

            results = []
            for i in range(len(subset)):
                results.append({
                    "Lat": round(float(lats[i]), 4),
                    "Lon": round(float(adjusted_lons[i]), 4),
                    "score": round(float(final_scores[i]), 2)
                })

            print(f"✅ Success: Calculated {len(results)} points.")
            return results

        except Exception as e:
            print(f"❌ Backend Error in calculate_goodness: {e}")
            raise e


    def get_tile_bounds(self, z, x, y):
        # Conversions for EPSG:4326 (Plate Carrée)
        # World is -180 to 180 lon, -90 to 90 lat
        # Zoom 0: 2 tiles wide (0=west, 1=east), 1 tile high
        # Leaflet assumes 256x256 tiles
        
        # Total tiles in x and y
        tiles_x = 2 ** (z + 1) # Global width is 360, tile is usually defined as covering some degrees
        # Wait, standard XYZ for EPSG:3857 (Web Mercator) is different.
        # But our frontend says EPSG:4326 (Plate Carrée).
        # In this CRS:
        # Level 0: 2x1 tiles covering the whole world.
        # Width: 360 degrees. Height: 180 degrees.
        # Tile width (deg): 360 / (2 * 2^z) -> 180 / 2^z
        # Tile height (deg): 180 / 2^z (usually matches for square tiles in pixels, but degrees differ)
        
        # Let's align with what frontend tile_proxy does/expects. 
        # Frontend: max_y = 2^z, max_x = 2 * 2^z.
        # So at z=0: x in [0,1], y in [0,0].
        # 360 deg width / 2 tiles = 180 deg/tile.
        # 180 deg height / 1 tile = 180 deg/tile.
        
        num_tiles_x = 2 * (2 ** z)
        num_tiles_y = 1 * (2 ** z)
        
        deg_per_tile_x = 360.0 / num_tiles_x
        deg_per_tile_y = 180.0 / num_tiles_y
        
        # x=0 is -180, x=max is +180
        min_lon = -180.0 + (x * deg_per_tile_x)
        max_lon = min_lon + deg_per_tile_x
        
        # y=0 is +90 (top), y=max is -90 (bottom)
        # So y varies from 0 (top) to N (bottom)
        # max_lat = 90 - (y * deg_per_tile_y) -> WRONG for standard TMS
        # XYZ usually has y=0 at top.
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

            # --- 0. Global Stats Synchronization Guard ---
            if self.last_global_stats['weights'] != weights:
                print("⚠️ Tile Engine: Global stats outdated for requested weights. Synchronizing...")
                self.calculate_goodness(weights)
            
            global_stats = self.last_global_stats
            
            # Safety: Ensure we have valid stats
            if 'p10' not in global_stats or 'p90' not in global_stats:
                print("⚠️ Tile Engine: Missing global stats. Computing now...")
                self.calculate_goodness(weights)
                global_stats = self.last_global_stats

        # --- 1. Tile Context & Buffered Bounds ---
        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(z, x, y)
        
        # Smooth cap to avoid excessive dilution
        sigma = min(3.0, 1.5 + z * 0.5) 
        pad = int(4 * sigma) # context to ensure seamless blending
        
        # Calculate degrees per pixel for the query expansion
        deg_per_px_x = (max_lon - min_lon) / 256.0
        deg_per_px_y = (max_lat - min_lat) / 256.0
        
        # Expanded bounds to pull neighboring data
        q_min_lon = min_lon - (pad * deg_per_px_x)
        q_max_lon = max_lon + (pad * deg_per_px_x)
        q_min_lat = min_lat - (pad * deg_per_px_y)
        q_max_lat = max_lat + (pad * deg_per_px_y)
        
        # --- 2. Data Filtering & Scoring (LAZY PARQUET LOADING) ---
        # The Parquet file has RAW coordinates (0-5127.5 lat, 0-9711.5 lon)
        # We need to convert our query bounds back to raw space for filtering
        SCALE = 27.0
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
        
        # Transform coordinates from raw pixel space to lat/lon
        subset['Lat'] = (subset['Lat'] / SCALE) - 90
        subset['Lon'] = (subset['Lon'] / SCALE) - 180
        subset['Lon'] = ((subset['Lon'] + 180) % 360) - 180  # Normalize to -180/180
        
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
            
            # Normalize subset relative to global stats
            feat_stats = self.normalized_data[feat]
            val = (subset[feat].values - feat_stats['min']) / (feat_stats['max'] - feat_stats['min'] + 1e-6)
            
            # ===== SCIENTIFIC SCALING =====
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

        # Global Absolute Normalization
        # Instead of local tile normalization, we map the goodness scores to the GLOBAL range
        # captured in last_global_stats. This ensures "Green" is the same value everywhere.
        goodness = (goodness - global_stats['p10']) / (global_stats['p90'] - global_stats['p10'] + 1e-6)
        goodness = np.clip(goodness, 0, 1)

        # --- 3. Grid Creation, Smoothing & Cropping ---
        grid_dim = 256 + 2 * pad
        grid_sum = np.zeros((grid_dim, grid_dim))
        grid_count = np.zeros((grid_dim, grid_dim))
        
        # Map relative to the PADDED bounds
        pixel_x = ((subset['Lon'].values - q_min_lon) / (q_max_lon - q_min_lon) * grid_dim).astype(int)
        pixel_y = ((q_max_lat - subset['Lat'].values) / (q_max_lat - q_min_lat) * grid_dim).astype(int) 
        
        pixel_x = np.clip(pixel_x, 0, grid_dim - 1)
        pixel_y = np.clip(pixel_y, 0, grid_dim - 1)

        # Injected normalized goodness onto the grid
        np.add.at(grid_sum, (pixel_y, pixel_x), goodness)
        np.add.at(grid_count, (pixel_y, pixel_x), 1)

        # Weighted Smoothing: Smooth the sums and counts separately
        # This prevents "dilution" at the edges behavior where a high-value point 
        # gets averaged with zero-value backgrounds early.
        sum_smoothed = gaussian_filter(grid_sum, sigma=sigma)
        count_smoothed = gaussian_filter(grid_count, sigma=sigma)

        # Recalculate Average AFTER smoothing (this spreads the color intensity properly)
        with np.errstate(divide='ignore', invalid='ignore'):
            final_grid = sum_smoothed / (count_smoothed + 1e-6)
            final_grid = np.nan_to_num(final_grid)
            final_grid = np.clip(final_grid, 0, 1)

        # Crop back to 256x256
        final_grid = final_grid[pad:pad+256, pad:pad+256]
        final_count = count_smoothed[pad:pad+256, pad:pad+256]

        # --- 4. Rendering & Alpha ---
        # Fixed alpha multiplier for stability
        alpha_mult = 4.0
        mask_threshold = 0.005
        
        mask_visibility = (final_count > mask_threshold)
        cmap = plt.get_cmap('RdYlGn')
        rgba_img = cmap(final_grid)

        # Apply alpha floor based on density to prevent vanishing at high zoom
        # We ensure a minimum alpha of 0.3 where data is present (mask_visibility).
        alpha_channel = np.clip(final_count * alpha_mult, 0.3, 0.9)
        alpha_channel[~mask_visibility] = 0
        rgba_img[..., 3] = alpha_channel

        # Convert to PNG Byte stream
        rgba_uint8 = (rgba_img * 255).astype(np.uint8)
        img = Image.fromarray(rgba_uint8)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

# Singleton instance
engine = NovaEngine()
