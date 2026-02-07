import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree

DATA_PATH = "backend/models/datasets/8ppd_extrapolated_targets.csv"

class NovaEngine:
    def __init__(self):
        self.df = None
        self.tree = None
        self.normalized_data = {}
        self.load_data()

    def load_data(self):
        if not os.path.exists(DATA_PATH):
            print(f"❌ Data file not found at {DATA_PATH}")
            return

        print(f"📖 Loading 8ppd data into memory...")
        self.df = pd.read_csv(DATA_PATH)
        
        # Normalize Longitude to -180 to 180 standard
        # 0-360 -> -180-180
        if 'Lon' in self.df.columns:
             # ((lon + 180) % 360) - 180
             self.df['Lon'] = ((self.df['Lon'] + 180) % 360) - 180
        
        # Pre-calculate min/max for normalization of key features
        features = ['Water', 'Dust', 'Surface_Temp', 'Temp_Variation', 'Gravity', 'TI_Variance']
        for feat in features:
            if feat in self.df.columns:
                f_min = self.df[feat].min()
                f_max = self.df[feat].max()
                # Store normalized version for fast math
                # formula: (x - min) / (max - min)
                self.normalized_data[feat] = (self.df[feat] - f_min) / (f_max - f_min)
                
                # Also normalize uncertainties
                unc_feat = f"{feat}_Uncertainty"
                if unc_feat in self.df.columns:
                    u_min = self.df[unc_feat].min()
                    u_max = self.df[unc_feat].max()
                    self.normalized_data[unc_feat] = (self.df[unc_feat] - u_min) / (u_max - u_min)

        print("🌲 Building Spatial Index (cKDTree)...")
        # Build tree for nearest neighboring lookups (double click)
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
            goodness = np.zeros(len(self.df))
            total_weight = 0
            penalty_weight = float(weights.get("Uncertainty_Penalty", 0.5))

            for feat, weight_val in weights.items():
                if feat in self.normalized_data and "_Uncertainty" not in feat:
                    weight = float(weight_val)
                    val = self.normalized_data[feat].values

                    # --- Feature-specific scientific scoring ---
                    if feat == "Water":
                        score = val / (val + 0.15)
                    elif feat == "Dust":
                        score = 1.0 - val
                    elif feat == "Surface_Temp":
                        diff = val - 0.7
                        score = np.where(
                            diff < 0, 
                            np.exp(-(diff**2) / 0.02),  # Cold penalty
                            np.exp(-(diff**2) / 0.12)   # Hot penalty
                        )
                    elif feat in ["Temp_Variation", "TI_Variance"]:
                        regional_mean = val.mean()
                        score = 1.0 - np.clip(np.abs(val - regional_mean) * 3.0, 0, 1)
                    elif feat == "Gravity":
                        score = np.exp(-((val - 0.5)**2) / 0.02)
                    else:
                        score = val  # fallback linear

                    goodness += score * weight
                    total_weight += abs(weight)

                    # --- Apply uncertainty penalty ---
                    unc_feat = f"{feat}_Uncertainty"
                    if unc_feat in self.normalized_data:
                        goodness -= self.normalized_data[unc_feat].values * penalty_weight

            # --- Normalize 0-100 for display ---
            if total_weight > 0:
                g_min, g_max = float(goodness.min()), float(goodness.max())
                if g_max > g_min:
                    goodness = 100 * (goodness - g_min) / (g_max - g_min)

            # --- Prepare subset for JSON/leaflet ---
            step = 500
            subset = self.df.iloc[::step]
            lons = subset['Lon'].values
            adjusted_lons = ((lons + 180) % 360) - 180
            lats = np.nan_to_num(subset['Lat'].values)
            final_scores = np.nan_to_num(goodness[::step])

            results = []
            for i in range(len(subset)):
                results.append({
                    "Lat": round(float(lats[i]), 4),
                    "Lon": round(float(adjusted_lons[i]), 4),
                    "score": round(float(final_scores[i]), 2)
                })

            print(f"✅ Success: Calculated {len(results)} points and cleansed NaNs.")
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
        import matplotlib
        matplotlib.use('Agg') # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from PIL import Image
        from scipy.ndimage import gaussian_filter
        import io

        # --- 1. Tile Context & Buffered Bounds ---
        # Get base bounds for a 256x256 tile
        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(z, x, y)
        
        # Define smoothing parameters (as per user instructions)
        sigma = 2.5
        pad = int(4 * sigma) # 10 pixels of context to ensure seamless blending
        
        # Calculate degrees per pixel for the query expansion
        deg_per_px_x = (max_lon - min_lon) / 256.0
        deg_per_px_y = (max_lat - min_lat) / 256.0
        
        # Expanded bounds to pull neighboring data
        q_min_lon = min_lon - (pad * deg_per_px_x)
        q_max_lon = max_lon + (pad * deg_per_px_x)
        q_min_lat = min_lat - (pad * deg_per_px_y)
        q_max_lat = max_lat + (pad * deg_per_px_y)
        
        # --- 2. Data Filtering & Scoring ---
        mask = (
            (self.df['Lon'] >= q_min_lon) & 
            (self.df['Lon'] < q_max_lon) & 
            (self.df['Lat'] >= q_min_lat) & 
            (self.df['Lat'] < q_max_lat)
        )
        subset = self.df[mask]
        
        if subset.empty:
            img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        goodness = np.zeros(len(subset))
        total_weight = 0
        penalty_weight = float(weights.get("Uncertainty_Penalty", 0.5))

        for feat, weight_val in weights.items():
            if feat in self.normalized_data and "_Uncertainty" not in feat:
                weight = float(weight_val)
                if feat in subset.columns:
                    val = self.normalized_data[feat].values[mask]
                    goodness += val * weight
                    total_weight += abs(weight)
                    
                    unc_feat = f"{feat}_Uncertainty"
                    if unc_feat in self.normalized_data:
                        u_val = self.normalized_data[unc_feat].values[mask]
                        goodness -= u_val * penalty_weight

        # Global Normalization (consistent across all tiles)
        if total_weight > 0:
            goodness = goodness / total_weight
        goodness = np.clip(goodness, 0, 1)

        # --- 3. Padded Grid Creation & Point Injection ---
        grid_dim = 256 + 2 * pad
        grid = np.zeros((grid_dim, grid_dim))
        count = np.zeros((grid_dim, grid_dim))
        
        # Map relative to the PADDED bounds
        pixel_x = ((subset['Lon'].values - q_min_lon) / (q_max_lon - q_min_lon) * grid_dim).astype(int)
        pixel_y = ((q_max_lat - subset['Lat'].values) / (q_max_lat - q_min_lat) * grid_dim).astype(int) 
        
        pixel_x = np.clip(pixel_x, 0, grid_dim - 1)
        pixel_y = np.clip(pixel_y, 0, grid_dim - 1)

        np.add.at(grid, (pixel_y, pixel_x), goodness)
        np.add.at(count, (pixel_y, pixel_x), 1)

        # Average grid values before smoothing for accuracy
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = grid / (count + 0.000001)
            grid[count == 0] = 0
        
        # --- 4. Smooth on Extended Context & Crop ---
        grid_smoothed = gaussian_filter(grid, sigma=sigma)
        count_smoothed = gaussian_filter(count, sigma=sigma)

        # Crop back to the central 256x256 region
        final_grid = grid_smoothed[pad:pad+256, pad:pad+256]
        final_count = count_smoothed[pad:pad+256, pad:pad+256]

        # --- 5. Rendering & Alpha ---
        mask_visibility = (final_count > 0.005)
        cmap = plt.get_cmap('RdYlGn')
        rgba_img = cmap(final_grid)

        # Compute alpha from the counts inside this tile
        # Cap alpha at 0.9 as per instructions
        alpha_channel = np.clip(final_count * 4.0, 0, 0.9)
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

