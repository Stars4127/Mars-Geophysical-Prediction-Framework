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
            # Start with a base score of zero
            goodness = np.zeros(len(self.df))
            total_weight = 0

            # Convert to float to be safe
            penalty_weight = float(weights.get("Uncertainty_Penalty", 0.5))

            for feat, weight_val in weights.items():
                if feat in self.normalized_data and "_Uncertainty" not in feat:
                    weight = float(weight_val)
                    # Use the pre-normalized Series
                    goodness += self.normalized_data[feat].values * weight
                    total_weight += abs(weight)
                    
                    # Apply uncertainty penalty for this specific feature
                    unc_feat = f"{feat}_Uncertainty"
                    if unc_feat in self.normalized_data:
                        goodness -= self.normalized_data[unc_feat].values * penalty_weight

            # Normalize goodness score to 0-100 for display
            if total_weight > 0:
                g_min = float(goodness.min())
                g_max = float(goodness.max())
                if g_max > g_min:
                    goodness = 100 * (goodness - g_min) / (g_max - g_min)

            # Prepare the result and ensure JSON compliance
            step = 500 
            subset = self.df.iloc[::step]
            
            # Normalize Longitude to -180 to 180 for Leaflet
            lons = subset['Lon'].values
            adjusted_lons = ((lons + 180) % 360) - 180
            
            # Cleanse ALL values (Lats, Lons, Scores) of NaNs/Infs
            lats = np.nan_to_num(subset['Lat'].values, nan=0, posinf=0, neginf=0)
            adjusted_lons = np.nan_to_num(adjusted_lons, nan=0, posinf=0, neginf=0)
            final_scores = np.nan_to_num(goodness[::step], nan=0, posinf=0, neginf=0)
            
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
        import io

        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(z, x, y)
        
        # Filter dataframe for efficiency
        # We can implement a more efficient spatial query with cKDTree or just boolean mask for now.
        # Given dataset size (100k-1M points), boolean mask on Pandas is "okay" but could be slow.
        # However, for 8ppd extrapolations (millions of points), we need the index.
        # But cKDTree is for nearest neighbor.
        
        # Let's simple filter for now. 
        # Note: df is large.
        mask = (
            (self.df['Lon'] >= min_lon) & 
            (self.df['Lon'] < max_lon) & 
            (self.df['Lat'] >= min_lat) & 
            (self.df['Lat'] < max_lat)
        )
        subset = self.df[mask]
        
        if subset.empty:
            # Return transparent tile
            img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        # Calculate scores for this subset
        # We duplicate logic from calculate_goodness but optimized for this subset
        goodness = np.zeros(len(subset))
        total_weight = 0
        penalty_weight = float(weights.get("Uncertainty_Penalty", 0.5))

        for feat, weight_val in weights.items():
            if feat in self.normalized_data and "_Uncertainty" not in feat:
                weight = float(weight_val)
                # We need to grab values from the global arrays using the index mechanism OR re-extract
                # Re-extraction from subset is safer/easier to read.
                
                # Normalize on the fly? No, uses global min/max for consistency across tiles.
                # But subset doesn't have the normalized columns.
                # We should have stored normalized columns in df for RAM tradeoff? 
                # Or just do the math again.
                
                if feat in subset.columns:
                    # Get global min/max
                    # This is slightly inefficient but safe. 
                    # Better: self.normalized_data is a dict of Series (aligned with self.df).
                    # So we can index it.
                    global_norm_series = self.normalized_data[feat] # This is a Series/Array matching len(df)
                    # We need the values corresponding to 'mask'
                    # Boolean indexing on the numpy array
                    val = global_norm_series.values[mask]
                    
                    goodness += val * weight
                    total_weight += abs(weight)
                    
                    unc_feat = f"{feat}_Uncertainty"
                    if unc_feat in self.normalized_data:
                        u_val = self.normalized_data[unc_feat].values[mask]
                        goodness -= u_val * penalty_weight

        # Normalize 0-1 for colormap
        # We want global consistency? 
        # If we normalize per tile, colors shift as you pan.
        # We should clamp based on "expected" range or 0-1 if goodness is roughly 0-1.
        # Our goodness sums could be anything.
        # Let's normalize by total_weight to get roughly 0-1 range.
        # Normalize 0-1 for colormap
        # LENIENT NORMALIZATION
        # Standardize based on distribution:
        # We want the MEDIAN to be around 0.7 (Good/Green)
        # So most spots are "okay"
        
        if total_weight > 0:
            # Contrast-Enhanced Lenient Normalization
            # Instead of strict min-max, we use percentiles to define anchors.
            # This handles outliers and ensures the 'middle' data is colorful.
            p10 = np.percentile(goodness, 10)
            p90 = np.percentile(goodness, 90)
            
            if p90 > p10:
                # Map p10 to 0 (Red), p90 to 1 (Green)
                goodness = (goodness - p10) / (p90 - p10)
            else:
                # Fallback to local min-max if distribution is very tight
                g_min, g_max = goodness.min(), goodness.max()
                if g_max > g_min:
                    goodness = (goodness - g_min) / (g_max - g_min)
                else:
                    goodness.fill(0.5) 
            
            # Additional 'leniency' boost: 
            # Slightly shift everything towards 'Good' (Green)
            goodness = goodness ** 1.2 # Slightly more contrasty than before
            goodness = 0.0 + (goodness * 1.0) # Full range usage
            
        # Clip to 0-1
        goodness = np.clip(goodness, 0, 1)

        # Create 2D Histogram (Heatmap)
        # We map Lat/Lon to 0-256 pixel coordinates
        
        pixel_x = ((subset['Lon'].values - min_lon) / (max_lon - min_lon) * 256).astype(int)
        pixel_y = ((max_lat - subset['Lat'].values) / (max_lat - min_lat) * 256).astype(int) 
        
        # Clip again to be safe
        pixel_x = np.clip(pixel_x, 0, 255)
        pixel_y = np.clip(pixel_y, 0, 255)

        # Accumulate score in grid
        grid = np.zeros((256, 256))
        count = np.zeros((256, 256))
        
        indices = (pixel_y, pixel_x)
        np.add.at(grid, indices, goodness)
        np.add.at(count, indices, 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = grid / count
            grid[np.isnan(grid)] = 0
            
        # Point Expansion / Smoothing
        from scipy.ndimage import gaussian_filter
        
        sigma = 1.5
        grid_smoothed = gaussian_filter(grid, sigma=sigma)
        count_smoothed = gaussian_filter(count, sigma=sigma)
        
        # We re-mask based on smoothed density
        mask_visibility = (count_smoothed > 0.005)

        # Apply colormap
        cmap = plt.get_cmap('RdYlGn')
        
        # Grid is 0-1. Map to RGBA.
        rgba_img = cmap(grid_smoothed) 
        
        # Set Alpha based on density - VERY STRONG
        alpha_channel = rgba_img[..., 3]
        
        # Boosted alpha: 4.0 multiplier and 0.9 cap for "very strong" look
        alpha_channel[:] = np.clip(count_smoothed * 4.0, 0, 0.9)
        
        alpha_channel[~mask_visibility] = 0
        
        rgba_img[..., 3] = alpha_channel
        
        # Convert to uint8
        rgba_uint8 = (rgba_img * 255).astype(np.uint8)
        
        img = Image.fromarray(rgba_uint8)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

# Singleton instance
engine = NovaEngine()

