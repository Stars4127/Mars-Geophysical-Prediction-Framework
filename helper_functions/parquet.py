import pandas as pd
import os

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "backend", "models", "datasets", "8ppd_extrapolated_targets.csv")
PARQUET_PATH = os.path.join(BASE_DIR, "backend", "models", "datasets", "8ppd_extrapolated_targets.parquet")
# =========================
# STEP 1: Load CSV
# =========================
print(f"📖 Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# =========================
# STEP 2: Downcast numeric types (optional, saves ~50% memory)
# =========================
print("🔧 Downcasting numeric types...")
for col in df.select_dtypes(include="float"):
    df[col] = df[col].astype("float32")

# =========================
# STEP 3: Write to Parquet (compressed)
# =========================
print(f"💾 Writing Parquet: {PARQUET_PATH}")
df.to_parquet(PARQUET_PATH, engine="pyarrow", index=False, compression="snappy")

# =========================
# DONE
# =========================
print("✅ Conversion complete!")
print(f"Original CSV size: {os.path.getsize(CSV_PATH)/1e9:.2f} GB")
print(f"Parquet size: {os.path.getsize(PARQUET_PATH)/1e9:.2f} GB")