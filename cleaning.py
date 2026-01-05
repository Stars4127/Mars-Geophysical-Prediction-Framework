import pandas as pd

# -------------------------
# Config
# -------------------------
CSV_PATH = "datasets/surface_temp.csv"  # replace with your CSV path
OUTPUT_PATH = "datasets/mars_surface_avg_frost.csv"

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

# -------------------------
# Select CRISM CO2 Frost columns
# -------------------------
frost_cols = [col for col in df.columns if "CRISM CO2 Frost Prob." in col]

# -------------------------
# Compute average frost probability
# -------------------------
df["Avg_CRISM_CO2_Frost_Prob"] = df[frost_cols].mean(axis=1)

# -------------------------
# Drop individual frost prob columns
# -------------------------
df = df.drop(columns=frost_cols)

# -------------------------
# Save cleaned CSV
# -------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved cleaned CSV with Avg_CRISM_CO2_Frost_Prob to {OUTPUT_PATH}")
