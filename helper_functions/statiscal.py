import pandas as pd
import numpy as np
import os

# Get the path relative to this script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "backend", "models", "datasets", "8ppd_extrapolated_targets.parquet")

df = pd.read_parquet(DATA_PATH)

FEATURES = [
    "Water",
    "Dust",
    "Surface_Temp",
    "Temp_Variation",
    "Gravity",
    "TI_Variance"
]

UNC = [f + "_Uncertainty" for f in FEATURES]

ALL = FEATURES + UNC


print("\n===== A) RAW SCALE CHECK =====")
print(df[ALL].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))


print("\n===== B) RANGE MAGNITUDE =====")
for c in ALL:
    print(f"{c:28s} range = {df[c].max() - df[c].min():.3f}")


print("\n===== C) SIGNAL VS NOISE RATIO =====")
for f,u in zip(FEATURES, UNC):
    signal = df[f].std()
    noise  = df[u].mean()
    snr = signal / noise if noise != 0 else np.inf
    print(f"{f:18s} SNR = {snr:6.2f}   (std(signal) / mean(unc))")


print("\n===== D) OUTLIER DOMINATION (1–99%) =====")
for c in ALL:
    p1,p99 = df[c].quantile([.01,.99])
    out = ((df[c]<p1)|(df[c]>p99)).mean()*100
    print(f"{c:28s} {out:5.2f}% outside 1–99")


print("\n===== E) CORRELATION TO WATER =====")
print(df[FEATURES].corr()["Water"].sort_values())


# ----- Simulate YOUR scoring logic -----

print("\n===== F) SIMULATED GOODNESS CONTRIBUTION =====")

norm = {}
for c in FEATURES + UNC:
    mn,mx = df[c].min(), df[c].max()
    norm[c] = (df[c]-mn)/(mx-mn) if mx>mn else df[c]*0

penalty = 0.5

for f in FEATURES:
    raw = norm[f].mean()
    pen = norm[f+"_Uncertainty"].mean() * penalty
    print(f"{f:18s} +{raw:5.3f}  -{pen:5.3f}  => {raw-pen:5.3f}")