import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler


# ==============================
# CONFIG
# ==============================
CSV_PATH = "backend/datasets/core_set_2ppd.csv"
TARGET = 'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)'

FEATURES = [
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

LAT_COL = "Latitude (N)"


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(CSV_PATH)

cols_needed = FEATURES + [TARGET]
df = df[cols_needed]

for c in cols_needed:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna()

print("Rows after cleaning:", len(df))


# ==============================
# BASIC TARGET DIAGNOSTICS
# ==============================
y = df[TARGET].values

print("\n=== TARGET STATS ===")
print(pd.Series(y).describe())

plt.figure()
plt.hist(y, bins=100)
plt.title("Target Distribution")
plt.xlabel(TARGET)
plt.ylabel("Count")
plt.show()


# ==============================
# SPATIAL VARIANCE CHECK
# ==============================
df["lat_bin"] = pd.cut(df[LAT_COL], bins=10)

lat_var = df.groupby("lat_bin")[TARGET].var()
print("\n=== Target Variance by Latitude Bin ===")
print(lat_var)

plt.figure()
lat_var.plot(kind="bar")
plt.title("Target Variance by Latitude Bin")
plt.ylabel("Variance")
plt.show()


# ==============================
# TRAIN / TEST SPLIT
# (simple random for diagnostics)
# ==============================
X = df[FEATURES].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# BASELINE CHECKS
# ==============================
print("\n=== BASELINES ===")

# Mean predictor
mean_pred = np.full_like(y_test, y_train.mean())
print("Mean baseline R²:", r2_score(y_test, mean_pred))

# Linear regression
lin = LinearRegression()
lin.fit(X_train, y_train)
lin_pred = lin.predict(X_test)

print("Linear Regression R²:", r2_score(y_test, lin_pred))


# ==============================
# FEATURE SIGNAL ANALYSIS
# ==============================
print("\n=== FEATURE CORRELATIONS ===")
corrs = df[FEATURES + [TARGET]].corr()[TARGET].sort_values()
print(corrs)

print("\n=== MUTUAL INFORMATION ===")
mi = mutual_info_regression(X_train, y_train)
for f, score in zip(FEATURES, mi):
    print(f"{f}: {score:.4f}")


# ==============================
# RESIDUAL ANALYSIS
# ==============================
residuals = y_test - lin_pred

plt.figure()
plt.scatter(lin_pred, residuals, alpha=0.3)
plt.axhline(0, color="red")
plt.title("Residuals vs Prediction")
plt.xlabel("Prediction")
plt.ylabel("Residual")
plt.show()


# ==============================
# FRACTIONAL MSE
# ==============================
mse = mean_squared_error(y_test, lin_pred)
frac_mse = mse / np.var(y_test)

print("\n=== ERROR METRICS ===")
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("Fractional MSE:", frac_mse)
print("Equivalent R²:", 1 - frac_mse)


# ==============================
# FEATURE SCALE CHECK
# ==============================
print("\n=== FEATURE STD DEV ===")
print(df[FEATURES].std().sort_values())