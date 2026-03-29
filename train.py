"""
train.py — RandomForest + Bagging for maintenance time prediction
Dataset: port equipment maintenance interventions (donnees_tiers.xlsx)
Target: H. réalisées (repair hours)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_ESTIMATORS_RANGE = range(50, 251, 50)

# ── 1. Load & parse dates ────────────────────────────────────────────────────
df = pd.read_excel("donnees_tiers.xlsx")

df["Date D.I"]           = pd.to_datetime(df["Date D.I"],           format="%d/%m/%Y %I:%M %p", dayfirst=True)
df["Date debut travail"] = pd.to_datetime(df["Date debut travail"], format="%d/%m/%Y %I:%M %p", dayfirst=True)
df["Date fin de travail"]= pd.to_datetime(df["Date fin de travail"],format="%d/%m/%Y %I:%M %p", dayfirst=True)

df["Durée calculée (h)"] = (df["Date fin de travail"] - df["Date debut travail"]).dt.total_seconds() / 3600
df["Jour semaine"]       = df["Date D.I"].dt.day_name()
df["Mois"]               = df["Date D.I"].dt.month

# ── 2. Encode features ───────────────────────────────────────────────────────
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["Libellé Equipement", "Titre D.I"]])
X = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
y = df["H. réalisées"]

# ── 3. Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ── 4. Hyperparameter search: n_estimators ───────────────────────────────────
print("RandomForest — n_estimators sweep:")
best_n, best_mae = None, float("inf")
results = []

for n in N_ESTIMATORS_RANGE:
    model = RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    results.append((n, mae))
    print(f"  n_estimators={n} → MAE = {mae:.3f} h")
    if mae < best_mae:
        best_mae, best_n = mae, n

print(f"\nBest n_estimators={best_n}  MAE={best_mae:.3f} h")

# ── 5. OOB error curve ───────────────────────────────────────────────────────
oob_errors = []
oob_range  = range(10, 201, 10)
for n in oob_range:
    m = RandomForestRegressor(n_estimators=n, oob_score=True,
                              bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train, y_train)
    oob_errors.append(1 - m.oob_score_)

plt.figure(figsize=(10, 5))
plt.plot(oob_range, oob_errors, marker='o')
plt.title("OOB Error vs n_estimators")
plt.xlabel("n_estimators"); plt.ylabel("OOB Error (1 - R²)")
plt.grid(True); plt.tight_layout()
plt.savefig("oob_curve.png", dpi=130); plt.close()

# ── 6. Final RF model ────────────────────────────────────────────────────────
rf_model = RandomForestRegressor(n_estimators=best_n, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"\nFinal RF  MAE={mae:.2f}h  RMSE={rmse:.2f}h  R²={r2:.3f}")

# ── 7. Bagging comparison ────────────────────────────────────────────────────
print("\nBagging sweep:")
best_bag_n, best_bag_mae = None, float("inf")
bag_results = []

for n in N_ESTIMATORS_RANGE:
    bag = BaggingRegressor(RandomForestRegressor(random_state=RANDOM_STATE),
                           n_estimators=n, random_state=RANDOM_STATE)
    bag.fit(X_train, y_train)
    mae_b = mean_absolute_error(y_test, bag.predict(X_test))
    bag_results.append((n, mae_b))
    print(f"  n={n} → MAE = {mae_b:.3f} h")
    if mae_b < best_bag_mae:
        best_bag_mae, best_bag_n = mae_b, n

# ── 8. Save model & encoder ──────────────────────────────────────────────────
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(encoder,  "onehot_encoder.pkl")
print("\nSaved: random_forest_model.pkl  onehot_encoder.pkl")
