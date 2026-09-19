"""
DEPLOYMENT STEP 1 — TRAIN FINAL MODELS (Random Forest + XGBoost)
----------------------------------------------------------------------
Run once before launching the dashboard: pip install xgboost shap joblib

Trains the two point-prediction models that made the final cut, and
saves everything app.py needs to load instantly (no retraining at
dashboard launch time):
  - random_forest.joblib, xgboost.joblib      (trained models)
  - test_predictions.csv                       (actual vs. both models' predictions, per district-year)
  - shap_global_importance.csv                 (for the dashboard's feature importance chart)
  - shap_values_test.pkl                       (per-row SHAP values, for per-district explanations)
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import shap

df = pd.read_csv("rice_dataset_final.csv")
TARGET = "RICE YIELD (Kg per ha)"

# Same collinearity drops used throughout this project (see 05_eda_plots_baseline.py)
COLLINEAR_DROPS = [
    "RAIN_JUN_SEP_mm", "ETo_Kharif_seasonal", "GDD_Kharif",
    "TEMP_JUN_SEP_C", "TEMP_OPTIMALITY", "N_TEMP_INTERACTION",
]
id_cols = ["Dist Code", "Dist Name", "State Name", "Year"]
drop_cols = id_cols + [TARGET, "RICE AREA (1000 ha)"] + COLLINEAR_DROPS
feature_cols = [c for c in df.columns if c not in drop_cols]

train_mask = df["Year"] <= 2005
test_mask = df["Year"] >= 2006
X_train, X_test = df.loc[train_mask, feature_cols], df.loc[test_mask, feature_cols]
y_train, y_test = df.loc[train_mask, TARGET], df.loc[test_mask, TARGET]

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows | Features: {len(feature_cols)}")

# ---- Naive persistence, for the dashboard's comparison panel ----
naive_pred = df.loc[test_mask, "YIELD_LAG1"].values
naive_r2 = r2_score(y_test, naive_pred)

# ---- Random Forest (deployed baseline) ----
print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"Random Forest Test R²: {rf_r2:.4f} | RMSE: {rf_rmse:.2f}")

# ---- XGBoost (for SHAP explainability) ----
print("\nTraining XGBoost (with hyperparameter search)...")
param_grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]}
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror"),
                     param_grid, cv=tscv, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
xgb_model = grid.best_estimator_
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"XGBoost best params: {grid.best_params_}")
print(f"XGBoost Test R²: {xgb_r2:.4f} | RMSE: {xgb_rmse:.2f}")

# ---- Save models ----
joblib.dump(rf, "../deployment/random_forest.joblib")
joblib.dump(xgb_model, "../deployment/xgboost.joblib")
joblib.dump(feature_cols, "../deployment/feature_cols.joblib")

# ---- Save test predictions with district/year identifiers, for the dashboard's history chart ----
test_predictions = df.loc[test_mask, id_cols + [TARGET]].copy()
test_predictions["naive_pred"] = naive_pred
test_predictions["rf_pred"] = rf_pred
test_predictions["xgb_pred"] = xgb_pred
test_predictions.to_csv("../deployment/test_predictions.csv", index=False)

# ---- SHAP (global + per-row, for the dashboard) ----
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

global_importance = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)
global_importance.to_csv("../deployment/shap_global_importance.csv", header=["mean_abs_shap"])

with open("../deployment/shap_values_test.pkl", "wb") as f:
    pickle.dump({"shap_values": shap_values, "X_test": X_test.reset_index(drop=True),
                 "ids": df.loc[test_mask, id_cols].reset_index(drop=True)}, f)

# ---- Summary table for the dashboard ----
summary = pd.DataFrame([
    {"model": "Naive persistence", "test_r2": naive_r2, "test_rmse": np.sqrt(mean_squared_error(y_test, naive_pred))},
    {"model": "Random Forest", "test_r2": rf_r2, "test_rmse": rf_rmse},
    {"model": "XGBoost", "test_r2": xgb_r2, "test_rmse": xgb_rmse},
])
summary.to_csv("../deployment/model_summary.csv", index=False)
print("\n" + summary.to_string(index=False))
print("""
Saved to ../deployment/:
  random_forest.joblib, xgboost.joblib, feature_cols.joblib
  test_predictions.csv, model_summary.csv
  shap_global_importance.csv, shap_values_test.pkl

Next: run 02_train_state_space_model.py (mandatory), then launch app.py
""")
