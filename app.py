"""
STREAMLIT DASHBOARD — Rice Yield Prediction
----------------------------------------------------------------------
Run: pip install streamlit plotly
     streamlit run app.py

Prerequisite: run 01_train_final_models.py and 02_train_state_space_model.py
first (this app loads their saved outputs — it does NOT retrain anything,
so it launches instantly).

Layout:
  - Model comparison metrics (Objective 3 / RQ2)
  - District-level history chart: actual yield, RF/XGBoost predictions,
    and SSM forecast with 95% confidence band (Objective 5 / RQ3)
  - Global feature importance via SHAP (Objective 4 / RQ4)
  - "What-if" interactive prediction panel — lets a user adjust
    rainfall/temperature/soil inputs and see XGBoost's predicted yield
    update live. This is the closest thing to your proposal's planned
    AI chatbot (Objective 6) without the LLM integration itself —
    frame it in your presentation as "the underlying prediction engine
    the chatbot would call."
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Rice Yield Prediction Dashboard", layout="wide")

# ------------------------------------------------------------------
# Load everything once (cached — the app should feel instant)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    full_history = pd.read_csv("rice_dataset_final.csv")
    test_predictions = pd.read_csv("test_predictions.csv")
    model_summary = pd.read_csv("model_summary.csv")
    shap_importance = pd.read_csv("shap_global_importance.csv", index_col=0)
    ssm_forecasts = pd.read_csv("ssm_forecasts.csv")
    ssm_diagnostics = pd.read_csv("ssm_diagnostics.csv")
    return full_history, test_predictions, model_summary, shap_importance, ssm_forecasts, ssm_diagnostics


@st.cache_resource
def load_models():
    rf = joblib.load("random_forest.joblib")
    xgb_model = joblib.load("xgboost.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return rf, xgb_model, feature_cols


try:
    full_history, test_predictions, model_summary, shap_importance, ssm_forecasts, ssm_diagnostics = load_data()
    rf_model, xgb_model, feature_cols = load_models()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Run 01_train_final_models.py and "
             "02_train_state_space_model.py in this folder before launching the dashboard.")
    st.stop()

TARGET = "RICE YIELD (Kg per ha)"

# ------------------------------------------------------------------
# Header + model comparison
# ------------------------------------------------------------------
st.title("🌾 Rice Yield Prediction — District-Level Dashboard")
st.caption("Crop Yield Prediction Based on Climate, Soil and Environmental Factors — "
           "JAIN Deemed-to-be University")

st.subheader("Model Comparison (temporal split: train 1966-2005, test 2006-2025)")
cols = st.columns(len(model_summary))
for col, (_, row) in zip(cols, model_summary.iterrows()):
    col.metric(row["model"], f"R² = {row['test_r2']:.3f}", f"RMSE {row['test_rmse']:.0f} kg/ha")

with st.expander("Why is a Random Forest R² of ~0.84 reported alongside a lower XGBoost R²?"):
    st.write("""
    Both are benchmarked against a naive persistence baseline (predict = last
    year's yield), which alone scores R² ≈ 0.90 on this data because rice
    yield is strongly autocorrelated. Neither tree model currently beats
    that trivial baseline — this is reported honestly rather than hidden,
    and is discussed further in the project README.
    """)

st.divider()

# ------------------------------------------------------------------
# District selector
# ------------------------------------------------------------------
st.subheader("District-Level Forecast Explorer")
states = sorted(full_history["State Name"].unique())
col1, col2 = st.columns(2)
selected_state = col1.selectbox("State", states)
districts_in_state = sorted(full_history.loc[full_history["State Name"] == selected_state, "Dist Name"].unique())
selected_district = col2.selectbox("District", districts_in_state)

dist_history = full_history[full_history["Dist Name"] == selected_district].sort_values("Year")
dist_test_pred = test_predictions[test_predictions["Dist Name"] == selected_district].sort_values("Year")
dist_ssm = ssm_forecasts[ssm_forecasts["Dist Name"] == selected_district].sort_values("Year")

# ------------------------------------------------------------------
# History + forecast chart
# ------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(x=dist_history["Year"], y=dist_history[TARGET],
                          mode="lines+markers", name="Actual Yield",
                          line=dict(color="black", width=2)))

if not dist_test_pred.empty:
    fig.add_trace(go.Scatter(x=dist_test_pred["Year"], y=dist_test_pred["rf_pred"],
                              mode="lines", name="Random Forest Prediction",
                              line=dict(color="#2ca25f", dash="dot")))
    fig.add_trace(go.Scatter(x=dist_test_pred["Year"], y=dist_test_pred["xgb_pred"],
                              mode="lines", name="XGBoost Prediction",
                              line=dict(color="#e34a33", dash="dot")))

if not dist_ssm.empty:
    fig.add_trace(go.Scatter(x=dist_ssm["Year"], y=dist_ssm["ssm_forecast"],
                              mode="lines", name="SSM Forecast (trend + climate)",
                              line=dict(color="#3182bd", width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([dist_ssm["Year"], dist_ssm["Year"][::-1]]),
        y=pd.concat([dist_ssm["ssm_ci_upper"], dist_ssm["ssm_ci_lower"][::-1]]),
        fill="toself", fillcolor="rgba(49,130,189,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="SSM 95% Confidence Interval", showlegend=True,
    ))
else:
    st.info(f"No SSM forecast available for {selected_district} "
            "(likely too few years of history — see ssm_diagnostics.csv)")

fig.add_vline(x=2005.5, line_dash="dash", line_color="gray",
              annotation_text="Train / Test split")
fig.update_layout(title=f"{selected_district}, {selected_state} — Yield History & Forecasts",
                   xaxis_title="Year", yaxis_title="Yield (kg/ha)", height=500,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# SSM calibration note for this district
if not dist_ssm.empty:
    diag_row = ssm_diagnostics[ssm_diagnostics["district"] == selected_district]
    if not diag_row.empty and diag_row.iloc[0]["status"] == "ok":
        coverage = diag_row.iloc[0]["ci_coverage"]
        st.caption(f"SSM 95% CI coverage for this district: {coverage:.0%} "
                    "(fraction of actual test-year values falling inside the shaded band — "
                    "close to 95% means the uncertainty estimate is trustworthy)")

st.divider()

# ------------------------------------------------------------------
# Feature importance (SHAP)
# ------------------------------------------------------------------
st.subheader("What Drives Yield Predictions? (SHAP Feature Importance — XGBoost)")
top_n = st.slider("Number of top features to show", 5, 20, 10)
top_features = shap_importance.head(top_n).sort_values("mean_abs_shap")
fig2 = go.Figure(go.Bar(x=top_features["mean_abs_shap"], y=top_features.index,
                         orientation="h", marker_color="#756bb1"))
fig2.update_layout(title="Mean |SHAP value| — higher means bigger influence on predicted yield",
                    xaxis_title="Mean |SHAP value|", height=400 + top_n * 15)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# What-if prediction panel
# ------------------------------------------------------------------
st.subheader("What-If Yield Predictor")
st.caption("Adjust climate/soil inputs below to see XGBoost's predicted yield update live — "
           "this is the prediction engine your proposal's future AI chatbot would call.")

# start from this district's most recent row as the baseline, let user override key drivers
baseline_full_row = dist_history.iloc[-1]
baseline_row = baseline_full_row[feature_cols].copy()
baseline_actual_yield = baseline_full_row[TARGET]
baseline_year = int(baseline_full_row["Year"])

wcol1, wcol2, wcol3 = st.columns(3)
rain = wcol1.slider("Annual Rainfall (mm)", 200, 3500, int(baseline_row.get("RAIN_ANNUAL_mm", 1000)))
temp = wcol2.slider("Annual Temperature (°C)", 22.0, 34.0, float(baseline_row.get("TEMP_ANNUAL_C", 28.0)), 0.1)
soil_ph = wcol3.slider("Soil pH", 4.5, 9.0, float(baseline_row.get("SOIL_PHH2O", 6.5)), 0.1)

input_row = baseline_row.copy()
input_row["RAIN_ANNUAL_mm"] = rain
input_row["TEMP_ANNUAL_C"] = temp
input_row["SOIL_PHH2O"] = soil_ph
input_df = pd.DataFrame([input_row])[feature_cols]

predicted_yield = xgb_model.predict(input_df)[0]
st.metric("Predicted Rice Yield", f"{predicted_yield:.0f} kg/ha",
          f"{predicted_yield - baseline_actual_yield:.0f} vs. {baseline_year} actual ({baseline_actual_yield:.0f} kg/ha)")

st.caption("⚠️ Data quality note: soil features for districts outside the original 67 with "
           "confirmed ISRIC coverage currently use an interim state-median imputation. "
           "See README.md for details and the re-fetch script (03_fetch_missing_soil.py).")
