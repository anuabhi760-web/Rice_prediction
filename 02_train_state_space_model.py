"""
DEPLOYMENT STEP 2 — STATE SPACE MODEL (MANDATORY)
----------------------------------------------------------------------
Run: pip install statsmodels

This is the project's core novel contribution (Research Gap #4): a
per-district Kalman filter that explicitly separates the LEVEL/TREND
component (slow-moving — irrigation, seed variety, MSP adoption) from
the CLIMATE SHOCK component (year-to-year rainfall/temperature
deviations), and produces calibrated 95% confidence intervals — which
neither Random Forest nor XGBoost give you natively.

Fits one local-level model per district (with rainfall/temperature/SPI
as exogenous covariates), forecasts the test period, and saves
per-district-year: fitted trend, forecast, and 95% CI bounds — this is
what the dashboard's uncertainty band comes from.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

df = pd.read_csv("rice_dataset_final.csv")  # full table (lag NaNs OK, not needed here)
TARGET = "RICE YIELD (Kg per ha)"
COVARIATES = ["RAIN_ANNUAL_mm", "TEMP_ANNUAL_C", "SPI_Kharif", "HEAT_STRESS"]

all_results = []
model_diagnostics = []

for dist_name, g in df.groupby("Dist Name"):
    g = g.sort_values("Year").reset_index(drop=True)
    if len(g) < 20:  # need enough years for a stable Kalman fit + meaningful train/test split
        model_diagnostics.append({"district": dist_name, "status": "skipped (too few years)"})
        continue

    y_full = g[TARGET].values
    X_full = g[COVARIATES].fillna(g[COVARIATES].median()).values
    years = g["Year"].values

    train_idx = years <= 2005
    test_idx = years >= 2006
    if test_idx.sum() < 3:
        model_diagnostics.append({"district": dist_name, "status": "skipped (too few test years)"})
        continue

    try:
        # Fit on train only, then forecast forward through the test period —
        # this is the honest, out-of-sample way to produce the CI band
        # (fitting on the full series and reading off in-sample CIs would
        # be optimistic, same leakage issue flagged earlier in the project).
        model = UnobservedComponents(
            endog=y_full[train_idx],
            level="local level",   # stochastic trend = the "technology" component
            exog=X_full[train_idx],
        )
        fit = model.fit(disp=False)

        # forecast the test period using the test period's own covariates
        n_test = test_idx.sum()
        forecast = fit.get_forecast(steps=n_test, exog=X_full[test_idx])
        pred_mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.05)  # 95% CI

        district_result = pd.DataFrame({
            "Dist Name": dist_name,
            "Year": years[test_idx],
            "actual": y_full[test_idx],
            "ssm_forecast": pred_mean,
            "ssm_ci_lower": ci[:, 0],
            "ssm_ci_upper": ci[:, 1],
        })
        all_results.append(district_result)

        rmse = np.sqrt(np.mean((y_full[test_idx] - pred_mean) ** 2))
        coverage = np.mean((y_full[test_idx] >= ci[:, 0]) & (y_full[test_idx] <= ci[:, 1]))
        model_diagnostics.append({
            "district": dist_name, "status": "ok", "n_train_years": train_idx.sum(),
            "n_test_years": n_test, "test_rmse": rmse,
            "ci_coverage": coverage,  # should be close to 0.95 if CIs are well-calibrated
        })
    except Exception as e:
        model_diagnostics.append({"district": dist_name, "status": f"failed: {e}"})

# ---- Save results ----
if all_results:
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv("../deployment/ssm_forecasts.csv", index=False)
else:
    results_df = pd.DataFrame()

diag_df = pd.DataFrame(model_diagnostics)
diag_df.to_csv("../deployment/ssm_diagnostics.csv", index=False)

ok_diag = diag_df[diag_df["status"] == "ok"]
print(f"Fit SSM successfully for {len(ok_diag)} / {len(diag_df)} districts")
if len(ok_diag) > 0:
    from sklearn.metrics import r2_score
    overall_r2 = r2_score(results_df["actual"], results_df["ssm_forecast"])
    print(f"\nOverall SSM Test R² (pooled across all districts): {overall_r2:.4f}")
    print(f"Mean per-district test RMSE: {ok_diag['test_rmse'].mean():.2f} kg/ha")
    print(f"Mean 95% CI coverage: {ok_diag['ci_coverage'].mean():.3f} "
          f"(should be close to 0.95 — {'well-calibrated' if 0.85 <= ok_diag['ci_coverage'].mean() <= 1.0 else 'check model — over/under-confident intervals'})")

print("\nSaved: ../deployment/ssm_forecasts.csv, ../deployment/ssm_diagnostics.csv")
print("""
NOTE FOR YOUR MENTOR: SSM's headline contribution isn't necessarily a
higher R² than Random Forest/XGBoost — it's CI coverage close to 0.95,
meaning the uncertainty band is trustworthy. A tree model gives you a
single number with no sense of confidence; SSM tells you both the
forecast AND how sure to be about it, per district. That's what the
dashboard's uncertainty band visualizes, and it directly answers RQ3.
""")
