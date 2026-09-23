"""
DEPLOYMENT STEP 2 — STATE SPACE MODEL, ROLLING ONE-STEP-AHEAD
----------------------------------------------------------------------
Run: pip install statsmodels

WHY THIS VERSION EXISTS:
The earlier version fit each district's model on data through 2005, then
forecast the ENTIRE 2006-2025 period (up to 20 years) in one blind leap.
That is an unusually hard test for any model — real deployment would
never actually need a 20-year blind forecast, because new yield data
arrives every year. It also produced a real, diagnosable failure: with
"local linear trend", the stochastic slope over-extrapolated across two
decades, making point forecasts WORSE than the simpler "local level"
despite being the better structural match for the data's known upward
trend (R²=-0.29 vs 0.22, tested and confirmed).

This version fixes the evaluation to match how a Kalman filter is
actually meant to be used and assessed: ROLLING ONE-STEP-AHEAD forecasts.
For each district:
  1. Fit once on data through 2005 (estimates the model's variance
     parameters via MLE — this is the expensive step, done ONCE).
  2. For each test year in order: forecast just that one year ahead,
     record it, then reveal the true value and extend the filtered
     state with .append(refit=False) — this updates the Kalman filter's
     internal state using the already-estimated parameters, without
     re-running the optimizer. Cheap and exactly what the Kalman filter
     recursion is designed for.
  3. Move to the next year, now one step closer, and repeat.

This should substantially improve both R² and CI coverage, because the
model is never asked to extrapolate more than one year past its most
recent true observation.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

df = pd.read_csv("rice_dataset_final.csv")
TARGET = "RICE YIELD (Kg per ha)"
COVARIATES = ["RAIN_ANNUAL_mm", "TEMP_ANNUAL_C", "SPI_Kharif", "HEAT_STRESS"]

all_results = []
model_diagnostics = []

districts = df["Dist Name"].unique()
print(f"Processing {len(districts)} districts (rolling one-step-ahead)...")

for dist_i, (dist_name, g) in enumerate(df.groupby("Dist Name")):
    g = g.sort_values("Year").reset_index(drop=True)
    if len(g) < 20:
        model_diagnostics.append({"district": dist_name, "status": "skipped (too few years)"})
        continue

    y_full = g[TARGET].values
    X_full = g[COVARIATES].fillna(g[COVARIATES].median()).values
    years = g["Year"].values

    train_idx = years <= 2005
    test_idx = years >= 2006
    n_train = int(train_idx.sum())
    n_test = int(test_idx.sum())
    if n_test < 3:
        model_diagnostics.append({"district": dist_name, "status": "skipped (too few test years)"})
        continue

    try:
        # Step 1: fit ONCE on the training window — this is the only
        # expensive MLE optimization for this district.
        model = UnobservedComponents(
            endog=y_full[train_idx],
            level="local level",
            exog=X_full[train_idx],
        )
        current = model.fit(disp=False)

        test_years_arr = years[test_idx]
        test_y_arr = y_full[test_idx]
        test_X_arr = X_full[test_idx]

        preds, ci_lowers, ci_uppers = [], [], []

        # Step 2-3: roll forward one year at a time
        for i in range(n_test):
            start = current.nobs  # position right after all data seen so far
            pred = current.get_prediction(start=start, end=start, exog=test_X_arr[i:i + 1])
            pred_mean_i = np.asarray(pred.predicted_mean)[0]
            ci_i = np.asarray(pred.conf_int(alpha=0.05))

            preds.append(pred_mean_i)
            ci_lowers.append(ci_i[0, 0])
            ci_uppers.append(ci_i[0, 1])

            # reveal the true value for this year, extend the filtered
            # state without re-optimizing (refit=False) — cheap update
            current = current.append(endog=test_y_arr[i:i + 1],
                                      exog=test_X_arr[i:i + 1], refit=False)

        preds = np.array(preds)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)

        district_result = pd.DataFrame({
            "Dist Name": dist_name,
            "Year": test_years_arr,
            "actual": test_y_arr,
            "ssm_forecast": preds,
            "ssm_ci_lower": ci_lowers,
            "ssm_ci_upper": ci_uppers,
        })
        all_results.append(district_result)

        rmse = np.sqrt(np.mean((test_y_arr - preds) ** 2))
        coverage = np.mean((test_y_arr >= ci_lowers) & (test_y_arr <= ci_uppers))
        model_diagnostics.append({
            "district": dist_name, "status": "ok", "n_train_years": n_train,
            "n_test_years": n_test, "test_rmse": rmse, "ci_coverage": coverage,
        })

    except Exception as e:
        model_diagnostics.append({"district": dist_name, "status": f"failed: {e}"})
        n_failed_so_far = sum(1 for d in model_diagnostics if str(d.get("status", "")).startswith("failed"))
        if n_failed_so_far <= 3:
            print(f"  FAILED [{dist_name}]: {e}")

    if (dist_i + 1) % 50 == 0:
        print(f"  ...{dist_i + 1}/{len(districts)} districts processed")

# ---- Save results ----
if all_results:
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv("ssm_forecasts.csv", index=False)
else:
    results_df = pd.DataFrame()

diag_df = pd.DataFrame(model_diagnostics)
diag_df.to_csv("ssm_diagnostics.csv", index=False)

ok_diag = diag_df[diag_df["status"] == "ok"]
print(f"\nFit SSM successfully for {len(ok_diag)} / {len(diag_df)} districts")
if len(ok_diag) > 0:
    from sklearn.metrics import r2_score
    overall_r2 = r2_score(results_df["actual"], results_df["ssm_forecast"])
    print(f"\nOverall SSM Test R² (pooled, rolling one-step-ahead): {overall_r2:.4f}")
    print(f"Mean per-district test RMSE: {ok_diag['test_rmse'].mean():.2f} kg/ha")
    print(f"Mean 95% CI coverage: {ok_diag['ci_coverage'].mean():.3f} "
          f"(should be close to 0.95 — {'well-calibrated' if 0.85 <= ok_diag['ci_coverage'].mean() <= 1.0 else 'still check calibration'})")

print("\nSaved: ssm_forecasts.csv, ssm_diagnostics.csv")
print("""
NOTE FOR YOUR MENTOR: this version evaluates the SSM with rolling
one-step-ahead forecasts rather than a single 20-year blind forecast —
the standard way Kalman filter models are assessed, and the way this
model would actually be used in deployment (predict next year, then
update once that year's true yield is known). Compare this R²/coverage
against the earlier 20-year single-shot version (R²=0.22, coverage=0.70
for local level; R²=-0.29, coverage=0.76 for local linear trend) — the
gap between them is itself a finding worth discussing: it shows how much
harder long-horizon forecasting is than the short-horizon, continuously-
updated forecasting this model is actually suited for.
""")
