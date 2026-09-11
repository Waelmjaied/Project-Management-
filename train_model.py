"""
Project Cost & Duration Prediction - Model Training
----------------------------------------------------
Trains and evaluates two models for project estimation:
  - model_cost.pkl      predicts total project cost
  - model_duration.pkl  predicts total project duration (in days)

NOTE ON DATA: This version uses synthetic data with realistic non-linear
feature interactions (not a simple additive formula) to better validate
the pipeline. It is designed to be retrained on real historical project
data once available (see README for instructions).

The cost model takes 5 features:
  task_complexity, team_size, effective_hours, experience, hourly_rate

The duration model takes the original 4 (hourly_rate doesn't affect how
long something takes, only what it costs):
  task_complexity, team_size, effective_hours, experience

Design notes:
  - hourly_rate is an explicit input rather than baked into the model, so
    the predicted cost is anchored to a rate the user actually controls
    (a freelancer's $15/hr and an agency's $150/hr should produce very
    different totals for the same amount of work) instead of a hidden
    constant guessed at training time.
  - Non-linear feature interactions (complexity x team size, diminishing
    returns on team size, experience discount) instead of a purely
    additive formula, so each model has to learn real structure rather
    than trivially recover a linear equation.
  - Bigger teams reduce duration (parallelization) but increase cost
    (more salaries) — a more realistic trade-off than reusing the same
    formula for both targets.
  - 5-fold cross-validation instead of a single train/test split, for a
    statistically sound estimate of generalization performance.
  - A naive baseline (predict the mean) to contextualize whether each
    model is actually learning something useful.
  - Metrics saved to a JSON report file, not just printed to console.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

RANDOM_STATE = 42
N_SAMPLES = 300

COST_FEATURES = ["task_complexity", "team_size", "effective_hours", "experience", "hourly_rate"]
DURATION_FEATURES = ["task_complexity", "team_size", "effective_hours", "experience"]

# ---------------------------------------------------------------------
# 1. Generate synthetic data with realistic, non-linear structure
# ---------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)

df = pd.DataFrame({
    "task_complexity": rng.integers(1, 6, N_SAMPLES),        # 1-5
    "team_size": rng.integers(1, 11, N_SAMPLES),              # 1-10
    "effective_hours": rng.uniform(2.0, 60.0, N_SAMPLES),     # hours/week
    "experience": rng.integers(0, 11, N_SAMPLES),             # years, 0-10
    "hourly_rate": rng.uniform(10.0, 200.0, N_SAMPLES),       # currency/hour, blended team rate
})

# --- Cost target -------------------------------------------------------
# Cost is built from a "billable hour equivalents" quantity, then
# multiplied by the hourly rate — so cost scales linearly and
# transparently with rate, the way real billing works. Complexity and
# team-size coordination overhead add extra effective hours on top of
# the raw effective_hours input; team size has diminishing returns via
# sqrt (coordination overhead eats into the gain of adding people).
billable_hour_equivalents = (
        df["effective_hours"]
        + df["task_complexity"] * 4
        + np.sqrt(df["team_size"]) * df["task_complexity"] * 7
)
base_cost = billable_hour_equivalents * df["hourly_rate"]

cost_experience_discount = 1 - (df["experience"] * 0.015).clip(upper=0.3)
cost_noise = rng.normal(0, 0.05, N_SAMPLES) * base_cost  # proportional noise

df["cost"] = (base_cost * cost_experience_discount + cost_noise).clip(lower=50).round(2)

# --- Duration target (days) --------------------------------------------
# More complex work and more total effective_hours both push duration up.
# A bigger team speeds things up (parallelization), but with diminishing
# returns (sqrt) since coordination overhead eats into the gains.
# Experienced teams work faster (discount), similar to the cost model but
# with a different magnitude, so the two targets aren't just rescalings
# of each other. Duration does not depend on hourly_rate.
base_duration = (
                        df["task_complexity"] * 1.8
                        + df["effective_hours"] * 0.9
                ) / np.sqrt(df["team_size"])
duration_experience_discount = 1 - (df["experience"] * 0.02).clip(upper=0.35)
duration_noise = rng.normal(0, 1.2, N_SAMPLES) + base_duration * rng.normal(0, 0.07, N_SAMPLES)

df["duration"] = (base_duration * duration_experience_discount + duration_noise).clip(lower=1).round(1)

targets = {
    "cost": (df[COST_FEATURES], df["cost"]),
    "duration": (df[DURATION_FEATURES], df["duration"]),
}

# ---------------------------------------------------------------------
# 2 & 3. Train + evaluate a model for each target
# ---------------------------------------------------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
report = {}

for target_name, (X, y) in targets.items():
    print("=" * 60)
    print(f"TARGET: {target_name}  (features: {list(X.columns)})")
    print("=" * 60)

    models = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
    }

    target_report = {}

    for name, model in models.items():
        r2_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
        mae_scores = -cross_val_score(
            model, X, y, cv=kfold, scoring="neg_mean_absolute_error"
        )

        target_report[name] = {
            "r2_mean": round(float(r2_scores.mean()), 4),
            "r2_std": round(float(r2_scores.std()), 4),
            "mae_mean": round(float(mae_scores.mean()), 2),
            "mae_std": round(float(mae_scores.std()), 2),
        }

        print(f"\n{name}:")
        print(f"  R^2  : {r2_scores.mean():.3f} (+/- {r2_scores.std():.3f})")
        print(f"  MAE  : {mae_scores.mean():.2f} (+/- {mae_scores.std():.2f})")

    # Held-out test set evaluation (for the final model to be deployed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    final_model = RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE
    )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    target_report["final_model_holdout_test"] = {
        "r2": round(float(test_r2), 4),
        "mae": round(float(test_mae), 2),
        "improvement_over_baseline_r2": round(
            float(test_r2 - target_report["baseline_mean"]["r2_mean"]), 4
        ),
    }

    print(f"\nFINAL MODEL ({target_name}, Random Forest) - HELD-OUT TEST SET")
    print(f"R^2 : {test_r2:.3f}")
    print(f"MAE : {test_mae:.2f}")

    # Feature importance
    importances = pd.Series(
        final_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    target_report["feature_importance"] = {
        k: round(float(v), 4) for k, v in importances.items()
    }

    print(f"\nFeature importance ({target_name}):")
    for feat, imp in importances.items():
        print(f"  {feat:<20} {imp:.3f}")

    # Save model
    model_path = f"model_{target_name}.pkl"
    joblib.dump(final_model, model_path)
    print(f"\nSaved model to {model_path}\n")

    report[target_name] = target_report

with open("training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Saved metrics report to training_report.json")