"""
RetailPulse AI - Model Training & Evaluation Pipeline
======================================================
Academic Level: 2nd Year 2nd Semester Undergraduate Data Science Project
Author: RetailPulse AI Team

Description:
    This script implements an end-to-end Machine Learning pipeline for multi-store
    weekly sales forecasting. It enforces time-series best practices:
      1. Chronological holdout test split (no future data leakage).
      2. Shifted rolling statistics (t-1 to t-4) to prevent lookahead bias.
      3. Store-level historical volume feature to properly represent categorical store scale.
      4. Time-series expanding-window cross-validation.
      5. Dual evaluation: 1-step-ahead metrics vs. Multi-step recursive simulation.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Walmart_Store_sales.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Explicit feature list
FEATURES = [
    "Store",
    "Store_Avg_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "WeekOfYear",
    "Quarter",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_52",
    "Rolling_Mean_4",
    "Rolling_Std_4",
]


def load_raw_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load raw Walmart store sales dataset, clean column names,
    and parse dates in chronological order.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    # Normalize column names
    df.columns = [c.strip().replace("\\_", "_") for c in df.columns]

    # Parse date (Day-Month-Year format in original Walmart dataset)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Drop missing dates or target values and remove any duplicate rows
    df = df.dropna(subset=["Date", "Weekly_Sales"]).drop_duplicates()

    # Sort strictly by Store and Date for time-series integrity
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, train_cutoff: pd.Timestamp = None) -> pd.DataFrame:
    """
    Generate time-series lag features, calendar features, and rolling statistics.
    Ensures zero lookahead bias by shifting rolling calculations.
    """
    data = df.copy()

    # 1. Calendar / Temporal Features
    iso_cal = data["Date"].dt.isocalendar()
    data["Year"] = data["Date"].dt.year.astype(int)
    data["Month"] = data["Date"].dt.month.astype(int)
    data["WeekOfYear"] = iso_cal.week.astype(int)
    data["Quarter"] = data["Date"].dt.quarter.astype(int)

    # 2. Store-level historical baseline (to properly capture store scale without arbitrary ordinality)
    # Calculated on training partition if train_cutoff is provided to prevent leakage
    if train_cutoff is not None:
        train_mask = data["Date"] < train_cutoff
        store_means = data[train_mask].groupby("Store")["Weekly_Sales"].mean().to_dict()
    else:
        store_means = data.groupby("Store")["Weekly_Sales"].mean().to_dict()
    data["Store_Avg_Sales"] = data["Store"].map(store_means)

    # 3. Lags (1 week, 2 weeks, 4 weeks, and 52 weeks / 1 year ago)
    store_sales_group = data.groupby("Store")["Weekly_Sales"]
    for lag in [1, 2, 4, 52]:
        data[f"Lag_{lag}"] = store_sales_group.shift(lag)

    # 4. Shifted Rolling Statistics (4-week window ending at t-1)
    # Crucial: shift(1) guarantees the current week's sales is NOT inside the rolling window
    shifted_sales = store_sales_group.shift(1)
    data["Rolling_Mean_4"] = shifted_sales.groupby(data["Store"]).transform(
        lambda s: s.rolling(window=4, min_periods=4).mean()
    )
    data["Rolling_Std_4"] = shifted_sales.groupby(data["Store"]).transform(
        lambda s: s.rolling(window=4, min_periods=4).std()
    )

    # Drop early rows where 52-week lag or 4-week rolling window are not yet available
    clean_data = data.dropna(subset=FEATURES).reset_index(drop=True)
    return clean_data


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate standard regression evaluation metrics: MAE, RMSE, MAPE, and R2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Guard against division by zero in MAPE
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2),
        "R2": round(float(r2), 4),
    }


def get_candidate_models() -> dict:
    """
    Initialize machine learning models with robust hyperparameter configurations.
    """
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.25,
            random_state=42,
        ),
    }


def simulate_recursive_forecast(
    model,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int = 12,
) -> dict:
    """
    Evaluate multi-step recursive forecasting on the holdout test period.
    Instead of using true lags, predicted sales are fed back into subsequent lag features.
    """
    all_stores = sorted(test_df["Store"].unique())
    rec_errors = []

    for store_id in all_stores:
        s_hist = history_df[history_df["Store"] == store_id].sort_values("Date")
        s_test = test_df[test_df["Store"] == store_id].sort_values("Date")

        if len(s_test) < horizon:
            continue

        sales_history = list(s_hist["Weekly_Sales"])
        store_avg = float(s_hist["Store_Avg_Sales"].iloc[-1])

        # Test the first 'horizon' weeks
        test_subset = s_test.iloc[:horizon]
        store_preds = []

        for _, row in test_subset.iterrows():
            dt = row["Date"]
            iso = dt.isocalendar()

            vals = {
                "Store": store_id,
                "Store_Avg_Sales": store_avg,
                "Holiday_Flag": row["Holiday_Flag"],
                "Temperature": row["Temperature"],
                "Fuel_Price": row["Fuel_Price"],
                "CPI": row["CPI"],
                "Unemployment": row["Unemployment"],
                "Year": dt.year,
                "Month": dt.month,
                "WeekOfYear": int(iso.week),
                "Quarter": dt.quarter,
                "Lag_1": sales_history[-1],
                "Lag_2": sales_history[-2],
                "Lag_4": sales_history[-4],
                "Lag_52": sales_history[-52] if len(sales_history) >= 52 else sales_history[-1],
                "Rolling_Mean_4": float(np.mean(sales_history[-4:])),
                "Rolling_Std_4": float(np.std(sales_history[-4:], ddof=1)) if len(sales_history) >= 4 else 0.0,
            }

            pred = max(0.0, float(model.predict(pd.DataFrame([vals])[FEATURES])[0]))
            store_preds.append(pred)
            sales_history.append(pred)

        actuals = test_subset["Weekly_Sales"].values
        rec_errors.extend(np.abs(actuals - np.array(store_preds)))

    recursive_mae = float(np.mean(rec_errors)) if rec_errors else 0.0
    return {"Recursive_12W_MAE": round(recursive_mae, 2)}


def train_and_evaluate():
    """
    Main execution pipeline: loads data, engineers features, trains candidate models,
    validates with TimeSeries expanding window CV, and saves the best model + metadata.
    """
    print("=" * 60)
    print(" RetailPulse AI: Training & Evaluation Pipeline")
    print("=" * 60)

    # 1. Load Data
    raw_df = load_raw_data()
    unique_dates = np.array(sorted(raw_df["Date"].unique()))
    split_date = pd.Timestamp(unique_dates[-20])
    print(f"[*] Total observations: {len(raw_df):,}")
    print(f"[*] Total stores: {raw_df['Store'].nunique()}")
    print(f"[*] Date range: {raw_df['Date'].min().strftime('%Y-%m-%d')} to {raw_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"[*] Test holdout split date: {split_date.strftime('%Y-%m-%d')} (Last 20 weeks)")

    # 2. Engineer Features with no leakage across the split date
    feat_df = engineer_features(raw_df, train_cutoff=split_date)

    # 3. Chronological Train/Test Split
    train_df = feat_df[feat_df["Date"] < split_date].copy()
    test_df = feat_df[feat_df["Date"] >= split_date].copy()

    X_train, y_train = train_df[FEATURES], train_df["Weekly_Sales"]
    X_test, y_test = test_df[FEATURES], test_df["Weekly_Sales"]

    print(f"[*] Training samples: {len(X_train):,}, Holdout test samples: {len(X_test):,}")

    # 4. Baselines (Naive Lag 1 & Seasonal Naive Lag 52)
    results = {
        "Naive (Lag 1)": calculate_metrics(y_test, test_df["Lag_1"]),
        "Seasonal Naive (Lag 52)": calculate_metrics(y_test, test_df["Lag_52"]),
    }

    # 5. Train Machine Learning Candidates
    fitted_models = {}
    test_predictions = {}

    for name, model in get_candidate_models().items():
        print(f"[*] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted_models[name] = model
        test_predictions[name] = y_pred
        results[name] = calculate_metrics(y_test, y_pred)
        print(f"    -> {name} Holdout RMSE: ${results[name]['RMSE']:,.2f} | R2: {results[name]['R2']:.4f}")

    # 6. Expanding-Window Cross-Validation (3 Folds)
    print("[*] Running 3-Fold Expanding-Window TimeSeriesSplit CV...")
    cv_scores = {name: [] for name in get_candidate_models()}
    tscv = TimeSeriesSplit(n_splits=3)
    clean_dates = np.array(sorted(feat_df["Date"].unique()))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(clean_dates), start=1):
        cutoff = pd.Timestamp(clean_dates[val_idx[0]])
        end = pd.Timestamp(clean_dates[val_idx[-1]])
        fold_train = feat_df[feat_df["Date"] < cutoff]
        fold_val = feat_df[(feat_df["Date"] >= cutoff) & (feat_df["Date"] <= end)]

        for name, model in get_candidate_models().items():
            model.fit(fold_train[FEATURES], fold_train["Weekly_Sales"])
            preds = model.predict(fold_val[FEATURES])
            fold_rmse = calculate_metrics(fold_val["Weekly_Sales"], preds)["RMSE"]
            cv_scores[name].append(fold_rmse)

    cv_summary = {
        name: {
            "Fold_RMSE": cv_scores[name],
            "Mean_RMSE": round(float(np.mean(cv_scores[name])), 2),
        }
        for name in cv_scores
    }

    # 7. Select Best Model by Holdout RMSE
    best_model_name = min(fitted_models, key=lambda n: results[n]["RMSE"])
    best_model = fitted_models[best_model_name]
    print(f"\n[+] Champion Model Selected: {best_model_name}")

    # 8. Recursive 12-Week Simulation
    print("[*] Evaluating multi-step recursive forecasting on holdout set...")
    rec_metric = simulate_recursive_forecast(best_model, train_df, test_df, horizon=12)
    print(f"    -> Multi-step 12-Week Recursive MAE: ${rec_metric['Recursive_12W_MAE']:,.2f}")

    # 9. Extract Feature Importance (fallback to Random Forest if HistGradientBoosting doesn't expose it directly)
    if hasattr(best_model, "feature_importances_"):
        importances = sorted(
            zip(FEATURES, [round(float(v), 4) for v in best_model.feature_importances_]),
            key=lambda z: z[1],
            reverse=True,
        )
    elif "Random Forest" in fitted_models and hasattr(fitted_models["Random Forest"], "feature_importances_"):
        rf_model = fitted_models["Random Forest"]
        importances = sorted(
            zip(FEATURES, [round(float(v), 4) for v in rf_model.feature_importances_]),
            key=lambda z: z[1],
            reverse=True,
        )
    else:
        importances = []

    # 10. Save Artifacts
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print(f"[+] Saved model to {MODELS_DIR / 'best_model.joblib'}")

    # Save test predictions for visual inspection
    export_preds = test_df[["Store", "Date", "Weekly_Sales"]].copy()
    export_preds["Naive_Lag_1"] = test_df["Lag_1"].values
    export_preds["Seasonal_Lag_52"] = test_df["Lag_52"].values
    for name, p in test_predictions.items():
        export_preds[name.replace(" ", "_")] = p
    export_preds.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    print(f"[+] Saved predictions to {MODELS_DIR / 'test_predictions.csv'}")

    # Save rich metadata
    metadata = {
        "best_model": best_model_name,
        "split_date": split_date.strftime("%Y-%m-%d"),
        "features": FEATURES,
        "metrics": results,
        "cv": cv_summary,
        "recursive_evaluation": rec_metric,
        "importance": importances,
        "rows": len(raw_df),
        "stores": int(raw_df["Store"].nunique()),
        "start": raw_df["Date"].min().strftime("%Y-%m-%d"),
        "end": raw_df["Date"].max().strftime("%Y-%m-%d"),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[+] Saved metadata to {MODELS_DIR / 'metadata.json'}")
    print("=" * 60)
    print(" Training pipeline completed successfully!")
    print("=" * 60)
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
