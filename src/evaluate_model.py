"""evaluate.py — model evaluation with charts and test-set metrics."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pickup_datetime" in df.columns:
        ts = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        if "pickup_year" not in df.columns:
            df["pickup_year"] = ts.dt.year
        if "pickup_month" not in df.columns:
            df["pickup_month"] = ts.dt.month
        if "pickup_day" not in df.columns:
            df["pickup_day"] = ts.dt.day
        if "pickup_hour" not in df.columns:
            df["pickup_hour"] = ts.dt.hour
        df = df.drop(columns=["pickup_datetime"])
    coord_cols = {"pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"}
    if coord_cols.issubset(df.columns):
        df["haversine_km"] = haversine_km(
            df["pickup_latitude"], df["pickup_longitude"],
            df["dropoff_latitude"], df["dropoff_longitude"],
        )
    return df


def drop_leakage(X: pd.DataFrame) -> pd.DataFrame:
    for col in ("key", "fare_outlier", "distance_outlier", "pickup_datetime"):
        if col in X.columns:
            X = X.drop(columns=[col])
    return X


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = np.abs(y_true - y_pred)
    return {
        "R2":          r2_score(y_true, y_pred),
        "RMSE":        mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE":         mean_absolute_error(y_true, y_pred),
        "MAPE":        mean_absolute_percentage_error(y_true, y_pred) * 100,
        "Median_AE":   float(np.median(errors)),
        "Std_AE":      float(np.std(errors)),
        "Within_$1":   float((errors <= 1).mean() * 100),
        "Within_$2":   float((errors <= 2).mean() * 100),
        "Within_$5":   float((errors <= 5).mean() * 100),
        "Within_$10":  float((errors <= 10).mean() * 100),
        "Max_AE":      float(errors.max()),
        "Min_AE":      float(errors.min()),
    }


# ---------------------------------------------------------------------------
# Chart helpers — each returns a Figure
# ---------------------------------------------------------------------------

STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8f8f8",
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
}

BLUE   = "#378ADD"
GREEN  = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
AMBER  = "#EF9F27"
GRAY   = "#888780"


def chart_predicted_vs_actual(y_true, y_pred, output_path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        lim = max(y_true.max(), y_pred.max()) * 1.05
        ax.scatter(y_true, y_pred, alpha=0.25, s=8, color=BLUE, rasterized=True, label="Predictions")
        ax.plot([0, lim], [0, lim], color=CORAL, lw=1.5, ls="--", label="Perfect fit")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Actual fare ($)")
        ax.set_ylabel("Predicted fare ($)")
        ax.set_title("Predicted vs actual fare")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {output_path}")


def chart_residuals(y_true, y_pred, output_path: Path) -> None:
    residuals = y_pred - y_true
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # Residuals vs predicted
        ax1.scatter(y_pred, residuals, alpha=0.2, s=6, color=PURPLE, rasterized=True)
        ax1.axhline(0, color=CORAL, lw=1.5, ls="--")
        ax1.set_xlabel("Predicted fare ($)")
        ax1.set_ylabel("Residual (predicted − actual) ($)")
        ax1.set_title("Residuals vs predicted")

        # Residual distribution
        ax2.hist(residuals, bins=80, color=BLUE, edgecolor="white", linewidth=0.3)
        ax2.axvline(0, color=CORAL, lw=1.5, ls="--", label="Zero error")
        ax2.axvline(residuals.mean(), color=AMBER, lw=1.5, ls="-", label=f"Mean {residuals.mean():.2f}")
        ax2.set_xlabel("Residual ($)")
        ax2.set_ylabel("Count")
        ax2.set_title("Residual distribution")
        ax2.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {output_path}")


def chart_error_thresholds(metrics: dict, output_path: Path) -> None:
    thresholds = ["Within_$1", "Within_$2", "Within_$5", "Within_$10"]
    labels     = ["±$1", "±$2", "±$5", "±$10"]
    values     = [metrics[t] for t in thresholds]
    colors     = [BLUE, GREEN, AMBER, CORAL]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_xlabel("Error threshold")
        ax.set_ylabel("% of predictions")
        ax.set_title("Accuracy within error threshold")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {output_path}")


def chart_feature_importance(pipeline, feature_names: list[str], output_path: Path, top_n: int = 20) -> None:
    rf = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]

    try:
        ohe_names = pre.named_transformers_["cat"]["onehot"].get_feature_names_out(
            pre.transformers_[1][2]  # categorical column names
        )
    except Exception:
        ohe_names = []

    num_names = pre.transformers_[0][2]
    all_names = list(num_names) + list(ohe_names)

    importances = rf.feature_importances_
    n = min(len(all_names), len(importances))
    imp_series = pd.Series(importances[:n], index=all_names[:n]).sort_values(ascending=True).tail(top_n)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, max(4, len(imp_series) * 0.32)))
        colors = [BLUE if imp_series.index[i].startswith(("distance", "haversine", "pickup_lat", "pickup_lon", "dropoff"))
                  else GREEN if imp_series.index[i].startswith("pickup_hour")
                  else GRAY for i in range(len(imp_series))]
        ax.barh(imp_series.index, imp_series.values, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Feature importance (mean decrease impurity)")
        ax.set_title(f"Top {len(imp_series)} feature importances")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {output_path}")


def chart_error_by_fare_bin(y_true, y_pred, output_path: Path) -> None:
    df = pd.DataFrame({"actual": y_true, "abs_error": np.abs(y_pred - y_true)})
    bins = [0, 5, 10, 15, 20, 30, 50, 100]
    labels = ["$0-5", "$5-10", "$10-15", "$15-20", "$20-30", "$30-50", "$50+"]
    df["fare_bin"] = pd.cut(df["actual"], bins=bins, labels=labels, right=False)
    grouped = df.groupby("fare_bin", observed=True)["abs_error"].median()

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(grouped.index.astype(str), grouped.values, color=PURPLE,
               edgecolor="white", linewidth=0.4, width=0.6)
        for i, (label, val) in enumerate(zip(grouped.index, grouped.values)):
            ax.text(i, val + 0.02, f"${val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Actual fare range")
        ax.set_ylabel("Median absolute error ($)")
        ax.set_title("Median error by fare range")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    BASE_DIR    = Path(__file__).resolve().parents[1]
    model_path  = BASE_DIR / "models" / "fare_model.joblib"
    output_dir  = BASE_DIR / "output"

    # parquet may live in output/ (pandas pipeline) or data/processed/ (Spark pipeline)
    _candidates_train = [
        BASE_DIR / "output" / "train_cleaned.parquet",
        BASE_DIR / "data" / "processed" / "train_cleaned.parquet",
    ]
    train_path = next((p for p in _candidates_train if p.exists()), _candidates_train[0])

    _candidates_test = [
        BASE_DIR / "output" / "test_cleaned.parquet",
        BASE_DIR / "data" / "processed" / "test_cleaned.parquet",
    ]
    test_path = next((p for p in _candidates_test if p.exists()), _candidates_test[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    target_col = "fare_amount"

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_model.py first.")
    print(f"Loading model from {model_path} ...")
    pipeline = joblib.load(model_path)

    # ------------------------------------------------------------------
    # 2. Validation metrics (from training data split)
    # ------------------------------------------------------------------
    print(f"\nLoading training data from {train_path} ...")
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    df_train = pd.read_parquet(train_path)
    if len(df_train) > 100_000:
        df_train = df_train.sample(n=100_000, random_state=42)

    y = df_train[target_col].copy()
    X = drop_leakage(df_train.drop(columns=[target_col]))
    X = add_features(X)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Running predictions on validation split ...")
    preds_val = pipeline.predict(X_val)
    val_metrics = compute_metrics(y_val.values, preds_val)

    # ------------------------------------------------------------------
    # 3. Test set metrics (if test data has target column)
    # ------------------------------------------------------------------
    test_metrics = None
    preds_test   = None
    y_test_vals  = None

    if test_path.exists():
        print(f"Loading test data from {test_path} ...")
        df_test = pd.read_parquet(test_path)
        if len(df_test) > 50_000:
            df_test = df_test.sample(n=50_000, random_state=42)

        has_target = target_col in df_test.columns
        if has_target:
            y_test_vals = df_test[target_col].copy()
            X_test = drop_leakage(df_test.drop(columns=[target_col]))
        else:
            X_test = drop_leakage(df_test.copy())

        X_test = add_features(X_test)
        preds_test = pipeline.predict(X_test)

        if has_target:
            test_metrics = compute_metrics(y_test_vals.values, preds_test)
    else:
        print(f"No test file found at {test_path} — skipping test evaluation.")

    # ------------------------------------------------------------------
    # 4. Print report
    # ------------------------------------------------------------------
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("TAXI FARE PREDICTION — EVALUATION REPORT".center(70))
    print(f"{SEP}")

    def print_metrics(name: str, m: dict) -> None:
        print(f"\n{'—'*70}")
        print(f"  {name}")
        print(f"{'—'*70}")
        print(f"  R² Score  : {m['R2']:.4f}  ({m['R2']*100:.2f}% variance explained)")
        print(f"  RMSE      : ${m['RMSE']:.4f}")
        print(f"  MAE       : ${m['MAE']:.4f}")
        print(f"  MAPE      : {m['MAPE']:.2f}%")
        print(f"  Median AE : ${m['Median_AE']:.4f}   |  Std AE: ${m['Std_AE']:.4f}")
        print(f"\n  Predictions within threshold:")
        print(f"    ±$1  → {m['Within_$1']:.2f}%")
        print(f"    ±$2  → {m['Within_$2']:.2f}%")
        print(f"    ±$5  → {m['Within_$5']:.2f}%")
        print(f"    ±$10 → {m['Within_$10']:.2f}%")
        print(f"\n  Error range: ${m['Min_AE']:.4f} (best) → ${m['Max_AE']:.4f} (worst)")

    print_metrics("VALIDATION SET", val_metrics)
    if test_metrics:
        print_metrics("TEST SET", test_metrics)

    # Sample predictions table
    print(f"\n{'—'*70}")
    print("  SAMPLE PREDICTIONS — VALIDATION SET (first 20 rows)")
    print(f"{'—'*70}")
    sample = pd.DataFrame({
        "Actual ($)":    y_val.iloc[:20].values,
        "Predicted ($)": preds_val[:20],
        "Abs Error ($)": np.abs(y_val.iloc[:20].values - preds_val[:20]),
        "Error %":       (np.abs(y_val.iloc[:20].values - preds_val[:20]) / y_val.iloc[:20].values * 100),
    }).round(3)
    print(sample.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Generate charts
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("GENERATING CHARTS".center(70))
    print(SEP)

    chart_predicted_vs_actual(
        y_val.values, preds_val,
        output_dir / "chart_predicted_vs_actual.png"
    )
    chart_residuals(
        y_val.values, preds_val,
        output_dir / "chart_residuals.png"
    )
    chart_error_thresholds(
        val_metrics,
        output_dir / "chart_error_thresholds.png"
    )
    chart_feature_importance(
        pipeline,
        feature_names=X_val.columns.tolist(),
        output_path=output_dir / "chart_feature_importance.png",
    )
    chart_error_by_fare_bin(
        y_val.values, preds_val,
        output_dir / "chart_error_by_fare_bin.png"
    )

    # ------------------------------------------------------------------
    # 6. Save predictions to CSV
    # ------------------------------------------------------------------
    val_pred_df = pd.DataFrame({
        "actual":    y_val.values,
        "predicted": preds_val,
        "abs_error": np.abs(y_val.values - preds_val),
    })
    val_pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    print(f"  Saved: {output_dir / 'val_predictions.csv'}")

    if preds_test is not None:
        test_pred_df = pd.DataFrame({"predicted": preds_test})
        if y_test_vals is not None:
            test_pred_df["actual"]    = y_test_vals.values
            test_pred_df["abs_error"] = np.abs(y_test_vals.values - preds_test)
        test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
        print(f"  Saved: {output_dir / 'test_predictions.csv'}")

    # ------------------------------------------------------------------
    # 7. Save text report
    # ------------------------------------------------------------------
    report_path = output_dir / "model_evaluation.txt"
    with open(report_path, "w") as f:
        f.write("TAXI FARE PREDICTION — MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        for name, m in [("VALIDATION SET", val_metrics)] + (
            [("TEST SET", test_metrics)] if test_metrics else []
        ):
            f.write(f"{name}\n{'-'*70}\n")
            for k, v in m.items():
                f.write(f"  {k:20s}: {v:.4f}\n")
            f.write("\n")
    print(f"  Saved: {report_path}")

    print(f"\n{SEP}")
    print("  Done. All outputs written to: output/")
    print(SEP + "\n")


if __name__ == "__main__":
    main()