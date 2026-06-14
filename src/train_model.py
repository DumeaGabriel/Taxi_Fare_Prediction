from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a taxi fare regression model.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train_cleaned.parquet"),
        help="Path to cleaned training parquet data.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="fare_amount",
        help="Target column name for regression.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/fare_model.joblib"),
        help="Where to write the trained model pipeline.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Sample size for training (to handle large datasets).",
    )
    return parser.parse_args()


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Compute great-circle distance in km between two lat/lon points (vectorised)."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetime parts and Haversine distance if raw columns are present.

    Spark already writes pickup_year/month/day/hour into the parquet, so we only
    derive them here when the raw pickup_datetime column is still present (i.e.
    the data was NOT pre-processed by clean_data.py).
    """
    df = df.copy()

    # Only expand pickup_datetime when the derived columns are absent.
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

    # Haversine distance (more accurate than the Euclidean distance_km from Spark).
    coord_cols = {"pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"}
    if coord_cols.issubset(df.columns):
        df["haversine_km"] = haversine_km(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

    return df


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,           # prevents overfitting on 100k sample
        min_samples_leaf=5,     # smooths leaves, reduces variance
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main() -> None:
    args = parse_args()

    if not args.train_path.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_path}")

    print("Loading and sampling training data...")
    df = pd.read_parquet(args.train_path)

    if len(df) > args.sample_size:
        print(f"Dataset size: {len(df):,} rows. Sampling {args.sample_size:,} rows for training...")
        df = df.sample(n=args.sample_size, random_state=42)

    print(f"Using {len(df):,} rows for training.")

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Columns: {list(df.columns)}"
        )

    y = df[args.target].copy()
    X = df.drop(columns=[args.target]).copy()

    # Drop identifier / leakage columns
    for col in ("key", "fare_outlier", "distance_outlier"):
        if col in X.columns:
            X = X.drop(columns=[col])

    X = add_features(X)

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)

    mse = mean_squared_error(y_val, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print("\nLearning method: Supervised Learning -> Regression")
    print("Model: RandomForestRegressor (max_depth=20, min_samples_leaf=5, n_estimators=300)")
    print(f"Validation RMSE : {rmse:.4f}")
    print(f"Validation MAE  : {mae:.4f}")
    print(f"Validation R²   : {r2:.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    print(f"\nSaved model to: {args.model_out}")


if __name__ == "__main__":
    main()