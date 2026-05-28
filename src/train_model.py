from __future__ import annotations

import argparse
from pathlib import Path

import joblib
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


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    # Expand pickup timestamp into explicit numeric features the model can learn from.
    if "pickup_datetime" in df.columns:
        pickup_ts = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df = df.copy()
        df["pickup_year"] = pickup_ts.dt.year
        df["pickup_month"] = pickup_ts.dt.month
        df["pickup_day"] = pickup_ts.dt.day
        df["pickup_hour"] = pickup_ts.dt.hour
        df = df.drop(columns=["pickup_datetime"])
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
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main() -> None:
    args = parse_args()

    if not args.train_path.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_path}")

    # Load and sample if needed (for memory efficiency with large datasets)
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

    X = add_datetime_features(X)

    # Drop identifier fields that do not carry stable predictive signal.
    if "key" in X.columns:
        X = X.drop(columns=["key"])

    # Drop fare_outlier (only exists in train, cannot compute on test without ground truth)
    if "fare_outlier" in X.columns:
        X = X.drop(columns=["fare_outlier"])

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

    print("Learning method: Supervised Learning -> Regression")
    print("Model: RandomForestRegressor")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE : {mae:.4f}")
    print(f"Validation R2  : {r2:.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    print(f"Saved model to: {args.model_out}")


if __name__ == "__main__":
    main()

