"""clean_data.py — pandas-based data cleaning (replaces PySpark version).

Reads raw train.csv / test.csv, applies the same cleaning logic as before,
and writes parquet files to data/processed/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
TEST_PATH  = BASE_DIR / "data" / "raw" / "test.csv"
OUT_DIR    = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: pd.Series, lon1: pd.Series,
                 lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def iqr_bounds(series: pd.Series,
               lower_k: float = 1.5,
               upper_k: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return max(0.0, q1 - lower_k * iqr), q3 + upper_k * iqr


# ---------------------------------------------------------------------------
# Clean train
# ---------------------------------------------------------------------------

def clean_train(path: Path) -> pd.DataFrame:
    print(f"Reading {path} ...")
    df = pd.read_csv(
        path,
        usecols=["key", "fare_amount", "pickup_datetime",
                 "pickup_longitude", "pickup_latitude",
                 "dropoff_longitude", "dropoff_latitude",
                 "passenger_count"],
        parse_dates=["pickup_datetime"],
    )
    print(f"  Loaded {len(df):,} rows")

    # --- cast dtypes ---
    float_cols = ["fare_amount", "pickup_longitude", "pickup_latitude",
                  "dropoff_longitude", "dropoff_latitude"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce").astype("Int64")

    # --- drop nulls in required columns ---
    required = ["pickup_datetime", "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude", "passenger_count"]
    before = len(df)
    df = df.dropna(subset=required)
    print(f"  Dropped {before - len(df):,} rows with nulls in required columns")

    # --- fill missing fare with median ---
    if df["fare_amount"].isna().any():
        median_fare = df["fare_amount"].median()
        df["fare_amount"] = df["fare_amount"].fillna(median_fare)

    # --- deduplicate ---
    before = len(df)
    df = df.drop_duplicates(subset=["pickup_datetime", "pickup_longitude",
                                    "pickup_latitude", "dropoff_longitude",
                                    "dropoff_latitude", "passenger_count"])
    print(f"  Removed {before - len(df):,} duplicate rows")

    # --- geographic bounds (NYC area) ---
    before = len(df)
    df = df[
        df["passenger_count"].between(1, 6) &
        df["pickup_latitude"].between(40.0, 43.0) &
        df["dropoff_latitude"].between(40.0, 43.0) &
        df["pickup_longitude"].between(-75.0, -72.0) &
        df["dropoff_longitude"].between(-75.0, -72.0)
    ]
    print(f"  Removed {before - len(df):,} rows outside geographic bounds")

    # --- positive fare ---
    before = len(df)
    df = df[df["fare_amount"] > 0]
    print(f"  Removed {before - len(df):,} rows with non-positive fare")

    # --- distance feature (Euclidean, matches original Spark script) ---
    df["distance_km"] = 111 * np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2 +
        (df["pickup_latitude"]  - df["dropoff_latitude"])  ** 2
    )

    # --- distance outliers (IQR, upper_k=2 as in original) ---
    dist_lo, dist_hi = iqr_bounds(df["distance_km"], lower_k=1.5, upper_k=2.0)
    df["distance_outlier"] = ~df["distance_km"].between(dist_lo, dist_hi)
    print(f"  Distance bounds: {dist_lo:.2f} km – {dist_hi:.2f} km")
    print(f"  Distance outliers: {df['distance_outlier'].sum():,}")

    # --- fare outliers (IQR 1.5 both sides) ---
    fare_lo, fare_hi = iqr_bounds(df["fare_amount"])
    df["fare_outlier"] = ~df["fare_amount"].between(fare_lo, fare_hi)
    print(f"  Fare outliers: {df['fare_outlier'].sum():,}")

    # --- keep only clean rows ---
    before = len(df)
    df = df[~df["distance_outlier"] & ~df["fare_outlier"]]
    print(f"  Removed {before - len(df):,} outlier rows → {len(df):,} clean rows remain")

    # --- datetime parts ---
    df["pickup_year"]  = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"]   = df["pickup_datetime"].dt.day
    df["pickup_hour"]  = df["pickup_datetime"].dt.hour

    df["passenger_count"] = df["passenger_count"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Clean test (no fare column, lighter cleaning)
# ---------------------------------------------------------------------------

def clean_test(path: Path) -> pd.DataFrame:
    print(f"\nReading {path} ...")
    df = pd.read_csv(
        path,
        parse_dates=["pickup_datetime"],
    )
    print(f"  Loaded {len(df):,} rows")

    float_cols = ["pickup_longitude", "pickup_latitude",
                  "dropoff_longitude", "dropoff_latitude"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "passenger_count" in df.columns:
        df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce").fillna(1).astype(int)

    # distance
    df["distance_km"] = 111 * np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2 +
        (df["pickup_latitude"]  - df["dropoff_latitude"])  ** 2
    )

    # datetime parts
    df["pickup_year"]  = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"]   = df["pickup_datetime"].dt.day
    df["pickup_hour"]  = df["pickup_datetime"].dt.hour

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Train
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Raw train CSV not found: {TRAIN_PATH}")

    train_clean = clean_train(TRAIN_PATH)
    out_train = OUT_DIR / "train_cleaned.parquet"
    train_clean.to_parquet(out_train, index=False, compression="snappy")
    print(f"\nSaved cleaned train to: {out_train}  ({len(train_clean):,} rows)")

    # Test (optional)
    if TEST_PATH.exists():
        test_clean = clean_test(TEST_PATH)
        out_test = OUT_DIR / "test_cleaned.parquet"
        test_clean.to_parquet(out_test, index=False, compression="snappy")
        print(f"Saved cleaned test  to: {out_test}  ({len(test_clean):,} rows)")
    else:
        print(f"\nNo test CSV found at {TEST_PATH} — skipping.")

    print("\nDone.")


if __name__ == "__main__":
    main()