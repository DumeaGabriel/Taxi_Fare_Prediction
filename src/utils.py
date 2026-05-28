"""Utility functions for taxi fare prediction pipeline."""

from pathlib import Path
import pandas as pd
import numpy as np


def load_parquet(path: Path | str, sample_size: int | None = None) -> pd.DataFrame:
    """Load parquet file with optional sampling.

    Args:
        path: Path to parquet file
        sample_size: If set, sample this many rows (useful for large datasets)

    Returns:
        DataFrame loaded from parquet
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)

    if sample_size and len(df) > sample_size:
        print(f"Dataset size: {len(df):,} rows. Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=42)

    return df


def clean_features(X: pd.DataFrame, target_col: str = "fare_amount") -> pd.DataFrame:
    """Drop problematic columns that shouldn't be in features.

    Args:
        X: Feature DataFrame
        target_col: Target column name (for reference)

    Returns:
        Cleaned feature DataFrame
    """
    drop_cols = []

    if "key" in X.columns:
        drop_cols.append("key")
    if "fare_outlier" in X.columns:
        drop_cols.append("fare_outlier")
    if "pickup_datetime" in X.columns:
        drop_cols.append("pickup_datetime")

    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str, width: int = 70) -> None:
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)

