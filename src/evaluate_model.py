from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import train_test_split


def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate comprehensive accuracy metrics for regression."""

    errors = np.abs(y_true - y_pred)

    # Regression metrics
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Percentage of predictions within certain error thresholds
    within_1 = (errors <= 1).sum() / len(errors) * 100
    within_2 = (errors <= 2).sum() / len(errors) * 100
    within_5 = (errors <= 5).sum() / len(errors) * 100
    within_10 = (errors <= 10).sum() / len(errors) * 100

    # Additional statistics
    median_error = np.median(errors)
    std_error = np.std(errors)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2_Score": r2,
        "Median_Error": median_error,
        "StdDev_Error": std_error,
        "Within_$1": within_1,
        "Within_$2": within_2,
        "Within_$5": within_5,
        "Within_$10": within_10,
        "Min_Error": errors.min(),
        "Max_Error": errors.max(),
        "Mean_Error": errors.mean()
    }


def main():
    model_path = Path("models/fare_model.joblib")
    train_path = Path("data/processed/train_cleaned.parquet")
    target_col = "fare_amount"

    # Load model and training data
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    print("=" * 70)
    print("MODEL ACCURACY EVALUATION")
    print("=" * 70)

    print("\n[1] Loading model and data...")
    pipeline = joblib.load(model_path)
    df = pd.read_parquet(train_path)

    # Sample if too large
    if len(df) > 100_000:
        print(f"    Dataset size: {len(df):,} rows. Sampling 100k rows...")
        df = df.sample(n=100_000, random_state=42)

    print(f"    Using {len(df):,} rows")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Prepare data
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Drop problematic columns
    if "key" in X.columns:
        X = X.drop(columns=["key"])
    if "fare_outlier" in X.columns:
        X = X.drop(columns=["fare_outlier"])
    if "pickup_datetime" in X.columns:
        X = X.drop(columns=["pickup_datetime"])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Make predictions
    print("\n[2] Making predictions on validation set...")
    preds_val = pipeline.predict(X_val)

    # Calculate metrics
    print("\n[3] Calculating accuracy metrics...")
    metrics = calculate_accuracy_metrics(y_val.values, preds_val)

    # Print results
    print("\n" + "=" * 70)
    print("REGRESSION METRICS")
    print("=" * 70)
    print(f"R² Score (Coefficient of Determination): {metrics['R2_Score']:.4f}")
    print(f"  -> Explains {metrics['R2_Score']*100:.2f}% of the variance in fares")
    print()
    print(f"RMSE (Root Mean Squared Error):          ${metrics['RMSE']:.4f}")
    print(f"  -> Average prediction error magnitude")
    print()
    print(f"MAE (Mean Absolute Error):               ${metrics['MAE']:.4f}")
    print(f"  -> Average absolute prediction error")
    print()
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.4f}%")
    print(f"  -> Average percentage error relative to actual fares")
    print()
    print(f"Median Error:                            ${metrics['Median_Error']:.4f}")
    print(f"Std Dev of Error:                        ${metrics['StdDev_Error']:.4f}")

    print("\n" + "=" * 70)
    print("ACCURACY AS PERCENTAGE WITHIN ERROR THRESHOLD")
    print("=" * 70)
    print(f"Predictions within ±$1:                  {metrics['Within_$1']:.2f}%")
    print(f"Predictions within ±$2:                  {metrics['Within_$2']:.2f}%")
    print(f"Predictions within ±$5:                  {metrics['Within_$5']:.2f}%")
    print(f"Predictions within ±$10:                 {metrics['Within_$10']:.2f}%")

    print("\n" + "=" * 70)
    print("ERROR RANGE")
    print("=" * 70)
    print(f"Minimum Error (best prediction):         ${metrics['Min_Error']:.4f}")
    print(f"Maximum Error (worst prediction):        ${metrics['Max_Error']:.4f}")
    print(f"Mean Error (avg error magnitude):        ${metrics['Mean_Error']:.4f}")

    # Sample predictions vs actual
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS vs ACTUAL (First 20)")
    print("=" * 70)
    sample_df = pd.DataFrame({
        "Actual_Fare": y_val.iloc[:20].values,
        "Predicted_Fare": preds_val[:20],
        "Error": np.abs(y_val.iloc[:20].values - preds_val[:20]),
        "Error_%": (np.abs(y_val.iloc[:20].values - preds_val[:20]) / y_val.iloc[:20].values * 100)
    })
    sample_df = sample_df.round(4)
    print(sample_df.to_string(index=False))

    # Save evaluation report
    print("\n[4] Saving evaluation report...")
    report_path = Path("output/model_evaluation.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TAXI FARE PREDICTION MODEL - ACCURACY EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("REGRESSION METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"R² Score:                              {metrics['R2_Score']:.4f}\n")
        f.write(f"RMSE (Root Mean Squared Error):        ${metrics['RMSE']:.4f}\n")
        f.write(f"MAE (Mean Absolute Error):             ${metrics['MAE']:.4f}\n")
        f.write(f"MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.4f}%\n")
        f.write(f"Median Error:                          ${metrics['Median_Error']:.4f}\n")
        f.write(f"Std Dev of Error:                      ${metrics['StdDev_Error']:.4f}\n\n")

        f.write("ACCURACY (% within error threshold)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Within ±$1:                            {metrics['Within_$1']:.2f}%\n")
        f.write(f"Within ±$2:                            {metrics['Within_$2']:.2f}%\n")
        f.write(f"Within ±$5:                            {metrics['Within_$5']:.2f}%\n")
        f.write(f"Within ±$10:                           {metrics['Within_$10']:.2f}%\n\n")

        f.write("ERROR STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Minimum Error:                         ${metrics['Min_Error']:.4f}\n")
        f.write(f"Maximum Error:                         ${metrics['Max_Error']:.4f}\n")
        f.write(f"Mean Error:                            ${metrics['Mean_Error']:.4f}\n")

    print(f"Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"\n[OK] Your model achieves {metrics['R2_Score']*100:.1f}% accuracy (R² Score)")
    print(f"[OK] Average prediction error is ${metrics['MAE']:.2f}")
    print(f"[OK] {metrics['Within_$2']:.1f}% of predictions are within ±$2 (good for taxi fares)")
    print("\n")


if __name__ == "__main__":
    main()

