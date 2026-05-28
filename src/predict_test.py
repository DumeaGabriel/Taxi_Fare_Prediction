from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    model_path = Path("models/fare_model.joblib")
    test_path = Path("data/processed/test_cleaned.parquet")
    target_col = "fare_amount"

    # 1) Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)

    # 2) Load test data
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    print(f"Loading test data from {test_path}...")
    df_test = pd.read_parquet(test_path)

    # Sample if too large
    if len(df_test) > 100_000:
        print(f"Test dataset size: {len(df_test):,} rows. Sampling 50k rows for testing...")
        df_test = df_test.sample(n=50_000, random_state=42)

    print(f"Using {len(df_test):,} rows for testing.")

    # 3) Check if target exists (for metric calculation)
    has_target = target_col in df_test.columns

    if has_target:
        y_test = df_test[target_col].copy()
        X_test = df_test.drop(columns=[target_col]).copy()
    else:
        X_test = df_test.copy()
        print(f"Warning: Target column '{target_col}' not found. Will generate predictions only.")

    # 4) Generate predictions
    print("Generating predictions...")
    predictions = pipeline.predict(X_test)

    # 5) Calculate metrics if target exists
    if has_target:
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\n=== Test Set Metrics ===")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE : {mae:.4f}")
        print(f"Test R2  : {r2:.4f}")

        # Show sample predictions vs actuals
        print("\n=== Sample Predictions vs Actual ===")
        results = pd.DataFrame({
            "actual": y_test.iloc[:10].values,
            "predicted": predictions[:10],
            "error": abs(y_test.iloc[:10].values - predictions[:10])
        })
        print(results.to_string(index=False))
    else:
        print(f"\nGenerated {len(predictions)} predictions (no ground truth available)")
        print(f"Sample predictions: {predictions[:10]}")

    # 6) Save predictions
    output_pred_path = Path("output/test_predictions.csv")
    pred_df = pd.DataFrame({
        "prediction": predictions,
    })
    if has_target:
        pred_df["actual"] = y_test.values
        pred_df["error"] = abs(y_test.values - predictions)

    pred_df.to_csv(output_pred_path, index=False)
    print(f"\nPredictions saved to: {output_pred_path}")


if __name__ == "__main__":
    main()

