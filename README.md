Taxi Fare Prediction — Quick Start

This project trains a model to predict taxi fare amounts from trip data (pickup/dropoff coordinates, time features, passenger count, distance). It uses a scikit-learn pipeline with a Random Forest regressor as a strong baseline.

Quick checklist for the next student

- [ ] Install dependencies
- [ ] Prepare data (place cleaned parquet files in `data/processed/` or run `clean_data.py`)
- [ ] Train the model
- [ ] Evaluate model accuracy
- [ ] Generate predictions on test set

Prerequisites

- Python 3.8+ (Windows PowerShell examples below)
- From project root, install dependencies:

```powershell
pip install -r requirements.txt
```

Data

- Place raw CSVs (if you have them) in `data/raw/` and run the cleaning script:

```powershell
python src/clean_data.py
```

- The pipeline expects cleaned Parquet files at:
  - `data/processed/train_cleaned.parquet` (training data with `fare_amount`)
  - `data/processed/test_cleaned.parquet` (test data; `fare_amount` may be absent)

Train the model

- Default (samples up to 100k rows to avoid memory issues):

```powershell
python src/train_model.py
```

- To change training sample size (useful if you have more RAM):

```powershell
python src/train_model.py --sample-size 50000
```

- Outputs: trained pipeline saved to `models/fare_model.joblib` by default.

Evaluate the model

```powershell
python src/evaluate_model.py
```

- This will print evaluation metrics (RMSE, MAE, R^2) and save a report to `output/model_evaluation.txt`.

Generate predictions on the test set

```powershell
python src/predict_test.py
```

- Output: `output/test_predictions.csv` (predictions and, if available, actuals and errors).

Notes

- Learning type: Supervised learning — Regression.
- Baseline algorithm: RandomForestRegressor (good for tabular data).
- If you run into memory errors, reduce `--sample-size` or run on a machine with more RAM.
- If you want the student not to receive the trained model, delete `models/fare_model.joblib` before sharing.

If you want a one-page summary or want me to remove the trained model from the repo, tell me and I will update the project accordingly.
