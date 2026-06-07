from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "output"
train_parquet = str(output_dir / "train_cleaned.parquet")
test_parquet = str(output_dir / "test_cleaned.parquet")

spark = (
    SparkSession.builder
    .appName("taxi-gbt-training")
    .master("local[*]")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
    .config("spark.sql.warehouse.dir", str(base_dir / "spark-warehouse"))
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.2")
    .getOrCreate()
)

train_df = spark.read.parquet(train_parquet).sample(fraction=0.1, seed=42)
test_df  = spark.read.parquet(test_parquet)

print("Train schema:")
train_df.printSchema()

def add_features(df):
    # Haversine distance (more accurate than the Euclidean approximation in cleaning)
    lat1 = F.toRadians(F.col("pickup_latitude"))
    lat2 = F.toRadians(F.col("dropoff_latitude"))
    lon1 = F.toRadians(F.col("pickup_longitude"))
    lon2 = F.toRadians(F.col("dropoff_longitude"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = F.pow(F.sin(dlat / 2), 2) + F.cos(lat1) * F.cos(lat2) * F.pow(F.sin(dlon / 2), 2)
    haversine = 6371.0 * 2 * F.asin(F.sqrt(a))   # km

    # Trip bearing: direction of travel in radians
    bearing = F.atan2(
        F.sin(dlon) * F.cos(lat2),
        F.cos(lat1) * F.sin(lat2) - F.sin(lat1) * F.cos(lat2) * F.cos(dlon)
    )

    # Temporal flags
    dow = F.dayofweek("pickup_datetime")           # 1=Sun … 7=Sat
    is_weekend = ((dow == 1) | (dow == 7)).cast("int")
    is_rush    = (
        ((F.col("pickup_hour") >= 7)  & (F.col("pickup_hour") <= 9)  & (is_weekend == 0)) |
        ((F.col("pickup_hour") >= 17) & (F.col("pickup_hour") <= 19) & (is_weekend == 0))
    ).cast("int")
    is_night   = ((F.col("pickup_hour") >= 20) | (F.col("pickup_hour") <= 5)).cast("int")

    # Airport proximity flags (within ~2 km of terminal)
    def near(lat, lon, clat, clon, thresh_deg=0.018):
        return (
            (F.abs(F.col(lat) - clat) < thresh_deg) &
            (F.abs(F.col(lon) - clon) < thresh_deg)
        )  # keep as boolean so | works correctly

    jfk_pickup  = near("pickup_latitude",  "pickup_longitude",   40.6413, -73.7781)
    jfk_dropoff = near("dropoff_latitude", "dropoff_longitude",  40.6413, -73.7781)
    lga_pickup  = near("pickup_latitude",  "pickup_longitude",   40.7769, -73.8740)
    lga_dropoff = near("dropoff_latitude", "dropoff_longitude",  40.7769, -73.8740)
    is_airport  = (jfk_pickup | jfk_dropoff | lga_pickup | lga_dropoff).cast("int")

    return (
        df
        .withColumn("haversine_km", haversine)
        .withColumn("bearing",      bearing)
        .withColumn("is_weekend",   is_weekend)
        .withColumn("is_rush",      is_rush)
        .withColumn("is_night",     is_night)
        .withColumn("is_airport",   is_airport)
    )

train_raw, val_raw = train_df.randomSplit([0.8, 0.2], seed=42)
train_split = add_features(train_raw)
val_split   = add_features(val_raw)
test_df     = add_features(test_df)

# Build pipeline
feature_cols = [
    "haversine_km",
    "distance_km",        # Euclidean already in parquet — keep as cross-check
    "bearing",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_year",
    "pickup_month",
    "pickup_hour",
    "is_weekend",
    "is_rush",
    "is_night",
    "is_airport",
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="fare_amount",
    maxIter=30,
    maxDepth=5,
    maxBins=16,
    stepSize=0.1,
    subsamplingRate=0.8,
    seed=42,
)

pipeline = Pipeline(stages=[assembler, gbt])

print("Split complete — starting training...")

model = pipeline.fit(train_split)

# Evaluate on validation set
val_preds = model.transform(val_split)

rmse_eval = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
mae_eval  = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="mae")
r2_eval   = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")

rmse = rmse_eval.evaluate(val_preds)
mae  = mae_eval.evaluate(val_preds)
r2   = r2_eval.evaluate(val_preds)

print("\n=== Validation Metrics ===")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")

# Feature importances from the GBT model
gbt_model   = model.stages[-1]
importances = list(zip(feature_cols, gbt_model.featureImportances.toArray()))
importances.sort(key=lambda x: x[1], reverse=True)

print("\n=== Feature Importances ===")
for feat, score in importances:
    print(f"  {feat:<22} {score:.4f}")

# Predict on test set and save
test_preds = model.transform(test_df)

predictions_path = str(output_dir / "gbt_predictions.parquet")
(
    test_preds
    .select("key", "prediction")
    .withColumnRenamed("prediction", "fare_amount")
    .coalesce(2)
    .write.mode("overwrite")
    .option("compression", "snappy")
    .parquet(predictions_path)
)

print(f"\nPredictions saved to: {predictions_path}")

spark.stop()
