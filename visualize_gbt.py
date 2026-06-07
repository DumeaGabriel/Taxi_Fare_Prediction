from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

base_dir   = Path(__file__).resolve().parent
output_dir = base_dir / "output"
train_parquet = str(output_dir / "train_cleaned.parquet")

spark = (
    SparkSession.builder
    .appName("taxi-gbt-visualize")
    .master("local[*]")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
    .config("spark.sql.warehouse.dir", str(base_dir / "spark-warehouse"))
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.2")
    .getOrCreate()
)

train_df = spark.read.parquet(train_parquet).sample(fraction=0.1, seed=42)

def add_features(df):
    lat1 = F.toRadians(F.col("pickup_latitude"))
    lat2 = F.toRadians(F.col("dropoff_latitude"))
    lon1 = F.toRadians(F.col("pickup_longitude"))
    lon2 = F.toRadians(F.col("dropoff_longitude"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = F.pow(F.sin(dlat / 2), 2) + F.cos(lat1) * F.cos(lat2) * F.pow(F.sin(dlon / 2), 2)
    haversine = 6371.0 * 2 * F.asin(F.sqrt(a))

    bearing = F.atan2(
        F.sin(dlon) * F.cos(lat2),
        F.cos(lat1) * F.sin(lat2) - F.sin(lat1) * F.cos(lat2) * F.cos(dlon)
    )

    dow        = F.dayofweek("pickup_datetime")
    is_weekend = ((dow == 1) | (dow == 7)).cast("int")
    is_rush    = (
        ((F.col("pickup_hour") >= 7)  & (F.col("pickup_hour") <= 9)  & (is_weekend == 0)) |
        ((F.col("pickup_hour") >= 17) & (F.col("pickup_hour") <= 19) & (is_weekend == 0))
    ).cast("int")
    is_night   = ((F.col("pickup_hour") >= 20) | (F.col("pickup_hour") <= 5)).cast("int")

    def near(lat, lon, clat, clon, thresh_deg=0.018):
        return (
            (F.abs(F.col(lat) - clat) < thresh_deg) &
            (F.abs(F.col(lon) - clon) < thresh_deg)
        )

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

# Split before featurising (same order as train_gbt)
train_raw, val_raw = train_df.randomSplit([0.8, 0.2], seed=42)
train_split = add_features(train_raw)
val_split   = add_features(val_raw)

feature_cols = [
    "haversine_km",
    "distance_km",
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

print("Training model for visualisation...")
model = pipeline.fit(train_split)

val_preds = model.transform(val_split)

rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse").evaluate(val_preds)
r2   = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2").evaluate(val_preds)

print(f"  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")

sample_pd = (
    val_preds
    .select("fare_amount", "prediction")
    .sample(fraction=0.005, seed=42)
    .limit(5000)
    .toPandas()
)

# Feature importances
gbt_model   = model.stages[-1]
importances = list(zip(feature_cols, gbt_model.featureImportances.toArray()))
importances.sort(key=lambda x: x[1])          # ascending so top bar is at the top

feat_names = [f for f, _ in importances]
feat_scores = [s for _, s in importances]

fig1, ax1 = plt.subplots(figsize=(9, 6))

colors = ["#1a6faf" if s == max(feat_scores) else "#5ba4cf" for s in feat_scores]
bars = ax1.barh(feat_names, feat_scores, color=colors, edgecolor="white", height=0.7)

for bar, score in zip(bars, feat_scores):
    ax1.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{score:.3f}",
        va="center", ha="left", fontsize=9, color="#333333"
    )

ax1.set_xlabel("Importance Score", fontsize=11)
ax1.set_title("GBT Feature Importances — Taxi Fare Prediction", fontsize=13, fontweight="bold", pad=14)
ax1.set_xlim(0, max(feat_scores) * 1.18)
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis="y", labelsize=10)
ax1.grid(axis="x", linestyle="--", alpha=0.4)

fig1.tight_layout()
chart1_path = str(output_dir / "gbt_feature_importances.png")
fig1.savefig(chart1_path, dpi=180, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {chart1_path}")

# Plot 2 - Predicted vs Actual
actual     = sample_pd["fare_amount"].values
predicted  = sample_pd["prediction"].values

# axis range: shared min/max with a small margin
lo = min(actual.min(), predicted.min())
hi = max(actual.max(), predicted.max())
margin = (hi - lo) * 0.04

fig2, ax2 = plt.subplots(figsize=(7, 7))

ax2.scatter(actual, predicted, alpha=0.25, s=18, color="#1a6faf", linewidths=0)

diag = np.linspace(lo - margin, hi + margin, 100)
ax2.plot(diag, diag, color="#e05c2d", linewidth=1.8, linestyle="--", label="Perfect prediction")

ax2.text(
    0.05, 0.93,
    f"RMSE = {rmse:.2f}\nR²   = {r2:.4f}",
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9)
)

ax2.set_xlim(lo - margin, hi + margin)
ax2.set_ylim(lo - margin, hi + margin)
ax2.set_xlabel("Actual Fare (USD)", fontsize=11)
ax2.set_ylabel("Predicted Fare (USD)", fontsize=11)
ax2.set_title("Predicted vs Actual Fare — GBT Validation Set", fontsize=13, fontweight="bold", pad=14)
ax2.legend(fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_aspect("equal")
ax2.grid(linestyle="--", alpha=0.35)

fig2.tight_layout()
chart2_path = str(output_dir / "gbt_predicted_vs_actual.png")
fig2.savefig(chart2_path, dpi=180, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {chart2_path}")

spark.stop()
