from pathlib import Path
from pyspark.sql import SparkSession, functions as F
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parents[1]
train_path = str(base_dir / "data" / "raw" / "train.csv")
test_path = str(base_dir / "data" / "raw" / "test.csv")
output_dir = base_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("taxi-data-cleaning")
    .master("local[*]")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
    .config("spark.sql.warehouse.dir", str(base_dir / "spark-warehouse"))
    .getOrCreate()
)

train_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(train_path)
)

test_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(test_path)
)

print("Train schema:")
train_df.printSchema()

print("Test schema:")
test_df.printSchema()

available_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

train_fixed = train_df.select(*[c for c in available_cols if c in train_df.columns])

numeric_cols = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

for col_name in numeric_cols:
    if col_name in train_fixed.columns:
        train_fixed = train_fixed.withColumn(col_name, F.col(col_name).cast("double"))

if "passenger_count" in train_fixed.columns:
    train_fixed = train_fixed.withColumn("passenger_count", F.col("passenger_count").cast("int"))

train_fixed = (
    train_fixed
    .withColumn("key", F.trim(F.col("key")))
    .withColumn("pickup_datetime_raw", F.trim(F.col("pickup_datetime")))
    .withColumn(
        "pickup_datetime",
        F.coalesce(
            F.to_timestamp("pickup_datetime_raw"),
            F.to_timestamp("pickup_datetime_raw", "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp("pickup_datetime_raw", "yyyy-MM-dd HH:mm:ss.SSS")
        )
    )
    .drop("pickup_datetime_raw")
)

print("Missing values before cleaning:")
train_fixed.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in train_fixed.columns
]).show(truncate=False)

required_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

train_clean = train_fixed.dropna(subset=required_cols)

if "fare_amount" in train_clean.columns:
    fare_values = train_clean.filter(F.col("fare_amount").isNotNull())
    if fare_values.count() > 0:
        fare_median = fare_values.approxQuantile("fare_amount", [0.5], 0.01)[0]
        train_clean = train_clean.fillna({"fare_amount": float(fare_median)})

# initial_rows = train_clean.count()
train_clean = train_clean.dropDuplicates([
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
])
# deduped_rows = train_clean.count()
# print(f"Duplicates removed: {initial_rows - deduped_rows:,}")

train_clean = (
    train_clean
    .filter(F.col("passenger_count").between(1, 6))
    .filter(F.col("pickup_latitude").between(40.0, 43.0))
    .filter(F.col("dropoff_latitude").between(40.0, 43.0))
    .filter(F.col("pickup_longitude").between(-75.0, -72.0))
    .filter(F.col("dropoff_longitude").between(-75.0, -72.0))
)

train_clean = train_clean.withColumn(
    "distance_km",
    111 * F.sqrt(
        F.pow(F.col("pickup_longitude") - F.col("dropoff_longitude"), 2) +
        F.pow(F.col("pickup_latitude") - F.col("dropoff_latitude"), 2)
    )
)

if "fare_amount" in train_clean.columns:
    train_clean = train_clean.filter(F.col("fare_amount") > 0)

distance_q1, distance_q3 = train_clean.approxQuantile("distance_km", [0.25, 0.75], 0.01)
distance_iqr = distance_q3 - distance_q1
distance_lower = max(0, distance_q1 - 1.5 * distance_iqr)
distance_upper = distance_q3 + 2 * distance_iqr

train_clean = train_clean.withColumn(
    "distance_outlier",
    ~F.col("distance_km").between(distance_lower, distance_upper)
)

if "fare_amount" in train_clean.columns:
    fare_q1, fare_q3 = train_clean.approxQuantile("fare_amount", [0.25, 0.75], 0.01)
    fare_iqr = fare_q3 - fare_q1
    fare_lower = max(0, fare_q1 - 1.5 * fare_iqr)
    fare_upper = fare_q3 + 1.5 * fare_iqr

    train_clean = train_clean.withColumn(
        "fare_outlier",
        ~F.col("fare_amount").between(fare_lower, fare_upper)
    )

    clean_final = train_clean.filter(~F.col("distance_outlier") & ~F.col("fare_outlier"))
else:
    clean_final = train_clean.filter(~F.col("distance_outlier"))

clean_final = (
    clean_final
    .withColumn("passenger_count", F.col("passenger_count").cast("int"))
    .withColumn("pickup_year", F.year("pickup_datetime"))
    .withColumn("pickup_month", F.month("pickup_datetime"))
    .withColumn("pickup_day", F.dayofmonth("pickup_datetime"))
    .withColumn("pickup_hour", F.hour("pickup_datetime"))
)

# print("Schema after cleaning:")
# clean_final.printSchema()
#
# print("Missing values after cleaning:")
# clean_final.select([
#     F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in clean_final.columns
# ]).show(truncate=False)

normal_data = train_clean.filter(~F.col("distance_outlier"))
outlier_data = train_clean.filter(F.col("distance_outlier"))

print(f"Normal rows: {normal_data.count():,}")
print(f"Outlier rows: {outlier_data.count():,}")
print(f"Distance bounds: {distance_lower:.2f} km - {distance_upper:.2f} km")

normal_pd = normal_data.sample(False, 0.05, seed=42).select("distance_km").toPandas()
outlier_pd = outlier_data.sample(False, 0.40, seed=42).select("distance_km").toPandas()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
#
# ax1.hist(
#     normal_pd["distance_km"],
#     bins=50,
#     color="blue",
#     alpha=0.7,
#     density=True,
#     edgecolor="darkblue",
#     linewidth=1,
#     label="Normal distribution"
# )
# ax1.axvline(distance_lower, color="green", ls="--", lw=2, label=f"Lower: {distance_lower:.1f}")
# ax1.axvline(distance_upper, color="green", ls="--", lw=2, label=f"Upper: {distance_upper:.1f}")
# ax1.set_xlabel("Distance (km)")
# ax1.set_ylabel("Density")
# ax1.set_title("Normal Trips Distribution")
# ax1.legend()
# ax1.grid(True, alpha=0.3)
#
# ax2.scatter(
#     normal_pd.index,
#     normal_pd["distance_km"],
#     color="lightblue",
#     alpha=0.5,
#     s=20,
#     label=f"Normal ({len(normal_pd):,})"
# )
# ax2.scatter(
#     outlier_pd.index,
#     outlier_pd["distance_km"],
#     color="red",
#     alpha=0.9,
#     s=60,
#     edgecolors="darkred",
#     linewidth=1.5,
#     label=f"Outliers ({len(outlier_pd):,})",
#     zorder=10
# )
# ax2.axhline(distance_lower, color="green", ls="--", lw=2, alpha=0.8)
# ax2.axhline(distance_upper, color="green", ls="--", lw=2, alpha=0.8)
# ax2.set_xlabel("Index")
# ax2.set_ylabel("Distance (km)")
# ax2.set_title("Distance Outliers")
# ax2.legend()
# ax2.grid(True, alpha=0.3)
#
# plt.tight_layout()
# chart_path = output_dir / "distance_outliers.png"
# plt.savefig(chart_path, dpi=200, bbox_inches="tight")
# plt.close()

parquet_path = str(output_dir / "train_cleaned.parquet")

clean_final = clean_final.coalesce(2)
clean_final.write.mode("overwrite").option("compression", "snappy").parquet(parquet_path)

# print(f"Chart saved to: {chart_path}")
print(f"Cleaned parquet saved to: {parquet_path}")

spark.stop()