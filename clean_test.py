from pathlib import Path
from pyspark.sql import SparkSession, functions as F

base_dir = Path(__file__).resolve().parent
test_path = str(base_dir / "raw data" / "test.csv")
output_dir = base_dir / "output"
output_dir.mkdir(exist_ok=True)

spark = (
    SparkSession.builder
    .appName("taxi-test-data-cleaning")
    .master("local[*]")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
    .config("spark.sql.warehouse.dir", str(base_dir / "spark-warehouse"))
    .getOrCreate()
)

test_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(test_path)
)

print("Test schema:")
test_df.printSchema()

available_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

numeric_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

required_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

test_fixed = test_df.select(*[c for c in available_cols if c in test_df.columns])

for col_name in numeric_cols:
    if col_name in test_fixed.columns:
        test_fixed = test_fixed.withColumn(col_name, F.col(col_name).cast("double"))

if "passenger_count" in test_fixed.columns:
    test_fixed = test_fixed.withColumn("passenger_count", F.col("passenger_count").cast("int"))

test_fixed = (
    test_fixed
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
test_fixed.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in test_fixed.columns
]).show(truncate=False)

test_clean = test_fixed.dropna(subset=required_cols)

test_clean = test_clean.dropDuplicates([
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
])

test_clean = (
    test_clean
    .filter(F.col("passenger_count").between(1, 6))
    .filter(F.col("pickup_latitude").between(40.0, 43.0))
    .filter(F.col("dropoff_latitude").between(40.0, 43.0))
    .filter(F.col("pickup_longitude").between(-75.0, -72.0))
    .filter(F.col("dropoff_longitude").between(-75.0, -72.0))
)

test_clean = test_clean.withColumn(
    "distance_km",
    111 * F.sqrt(
        F.pow(F.col("pickup_longitude") - F.col("dropoff_longitude"), 2) +
        F.pow(F.col("pickup_latitude") - F.col("dropoff_latitude"), 2)
    )
)

distance_q1, distance_q3 = test_clean.approxQuantile("distance_km", [0.25, 0.75], 0.01)
distance_iqr = distance_q3 - distance_q1
distance_lower = max(0, distance_q1 - 1.5 * distance_iqr)
distance_upper = distance_q3 + 2 * distance_iqr

test_clean = test_clean.withColumn(
    "distance_outlier",
    ~F.col("distance_km").between(distance_lower, distance_upper)
)

clean_final = test_clean.filter(~F.col("distance_outlier"))

clean_final = (
    clean_final
    .withColumn("passenger_count", F.col("passenger_count").cast("int"))
    .withColumn("pickup_year", F.year("pickup_datetime"))
    .withColumn("pickup_month", F.month("pickup_datetime"))
    .withColumn("pickup_day", F.dayofmonth("pickup_datetime"))
    .withColumn("pickup_hour", F.hour("pickup_datetime"))
)

print(f"Clean test rows: {clean_final.count():,}")
print(f"Distance bounds: {distance_lower:.2f} km - {distance_upper:.2f} km")

parquet_path = str(output_dir / "test_cleaned.parquet")

clean_final = clean_final.coalesce(2)
clean_final.write.mode("overwrite").option("compression", "snappy").parquet(parquet_path)

print(f"Cleaned parquet saved to: {parquet_path}")

spark.stop()