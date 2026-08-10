"""Batch feature engineering example for transaction risk using PySpark."""
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


def build_features(input_path: str, output_path: str) -> None:
    spark = SparkSession.builder.appName("banking-risk-features").getOrCreate()
    tx = spark.read.parquet(input_path)
    tx = tx.withColumn("event_time", F.to_timestamp("event_time"))
    tx = tx.withColumn("event_ts", F.col("event_time").cast("long"))

    w_24h = Window.partitionBy("customer_id").orderBy("event_ts").rangeBetween(-86400, 0)
    w_7d = Window.partitionBy("customer_id").orderBy("event_ts").rangeBetween(-7 * 86400, 0)

    features = (
        tx.withColumn("txn_count_24h", F.count("transaction_id").over(w_24h))
        .withColumn("amount_sum_24h", F.sum("amount").over(w_24h))
        .withColumn("amount_avg_7d", F.avg("amount").over(w_7d))
        .withColumn("amount_std_7d", F.stddev_pop("amount").over(w_7d))
        .withColumn("country_changed", F.when(F.lag("country_code").over(Window.partitionBy("customer_id").orderBy("event_time")) != F.col("country_code"), 1).otherwise(0))
        .fillna({"amount_std_7d": 0.0})
    )

    features.write.mode("overwrite").partitionBy("country_code").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_features(args.input, args.output)
