"""Optional PySpark batch analytics for an exported BDSMA JSONL event file.

Run from the project root:
    spark-submit bdsma/pyspark_job.py data/bdsma_events.jsonl
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main(path: str) -> None:
    spark = (SparkSession.builder.appName("SupplyChainBDSMAAnalytics")
             .master("local[*]").getOrCreate())
    events = spark.read.json(path)
    print("\n=== Event type counts ===")
    events.groupBy("event_type").count().orderBy(F.desc("count")).show(truncate=False)

    print("\n=== Product-wise sales aggregation ===")
    (events.filter(F.col("event_type") == "sale")
     .groupBy("product_id", "product_name")
     .agg(F.sum("units_sold").alias("total_units"),
          F.round(F.sum("revenue"), 2).alias("total_revenue"))
     .orderBy(F.desc("total_units")).show(20, truncate=False))

    print("\n=== Warehouse-wise sales aggregation ===")
    (events.filter(F.col("event_type") == "sale")
     .groupBy("warehouse_id")
     .agg(F.sum("units_sold").alias("total_units"),
          F.round(F.sum("revenue"), 2).alias("total_revenue"))
     .orderBy(F.desc("total_revenue")).show(truncate=False))

    print("\n=== Customer feedback sentiment ===")
    (events.filter(F.col("event_type") == "customer_feedback")
     .groupBy("sentiment", "topic").count()
     .orderBy(F.desc("count")).show(truncate=False))
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: spark-submit bdsma/pyspark_job.py <events.jsonl>")
    main(sys.argv[1])
