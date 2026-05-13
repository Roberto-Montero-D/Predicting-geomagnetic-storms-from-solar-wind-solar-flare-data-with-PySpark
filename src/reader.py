from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import (
    col, split, trim, format_string, to_timestamp,
    date_trunc, count, avg, sum as spark_sum,
    when, concat_ws, size, get
)

class Reader:
    def __init__(self, spark: SparkSession, base_path: str):
        self.spark = spark
        self.base_path = base_path

    def load_omni(self):
        raw = self.spark.read.text(f"{self.base_path}/omni.lst")

        omni = (
            raw.select(split(trim(col("value")), r"\s+").alias("cols"))
            .select(
                col("cols")[0].cast("int").alias("year"),
                col("cols")[1].cast("int").alias("doy"),
                col("cols")[2].cast("int").alias("hour"),
                col("cols")[3].cast("double").alias("B_scalar"),
                col("cols")[4].cast("double").alias("Bx"),
                col("cols")[5].cast("double").alias("By"),
                col("cols")[6].cast("double").alias("Bz"),
                col("cols")[7].cast("double").alias("temperature"),
                col("cols")[8].cast("double").alias("density"),
                col("cols")[9].cast("double").alias("speed"),
                col("cols")[10].cast("double").alias("flow_pressure"),
                col("cols")[11].cast("double").alias("electric_field"),
                col("cols")[12].cast("double").alias("plasma_beta"),
                col("cols")[13].cast("double").alias("alfven_mach"),
                col("cols")[14].cast("double").alias("kp"),
                col("cols")[15].cast("int").alias("sunspots"),
                col("cols")[16].cast("double").alias("f107")
            )
            .withColumn(
                "timestamp",
                to_timestamp(
                    format_string(
                        "%04d-%03d %02d:00:00",
                        col("year"), col("doy"), col("hour")
                    ),
                    "yyyy-DDD HH:mm:ss"
                )
            )
        )
        return omni
    
    

    def load_gbm(self):
        raw = self.spark.read.text(f"{self.base_path}/gbm_flare_list.txt")

        gbm = (
            raw.withColumn("line", trim(col("value")))
            .filter(~col("line").startswith("Fermi"))
            .filter(~col("line").startswith("Total"))
            .filter(~col("line").startswith("Flare"))
            .filter(col("line") != "")
            .withColumn("tokens", split(col("line"), r"\s+"))
            .filter(size(col("tokens")) >= 8)
            .select(
                get(col("tokens"), 0).alias("flare_id"),
                get(col("tokens"), 1).alias("start_date"),
                get(col("tokens"), 2).alias("start_time"),
                get(col("tokens"), 5).cast("int").alias("duration_sec"), 
                get(col("tokens"), 6).cast("int").alias("peak_counts"), 
                get(col("tokens"), 7).cast("int").alias("total_counts")
            )
        )
        return gbm

    