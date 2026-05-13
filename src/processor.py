from pyspark.sql import SparkSession
import os
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, split, trim, format_string, to_timestamp,
    date_trunc, count, avg, sum as spark_sum,
    when, concat_ws, size, get
)
from pyspark.sql.window import Window

class Processor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    
    def encode_kp(self, df):
        df = df.withColumn("kp", F.col("kp") / 10)
        df = df.withColumn(
            "target",
            when(col("kp") <= 5, 0).otherwise(1)
        )
        return df

    def filter_omni(self, omni):
        hard_flags = [
            99.99, 999.9, 9999.0, 99999.0,
            9999999.0, 99999999.0
        ]

        soft_flag_thresholds = {
            "B_scalar": 200,
            "Bx": 200,
            "By": 200,
            "Bz": 200,
            "density": 500,
            "speed": 3000,
            "temperature": 1e6,
            "flow_pressure": 50,
            "electric_field": 200,
            "plasma_beta": 500,
            "alfven_mach": 500,
            "f107": 500
        }
        
        for c, max_phys in soft_flag_thresholds.items():
            omni = omni.withColumn(
                c,
                when(
                    (col(c).isin(hard_flags)) | (col(c) > max_phys),
                    None
                ).otherwise(col(c))
            )

        return omni

        
    def process_gbm_timestamps(self, gbm):
        gbm = (
            gbm.withColumn(
                "timestamp",
                when(
                    col("start_date").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}$"),
                    to_timestamp(
                        concat_ws(" ", "start_date", "start_time"),
                        "yyyy-MM-dd HH:mm:ss"
                    )
                ).when(
                    col("start_date").rlike("^[0-9]{1,2}-[A-Za-z]{3}-[0-9]{4}$"),
                    to_timestamp(
                        concat_ws(" ", "start_date", "start_time"),
                        "d-MMM-yyyy HH:mm:ss"
                    )
                )
            )
            .filter(col("timestamp").isNotNull())
        )
        return gbm

    def gbm_group_by_hour(self, gbm):
        gbm_hourly = (
            gbm.withColumn("timestamp", date_trunc("hour", col("timestamp")))
            .groupBy("timestamp")
            .agg(
                count("*").alias("flare_count"),
                avg("peak_counts").alias("avg_peak_counts"),
                spark_sum("total_counts").alias("total_energy_proxy")
            )
        )

        return gbm_hourly
    
    def add_lagged_windows(self, df, feature, config, window):
       
        # LAGS
        if "lags" in config:
            for n in config["lags"]:
                df = df.withColumn(
                    f"{feature}_lag_{n}",
                    F.lag(feature, n).over(window)
                )

        # ROLLING MEAN
        if "roll_mean" in config:
            for n in config["roll_mean"]:
                df = df.withColumn(
                    f"avg_{feature}_{n}",
                    F.avg(feature).over(window.rowsBetween(-n, -1))
                )

        # ROLLING MIN
        if "roll_min" in config:
            for n in config["roll_min"]:
                df = df.withColumn(
                    f"min_{feature}_{n}",
                    F.min(feature).over(window.rowsBetween(-n, -1))
                )

        # ROLLING MAX
        if "roll_max" in config:
            for n in config["roll_max"]:
                df = df.withColumn(
                    f"max_{feature}_{n}",
                    F.max(feature).over(window.rowsBetween(-n, -1))
                )

        # ROLLING SUM
        if "roll_sum" in config:
            for n in config["roll_sum"]:
                df = df.withColumn(
                    f"sum_{feature}_{n}",
                    F.sum(feature).over(window.rowsBetween(-n, -1))
                )

        return df


    def apply_lag_roll(self, df, feature_config):

        window = Window.orderBy("timestamp")

        for feature, config in feature_config.items():
            df = self.add_lagged_windows(
                df=df,
                feature=feature,
                config=config,
                window=window
            )

        return df
    
    def apply_forward_fill(self, df, cols):
        window = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0) # 2000
        for col in cols:
            df = df.withColumn(
                col,
                F.last(col, ignorenulls=True).over(window)
            )
        return df


    def cyclical_encoding(self, df, col_name: str, period: int):
        df = df.withColumn(
            f"{col_name}_sin",
            F.sin(2 * F.pi() * F.col(col_name) / F.lit(period))
        ).withColumn(
            f"{col_name}_cos",
            F.cos(2 * F.pi() * F.col(col_name) / F.lit(period))
        )
        return df
