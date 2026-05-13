import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from reader import Reader
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, TimestampType
)
from datetime import datetime


def test_load_omni(spark_session, tmp_path):
    base_path = tmp_path
    file_path = base_path / "omni.lst"

    file_path.write_text(
        "2025 5 23 4.9 -2.1 2.2 -0.1 61874. 5.4 378. 1.31 0.13 1.75 9.4 40 10 77.7\n"
    )

    reader = Reader(spark_session, base_path=str(base_path))
    result_df = reader.load_omni()

    expected_schema = StructType([
        StructField("year", IntegerType()),
        StructField("doy", IntegerType()),
        StructField("hour", IntegerType()),
        StructField("B_scalar", DoubleType()),
        StructField("Bx", DoubleType()),
        StructField("By", DoubleType()),
        StructField("Bz", DoubleType()),
        StructField("temperature", DoubleType()),
        StructField("density", DoubleType()),
        StructField("speed", DoubleType()),
        StructField("flow_pressure", DoubleType()),
        StructField("electric_field", DoubleType()),
        StructField("plasma_beta", DoubleType()),
        StructField("alfven_mach", DoubleType()),
        StructField("kp", DoubleType()),
        StructField("sunspots", IntegerType()),
        StructField("f107", DoubleType()),
        StructField("timestamp", TimestampType()),
    ])

    expected_df = spark_session.createDataFrame(
        [
            (
                2025, 5, 23,
                4.9, -2.1, 2.2, -0.1,
                61874.0, 5.4, 378.0,
                1.31, 0.13, 1.75, 9.4,
                40.0, 10, 77.7,
                datetime(2025, 1, 5, 23, 0, 0)
            )
        ],
        expected_schema
    )

    assert result_df.collect() == expected_df.collect()

from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType
)


def test_load_gbm(spark_session, tmp_path):
    base_path = tmp_path
    file_path = base_path / "gbm_flare_list.txt"

    file_path.write_text(
        """Fermi GBM Solar Flare List
1234 2024-01-01 12:00:00 X X 60 100 200
Total something
"""
    )

    reader = Reader(spark_session, base_path=str(base_path))
    result_df = reader.load_gbm()

    expected_schema = StructType([
        StructField("flare_id", StringType()),
        StructField("start_date", StringType()),
        StructField("start_time", StringType()),
        StructField("duration_sec", IntegerType()),
        StructField("peak_counts", IntegerType()),
        StructField("total_counts", IntegerType()),
    ])

    expected_df = spark_session.createDataFrame(
        [
            ("1234", "2024-01-01", "12:00:00", 60, 100, 200)
        ],
        expected_schema
    )

    assert result_df.collect() == expected_df.collect()