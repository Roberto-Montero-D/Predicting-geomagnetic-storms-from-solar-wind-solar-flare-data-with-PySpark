import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from processor import Processor
from pyspark.sql import SparkSession
from pyspark.sql.types import (StringType, StructType, StructField, IntegerType, DoubleType, TimestampType)
import math
from datetime import datetime

def test_encode_kp(spark_session: SparkSession):
    processor = Processor(spark_session)
    
    schema_input = StructType([
        StructField("id", IntegerType()),
        StructField("kp", DoubleType()),
    ])
    
    df_input = spark_session.createDataFrame(
        [(1, 30.0), (2, 50.0), (3, 60.0)],
        schema_input
    )
    
    result_df = processor.encode_kp(df_input)
    
    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("kp", DoubleType()),
        StructField("target", IntegerType()),
    ])
    
    expected_df = spark_session.createDataFrame(
        [(1, 3.0, 0), (2, 5.0, 0), (3, 6.0, 1)],
        expected_schema
    )
    
    assert result_df.orderBy("id").collect() == expected_df.orderBy("id").collect()
    
def test_filter_omni(spark_session: SparkSession):
    processor = Processor(spark_session)
    
    schema_input = StructType([
        StructField("id", IntegerType()),
        StructField("B_scalar", DoubleType()),
        StructField("Bx", DoubleType()),
        StructField("By", DoubleType()),
        StructField("Bz", DoubleType()),
        StructField("density", DoubleType()),
        StructField("speed", DoubleType()),
        StructField("temperature", DoubleType()),
        StructField("flow_pressure", DoubleType()),
        StructField("electric_field", DoubleType()),
        StructField("plasma_beta", DoubleType()),
        StructField("alfven_mach", DoubleType()),
        StructField("f107", DoubleType()),
    ])
    
    df_input = spark_session.createDataFrame(
        [
            (1, 100.0, 150.0, 120.0, 80.0, 300.0, 400.0, 500000.0, 30.0, 150.0, 400.0, 300.0, 200.0),  # Válidos
            (2, 99.99, 999.9, 9999.0, 99999.0, 9999999.0, 99999999.0, 100.0, 200.0, 50.0, 100.0, 150.0, 250.0),  # Hard flags
            (3, 250.0, 300.0, 250.0, 220.0, 600.0, 3500.0, 2000000.0, 60.0, 250.0, 600.0, 550.0, 600.0),  # Exceden thresholds
        ],
        schema_input
    )
    
    result_df = processor.filter_omni(df_input)
    
    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("B_scalar", DoubleType()),
        StructField("Bx", DoubleType()),
        StructField("By", DoubleType()),
        StructField("Bz", DoubleType()),
        StructField("density", DoubleType()),
        StructField("speed", DoubleType()),
        StructField("temperature", DoubleType()),
        StructField("flow_pressure", DoubleType()),
        StructField("electric_field", DoubleType()),
        StructField("plasma_beta", DoubleType()),
        StructField("alfven_mach", DoubleType()),
        StructField("f107", DoubleType()),
    ])
    
    expected_df = spark_session.createDataFrame(
        [
            (1, 100.0, 150.0, 120.0, 80.0, 300.0, 400.0, 500000.0, 30.0, 150.0, 400.0, 300.0, 200.0),
            (2, None, None, None, None, None, None, 100.0, None, 50.0, 100.0, 150.0, 250.0),
            (3, None, None, None, None, None, None, None, None, None, None, None, None),
        ],
        expected_schema
    )
    
    assert result_df.orderBy("id").collect() == expected_df.orderBy("id").collect()
    
def test_process_gbm_timestamps(spark_session):
    processor = Processor(spark_session)

    schema = StructType([
        StructField("flare_id", StringType()),
        StructField("start_date", StringType()),
        StructField("start_time", StringType()),
        StructField("duration_sec", IntegerType()),
        StructField("peak_counts", IntegerType()),
        StructField("total_counts", IntegerType()),
    ])

    df_input = spark_session.createDataFrame(
        [
            ("1", "2024-01-01", "12:00:00", 60, 100, 200),
            ("2", "1-Jan-2024", "13:00:00", 70, 150, 300),
        ],
        schema
    )

    result_df = processor.process_gbm_timestamps(df_input)

    expected_schema = StructType([
        StructField("flare_id", StringType()),
        StructField("start_date", StringType()),
        StructField("start_time", StringType()),
        StructField("duration_sec", IntegerType()),
        StructField("peak_counts", IntegerType()),
        StructField("total_counts", IntegerType()),
        StructField("timestamp", TimestampType()),
    ])

    expected_df = spark_session.createDataFrame(
        [
            (
                "1", "2024-01-01", "12:00:00",
                60, 100, 200,
                datetime(2024, 1, 1, 12, 0, 0)
            ),
            (
                "2", "1-Jan-2024", "13:00:00",
                70, 150, 300,
                datetime(2024, 1, 1, 13, 0, 0)
            ),
        ],
        expected_schema
    )

    assert result_df.orderBy("flare_id").collect() == \
           expected_df.orderBy("flare_id").collect()

from pyspark.sql.types import (
    StructType, StructField,
    TimestampType, IntegerType, DoubleType
)
from datetime import datetime


def test_gbm_group_by_hour(spark_session):
    processor = Processor(spark_session)

    schema = StructType([
        StructField("timestamp", TimestampType()),
        StructField("peak_counts", IntegerType()),
        StructField("total_counts", IntegerType()),
    ])

    df_input = spark_session.createDataFrame(
        [
            (datetime(2024, 1, 1, 12, 10), 100, 200),
            (datetime(2024, 1, 1, 12, 40), 200, 300),
            (datetime(2024, 1, 1, 13, 0),  50, 100),
        ],
        schema
    )

    result_df = processor.gbm_group_by_hour(df_input)

    expected_schema = StructType([
        StructField("timestamp", TimestampType()),
        StructField("flare_count", IntegerType()),
        StructField("avg_peak_counts", DoubleType()),
        StructField("total_energy_proxy", IntegerType()),
    ])

    expected_df = spark_session.createDataFrame(
        [
            (
                datetime(2024, 1, 1, 12, 0, 0),
                2,
                150.0,
                500
            ),
            (
                datetime(2024, 1, 1, 13, 0, 0),
                1,
                50.0,
                100
            ),
        ],
        expected_schema
    )

    assert result_df.orderBy("timestamp").collect() == \
           expected_df.orderBy("timestamp").collect()



def test_apply_lag_roll(spark_session: SparkSession):
    processor = Processor(spark_session)
    
    schema_input = StructType([
        StructField("id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("temperature", DoubleType()),
        StructField("speed", DoubleType()),
    ])
    
    df_input = spark_session.createDataFrame(
        [
            (1, datetime(2024, 1, 1, 0, 0), 10.0, 100.0),
            (2, datetime(2024, 1, 1, 1, 0), 20.0, 200.0),
            (3, datetime(2024, 1, 1, 2, 0), 30.0, 300.0),
            (4, datetime(2024, 1, 1, 3, 0), 40.0, 400.0),
            (5, datetime(2024, 1, 1, 4, 0), 50.0, 500.0),
        ],
        schema_input
    )
    
    feature_config = {
        "temperature": {
            "lags": [1, 2],
            "roll_mean": [2],
            "roll_min": [2],
            "roll_max": [2],
            "roll_sum": [2]
        }
    }
    
    result_df = processor.apply_lag_roll(df_input, feature_config)
    
    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("temperature", DoubleType()),
        StructField("speed", DoubleType()),
        StructField("temperature_lag_1", DoubleType()),
        StructField("temperature_lag_2", DoubleType()),
        StructField("avg_temperature_2", DoubleType()),
        StructField("min_temperature_2", DoubleType()),
        StructField("max_temperature_2", DoubleType()),
        StructField("sum_temperature_2", DoubleType()),
    ])
    
    expected_df = spark_session.createDataFrame(
        [
            (1, datetime(2024, 1, 1, 0, 0), 10.0, 100.0, None, None, None, None, None, None),
            (2, datetime(2024, 1, 1, 1, 0), 20.0, 200.0, 10.0, None, 10.0, 10.0, 10.0, 10.0),
            (3, datetime(2024, 1, 1, 2, 0), 30.0, 300.0, 20.0, 10.0, 15.0, 10.0, 20.0, 30.0),
            (4, datetime(2024, 1, 1, 3, 0), 40.0, 400.0, 30.0, 20.0, 25.0, 20.0, 30.0, 50.0),
            (5, datetime(2024, 1, 1, 4, 0), 50.0, 500.0, 40.0, 30.0, 35.0, 30.0, 40.0, 70.0),
        ],
        expected_schema
    )
    
    assert result_df.orderBy("id").collect() == expected_df.orderBy("id").collect()





def test_apply_forward_fill(spark_session: SparkSession):
    processor = Processor(spark_session)
    
    schema_input = StructType([
        StructField("id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("value1", DoubleType()),
        StructField("value2", DoubleType()),
    ])
    
    df_input = spark_session.createDataFrame(
        [
            (1, datetime(2024, 1, 1, 0, 0), 10.0, 100.0),
            (2, datetime(2024, 1, 1, 1, 0), None, 200.0),
            (3, datetime(2024, 1, 1, 2, 0), 30.0, None),
            (4, datetime(2024, 1, 1, 3, 0), None, None),
        ],
        schema_input
    )
    
    result_df = processor.apply_forward_fill(df_input, ["value1", "value2"])
    
    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("value1", DoubleType()),
        StructField("value2", DoubleType()),
    ])
    
    expected_df = spark_session.createDataFrame(
        [
            (1, datetime(2024, 1, 1, 0, 0), 10.0, 100.0),
            (2, datetime(2024, 1, 1, 1, 0), 10.0, 200.0),
            (3, datetime(2024, 1, 1, 2, 0), 30.0, 200.0),
            (4, datetime(2024, 1, 1, 3, 0), 30.0, 200.0),
        ],
        expected_schema
    )
    
    assert result_df.orderBy("id").collect() == expected_df.orderBy("id").collect()

def test_cyclical_encoding(spark_session: SparkSession):
    processor = Processor(spark_session)
    
    schema_input = StructType([
        StructField("id", IntegerType()),
        StructField("hour", IntegerType()),
    ])
    
    df_input = spark_session.createDataFrame(
        [(1, 0), (2, 6), (3, 12), (4, 23)],
        schema_input
    )
    
    result_df = processor.cyclical_encoding(df_input, "hour", 24)
    
    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("hour", IntegerType()),
        StructField("hour_sin", DoubleType()),
        StructField("hour_cos", DoubleType()),
    ])
    
    expected_df = spark_session.createDataFrame(
        [
            (1, 0, 0.0, 1.0),
            (2, 6, 1.0, 0.0),
            (3, 12, 0.0, -1.0),
            (4, 23, math.sin(2 * math.pi * 23 / 24), math.cos(2 * math.pi * 23 / 24)),
        ],
        expected_schema
    )
    
    result_rows = result_df.orderBy("id").collect()
    expected_rows = expected_df.orderBy("id").collect()
    
    for r, e in zip(result_rows, expected_rows):
        assert r.id == e.id
        assert r.hour == e.hour
        assert abs(r.hour_sin - e.hour_sin) < 1e-6
        assert abs(r.hour_cos - e.hour_cos) < 1e-6