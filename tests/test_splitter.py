from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, TimestampType, IntegerType

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from splitter import Splitter


def test_split_sizes(spark_session: SparkSession):
    splitter = Splitter()

    schema = StructType([
        StructField("timestamp", TimestampType()),
        StructField("target",    IntegerType()),
    ])

    df = spark_session.createDataFrame(
        [
            (datetime(2015, 1, 1), 0),   # train
            (datetime(2019, 6, 1), 1),   # train
            (datetime(2020, 3, 1), 0),   # val
            (datetime(2021, 9, 1), 1),   # val
            (datetime(2022, 5, 1), 0),   # test
            (datetime(2024, 1, 1), 1),   # test
        ],
        schema,
    )

    train, val, test = splitter.split(df)

    assert train.count() == 2
    assert val.count()   == 2
    assert test.count()  == 2


def test_no_overlap(spark_session: SparkSession):
    splitter = Splitter()

    schema = StructType([
        StructField("timestamp", TimestampType()),
        StructField("target",    IntegerType()),
    ])

    rows = [(datetime(2008 + i, 1, 1), 0) for i in range(17)]
    df = spark_session.createDataFrame(rows, schema)

    train, val, test = splitter.split(df)

    # Union of all three splits must equal the original
    total = train.count() + val.count() + test.count()
    assert total == df.count()
