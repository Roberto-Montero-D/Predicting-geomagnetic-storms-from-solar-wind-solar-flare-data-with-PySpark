import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from joiner import Joiner

from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField,IntegerType, StringType, FloatType, DateType)

def test_basic_join(spark_session: SparkSession):

    joiner = Joiner()

    schema_df1 = StructType([
        StructField("id", IntegerType()),
        StructField("value", StringType()),
    ])

    schema_df2 = StructType([
        StructField("id", IntegerType()),
        StructField("value2", StringType()),
    ])

    df1 = spark_session.createDataFrame(
        [(1, 'A'), (2, 'B')],
        schema_df1
    )

    df2 = spark_session.createDataFrame(
        [(1, 'X'), (2, 'Y')],
        schema_df2
    )

    join_df = joiner.joinLeftDataset(df1, df2, 'id')

    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("value", StringType()),
        StructField("value2", StringType()),
    ])

    expected_df = spark_session.createDataFrame(
        [(1, 'A', 'X'), (2, 'B', 'Y')],
        expected_schema
    )

    assert join_df.orderBy("id").collect() == expected_df.orderBy("id").collect()


def test_left_join_missing_data(spark_session: SparkSession):

    joiner = Joiner()

    schema_df1 = StructType([
        StructField("id", IntegerType()),
        StructField("value", StringType()),
    ])

    schema_df2 = StructType([
        StructField("id", IntegerType()),
        StructField("value2", StringType()),
    ])

    df1 = spark_session.createDataFrame(
        [(1, 'A'), (2, 'B'), (3, 'C')],
        schema_df1
    )

    df2 = spark_session.createDataFrame(
        [(1, 'X'), (2, 'Y')],
        schema_df2
    )

    join_df = joiner.joinLeftDataset(df1, df2, 'id')

    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("value", StringType()),
        StructField("value2", StringType()),
    ])

    expected_df = spark_session.createDataFrame(
        [(1, 'A', 'X'), (2, 'B', 'Y'), (3, 'C', None)],
        expected_schema
    )

    assert join_df.orderBy("id").collect() == expected_df.orderBy("id").collect()


def test_left_join_missing_data_right(spark_session: SparkSession):

    joiner = Joiner()

    schema_df1 = StructType([
        StructField("id", IntegerType()),
        StructField("value", StringType()),
    ])

    schema_df2 = StructType([
        StructField("id", IntegerType()),
        StructField("value2", StringType()),
    ])

    df1 = spark_session.createDataFrame(
        [(1, 'A')],
        schema_df1
    )

    df2 = spark_session.createDataFrame(
        [(1, 'X'), (2, 'Y')],
        schema_df2
    )

    join_df = joiner.joinLeftDataset(df1, df2, 'id')
    
    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("value", StringType()),
        StructField("value2", StringType()),
    ])

    expected_df = spark_session.createDataFrame(
        [(1, 'A', 'X')],
        expected_schema
    )

    assert join_df.orderBy("id").collect() == expected_df.orderBy("id").collect()
