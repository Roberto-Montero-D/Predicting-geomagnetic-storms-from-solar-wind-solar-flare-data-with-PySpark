import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from evaluator import Evaluator


def test_perfect_predictions(spark_session: SparkSession):
    evaluator = Evaluator()

    schema = StructType([
        StructField("target",     IntegerType()),
        StructField("prediction", DoubleType()),
    ])
    df = spark_session.createDataFrame(
        [(1, 1.0), (1, 1.0), (0, 0.0), (0, 0.0)],
        schema,
    )

    m = evaluator.compute_metrics(df)
    assert m["f1"]        == 1.0
    assert m["precision"] == 1.0
    assert m["recall"]    == 1.0
    assert m["accuracy"]  == 1.0


def test_all_false_negatives(spark_session: SparkSession):
    """Model never predicts positive — recall must be 0, F1 must be 0."""
    evaluator = Evaluator()

    schema = StructType([
        StructField("target",     IntegerType()),
        StructField("prediction", DoubleType()),
    ])
    df = spark_session.createDataFrame(
        [(1, 0.0), (1, 0.0), (0, 0.0)],
        schema,
    )

    m = evaluator.compute_metrics(df)
    assert m["recall"] == 0.0
    assert m["f1"]     == 0.0


def test_imbalanced_but_correct(spark_session: SparkSession):
    """10:1 ratio, all correctly classified."""
    evaluator = Evaluator()

    schema = StructType([
        StructField("target",     IntegerType()),
        StructField("prediction", DoubleType()),
    ])
    rows = [(0, 0.0)] * 10 + [(1, 1.0)]
    df = spark_session.createDataFrame(rows, schema)

    m = evaluator.compute_metrics(df)
    assert m["f1"]       == 1.0
    assert m["accuracy"] == 1.0
