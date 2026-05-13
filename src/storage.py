import os
from pyspark.sql import DataFrame, SparkSession


class ParquetStorage:
    """
    Replaces the PostgreSQL connector with local Parquet files.

    Parquet is the standard columnar format used in production Spark pipelines
    (data lakes, Databricks, AWS Glue, etc.).  It is faster to read/write than
    CSV, stores schema information, and enables predicate pushdown — making it
    a better portfolio showcase than a local RDBMS dependency.

    Directory layout written to base_dir:
        base_dir/
            omni/           ← raw OMNI solar wind data after ETL
            gbm/            ← raw GBM flare data after ETL
            geomagstorm/    ← joined, feature-engineered dataset
    """

    def __init__(self, base_dir: str = "data/processed"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, table: str) -> str:
        return os.path.join(self.base_dir, table)

    def write(self, df: DataFrame, table: str, mode: str = "overwrite") -> None:
        path = self._path(table)
        df.write.mode(mode).parquet(path)
        print(f"[Storage] Wrote '{table}' → {path}")

    def read(self, spark: SparkSession, table: str) -> DataFrame:
        path = self._path(table)
        df = spark.read.parquet(path)
        print(f"[Storage] Read '{table}' ← {path}  ({df.count()} rows)")
        return df

    def exists(self, table: str) -> bool:
        return os.path.isdir(self._path(table))
