from pyspark.sql import DataFrame
from pyspark.sql.functions import col


class Splitter:
    """
    Time-based train/validation/test split for time-series data.

    A random split must NOT be used here because it would cause data leakage:
    training with future data and predicting the past inflates metrics artificially.
    Instead, data is partitioned chronologically so the model always predicts
    genuinely unseen future observations.

    Split ranges:
        train : 2008-01-01 – 2019-12-31
        val   : 2020-01-01 – 2021-12-31
        test  : 2022-01-01 – present
    """

    TRAIN_END = "2020-01-01"
    VAL_END   = "2022-01-01"

    def split(self, df: DataFrame, timestamp_col: str = "timestamp"):
        train = df.filter(col(timestamp_col) < self.TRAIN_END)
        val   = df.filter(
            (col(timestamp_col) >= self.TRAIN_END) &
            (col(timestamp_col) <  self.VAL_END)
        )
        test  = df.filter(col(timestamp_col) >= self.VAL_END)
        return train, val, test
