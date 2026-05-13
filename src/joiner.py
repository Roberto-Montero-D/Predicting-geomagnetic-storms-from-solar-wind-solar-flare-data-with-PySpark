from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import (
    col, split, trim, format_string, to_timestamp,
    date_trunc, count, avg, sum as spark_sum,
    when, concat_ws, size, get
)

class Joiner:
    def __init__(self):
        pass
        
    def joinLeftDataset(self, dataset1, dataset2, column):
        return dataset1.join(dataset2, on=column, how='left')

