import itertools
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

from evaluator import Evaluator


FEATURE_COLS = [
    "B_scalar", "Bx", "By", "Bz",
    "temperature", "density", "speed",
    "flow_pressure", "electric_field",
    "plasma_beta", "alfven_mach",
    "sunspots", "f107",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    "flare_count", "total_energy_proxy",
    "Bz_lag_1", "Bz_lag_3", "Bz_lag_6", "Bz_lag_12", "Bz_lag_24",
    "avg_Bz_3", "avg_Bz_6", "avg_Bz_12",
    "min_Bz_3", "min_Bz_6", "min_Bz_12",
    "speed_lag_1", "speed_lag_3", "speed_lag_6",
    "avg_speed_6", "max_speed_6",
    "electric_field_lag_1",
    "avg_electric_field_6", "max_electric_field_6",
    "kp_lag_1", "kp_lag_2", "kp_lag_3", "max_kp_6",
    "sum_flare_count_6",
    "avg_total_energy_proxy_6",
]


class Trainer:
    """
    Trains Random Forest and Gradient Boosted Tree classifiers.

    Hyperparameter search is implemented manually (no CrossValidator) because
    Spark's CrossValidator shuffles data randomly, which would mix past and
    future observations and cause data leakage on time-series data.
    Instead we evaluate each parameter combination on the chronological
    validation set and keep the model with the best F1 score.

    Class imbalance (~2.3% positive): accuracy is not a reliable metric here.
    F1 is used as the selection criterion to balance precision and recall.
    """

    def __init__(self):
        self.evaluator = Evaluator()

    def vectorize(self, df: DataFrame, feature_cols: list = None) -> DataFrame:
        cols = feature_cols or FEATURE_COLS
        assembler = VectorAssembler(inputCols=cols, outputCol="features")
        return assembler.transform(df).select("features", "target", "timestamp")

    def _tune(
        self,
        estimator_class,
        param_grid: dict,
        base_params: dict,
        train_df: DataFrame,
        val_df: DataFrame,
    ):
        best_f1    = float("-inf")
        best_model = None
        best_params = None

        keys   = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))

        for combo in combos:
            params     = dict(zip(keys, combo))
            full_params = {**base_params, **params}
            model  = estimator_class(**full_params).fit(train_df)
            preds  = model.transform(val_df)
            metrics = self.evaluator.compute_metrics(preds)
            f1 = metrics["f1"]
            print(f"  {params} → F1={f1:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

            if f1 > best_f1:
                best_f1     = f1
                best_model  = model
                best_params = params

        print(f"\n  Best F1: {best_f1:.4f} | Best params: {best_params}")
        return best_model, best_f1, best_params

    def train_random_forest(self, train_df: DataFrame, val_df: DataFrame):
        print("\n[Trainer] Random Forest – hyperparameter search")
        param_grid = {
            "maxDepth":             [5, 10, 15],
            "numTrees":             [50, 100, 200],
            "minInstancesPerNode":  [1, 5],
            "featureSubsetStrategy":["sqrt", "log2", "onethird"],
        }
        base_params = {
            "labelCol":    "target",
            "featuresCol": "features",
            "seed":        42,
        }
        return self._tune(RandomForestClassifier, param_grid, base_params, train_df, val_df)

    def train_gbt(self, train_df: DataFrame, val_df: DataFrame):
        print("\n[Trainer] GBTClassifier – hyperparameter search")
        param_grid = {
            "maxDepth":        [3, 4, 5],
            "maxIter":         [50, 100, 150],
            "stepSize":        [0.05, 0.1],
            "subsamplingRate": [0.7, 0.9],
        }
        base_params = {
            "labelCol":    "target",
            "featuresCol": "features",
            "seed":        42,
        }
        return self._tune(GBTClassifier, param_grid, base_params, train_df, val_df)
