from pyspark.sql import DataFrame
from pyspark.sql.functions import col


class Evaluator:
    """
    Computes binary classification metrics from a predictions DataFrame.

    Accuracy alone is misleading for imbalanced datasets like this one
    (~2.3% positive class). We prioritise F1, Precision, and Recall so
    that the cost of missing a real geomagnetic storm (false negative)
    is visible in the reported metrics.
    """

    def compute_metrics(
        self,
        predictions: DataFrame,
        label_col: str = "target",
        prediction_col: str = "prediction",
    ) -> dict:
        cm = (
            predictions
            .groupBy(label_col, prediction_col)
            .count()
            .collect()
        )

        TP = FP = TN = FN = 0
        for row in cm:
            label = row[label_col]
            pred  = row[prediction_col]
            count = row["count"]
            if   label == 1 and pred == 1: TP = count
            elif label == 0 and pred == 1: FP = count
            elif label == 0 and pred == 0: TN = count
            elif label == 1 and pred == 0: FN = count

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

        return {
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

    def print_report(self, model_name: str, metrics: dict) -> None:
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")
        print(f"  Confusion Matrix:")
        print(f"    TP={metrics['TP']:6d}  FP={metrics['FP']:6d}")
        print(f"    FN={metrics['FN']:6d}  TN={metrics['TN']:6d}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f}")
        print(f"{'='*50}\n")
