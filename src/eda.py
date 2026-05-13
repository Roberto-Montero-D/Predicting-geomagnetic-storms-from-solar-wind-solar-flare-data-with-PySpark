import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation


NUMERIC_COLS = [
    "B_scalar", "Bx", "By", "Bz",
    "temperature", "density", "speed",
    "flow_pressure", "electric_field",
    "plasma_beta", "alfven_mach",
    "sunspots", "f107", "kp",
    "flare_count", "avg_peak_counts", "total_energy_proxy",
]


class EDA:
    """
    Generates and saves exploratory plots to the outputs directory.

    All plots are written to disk (PNG) instead of displayed interactively
    so the EDA can run inside a non-GUI Spark environment (Docker / cluster).

    Physics context is embedded in the plot titles and file names so
    the repository is self-documenting.
    """

    def __init__(self, output_dir: str = "outputs/eda"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _sample_to_pandas(self, df: DataFrame, column: str, n: int = 100_000) -> pd.DataFrame:
        total = df.count()
        fraction = min(1.0, n / total)
        return df.select(column).sample(withReplacement=False, fraction=fraction).toPandas()

    # ── Individual column plots ──────────────────────────────────────────────

    def plot_histogram(self, df: DataFrame, column: str, bins: int = 30) -> None:
        pdf = self._sample_to_pandas(df, column)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(pdf[column], bins=bins, kde=True, ax=ax)
        ax.set_title(f"Distribution – {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"hist_{column}.png"), dpi=100)
        plt.close(fig)

    def plot_boxplot(self, df: DataFrame, column: str) -> None:
        pdf = self._sample_to_pandas(df, column)
        fig, ax = plt.subplots(figsize=(4, 5))
        sns.boxplot(y=pdf[column], ax=ax)
        ax.set_title(f"Boxplot – {column}")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"box_{column}.png"), dpi=100)
        plt.close(fig)

    def plot_class_balance(self, df: DataFrame, column: str = "target") -> None:
        counts = df.groupBy(column).count().toPandas()
        counts.columns = [column, "count"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=column, y="count", data=counts, ax=ax)
        ax.set_title("Class balance (0 = no storm, 1 = geomagnetic storm)")
        ax.set_xlabel("Target")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "class_balance.png"), dpi=100)
        plt.close(fig)

    # ── Correlation matrix ───────────────────────────────────────────────────

    def plot_correlation_matrix(self, df: DataFrame, cols: list = None) -> None:
        use_cols = cols or NUMERIC_COLS
        # Drop rows with nulls in these columns before computing correlation
        df_clean = df.select(use_cols).dropna()
        assembler = VectorAssembler(inputCols=use_cols, outputCol="_corr_vec")
        vec_df = assembler.transform(df_clean).select("_corr_vec")
        corr_matrix = Correlation.corr(vec_df, "_corr_vec").head()[0].toArray()
        corr_df = pd.DataFrame(corr_matrix, index=use_cols, columns=use_cols)

        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(
            corr_df, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.3, ax=ax, annot_kws={"size": 7}
        )
        ax.set_title("Pearson Correlation Matrix – Solar Wind & Geomagnetic Features")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "correlation_matrix.png"), dpi=120)
        plt.close(fig)
        print(f"[EDA] Correlation matrix saved.")

    # ── Time series ──────────────────────────────────────────────────────────

    def plot_timeseries(self, df: DataFrame, column: str, sample_n: int = 5_000) -> None:
        pdf = (
            df.select("timestamp", column)
            .dropna()
            .sample(withReplacement=False, fraction=min(1.0, sample_n / df.count()))
            .toPandas()
            .sort_values("timestamp")
        )
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(pdf["timestamp"], pdf[column], linewidth=0.5, alpha=0.7)
        ax.set_title(f"Time series – {column}")
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"ts_{column}.png"), dpi=100)
        plt.close(fig)

    # ── Feature importance ───────────────────────────────────────────────────

    def plot_feature_importance(self, model, feature_cols: list, model_name: str) -> None:
        importances = model.featureImportances.toArray()
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feature_cols, importances)
        ax.set_title(f"Feature Importances – {model_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        fig.savefig(os.path.join(self.output_dir, f"feature_importance_{safe_name}.png"), dpi=100)
        plt.close(fig)
        print(f"[EDA] Feature importance plot saved for {model_name}.")

    # ── Run full EDA ─────────────────────────────────────────────────────────

    def run(self, df: DataFrame) -> None:
        print(f"[EDA] Running full EDA — saving plots to {self.output_dir}/")

        print("[EDA] Descriptive statistics:")
        df.describe().show()

        print("[EDA] Class balance:")
        df.groupBy("target").count().show()

        self.plot_class_balance(df)

        for col in NUMERIC_COLS:
            print(f"[EDA]   {col}")
            self.plot_histogram(df, col)
            self.plot_boxplot(df, col)

        key_ts_cols = ["kp", "Bz", "speed", "electric_field", "flare_count"]
        for col in key_ts_cols:
            self.plot_timeseries(df, col)

        self.plot_correlation_matrix(df)

        print(f"[EDA] Done. All plots saved to {self.output_dir}/")
