"""
Geomagstorm – Geomagnetic Storm Prediction Pipeline
====================================================
End-to-end Spark pipeline that:
  1. Loads raw OMNI solar-wind and GBM solar-flare data
  2. Cleans, transforms, and engineers time-series features
  3. Joins the two sources on hourly timestamps
  4. Runs exploratory data analysis (plots saved to outputs/eda/)
  5. Trains and evaluates Random Forest and GBT classifiers
  6. Persists every stage as Parquet (no database dependency)

Usage
-----
    spark-submit src/main.py [--skip-eda] [--skip-training]

Arguments
---------
    --skip-eda       Skip EDA plots (faster iteration on ETL/models)
    --skip-training  Only run ETL and EDA, skip model training
"""

import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# Make src/ importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from reader    import Reader
from processor import Processor
from joiner    import Joiner
from storage   import ParquetStorage
from eda       import EDA
from splitter  import Splitter
from trainer   import Trainer, FEATURE_COLS
from evaluator import Evaluator


# ── Configuration ────────────────────────────────────────────────────────────

RAW_DATA_PATH   = "data/raw"
PROCESSED_PATH  = "data/processed"
OUTPUTS_PATH    = "outputs"

FORWARD_FILL_COLS = [
    "B_scalar", "Bx", "By", "Bz",
    "temperature", "density", "speed",
    "flow_pressure", "electric_field",
    "plasma_beta", "alfven_mach",
    "kp", "sunspots", "f107",
]

FEATURE_CONFIG = {
    "Bz": {
        "lags":      [1, 3, 6, 12, 24],
        "roll_mean": [3, 6, 12],
        "roll_min":  [3, 6, 12],
    },
    "speed": {
        "lags":      [1, 3, 6],
        "roll_mean": [6],
        "roll_max":  [6],
    },
    "electric_field": {
        "lags":      [1],
        "roll_mean": [6],
        "roll_max":  [6],
    },
    "kp": {
        "lags":    [1, 2, 3],
        "roll_max":[6],
    },
    "flare_count":        {"roll_sum":  [6]},
    "total_energy_proxy": {"roll_mean": [6]},
}


# ── Spark session ─────────────────────────────────────────────────────────────

def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Geomagstorm")
        .config("spark.driver.memory", "10g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.memory.fraction", "0.8")
        .getOrCreate()
    )


# ── ETL ───────────────────────────────────────────────────────────────────────

def run_etl(spark: SparkSession, storage: ParquetStorage):
    print("\n" + "="*60)
    print("  STAGE 1 – ETL")
    print("="*60)

    reader    = Reader(spark, RAW_DATA_PATH)
    processor = Processor(spark)
    joiner    = Joiner()

    # ── Load ────────────────────────────────────────────────────
    omni = reader.load_omni()
    gbm  = reader.load_gbm()
    print(f"[ETL] Loaded OMNI ({omni.count()} rows) and GBM ({gbm.count()} rows)")

    # ── OMNI: target encoding & physical filter ─────────────────
    omni = processor.encode_kp(omni)
    omni = processor.filter_omni(omni)

    print("[ETL] Target distribution after filtering:")
    omni.groupBy("target").count().orderBy("target").show()

    # ── OMNI: impute missing values via forward fill ─────────────
    omni = processor.apply_forward_fill(omni, FORWARD_FILL_COLS)

    # ── OMNI: cyclical time encoding ─────────────────────────────
    omni = processor.cyclical_encoding(omni, "hour", 24)
    omni = processor.cyclical_encoding(omni, "doy",  366)

    # ── GBM: parse timestamps & aggregate per hour ──────────────
    gbm = processor.process_gbm_timestamps(gbm)
    gbm = processor.gbm_group_by_hour(gbm)

    # ── Persist intermediate tables ──────────────────────────────
    storage.write(omni, "omni")
    storage.write(gbm,  "gbm")

    # ── Join on hourly timestamp ──────────────────────────────────
    df = joiner.joinLeftDataset(omni, gbm, "timestamp")

    # Rows with no solar flare in that hour → fill GBM columns with 0
    df = df.fillna({"flare_count": 0, "avg_peak_counts": 0, "total_energy_proxy": 0})

    # ── Feature engineering: lags and rolling windows ────────────
    df = processor.apply_lag_roll(df, FEATURE_CONFIG)

    # Drop rows that are NaN from lag/roll lookback warmup
    rows_before = df.count()
    df = df.dropna()
    rows_after  = df.count()
    print(f"[ETL] Dropped {rows_before - rows_after} warmup rows (lag/roll NaN). Final: {rows_after} rows.")

    # Put target last (convention)
    df = df.select([c for c in df.columns if c != "target"] + ["target"])

    # ── Persist final dataset ─────────────────────────────────────
    storage.write(df, "geomagstorm_dataset")

    return df


# ── EDA ───────────────────────────────────────────────────────────────────────

def run_eda(df):
    print("\n" + "="*60)
    print("  STAGE 2 - EDA")
    print("="*60)
    eda = EDA(output_dir=os.path.join(OUTPUTS_PATH, "eda"))
    eda.run(df)


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(df):
    print("\n" + "="*60)
    print("  STAGE 3 – TRAINING")
    print("="*60)

    trainer   = Trainer()
    splitter  = Splitter()
    evaluator = Evaluator()

    # Vectorise features
    vec_df = trainer.vectorize(df)

    # Time-based split (no random shuffle — prevents data leakage)
    train, val, test = splitter.split(vec_df)
    train = train.select("features", "target").repartition(32).cache()
    val   = val.select("features", "target").repartition(32).cache()
    test  = test.select("features", "target").repartition(32).cache()

    print("[Training] Split sizes:")
    print(f"  train={train.count()}  val={val.count()}  test={test.count()}")

    # ── Random Forest ────────────────────────────────────────────
    rf_model, rf_val_f1, rf_params = trainer.train_random_forest(train, val)
    rf_preds   = rf_model.transform(test)
    rf_metrics = evaluator.compute_metrics(rf_preds)
    evaluator.print_report(f"Random Forest (test) – best params: {rf_params}", rf_metrics)

    # ── GBT ──────────────────────────────────────────────────────
    gbt_model, gbt_val_f1, gbt_params = trainer.train_gbt(train, val)
    gbt_preds   = gbt_model.transform(test)
    gbt_metrics = evaluator.compute_metrics(gbt_preds)
    evaluator.print_report(f"GBTClassifier (test) – best params: {gbt_params}", gbt_metrics)

    # ── Feature importance plots ──────────────────────────────────
    eda = EDA(output_dir=os.path.join(OUTPUTS_PATH, "eda"))
    eda.plot_feature_importance(rf_model,  FEATURE_COLS, "Random Forest")
    eda.plot_feature_importance(gbt_model, FEATURE_COLS, "GBTClassifier")

    # ── Model comparison summary ──────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Model':<20} {'Val F1':>8} {'Test F1':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Random Forest':<20} {rf_val_f1:>8.4f} {rf_metrics['f1']:>8.4f} {rf_metrics['precision']:>10.4f} {rf_metrics['recall']:>8.4f}")
    print(f"  {'GBTClassifier':<20} {gbt_val_f1:>8.4f} {gbt_metrics['f1']:>8.4f} {gbt_metrics['precision']:>10.4f} {gbt_metrics['recall']:>8.4f}")
    print("="*60)

    return rf_model, gbt_model


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args        = set(sys.argv[1:])
    skip_eda      = "--skip-eda"      in args
    skip_training = "--skip-training" in args

    spark   = build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    storage = ParquetStorage(PROCESSED_PATH)

    # ETL: always runs (unless processed data already exists and you load it)
    if storage.exists("geomagstorm_dataset"):
        print("[Pipeline] Found existing processed dataset — loading from Parquet.")
        df = storage.read(spark, "geomagstorm_dataset")
    else:
        df = run_etl(spark, storage)

    if not skip_eda:
        run_eda(df)

    if not skip_training:
        run_training(df)

    spark.stop()
    print("\n[Pipeline] Done.")


if __name__ == "__main__":
    main()
