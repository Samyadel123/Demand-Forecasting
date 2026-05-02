"""
src/pipeline/training_pipeline.py
───────────────────────────────────
Orchestrates the model training and evaluation pipeline:
  Load Cleaned Data → Engineer Features → Evaluate Models → Leaderboard

Usage:
  python src/pipeline/training_pipeline.py --config src/config/training_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pyspark.sql.functions as F
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# ── Ensure project root is on the path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.reader import load_cleaned_data
from src.features.engineer import engineer_features

# Model imports
from src.models.baseline_lr import LinearForecaster
from src.models.ts_prophet import ProphetForecaster
from src.models.whale_xgb import WhaleForecaster
from src.models.whale_rf import RandomForestForecaster
from src.models.whale_lgbm import WhaleLGBMForecaster

# ─── MLflow setup ─────────────────────────────────────────────────────────────


def _init_mlflow(cfg: dict) -> None:
    ml_cfg = cfg.get("mlflow", {})
    tracking_uri = ml_cfg.get("tracking_uri", "http://mlflow:5000")
    experiment_name = ml_cfg.get("experiment_name", "Demand_Forecasting")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow initialized. Tracking URI: %s", tracking_uri)


# ─── Logging setup ────────────────────────────────────────────────────────────


def _setup_logging(log_path: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)


# ─── Config loader ────────────────────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from: %s", config_path)
    return cfg


# ─── SparkSession factory ─────────────────────────────────────────────────────


def build_spark_session(cfg: dict):
    """
    Build a SparkSession from the 'spark' section of the config.
    Adds hadoop-aws and aws-java-sdk JARs for S3A/MinIO support when needed.
    """
    from pyspark.sql import SparkSession

    spark_cfg = cfg.get("spark", {})
    app_name = spark_cfg.get("app_name", "DemandForecastingTrainer")
    master = spark_cfg.get("master", "local[*]")

    builder = SparkSession.builder.appName(app_name).master(master)

    # Apply extra Spark config keys
    for key, value in spark_cfg.get("config", {}).items():
        builder = builder.config(key, str(value))

    # Fix Spark 3.0+ date parsing issues
    builder = builder.config("spark.sql.legacy.timeParserPolicy", "LEGACY")

    # S3A JARs — required for MinIO access
    if cfg["data"]["source"] == "minio":
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        import os
        minio_cfg = cfg["data"]["minio"]
        endpoint = os.environ.get("MINIO_ENDPOINT", minio_cfg["endpoint"])
        access_key = os.environ.get("MINIO_ROOT_USER", minio_cfg["access_key"])
        secret_key = os.environ.get("MINIO_ROOT_PASSWORD", minio_cfg["secret_key"])

        builder = (
            builder
            .config("spark.hadoop.fs.s3a.endpoint", endpoint)
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "WARN"))
    logger.info("SparkSession created: %s (master=%s)", app_name, master)
    return spark


# ─── Evaluation Logic ─────────────────────────────────────────────────────────


def run_evaluation(spark, cfg: dict):
    train_cfg = cfg.get("training", {})
    target_whse = train_cfg.get("target_warehouse", "WHSE_S")
    target_cat = train_cfg.get("target_category", "CATEGORY_019")
    test_year = train_cfg.get("test_year", 2017)

    logger.info("═" * 60)
    logger.info("STEP 1/4 — Loading Cleaned Data")
    logger.info("═" * 60)
    df_clean = load_cleaned_data(spark, cfg)

    logger.info("═" * 60)
    logger.info("STEP 2/4 — Feature Engineering")
    logger.info("═" * 60)
    df_featured = engineer_features(df_clean, cfg)

    logger.info("═" * 60)
    logger.info("STEP 3/4 — Data Preparation (Filtering & Split)")
    logger.info("═" * 60)
    
    # Isolate the "Whale" for evaluation
    logger.info("Filtering for Warehouse: %s, Category: %s", target_whse, target_cat)
    whale_df = df_featured.filter(
        (F.col("Warehouse") == target_whse) & 
        (F.col("Product_Category") == target_cat)
    ).toPandas()

    # Drop nulls from rolling/lag windows at start of series
    initial_len = len(whale_df)
    whale_df = whale_df.dropna()
    logger.info("Dropped %d rows with NaNs (start-of-series).", initial_len - len(whale_df))

    # Chronological Split
    train_df = whale_df[whale_df["year"] < test_year].copy()
    test_df = whale_df[whale_df["year"] == test_year].copy()
    
    logger.info("Split: Train (Before %d) = %d rows, Test (%d) = %d rows", 
                test_year, len(train_df), test_year, len(test_df))

    y_train = train_df.pop("Order_Demand")
    X_train = train_df

    y_test = test_df.pop("Order_Demand")
    X_test = test_df

    # Identifiers to exclude from training
    meta_cols = ["Product_Code", "Product_Category", "Warehouse", "Date"]

    logger.info("═" * 60)
    logger.info("STEP 4/4 — Model Tournament")
    logger.info("═" * 60)

    models_to_run = {
        "Baseline Linear": LinearForecaster(),
        "Prophet (Time Series)": ProphetForecaster(),
        "Random Forest": RandomForestForecaster(),
        "XGBoost": WhaleForecaster(),
        "LightGBM": WhaleLGBMForecaster(),
    }

    results = {}
    mlflow_runs = {}

    # Get threshold for classification metrics
    threshold = train_cfg.get("high_demand_threshold", 10000)
    y_test_binary = (y_test > threshold).astype(int)

    for name, model in models_to_run.items():
        with mlflow.start_run(run_name=name) as run:
            logger.info("--> Evaluating %s...", name)

            # Preprocessing per model type
            if "Prophet" in name:
                X_train_m = X_train.drop(columns=["Product_Code", "Product_Category", "Warehouse"])
                X_test_m = X_test.drop(columns=["Product_Code", "Product_Category", "Warehouse"])
            else:
                X_train_m = X_train.drop(columns=meta_cols)
                X_test_m = X_test.drop(columns=meta_cols)

            t_start = time.time()
            model.train(X_train_m, y_train)
            predictions = model.predict(X_test_m)
            t_elapsed = time.time() - t_start

            # Regression Metric
            mae = mean_absolute_error(y_test, predictions)
            
            # Classification Metrics
            preds_binary = (predictions > threshold).astype(int)
            acc = accuracy_score(y_test_binary, preds_binary)
            f1 = f1_score(y_test_binary, preds_binary, zero_division=0)
            prec = precision_score(y_test_binary, preds_binary, zero_division=0)
            rec = recall_score(y_test_binary, preds_binary, zero_division=0)

            results[name] = {
                "MAE": float(mae),
                "Accuracy": float(acc),
                "F1": float(f1),
                "Precision": float(prec),
                "Recall": float(rec)
            }
            mlflow_runs[name] = run.info.run_id
            
            # Log to MLflow
            mlflow.log_params({
                "model_type": name,
                "high_demand_threshold": threshold,
                "target_warehouse": target_whse,
                "target_category": target_cat
            })
            mlflow.log_metrics({
                "MAE": float(mae),
                "Accuracy": float(acc),
                "F1": float(f1),
                "Precision": float(prec),
                "Recall": float(rec),
                "training_time": t_elapsed
            })

            # Log model artifact based on flavor
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model.model, "model")
            elif "LightGBM" in name:
                mlflow.lightgbm.log_model(model.model, "model")
            elif "Prophet" in name:
                # Prophet doesn't have a direct log_model in standard flavors without extra plugins, 
                # but we can use sklearn wrapper or just pickle it for now.
                mlflow.sklearn.log_model(model.model, "model")
            else:
                mlflow.sklearn.log_model(model.model, "model")

            logger.info(
                "    %s completed in %.2fs. MAE: %s | F1: %.2f", 
                name, t_elapsed, f"{mae:,.2f}", f1
            )

    # Results Summary
    logger.info("═" * 60)
    logger.info("WHALE LEADERBOARD")
    logger.info("═" * 60)
    sorted_res = sorted(results.items(), key=lambda x: x[1]["MAE"])
    for i, (name, metrics) in enumerate(sorted_res, 1):
        logger.info(
            "%d. %-25s: %12s MAE | F1: %.2f", 
            i, name, f"{metrics['MAE']:,.2f}", metrics["F1"]
        )
    
    winner, winner_metrics = sorted_res[0]
    logger.info("═" * 60)
    logger.info("WINNER (by MAE): %s", winner)
    logger.info("═" * 60)

    # ── Step 5: Model Registration ──────────────────────────────────────────
    ml_cfg = cfg.get("mlflow", {})
    registry_name = ml_cfg.get("model_registry_name", "Whale_Forecaster")
    
    winning_run_id = mlflow_runs[winner]
    model_uri = f"runs:/{winning_run_id}/model"
    
    logger.info("Registering winning model: %s", registry_name)
    mv = mlflow.register_model(model_uri, registry_name)
    
    # Transition to Production
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    # Note: Transitioning stages is deprecated in newer MLflow in favor of Aliases, 
    # but for compatibility with standard patterns:
    client.set_registered_model_alias(registry_name, "Production", mv.version)
    logger.info("Model version %s set to Production alias", mv.version)

    # ── Step 6: Local Persistence (Legacy) ───────────────────────────────────
    out_cfg = cfg.get("output", {})
    
    # 1. Save Metrics
    report_dir = Path(out_cfg.get("report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / out_cfg.get("metrics_filename", "leaderboard.json")
    
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Leaderboard saved to: %s", metrics_path)

    # 2. Save All Models
    model_dir = Path(out_cfg.get("model_dir", "src/models/artifacts"))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models_to_run.items():
        # Sanitize name for filename
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        suffix = "_best" if name == winner else ""
        model_path = model_dir / f"{safe_name}{suffix}.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved: %s -> %s", name, model_path)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument(
        "--config",
        default="src/config/training_config.yaml",
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--source",
        choices=["minio", "local"],
        default=None,
        help="Override the data source in the config file.",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def run(config_path: str, source_override: str | None = None) -> None:
    cfg = load_config(config_path)
    _setup_logging(cfg.get("logging", {}).get("path"))

    if source_override:
        cfg["data"]["source"] = source_override
        logger.info("Source overridden to: %s", source_override)

    _init_mlflow(cfg)
    spark = build_spark_session(cfg)
    
    t0 = time.time()
    try:
        run_evaluation(spark, cfg)
    finally:
        elapsed = time.time() - t0
        logger.info("Pipeline completed in %.1f seconds.", elapsed)
        spark.stop()


if __name__ == "__main__":
    args = parse_args()
    run(config_path=args.config, source_override=args.source)
