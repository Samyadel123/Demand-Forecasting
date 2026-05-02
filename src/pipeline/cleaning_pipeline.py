"""
src/pipeline/cleaning_pipeline.py
───────────────────────────────────
Orchestrates the end-to-end cleaning run:
  Reader → Cleaner → Writer

Can be submitted directly to a Spark cluster:
  spark-submit src/pipeline/cleaning_pipeline.py \
      --config src/config/cleaning_config.yaml \
      --source local

Or run locally (SparkSession created internally):
  python src/pipeline/cleaning_pipeline.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# ── Ensure project root is on the path when running as a script ───────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.reader import read
from src.infrastructure.writer import write
from src.processing.cleaner import clean

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
    app_name = spark_cfg.get("app_name", "DemandForecastingCleaner")
    master = spark_cfg.get("master", "local[*]")

    builder = SparkSession.builder.appName(app_name).master(master)

    # Apply extra Spark config keys
    for key, value in spark_cfg.get("config", {}).items():
        builder = builder.config(key, str(value))

    # Fix Spark 3.0+ date parsing issues for single-digit months/days
    builder = builder.config("spark.sql.legacy.timeParserPolicy", "LEGACY")

    # S3A JARs — required for MinIO access; harmless in local mode
    if cfg["data"]["source"] == "minio":
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        # Inject Hadoop S3A configurations via Spark properties (spark.hadoop.*)
        # This ensures they are available immediately when the FileSystem is first accessed.
        # We prioritize environment variables (common in Docker) over the YAML config.
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
            # Override potential "60s" defaults with integer milliseconds
            .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
            .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "WARN"))
    logger.info("SparkSession created: %s (master=%s)", app_name, master)
    return spark


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the demand-forecasting data cleaning pipeline."
    )
    parser.add_argument(
        "--config",
        default="src/config/cleaning_config.yaml",
        help="Path to the YAML config file (default: src/config/cleaning_config.yaml).",
    )
    parser.add_argument(
        "--source",
        choices=["minio", "local"],
        default=None,
        help="Override the data source in the config file.",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv", "delta"],
        default=None,
        help="Override the output format (default: parquet).",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def run(
    config_path: str,
    source_override: str | None = None,
    fmt_override: str | None = None,
) -> None:
    cfg = load_config(config_path)
    _setup_logging(cfg.get("logging", {}).get("path"))

    # Apply CLI overrides
    if source_override:
        cfg["data"]["source"] = source_override
        logger.info("Source overridden to: %s", source_override)
    if fmt_override:
        cfg["data"]["output_format"] = fmt_override
        logger.info("Output format overridden to: %s", fmt_override)

    spark = build_spark_session(cfg)

    t0 = time.time()

    logger.info("═" * 60)
    logger.info("STEP 1/3 — Reading raw data")
    logger.info("═" * 60)
    raw_df = read(spark, cfg)
    raw_df.printSchema()

    logger.info("═" * 60)
    logger.info("STEP 2/3 — Cleaning")
    logger.info("═" * 60)
    cleaned_df = clean(raw_df, cfg)
    cleaned_df.printSchema()

    logger.info("═" * 60)
    logger.info("STEP 3/3 — Writing processed data")
    logger.info("═" * 60)
    write(spark, cleaned_df, cfg)

    elapsed = time.time() - t0
    logger.info("Pipeline completed in %.1f seconds.", elapsed)

    spark.stop()


if __name__ == "__main__":
    args = parse_args()
    run(
        config_path=args.config,
        source_override=args.source,
        fmt_override=args.output_format,
    )

