"""
src/infrastructure/writer.py
─────────────────────────────
Spark-based writers that abstract over output destinations:
  - MinIO (s3a://) — processed bucket
  - Local filesystem

Supports Parquet (recommended), CSV, and Delta (if delta-spark is installed).
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _configure_s3a(spark: SparkSession, endpoint: str, access_key: str, secret_key: str) -> None:
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.endpoint", endpoint)
    hadoop_conf.set("fs.s3a.access.key", access_key)
    hadoop_conf.set("fs.s3a.secret.key", secret_key)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Explicitly set timeouts as integers (milliseconds) to avoid "60s" NumberFormatException
    hadoop_conf.set("fs.s3a.connection.timeout", "60000")
    hadoop_conf.set("fs.s3a.connection.establish.timeout", "60000")


def _write_df(df: DataFrame, path: str, fmt: str, partition_cols: list[str] | None) -> None:
    writer = df.write.mode("overwrite").format(fmt)

    if fmt == "csv":
        writer = writer.option("header", "true")

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.save(path)
    logger.info("Wrote %s data to %s (partitioned by %s).", fmt.upper(), path, partition_cols)


# ─── Public API ───────────────────────────────────────────────────────────────

def write_to_minio(
    spark: SparkSession,
    df: DataFrame,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    prefix: str,
    fmt: str = "parquet",
    partition_cols: list[str] | None = None,
) -> None:
    """
    Write a cleaned Spark DataFrame to MinIO.

    Parameters
    ----------
    spark          : Active SparkSession.
    df             : Cleaned DataFrame to persist.
    endpoint       : MinIO endpoint, e.g. "http://localhost:9000".
    access_key     : MinIO root user.
    secret_key     : MinIO root password.
    bucket         : Destination bucket (e.g. "processed").
    prefix         : Object prefix / folder (e.g. "demand_cleaned").
    fmt            : "parquet", "csv", or "delta".
    partition_cols : Optional list of columns to partition by (e.g. ["Warehouse"]).
    """
    path = f"s3a://{bucket}/{prefix}"
    logger.info("Writing to MinIO: %s (format=%s)", path, fmt)
    _write_df(df, path, fmt, partition_cols)


def write_to_local(
    df: DataFrame,
    path: str,
    fmt: str = "parquet",
    partition_cols: list[str] | None = None,
) -> None:
    """
    Write a cleaned Spark DataFrame to the local (or HDFS) filesystem.

    Parameters
    ----------
    df             : Cleaned DataFrame.
    path           : Output directory path.
    fmt            : "parquet", "csv", or "delta".
    partition_cols : Optional partitioning columns.
    """
    logger.info("Writing to local path: %s (format=%s)", path, fmt)
    _write_df(df, path, fmt, partition_cols)


def write(spark: SparkSession, df: DataFrame, cfg: dict) -> None:
    """
    Dispatch writer based on `cfg['data']['source']`.

    Processed data goes back to the same environment the raw data was read from,
    but into the "processed" sub-path / bucket.

    Parameters
    ----------
    spark : Active SparkSession.
    df    : Cleaned DataFrame.
    cfg   : Full config dict.
    """
    source = cfg["data"]["source"]
    fmt = cfg["data"].get("output_format", "parquet")

    # Always partition by Warehouse so downstream jobs can read efficiently
    partition_cols = ["Warehouse"]

    if source == "minio":
        minio_cfg = cfg["data"]["minio"]
        write_to_minio(
            spark,
            df,
            endpoint=minio_cfg["endpoint"],
            access_key=minio_cfg["access_key"],
            secret_key=minio_cfg["secret_key"],
            bucket="processed",               # write to a separate bucket
            prefix=cfg["data"]["local"]["processed_path"].split("/")[-1],
            fmt=fmt,
            partition_cols=partition_cols,
        )
    elif source == "local":
        write_to_local(
            df,
            path=cfg["data"]["local"]["processed_path"],
            fmt=fmt,
            partition_cols=partition_cols,
        )
    else:
        raise ValueError(f"Unknown data source '{source}'.")