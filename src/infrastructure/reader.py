"""
src/infrastructure/reader.py
─────────────────────────────
Spark-based readers that abstract over data sources:
  - MinIO (S3-compatible) via s3a://
  - Local filesystem
  - HDFS via hdfs://

All return a raw Spark DataFrame with no transformations applied.
"""

from __future__ import annotations

import logging
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _configure_s3a(spark: SparkSession, endpoint: str, access_key: str, secret_key: str) -> None:
    """Inject S3A / MinIO Hadoop configuration into an existing SparkSession."""
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.endpoint", endpoint)
    hadoop_conf.set("fs.s3a.access.key", access_key)
    hadoop_conf.set("fs.s3a.secret.key", secret_key)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Explicitly set timeouts as integers (milliseconds) to avoid "60s" NumberFormatException
    # seen in some Spark/Hadoop versions.
    hadoop_conf.set("fs.s3a.connection.timeout", "60000")
    hadoop_conf.set("fs.s3a.connection.establish.timeout", "60000")

    logger.debug("S3A configuration applied for endpoint=%s", endpoint)


# ─── Public API ───────────────────────────────────────────────────────────────

def read_from_minio(
    spark: SparkSession,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_key: str,
    header: bool = True,
    infer_schema: bool = False,   # keep everything as string; cleaner casts explicitly
) -> DataFrame:
    """
    Read a CSV from a MinIO bucket into a Spark DataFrame.

    Parameters
    ----------
    spark       : Active SparkSession.
    endpoint    : MinIO endpoint, e.g. "http://localhost:9000".
    access_key  : MinIO root user / AWS access key.
    secret_key  : MinIO root password / AWS secret key.
    bucket      : Bucket name (e.g. "raw").
    object_key  : Path inside the bucket (e.g. "Historical Product Demand.csv").
    header      : Whether the CSV has a header row.
    infer_schema: If False (default) all columns are read as strings and cast
                  explicitly during cleaning — safer for dirty data.

    Returns
    -------
    Raw Spark DataFrame.
    """
    s3_path = f"s3a://{bucket}/{object_key}"
    logger.info("Reading from MinIO: %s", s3_path)

    df = (
        spark.read
        .option("header", str(header).lower())
        .option("inferSchema", str(infer_schema).lower())
        .option("multiLine", "false")
        .option("quote", '"')
        .option("escape", '"')
        .csv(s3_path)
    )

    logger.info("Loaded %d columns from MinIO source.", len(df.columns))
    return df


def read_from_local(
    spark: SparkSession,
    path: str,
    header: bool = True,
    infer_schema: bool = False,
) -> DataFrame:
    """
    Read a CSV from the local filesystem.

    Parameters
    ----------
    spark        : Active SparkSession.
    path         : Absolute or relative path to the CSV file.
    header       : Whether the CSV has a header row.
    infer_schema : Same rationale as in read_from_minio.

    Returns
    -------
    Raw Spark DataFrame.
    """
    logger.info("Reading from local path: %s", path)

    df = (
        spark.read
        .option("header", str(header).lower())
        .option("inferSchema", str(infer_schema).lower())
        .option("quote", '"')
        .option("escape", '"')
        .csv(path)
    )

    logger.info("Loaded %d columns from local source.", len(df.columns))
    return df


def read_from_hdfs(
    spark: SparkSession,
    path: str,
    header: bool = True,
    infer_schema: bool = False,
) -> DataFrame:
    """Read a CSV from HDFS using Spark's built-in HDFS support."""
    logger.info("Reading from HDFS path: %s", path)
    return read_from_local(spark, path, header=header, infer_schema=infer_schema)


def read(spark: SparkSession, cfg: dict) -> DataFrame:
    """
    Dispatch reader based on `cfg['data']['source']`.

    Parameters
    ----------
    spark : Active SparkSession.
    cfg   : Full config dict (as loaded from cleaning_config.yaml).

    Returns
    -------
    Raw Spark DataFrame.
    """
    source = cfg["data"]["source"]

    if source == "minio":
        minio_cfg = cfg["data"]["minio"]
        return read_from_minio(
            spark,
            endpoint=minio_cfg["endpoint"],
            access_key=minio_cfg["access_key"],
            secret_key=minio_cfg["secret_key"],
            bucket=minio_cfg["bucket"],
            object_key=minio_cfg["object_key"],
        )
    elif source == "local":
        return read_from_local(spark, cfg["data"]["local"]["raw_path"])
    elif source == "hdfs":
        return read_from_hdfs(spark, cfg["data"]["hdfs"]["raw_path"])
    else:
        raise ValueError(f"Unknown data source '{source}'. Use 'minio', 'local', or 'hdfs'.")