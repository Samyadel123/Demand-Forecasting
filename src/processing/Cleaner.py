"""
src/processing/cleaner.py
──────────────────────────
PySpark data-cleaning pipeline for the Kaggle "Forecasts for Product Demand"
dataset (felixzhao/productdemandforecasting).

Dataset columns
───────────────
  Product_Code      – Categorical ID  (e.g. Product_0001)
  Warehouse         – One of four central warehouses
  Product_Category  – Category label  (e.g. Category_001)
  Date              – Raw string, formats vary (yyyy/MM/dd, MM/dd/yyyy, etc.)
  Order_Demand      – Raw string; legitimate negatives encoded as "(1234)"
                      which is accounting notation for a negative number.

Known data-quality issues (from community notebooks)
──────────────────────────────────────────────────────
  1. Order_Demand stored as STRING with accounting-style negative notation.
  2. Multiple date formats coexist across rows.
  3. NULL values in every column.
  4. Duplicate rows.
  5. Negative demand values (returns / cancellations) — flagged, not dropped.
  6. Zero-demand rows — kept but flagged.
  7. Sparse products (very few records) — flagged.
  8. Extreme outliers in Order_Demand (right-skewed distribution).
  9. Inconsistent whitespace / casing in categorical columns.
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, DateType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Schema normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_column_names(df: DataFrame) -> DataFrame:
    """
    Strip whitespace from column names and normalise to Title_Case.
    The raw CSV sometimes ships with a leading space on headers.
    """
    rename_map = {c: c.strip() for c in df.columns}
    for old, new in rename_map.items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    logger.info("Columns after normalisation: %s", df.columns)
    return df


EXPECTED_COLUMNS = {"Product_Code", "Warehouse", "Product_Category", "Date", "Order_Demand"}

def validate_schema(df: DataFrame) -> None:
    """Raise ValueError if any expected column is missing."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing columns: {missing}. "
            f"Found: {df.columns}"
        )
    logger.info("Schema validation passed.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Duplicate removal
# ═══════════════════════════════════════════════════════════════════════════════

def drop_duplicates(df: DataFrame) -> DataFrame:
    """
    Remove exact duplicate rows across all five columns.
    Community notebooks confirm these are data entry duplicates, not valid
    repeat orders on the same day (those would share all fields too).
    """
    before = df.count()
    df = df.dropDuplicates()
    after = df.count()
    logger.info("Duplicates removed: %d rows dropped (%d → %d).", before - after, before, after)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Categorical column cleaning
# ═══════════════════════════════════════════════════════════════════════════════

def clean_categoricals(df: DataFrame) -> DataFrame:
    """
    Trim whitespace and upper-case the three categorical ID columns.
    Null rows in these columns are unfixable and will be handled in step 5.
    """
    for col in ("Product_Code", "Warehouse", "Product_Category"):
        df = df.withColumn(col, F.upper(F.trim(F.col(col))))
    logger.info("Categorical columns trimmed and upper-cased.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Date parsing  (multiple formats)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_date(df: DataFrame, date_formats: list[str]) -> DataFrame:
    """
    Try each date format in order; keep the first successful parse per row.
    Rows where no format matched are left as NULL and flagged with
    `date_parse_failed = True`.

    Parameters
    ----------
    df           : Input DataFrame (Date column still a string).
    date_formats : Ordered list of Java SimpleDateFormat patterns to attempt.
    """
    # Build a coalesce() chain — the first non-null result wins
    parse_attempts = [
        F.to_date(F.col("Date"), fmt) for fmt in date_formats
    ]
    df = df.withColumn("Date_parsed", F.coalesce(*parse_attempts))
    df = df.withColumn(
        "date_parse_failed",
        F.col("Date_parsed").isNull() & F.col("Date").isNotNull(),
    )

    # Log how many rows failed every format
    failed = df.filter(F.col("date_parse_failed")).count()
    if failed:
        logger.warning("%d rows could not be parsed with any date format.", failed)

    # Replace original with parsed version
    df = df.drop("Date").withColumnRenamed("Date_parsed", "Date")
    logger.info("Date column parsed to DateType.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Order_Demand  — the most complex column
# ═══════════════════════════════════════════════════════════════════════════════

def parse_order_demand(df: DataFrame) -> DataFrame:
    """
    Convert the raw Order_Demand string to a signed LongType.

    Transformations applied in order
    ─────────────────────────────────
      a. Strip whitespace.
      b. Convert accounting-style negatives: "(1234)" → "-1234".
         This is the single most common data quality issue in this dataset,
         documented across every community notebook.
      c. Cast to Long; non-numeric values become NULL (cast errors coerced).
      d. Flag cast failures with `demand_cast_failed`.
    """
    df = df.withColumn(
        "Order_Demand",
        # Replace opening paren with minus sign, remove closing paren
        F.regexp_replace(
            F.regexp_replace(F.trim(F.col("Order_Demand")), r"^\(", "-"),
            r"\)$",
            "",
        ),
    )
    df = df.withColumn("Order_Demand_raw_str", F.col("Order_Demand"))  # keep for audit
    df = df.withColumn("Order_Demand", F.col("Order_Demand").cast(LongType()))
    df = df.withColumn(
        "demand_cast_failed",
        F.col("Order_Demand").isNull() & F.col("Order_Demand_raw_str").isNotNull(),
    )

    failed = df.filter(F.col("demand_cast_failed")).count()
    if failed:
        logger.warning("%d Order_Demand values could not be cast to Long.", failed)

    df = df.drop("Order_Demand_raw_str")
    logger.info("Order_Demand parsed to LongType.")
    return df


def flag_demand_anomalies(df: DataFrame) -> DataFrame:
    """
    Add boolean flag columns for downstream model awareness:
      - is_negative_demand  : demand < 0  (returns / cancellations)
      - is_zero_demand      : demand == 0 (no-order days, pipeline gaps)

    Rows are NOT dropped here — modelling teams may want to include or
    exclude them based on the forecasting strategy.
    """
    df = df.withColumn("is_negative_demand", F.col("Order_Demand") < 0)
    df = df.withColumn("is_zero_demand", F.col("Order_Demand") == 0)

    neg = df.filter(F.col("is_negative_demand")).count()
    zero = df.filter(F.col("is_zero_demand")).count()
    logger.info("Negative demand rows flagged: %d", neg)
    logger.info("Zero demand rows flagged: %d", zero)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Null handling
# ═══════════════════════════════════════════════════════════════════════════════

def handle_nulls(df: DataFrame, null_demand_strategy: str = "drop") -> DataFrame:
    """
    Handle NULL values in each column:

    Product_Code / Warehouse / Product_Category
        → Cannot be imputed — rows are dropped (no meaningful group key).

    Date
        → Cannot be imputed — rows where date_parse_failed=True are dropped.

    Order_Demand
        → Strategy controlled by `null_demand_strategy`:
            "drop"   – drop rows (default; safe for supervised learning)
            "zero"   – fill with 0  (appropriate if NULL means no order)
            "median" – fill with per-product median (expensive, use carefully)

    Parameters
    ----------
    df                   : DataFrame post date-parsing and demand casting.
    null_demand_strategy : One of "drop", "zero", "median".
    """
    # ── Key columns: drop anything without a group key ──────────────────────
    before = df.count()
    df = df.filter(
        F.col("Product_Code").isNotNull()
        & F.col("Warehouse").isNotNull()
        & F.col("Product_Category").isNotNull()
    )
    df = df.filter(F.col("Date").isNotNull())  # also covers date_parse_failed rows

    logger.info(
        "Rows dropped (null key/date): %d", before - df.count()
    )

    # ── Order_Demand nulls ──────────────────────────────────────────────────
    null_demand_count = df.filter(F.col("Order_Demand").isNull()).count()
    logger.info("NULL Order_Demand rows: %d (strategy: %s)", null_demand_count, null_demand_strategy)

    if null_demand_strategy == "drop":
        df = df.filter(F.col("Order_Demand").isNotNull())

    elif null_demand_strategy == "zero":
        df = df.fillna({"Order_Demand": 0})

    elif null_demand_strategy == "median":
        # Compute per-product median and broadcast as a fill map
        median_df = (
            df.filter(F.col("Order_Demand").isNotNull())
            .groupBy("Product_Code")
            .agg(F.expr("percentile_approx(Order_Demand, 0.5)").alias("median_demand"))
        )
        df = df.join(median_df, on="Product_Code", how="left")
        df = df.withColumn(
            "Order_Demand",
            F.when(F.col("Order_Demand").isNull(), F.col("median_demand"))
            .otherwise(F.col("Order_Demand")),
        )
        df = df.drop("median_demand")

    else:
        raise ValueError(f"Unknown null_demand_strategy '{null_demand_strategy}'.")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Outlier detection
# ═══════════════════════════════════════════════════════════════════════════════

def flag_outliers_iqr(df: DataFrame, multiplier: float = 3.0) -> DataFrame:
    """
    Compute per-warehouse IQR bounds and flag rows outside
    [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

    Using IQR (not z-score) because demand is heavily right-skewed.
    A multiplier of 3.0 is conservative — only genuine extremes are flagged.
    Outliers are FLAGGED with `is_outlier`, not removed.

    Parameters
    ----------
    df         : Cleaned DataFrame with numeric Order_Demand.
    multiplier : IQR fence multiplier (default 3.0).
    """
    # Compute per-warehouse quartiles
    quantiles = (
        df.filter(F.col("Order_Demand") > 0)   # exclude negatives / zeros from IQR
        .groupBy("Warehouse")
        .agg(
            F.expr("percentile_approx(Order_Demand, 0.25)").alias("Q1"),
            F.expr("percentile_approx(Order_Demand, 0.75)").alias("Q3"),
        )
        .withColumn("IQR", F.col("Q3") - F.col("Q1"))
        .withColumn("lower_fence", F.col("Q1") - multiplier * F.col("IQR"))
        .withColumn("upper_fence", F.col("Q3") + multiplier * F.col("IQR"))
        .select("Warehouse", "lower_fence", "upper_fence")
    )

    df = df.join(quantiles, on="Warehouse", how="left")
    df = df.withColumn(
        "is_outlier",
        F.when(
            F.col("Order_Demand").isNotNull(),
            (F.col("Order_Demand") < F.col("lower_fence"))
            | (F.col("Order_Demand") > F.col("upper_fence")),
        ).otherwise(False),
    )
    df = df.drop("lower_fence", "upper_fence")

    outlier_count = df.filter(F.col("is_outlier")).count()
    logger.info("Outliers flagged (IQR × %.1f): %d rows.", multiplier, outlier_count)
    return df


def flag_outliers_zscore(df: DataFrame, threshold: float = 4.0) -> DataFrame:
    """
    Alternative: flag rows where |z-score| > threshold per warehouse.
    Use when distribution is approximately normal (rarely the case here).
    """
    stats = (
        df.filter(F.col("Order_Demand") > 0)
        .groupBy("Warehouse")
        .agg(
            F.mean("Order_Demand").alias("mean_demand"),
            F.stddev("Order_Demand").alias("std_demand"),
        )
    )
    df = df.join(stats, on="Warehouse", how="left")
    df = df.withColumn(
        "z_score",
        F.when(
            F.col("std_demand") > 0,
            F.abs(F.col("Order_Demand") - F.col("mean_demand")) / F.col("std_demand"),
        ).otherwise(F.lit(0)),
    )
    df = df.withColumn("is_outlier", F.col("z_score") > threshold)
    df = df.drop("mean_demand", "std_demand", "z_score")

    outlier_count = df.filter(F.col("is_outlier")).count()
    logger.info("Outliers flagged (z-score > %.1f): %d rows.", threshold, outlier_count)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Sparse-product detection
# ═══════════════════════════════════════════════════════════════════════════════

def flag_sparse_products(df: DataFrame, min_records: int = 10) -> DataFrame:
    """
    Products with fewer than `min_records` non-null demand rows cannot be
    modelled reliably. Flag them so forecasting pipelines can exclude or
    handle them separately.
    """
    counts = (
        df.filter(F.col("Order_Demand").isNotNull())
        .groupBy("Product_Code")
        .agg(F.count("*").alias("record_count"))
    )
    df = df.join(counts, on="Product_Code", how="left")
    df = df.withColumn("is_sparse_product", F.col("record_count") < min_records)
    df = df.drop("record_count")

    sparse = df.filter(F.col("is_sparse_product")).select("Product_Code").distinct().count()
    logger.info("Sparse products flagged (< %d records): %d products.", min_records, sparse)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Temporal enrichment  (derived columns useful for time-series models)
# ═══════════════════════════════════════════════════════════════════════════════

def add_temporal_columns(df: DataFrame) -> DataFrame:
    """
    Extract year, month, day-of-week, week-of-year, and a YearMonth period
    string from the parsed Date column.

    These are lightweight derivations — heavier lag / rolling features live
    in src/processing/features.py.
    """
    df = (
        df
        .withColumn("year",        F.year("Date"))
        .withColumn("month",       F.month("Date"))
        .withColumn("day_of_week", F.dayofweek("Date"))   # 1=Sunday … 7=Saturday
        .withColumn("week_of_year", F.weekofyear("Date"))
        .withColumn("year_month",  F.date_format("Date", "yyyy-MM"))
    )
    logger.info("Temporal enrichment columns added.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Quality summary  (logged, not written to disk)
# ═══════════════════════════════════════════════════════════════════════════════

def log_quality_summary(df: DataFrame) -> None:
    """Print a concise quality report to the logger after all cleaning steps."""
    total = df.count()
    logger.info("═" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("  Total rows          : %d", total)
    logger.info("  Distinct products   : %d", df.select("Product_Code").distinct().count())
    logger.info("  Distinct warehouses : %d", df.select("Warehouse").distinct().count())
    logger.info("  Date range          : %s → %s",
                df.agg(F.min("Date")).collect()[0][0],
                df.agg(F.max("Date")).collect()[0][0])
    logger.info("  NULL Order_Demand   : %d", df.filter(F.col("Order_Demand").isNull()).count())
    logger.info("  Negative demand     : %d", df.filter(F.col("is_negative_demand")).count())
    logger.info("  Zero demand         : %d", df.filter(F.col("is_zero_demand")).count())
    logger.info("  Outlier rows        : %d", df.filter(F.col("is_outlier")).count())
    logger.info("  Sparse products     : %d rows",
                df.filter(F.col("is_sparse_product")).count())
    logger.info("  Date parse failures : %d",
                df.filter(F.col("date_parse_failed")).count())
    logger.info("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Master cleaning function
# ═══════════════════════════════════════════════════════════════════════════════

def clean(df: DataFrame, cfg: dict) -> DataFrame:
    """
    Run the full cleaning pipeline and return a cleaned, annotated DataFrame.

    Pipeline steps
    ──────────────
      1.  Normalise column names
      2.  Validate expected schema
      3.  Drop exact duplicates
      4.  Clean categorical columns (trim, upper-case)
      5.  Parse Date column (multiple format fallback)
      6.  Parse Order_Demand (accounting negatives → numeric)
      7.  Flag demand anomalies (negative, zero)
      8.  Handle NULL values
      9.  Flag outliers (IQR or z-score, per config)
      10. Flag sparse products
      11. Add temporal derived columns
      12. Log quality summary

    Parameters
    ----------
    df  : Raw Spark DataFrame as returned by reader.read().
    cfg : Full config dict (from cleaning_config.yaml).

    Returns
    -------
    Cleaned, annotated Spark DataFrame.
    """
    cleaning_cfg = cfg.get("cleaning", {})

    # ── Step 1 & 2 ──────────────────────────────────────────────────────────
    df = normalise_column_names(df)
    validate_schema(df)

    # ── Step 3 ───────────────────────────────────────────────────────────────
    df = drop_duplicates(df)

    # ── Step 4 ───────────────────────────────────────────────────────────────
    df = clean_categoricals(df)

    # ── Step 5 ───────────────────────────────────────────────────────────────
    date_formats = cleaning_cfg.get("date_formats", ["yyyy/MM/dd", "MM/dd/yyyy", "yyyy-MM-dd"])
    df = parse_date(df, date_formats)

    # ── Step 6 ───────────────────────────────────────────────────────────────
    df = parse_order_demand(df)

    # ── Step 7 ───────────────────────────────────────────────────────────────
    df = flag_demand_anomalies(df)

    # ── Step 8 ───────────────────────────────────────────────────────────────
    null_strategy = cleaning_cfg.get("null_demand_strategy", "drop")
    df = handle_nulls(df, null_strategy)

    # ── Step 9 ───────────────────────────────────────────────────────────────
    outlier_cfg = cleaning_cfg.get("outlier", {})
    strategy = outlier_cfg.get("strategy", "iqr")
    if strategy == "iqr":
        df = flag_outliers_iqr(df, multiplier=outlier_cfg.get("iqr_multiplier", 3.0))
    elif strategy == "zscore":
        df = flag_outliers_zscore(df, threshold=outlier_cfg.get("zscore_threshold", 4.0))

    # ── Step 10 ──────────────────────────────────────────────────────────────
    min_records = cleaning_cfg.get("min_records_per_product", 10)
    df = flag_sparse_products(df, min_records)

    # ── Step 11 ──────────────────────────────────────────────────────────────
    df = add_temporal_columns(df)

    # ── Step 12 ──────────────────────────────────────────────────────────────
    log_quality_summary(df)

    return df