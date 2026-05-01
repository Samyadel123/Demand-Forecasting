"""
tests/test_cleaner.py
──────────────────────
Unit tests for src/processing/cleaner.py.
Uses a local SparkSession with small synthetic DataFrames so no external
data sources are needed.

Run with:
  pytest tests/test_cleaner.py -v
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.processing.cleaner import (
    add_temporal_columns,
    clean_categoricals,
    drop_duplicates,
    flag_demand_anomalies,
    flag_outliers_iqr,
    flag_sparse_products,
    handle_nulls,
    normalise_column_names,
    parse_date,
    parse_order_demand,
    validate_schema,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for all tests in this module."""
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("cleaner_unit_tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


RAW_SCHEMA = StructType([
    StructField("Product_Code",     StringType(), True),
    StructField("Warehouse",        StringType(), True),
    StructField("Product_Category", StringType(), True),
    StructField("Date",             StringType(), True),
    StructField("Order_Demand",     StringType(), True),
])


def make_df(spark, rows):
    return spark.createDataFrame(rows, schema=RAW_SCHEMA)


# ─── Schema tests ─────────────────────────────────────────────────────────────

class TestNormaliseColumnNames:
    def test_strips_whitespace(self, spark):
        df = spark.createDataFrame(
            [("A", "B", "C", "2015/01/01", "100")],
            schema=StructType([
                StructField(" Product_Code",     StringType(), True),
                StructField("Warehouse ",        StringType(), True),
                StructField("Product_Category",  StringType(), True),
                StructField("Date",              StringType(), True),
                StructField("Order_Demand",      StringType(), True),
            ]),
        )
        df = normalise_column_names(df)
        assert "Product_Code" in df.columns
        assert "Warehouse" in df.columns

    def test_clean_names_unchanged(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "2015/01/01", "10")])
        df2 = normalise_column_names(df)
        assert df.columns == df2.columns


class TestValidateSchema:
    def test_raises_on_missing_column(self, spark):
        df = spark.createDataFrame([("A",)], ["Product_Code"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)

    def test_passes_with_all_columns(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "2015/01/01", "100")])
        validate_schema(df)  # should not raise


# ─── Duplicate removal ────────────────────────────────────────────────────────

class TestDropDuplicates:
    def test_removes_exact_duplicates(self, spark):
        rows = [
            ("P1", "Whse_A", "C1", "2015/01/01", "100"),
            ("P1", "Whse_A", "C1", "2015/01/01", "100"),  # duplicate
        ]
        df = make_df(spark, rows)
        df = drop_duplicates(df)
        assert df.count() == 1

    def test_keeps_distinct_rows(self, spark):
        rows = [
            ("P1", "Whse_A", "C1", "2015/01/01", "100"),
            ("P1", "Whse_A", "C1", "2015/01/02", "200"),
        ]
        df = make_df(spark, rows)
        df = drop_duplicates(df)
        assert df.count() == 2


# ─── Categorical cleaning ─────────────────────────────────────────────────────

class TestCleanCategoricals:
    def test_trims_whitespace(self, spark):
        df = make_df(spark, [("  P1  ", " Whse_A ", " Cat_1 ", "2015/01/01", "10")])
        df = clean_categoricals(df)
        row = df.collect()[0]
        assert row["Product_Code"] == "P1"
        assert row["Warehouse"] == "WHSE_A"
        assert row["Product_Category"] == "CAT_1"

    def test_uppercases(self, spark):
        df = make_df(spark, [("product_abc", "whse_j", "cat_x", "2015/01/01", "10")])
        df = clean_categoricals(df)
        row = df.collect()[0]
        assert row["Product_Code"] == "PRODUCT_ABC"


# ─── Date parsing ─────────────────────────────────────────────────────────────

class TestParseDate:
    FORMATS = ["yyyy/MM/dd", "MM/dd/yyyy", "yyyy-MM-dd"]

    def test_parses_standard_format(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "2015/01/15", "100")])
        df = parse_date(df, self.FORMATS)
        row = df.collect()[0]
        assert str(row["Date"]) == "2015-01-15"

    def test_parses_alternate_format(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "01/15/2015", "100")])
        df = parse_date(df, self.FORMATS)
        row = df.collect()[0]
        assert str(row["Date"]) == "2015-01-15"

    def test_flags_unparseable_date(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "not-a-date", "100")])
        df = parse_date(df, self.FORMATS)
        row = df.collect()[0]
        assert row["Date"] is None
        assert row["date_parse_failed"] is True

    def test_null_date_not_flagged_as_parse_failed(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", None, "100")])
        df = parse_date(df, self.FORMATS)
        row = df.collect()[0]
        assert row["date_parse_failed"] is False


# ─── Order_Demand parsing ─────────────────────────────────────────────────────

class TestParseOrderDemand:
    def test_parses_positive_integer(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "2015/01/01", "1500")])
        df = parse_order_demand(df)
        assert df.collect()[0]["Order_Demand"] == 1500

    def test_parses_accounting_negative(self, spark):
        """(1234) → -1234 — the most critical transformation in this dataset."""
        df = make_df(spark, [("P1", "W1", "C1", "2015/01/01", "(300)")])
        df = parse_order_demand(df)
        assert df.collect()[0]["Order_Demand"] == -300

    def test_flags_non_numeric(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "2015/01/01", "N/A")])
        df = parse_order_demand(df)
        row = df.collect()[0]
        assert row["Order_Demand"] is None
        assert row["demand_cast_failed"] is True

    def test_null_demand_not_flagged(self, spark):
        df = make_df(spark, [("P1", "W1", "C1", "2015/01/01", None)])
        df = parse_order_demand(df)
        row = df.collect()[0]
        assert row["demand_cast_failed"] is False


# ─── Anomaly flags ────────────────────────────────────────────────────────────

class TestFlagDemandAnomalies:
    def _make_numeric_df(self, spark, demand_val):
        """Helper: create a single-row DF with a cast Order_Demand."""
        schema = StructType([
            StructField("Product_Code",     StringType(), True),
            StructField("Warehouse",        StringType(), True),
            StructField("Product_Category", StringType(), True),
            StructField("Date",             StringType(), True),
            StructField("Order_Demand",     LongType(),   True),
        ])
        return spark.createDataFrame(
            [("P1", "W1", "C1", "2015-01-01", demand_val)], schema=schema
        )

    def test_flags_negative(self, spark):
        df = self._make_numeric_df(spark, -100)
        df = flag_demand_anomalies(df)
        assert df.collect()[0]["is_negative_demand"] is True

    def test_flags_zero(self, spark):
        df = self._make_numeric_df(spark, 0)
        df = flag_demand_anomalies(df)
        assert df.collect()[0]["is_zero_demand"] is True

    def test_positive_not_flagged(self, spark):
        df = self._make_numeric_df(spark, 500)
        df = flag_demand_anomalies(df)
        row = df.collect()[0]
        assert row["is_negative_demand"] is False
        assert row["is_zero_demand"] is False


# ─── Null handling ────────────────────────────────────────────────────────────

class TestHandleNulls:
    def _make_postparse_df(self, spark, rows):
        schema = StructType([
            StructField("Product_Code",     StringType(), True),
            StructField("Warehouse",        StringType(), True),
            StructField("Product_Category", StringType(), True),
            StructField("Date",             StringType(), True),
            StructField("Order_Demand",     LongType(),   True),
            StructField("date_parse_failed", StringType(), True),
        ])
        return spark.createDataFrame(rows, schema=schema)

    def test_drops_null_product_code(self, spark):
        rows = [
            (None,  "W1", "C1", "2015-01-01", 100, "false"),
            ("P1",  "W1", "C1", "2015-01-01", 200, "false"),
        ]
        df = self._make_postparse_df(spark, rows)
        df = handle_nulls(df, "drop")
        assert df.count() == 1

    def test_drop_strategy_removes_null_demand(self, spark):
        rows = [
            ("P1", "W1", "C1", "2015-01-01", None, "false"),
            ("P2", "W1", "C1", "2015-01-01", 100,  "false"),
        ]
        df = self._make_postparse_df(spark, rows)
        df = handle_nulls(df, "drop")
        assert df.count() == 1

    def test_zero_strategy_fills_null_demand(self, spark):
        rows = [
            ("P1", "W1", "C1", "2015-01-01", None, "false"),
        ]
        df = self._make_postparse_df(spark, rows)
        df = handle_nulls(df, "zero")
        assert df.collect()[0]["Order_Demand"] == 0


# ─── Outlier flagging ─────────────────────────────────────────────────────────

class TestFlagOutliersIQR:
    def test_extreme_value_flagged(self, spark):
        schema = StructType([
            StructField("Warehouse",    StringType(), True),
            StructField("Order_Demand", LongType(),   True),
        ])
        rows = [(f"W1", i * 100) for i in range(1, 20)]   # 100..1900
        rows.append(("W1", 999_999))                        # extreme outlier
        df = spark.createDataFrame(rows, schema=schema)
        df = flag_outliers_iqr(df, multiplier=3.0)
        outliers = df.filter(F.col("is_outlier")).collect()
        assert any(r["Order_Demand"] == 999_999 for r in outliers)

    def test_normal_values_not_flagged(self, spark):
        schema = StructType([
            StructField("Warehouse",    StringType(), True),
            StructField("Order_Demand", LongType(),   True),
        ])
        rows = [("W1", 100 * i) for i in range(1, 15)]
        df = spark.createDataFrame(rows, schema=schema)
        df = flag_outliers_iqr(df, multiplier=3.0)
        outlier_count = df.filter(F.col("is_outlier")).count()
        assert outlier_count == 0


# ─── Sparse product flagging ──────────────────────────────────────────────────

class TestFlagSparseProducts:
    def test_flags_product_below_threshold(self, spark):
        schema = StructType([
            StructField("Product_Code", StringType(), True),
            StructField("Order_Demand", LongType(),   True),
        ])
        rows = [("P_SPARSE", 10)] * 3 + [("P_RICH", 50)] * 25
        df = spark.createDataFrame(rows, schema=schema)
        df = flag_sparse_products(df, min_records=10)
        sparse = df.filter(F.col("is_sparse_product")).select("Product_Code").distinct().collect()
        codes = [r["Product_Code"] for r in sparse]
        assert "P_SPARSE" in codes
        assert "P_RICH" not in codes


# ─── Temporal enrichment ─────────────────────────────────────────────────────

class TestAddTemporalColumns:
    def test_adds_expected_columns(self, spark):
        from pyspark.sql.types import DateType
        schema = StructType([
            StructField("Date", DateType(), True),
        ])
        from datetime import date
        df = spark.createDataFrame([(date(2015, 6, 15),)], schema=schema)
        df = add_temporal_columns(df)
        assert "year"        in df.columns
        assert "month"       in df.columns
        assert "day_of_week" in df.columns
        assert "year_month"  in df.columns

    def test_correct_year_month(self, spark):
        from pyspark.sql.types import DateType
        from datetime import date
        schema = StructType([StructField("Date", DateType(), True)])
        df = spark.createDataFrame([(date(2016, 3, 7),)], schema=schema)
        df = add_temporal_columns(df)
        row = df.collect()[0]
        assert row["year"] == 2016
        assert row["month"] == 3
        assert row["year_month"] == "2016-03"