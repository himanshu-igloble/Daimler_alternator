"""Shared constants + per-VIN loader for the V12 GED investigation.

Step-1 schema discovery (2026-06-26):
  - datetime column : DATETIME  (Polars Datetime(us), already typed)
  - dtf column      : DAYS_TO_FAILURE  (Int32; present in all 25 parquets,
                       all-null for non-failed trucks)
  - parquet template: V5.2_20_5_ALT_{vin}.parquet
  - signals present : GED (UInt8), VSI (Float32), RPM (Float32),
                      ANR (Float32), CSP (Float32), SMA (UInt8)
"""
import pathlib
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]          # repo root
PARQUET_DIR = ROOT / "V5.2_ALT" / "features" / "parquets"
PARQUET_TMPL = "V5.2_20_5_ALT_{vin}.parquet"               # confirmed Step 1
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
DAILY_CACHE = RESULTS / "ged_daily_cache.parquet"

FAILED_VINS = [f"VIN{i}_F_ALT" for i in range(1, 11)]      # 10
NONFAILED_VINS = [f"VIN{i}_NF_ALT" for i in range(1, 16)]  # 15
ALL_VINS = FAILED_VINS + NONFAILED_VINS                     # 25

DT_COL = "DATETIME"        # confirmed Step 1 (not "timestamp")
DTF_COL = "DAYS_TO_FAILURE"  # present in all parquets; null for NF trucks

GED_STATES = [0, 1, 2, 3]


# KT-anchored operating regimes (RPM rev/min; CSP km/h; SMA crank flag)
def regime_expr() -> pl.Expr:
    """Return a polars Expr classifying each row into an operating regime."""
    return (
        pl.when(pl.col("SMA") == 1).then(pl.lit("crank"))
        .when(pl.col("RPM").is_null() | (pl.col("RPM") <= 0)).then(pl.lit("engine_off"))
        .when(pl.col("RPM") <= 700).then(pl.lit("idle"))
        .when(pl.col("RPM") <= 1800).then(pl.lit("cruise"))
        .otherwise(pl.lit("heavy"))
        .alias("regime")
    )


def load_vin(vin: str) -> pl.DataFrame:
    """Load one cleaned per-VIN parquet, add `day` (date) and `regime`.

    DATETIME is already Polars Datetime(us); .dt.date() extracts the Date.
    DAYS_TO_FAILURE is present in all parquets but null for NF trucks.
    """
    df = pl.read_parquet(PARQUET_DIR / PARQUET_TMPL.format(vin=vin))
    df = df.with_columns(pl.col(DT_COL).cast(pl.Datetime).dt.date().alias("day"))
    df = df.with_columns(regime_expr())
    return df


def is_failed(vin: str) -> bool:
    return vin.endswith("_F_ALT")
