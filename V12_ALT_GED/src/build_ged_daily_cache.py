"""Phase 0: per-VIN per-day GED aggregate cache.

Cache schema (one row per VIN per calendar day):
  vin, failed, day, dtf,
  n_rows, ged_null, ged_cnt_0, ged_cnt_1, ged_cnt_2, ged_cnt_3, ged2_rate,
  regime_<r>_rows, ged2_in_<regime>,
  vsi_mean, vsi_p10, vsi_p50, vsi_p90, vsi_when_ged2_mean,
  rpm_mean, anr_mean, csp_mean, sma_sum
"""
import importlib.util
import pathlib

import polars as pl

_SRC = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _load("ged_common")

REGIMES = ["crank", "engine_off", "idle", "cruise", "heavy"]


def daily_from_df(df: pl.DataFrame, vin: str, failed: bool) -> pl.DataFrame:
    """Aggregate a single VIN's raw frame to one row per calendar day.

    Parameters
    ----------
    df     : Raw per-VIN frame (already has `day` Date and `regime` String cols).
    vin    : VIN label string (e.g. "VIN1_F_ALT").
    failed : Whether this VIN is a failure truck.

    Returns
    -------
    Polars DataFrame — one row per calendar day present in `df`.
    """
    g = df.group_by("day").agg([
        # row counts
        pl.len().alias("n_rows"),
        # GED state accounting
        pl.col("GED").is_null().sum().alias("ged_null"),
        *[(pl.col("GED") == s).sum().alias(f"ged_cnt_{s}") for s in C.GED_STATES],
        # _ged_obs = non-null GED count (denominator for ged2_rate)
        pl.col("GED").filter(pl.col("GED").is_not_null()).count().alias("_ged_obs"),
        # VSI statistics
        pl.col("VSI").mean().alias("vsi_mean"),
        pl.col("VSI").quantile(0.10).alias("vsi_p10"),
        pl.col("VSI").quantile(0.50).alias("vsi_p50"),
        pl.col("VSI").quantile(0.90).alias("vsi_p90"),
        # VSI conditional on GED==2
        pl.col("VSI").filter(pl.col("GED") == 2).mean().alias("vsi_when_ged2_mean"),
        # Drive signal means
        pl.col("RPM").mean().alias("rpm_mean"),
        pl.col("ANR").mean().alias("anr_mean"),
        pl.col("CSP").mean().alias("csp_mean"),
        # Crank event count (SMA is binary 0/1)
        pl.col("SMA").cast(pl.Int32).sum().alias("sma_sum"),
        # Regime row counts
        *[(pl.col("regime") == r).sum().alias(f"regime_{r}_rows") for r in REGIMES],
        # GED=2 counts per regime (cross-tab)
        *[
            ((pl.col("GED") == 2) & (pl.col("regime") == r)).sum().alias(f"ged2_in_{r}")
            for r in REGIMES
        ],
    ])

    # Compute ged2_rate = ged_cnt_2 / _ged_obs; null when no observations
    g = g.with_columns(
        pl.when(pl.col("_ged_obs") > 0)
        .then(pl.col("ged_cnt_2").cast(pl.Float64) / pl.col("_ged_obs").cast(pl.Float64))
        .otherwise(None)
        .alias("ged2_rate")
    ).drop("_ged_obs")

    # Attach VIN metadata
    g = g.with_columns([
        pl.lit(vin).alias("vin"),
        pl.lit(failed).alias("failed"),
    ])

    # Attach dtf (days-to-failure) if present in source; else null column
    if C.DTF_COL in df.columns:
        dtf = df.group_by("day").agg(pl.col(C.DTF_COL).first().alias("dtf"))
        g = g.join(dtf, on="day", how="left")
    else:
        g = g.with_columns(pl.lit(None, dtype=pl.Int32).alias("dtf"))

    return g


def main():
    frames = []
    for vin in C.ALL_VINS:
        df = C.load_vin(vin)
        agg = daily_from_df(df, vin, C.is_failed(vin))
        frames.append(agg)
        print(f"{vin}: {df.height:>9,} rows -> {agg.height:>4} days")

    cache = pl.concat(frames, how="diagonal_relaxed").sort(["vin", "day"])

    C.DAILY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache.write_parquet(C.DAILY_CACHE)
    print(f"\nWrote {C.DAILY_CACHE} ({C.DAILY_CACHE.stat().st_size // 1024:,} KB)")

    # --- Reconciliation table ---
    print("\n=== GED=2 totals per VIN ===")
    ged2 = (
        cache.group_by("vin")
        .agg(pl.col("ged_cnt_2").sum().alias("ged2_total"))
        .sort("vin")
    )
    print(ged2)

    print("\n=== GED=1 totals per VIN (should all be 0) ===")
    ged1 = (
        cache.group_by("vin")
        .agg(pl.col("ged_cnt_1").sum().alias("ged1_total"))
        .sort("vin")
    )
    print(ged1)

    print("\n=== GED null fraction per VIN ===")
    null_frac = (
        cache.group_by("vin")
        .agg(
            (pl.col("ged_null").sum().cast(pl.Float64) / pl.col("n_rows").sum().cast(pl.Float64))
            .alias("null_frac")
        )
        .sort("vin")
    )
    print(null_frac)

    print(f"\nnull_frac range: [{null_frac['null_frac'].min():.4f}, {null_frac['null_frac'].max():.4f}]")


if __name__ == "__main__":
    main()
