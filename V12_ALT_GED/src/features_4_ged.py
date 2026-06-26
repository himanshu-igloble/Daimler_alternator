"""Phase 4: per-VIN scalar GED prognostic features (offline candidates).

Four candidate features extracted from the per-VIN daily GED cache:
  ged2_acceleration  — slope of ged2_rate over the trailing 30 days
  ged2_onset_slope   — max 7-day forward step-change in ged2_rate
  ged2_rate_idle     — fleet-lifetime GED=2 rate during idle regime
  ged2_rate_cruise   — fleet-lifetime GED=2 rate during cruise regime

Plus three regulation residual features joined from 3b_regulation_features.csv:
  resid_mean, resid_slope_30d, resid_oscillation
"""
import pathlib
import polars as pl
import numpy as np


# ---------------------------------------------------------------------------
# Core scalar extractors
# ---------------------------------------------------------------------------

def _slope_last(d: pl.DataFrame, col: str, n: int) -> float:
    """OLS slope of the last *n* non-NaN values in *col* (sorted by 'day')."""
    s = d.sort("day")[col].to_numpy().astype(float)
    s = s[~np.isnan(s)]
    if len(s) < 2:
        return 0.0
    k = min(n, len(s))
    x = np.arange(k)
    return float(np.polyfit(x, s[-k:], 1)[0])


def ged2_acceleration(daily: pl.DataFrame) -> float:
    """OLS slope of ged2_rate over the trailing 30 observations.

    Positive value → rate is accelerating toward failure.
    NaN days (no GED observation) are dropped before fitting.
    """
    return _slope_last(daily, "ged2_rate", 30)


def ged2_onset_slope(daily: pl.DataFrame) -> float:
    """Maximum 7-day forward step-change in ged2_rate (null → 0).

    Captures sudden onset events (e.g. GED=2 storm beginning).
    Returns 0.0 for series shorter than 8 rows.
    """
    s = daily.sort("day")["ged2_rate"].fill_null(0).to_numpy().astype(float)
    if len(s) < 8:
        return 0.0
    diffs = [s[i] - s[i - 7] for i in range(7, len(s))]
    return float(max(diffs)) if diffs else 0.0


def ged2_rate_regime(daily: pl.DataFrame, regime: str) -> float:
    """Lifetime GED=2 fraction within a given operating regime.

    regime: 'idle' | 'cruise'  (matches column names ged2_in_<regime>
    and regime_<regime>_rows in the daily cache).
    Returns 0.0 when the denominator is zero.
    """
    num = daily[f"ged2_in_{regime}"].sum()
    den = daily[f"regime_{regime}_rows"].sum()
    return float(num / den) if den else 0.0


def build_feature_row(daily_vin: pl.DataFrame) -> dict:
    """Return a dict of the 4 GED candidate features for one VIN."""
    return {
        "ged2_acceleration": ged2_acceleration(daily_vin),
        "ged2_onset_slope": ged2_onset_slope(daily_vin),
        "ged2_rate_idle": ged2_rate_regime(daily_vin, "idle"),
        "ged2_rate_cruise": ged2_rate_regime(daily_vin, "cruise"),
    }


# ---------------------------------------------------------------------------
# Main: build 25-row feature table
# ---------------------------------------------------------------------------

def main():
    import importlib.util

    _SRC = pathlib.Path(__file__).resolve().parent

    def _load(n):
        s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        return m

    C = _load("ged_common")

    cache = pl.read_parquet(C.DAILY_CACHE)
    reg = pl.read_csv(C.RESULTS / "3b_regulation_features.csv")

    rows = []
    for vin in C.ALL_VINS:
        dv = cache.filter(pl.col("vin") == vin)
        row = {"VIN": vin, **build_feature_row(dv)}
        rows.append(row)

    feats = pl.DataFrame(rows).join(
        reg.select(["vin", "resid_mean", "resid_slope_30d", "resid_oscillation"]).rename(
            {"vin": "VIN"}
        ),
        on="VIN",
        how="left",
    )

    out_path = C.RESULTS / "4_ged_features.csv"
    feats.write_csv(out_path)
    print(feats)
    print(f"\nWritten: {out_path}  ({len(feats)} rows)")


if __name__ == "__main__":
    main()
