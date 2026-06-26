"""Tests for build_ged_daily_cache: validate daily_from_df aggregation logic."""
import importlib.util
import pathlib
import datetime as dt

import polars as pl

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("build_ged_daily_cache")


def test_daily_from_df_counts_states_and_rate():
    d = dt.date(2025, 1, 1)
    df = pl.DataFrame({
        "day": [d, d, d, d, d],
        "GED": pl.Series([0, 2, 2, 3, None], dtype=pl.UInt8),
        "VSI": [28.0, 24.0, 25.0, 28.0, 28.0],
        "RPM": [800.0, 800.0, 800.0, 0.0, 800.0],
        "ANR": [100.0] * 5,
        "CSP": [40.0] * 5,
        "SMA": pl.Series([0, 0, 0, 0, 0], dtype=pl.UInt8),
        "regime": ["cruise", "cruise", "cruise", "engine_off", "cruise"],
    })
    out = B.daily_from_df(df, vin="VIN1_F_ALT", failed=True).sort("day")
    r = out.row(0, named=True)

    assert r["n_rows"] == 5, f"n_rows={r['n_rows']}"
    assert r["ged_null"] == 1, f"ged_null={r['ged_null']}"
    assert r["ged_cnt_2"] == 2, f"ged_cnt_2={r['ged_cnt_2']}"
    assert r["ged_cnt_0"] == 1, f"ged_cnt_0={r['ged_cnt_0']}"
    assert r["ged_cnt_3"] == 1, f"ged_cnt_3={r['ged_cnt_3']}"
    assert r["ged_cnt_1"] == 0, f"ged_cnt_1={r['ged_cnt_1']}"
    # ged2_rate denominator = non-null GED (4) -> 2/4 = 0.5
    assert abs(r["ged2_rate"] - 0.5) < 1e-9, f"ged2_rate={r['ged2_rate']}"
    # vsi when GED==2 is rows 1 and 2: (24.0 + 25.0) / 2 = 24.5
    assert abs(r["vsi_when_ged2_mean"] - 24.5) < 1e-9, f"vsi_when_ged2_mean={r['vsi_when_ged2_mean']}"
    # vin and failed metadata
    assert r["vin"] == "VIN1_F_ALT"
    assert r["failed"] is True


def test_daily_from_df_dtf_absent_gives_null():
    """When no DAYS_TO_FAILURE column present, dtf column must exist and be null."""
    d = dt.date(2025, 1, 2)
    df = pl.DataFrame({
        "day": [d],
        "GED": pl.Series([0], dtype=pl.UInt8),
        "VSI": [28.0],
        "RPM": [800.0],
        "ANR": [100.0],
        "CSP": [40.0],
        "SMA": pl.Series([0], dtype=pl.UInt8),
        "regime": ["cruise"],
    })
    out = B.daily_from_df(df, vin="VIN1_NF_ALT", failed=False)
    assert "dtf" in out.columns
    assert out["dtf"][0] is None


def test_daily_from_df_ged2_rate_zero_obs_is_null():
    """When all GED values are null, ged2_rate must be null (not 0 or NaN)."""
    d = dt.date(2025, 1, 3)
    df = pl.DataFrame({
        "day": [d, d],
        "GED": pl.Series([None, None], dtype=pl.UInt8),
        "VSI": [28.0, 28.0],
        "RPM": [800.0, 800.0],
        "ANR": [100.0, 100.0],
        "CSP": [40.0, 40.0],
        "SMA": pl.Series([0, 0], dtype=pl.UInt8),
        "regime": ["cruise", "cruise"],
    })
    out = B.daily_from_df(df, vin="VIN2_F_ALT", failed=True)
    r = out.row(0, named=True)
    assert r["ged2_rate"] is None, f"ged2_rate should be null but got {r['ged2_rate']}"
