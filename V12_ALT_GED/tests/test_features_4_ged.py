"""Tests for features_4_ged.py — Phase-4 candidate GED prognostic features.

TDD: written before the implementation exists so the first run is RED.
"""
import importlib.util
import pathlib
import datetime as dt

import polars as pl
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


F = _load("features_4_ged")


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _ramp_daily():
    """40-day series: flat 0.0 for first 30 days, then ramps up over 10 days."""
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    rate = [0.0] * 30 + [0.01 * i for i in range(10)]
    return pl.DataFrame({"day": days, "ged2_rate": rate})


def _flat_daily():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    return pl.DataFrame({"day": days, "ged2_rate": [0.0] * 40})


def _regime_daily(idle_ged2=100, idle_total=1000, cruise_ged2=200, cruise_total=2000):
    """Single-row daily frame with explicit regime counts."""
    days = [dt.date(2025, 1, 1)]
    return pl.DataFrame(
        {
            "day": days,
            "ged2_in_idle": [idle_ged2],
            "regime_idle_rows": [idle_total],
            "ged2_in_cruise": [cruise_ged2],
            "regime_cruise_rows": [cruise_total],
        }
    )


# ---------------------------------------------------------------------------
# ged2_acceleration
# ---------------------------------------------------------------------------

def test_ged2_acceleration_positive_on_ramp():
    """Acceleration should be positive when ged2_rate rises in the last 30 days."""
    assert F.ged2_acceleration(_ramp_daily()) > 0


def test_ged2_acceleration_zero_on_flat():
    """Acceleration should be ~0 on a constant-zero series."""
    assert abs(F.ged2_acceleration(_flat_daily())) < 1e-9


def test_ged2_acceleration_type():
    """Must return a Python float."""
    assert isinstance(F.ged2_acceleration(_ramp_daily()), float)


# ---------------------------------------------------------------------------
# ged2_onset_slope
# ---------------------------------------------------------------------------

def test_ged2_onset_slope_positive_on_step():
    """onset_slope > 0 when there is a step-up between consecutive 7-day windows."""
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(20)]
    # first 10 days: 0.0; next 10 days: 0.5 → big 7-day diff
    rate = [0.0] * 10 + [0.5] * 10
    df = pl.DataFrame({"day": days, "ged2_rate": rate})
    assert F.ged2_onset_slope(df) > 0


def test_ged2_onset_slope_zero_on_flat():
    """onset_slope should be 0 on a flat series."""
    assert F.ged2_onset_slope(_flat_daily()) == 0.0


def test_ged2_onset_slope_short_series():
    """Should return 0.0 (not crash) for a series shorter than 8 rows."""
    short = pl.DataFrame(
        {
            "day": [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(5)],
            "ged2_rate": [0.01 * i for i in range(5)],
        }
    )
    assert F.ged2_onset_slope(short) == 0.0


# ---------------------------------------------------------------------------
# ged2_rate_regime
# ---------------------------------------------------------------------------

def test_ged2_rate_idle_correct():
    df = _regime_daily(idle_ged2=100, idle_total=1000)
    assert abs(F.ged2_rate_regime(df, "idle") - 0.1) < 1e-9


def test_ged2_rate_cruise_correct():
    df = _regime_daily(cruise_ged2=200, cruise_total=2000)
    assert abs(F.ged2_rate_regime(df, "cruise") - 0.1) < 1e-9


def test_ged2_rate_regime_zero_denom():
    """Should return 0.0 when the regime has no rows (avoid division by zero)."""
    df = _regime_daily(idle_ged2=0, idle_total=0)
    assert F.ged2_rate_regime(df, "idle") == 0.0


# ---------------------------------------------------------------------------
# build_feature_row
# ---------------------------------------------------------------------------

def test_build_feature_row_keys():
    """build_feature_row must return a dict with all 4 expected keys."""
    combined = _ramp_daily().with_columns(
        pl.lit(100).alias("ged2_in_idle"),
        pl.lit(1000).alias("regime_idle_rows"),
        pl.lit(200).alias("ged2_in_cruise"),
        pl.lit(2000).alias("regime_cruise_rows"),
    )
    row = F.build_feature_row(combined)
    assert set(row.keys()) == {
        "ged2_acceleration",
        "ged2_onset_slope",
        "ged2_rate_idle",
        "ged2_rate_cruise",
    }


def test_build_feature_row_values_float():
    """All values in the dict must be Python floats."""
    combined = _ramp_daily().with_columns(
        pl.lit(100).alias("ged2_in_idle"),
        pl.lit(1000).alias("regime_idle_rows"),
        pl.lit(200).alias("ged2_in_cruise"),
        pl.lit(2000).alias("regime_cruise_rows"),
    )
    row = F.build_feature_row(combined)
    for k, v in row.items():
        assert isinstance(v, float), f"{k} is {type(v)}, expected float"
