"""Tests for ged_common: VIN lists, loader contract, regime labels."""
import importlib.util
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _load("ged_common")


def test_vin_lists():
    assert len(C.FAILED_VINS) == 10 and len(C.NONFAILED_VINS) == 15
    assert C.ALL_VINS[0] == "VIN1_F_ALT" and C.ALL_VINS[-1] == "VIN15_NF_ALT"
    assert C.is_failed("VIN1_F_ALT") and not C.is_failed("VIN1_NF_ALT")


def test_load_vin_has_day_and_regime():
    df = C.load_vin("VIN1_F_ALT")
    assert "day" in df.columns and "regime" in df.columns
    assert df["regime"].is_in(["crank", "engine_off", "idle", "cruise", "heavy"]).all()
    assert {"GED", "VSI", "RPM", "ANR", "CSP", "SMA"}.issubset(set(df.columns))
