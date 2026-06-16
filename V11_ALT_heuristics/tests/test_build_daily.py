import importlib.util
import pathlib
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FOR = _load("V11_ALT_heuristics_forensic")


def test_build_daily_has_all_feat_cols():
    vin = "VIN1_F_ALT"
    ref = FOR.build_reference()
    d = FOR.build_daily(vin, ref)
    for col in cfg.FEAT_COLS:
        assert col in d.columns, f"missing {col}"
    assert "dtf" in d.columns and "day" in d.columns and "n_eo" in d.columns
    assert len(d) > 30
