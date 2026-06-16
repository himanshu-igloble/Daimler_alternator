import importlib.util, pathlib
import numpy as np, pandas as pd
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
E = _load("V11_1_ALT_emergency")

def test_exceedance_trailing_window():
    d = pd.DataFrame({"day": range(1, 61), "n_eo": 500, "crank_recovery_t": 0.0})
    d.loc[d.day.isin([40, 45]), "crank_recovery_t"] = 9.0
    assert E.exceedance_first_fire(d, p95=0.05, k=2, trail=30) == 45
    assert E.exceedance_first_fire(d, p95=0.05, k=3, trail=30) is None

def test_exceedance_respects_window():
    d = pd.DataFrame({"day": range(1, 101), "n_eo": 500, "crank_recovery_t": 0.0})
    d.loc[d.day.isin([10, 80]), "crank_recovery_t"] = 9.0
    assert E.exceedance_first_fire(d, p95=0.05, k=2, trail=30) is None
