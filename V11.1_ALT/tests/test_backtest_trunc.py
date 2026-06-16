import importlib.util, pathlib
import numpy as np, pandas as pd
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
B = _load("V11_1_ALT_backtest")

def test_rewind_time_and_truncation_contract():
    calls = []
    def spy_cov(vin, t, p95, nfb):
        calls.append((vin, t)); return 0.5, 1
    x = B.rewind_covariates("VINX", ttf=600.0, horizon=90, cov_fn=spy_cov, p95=0.05, nfb=None)
    assert calls == [("VINX", 510.0)] and x == (0.5, 1)
