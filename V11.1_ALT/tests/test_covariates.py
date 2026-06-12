import importlib.util, pathlib
import numpy as np, pandas as pd
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cfg = _load("V11_1_ALT_config")
C = _load("V11_1_ALT_covariates")

def _panel():
    d = pd.DataFrame({"day": range(1, 101), "n_eo": 500})
    d["crank_recovery_t"] = 0.0
    d.loc[d.day.isin([10, 50, 90]), "crank_recovery_t"] = 5.0
    for ch in cfg.VOTE_CHANNELS:
        if ch not in d.columns:
            d[ch] = 100.0
    return d

def test_x1_counts_only_up_to_t():
    d = _panel()
    assert C.x1_exceedance(d, t=60, p95=0.05) == np.log1p(2)
    assert C.x1_exceedance(d, t=100, p95=0.05) == np.log1p(3)
    assert C.x1_exceedance(d, t=5, p95=0.05) == np.log1p(0)

def test_x2_trailing_window_votes():
    d = _panel()
    nfb = pd.DataFrame({"feature": cfg.VOTE_CHANNELS,
                        "nf_p05": [0.0]*5, "nf_p95": [50.0]*5}).set_index("feature")
    assert C.x2_compound(d, t=100, nfb=nfb) == 1
    nfb2 = nfb.copy(); nfb2["nf_p95"] = 1e6
    assert C.x2_compound(d, t=100, nfb=nfb2) == 0

def test_x2_ignores_data_after_t():
    d = _panel()
    d.loc[d.day > 50, "ged_churn"] = 1e9
    nfb = pd.DataFrame({"feature": cfg.VOTE_CHANNELS,
                        "nf_p05": [-1e12]*5, "nf_p95": [1e6]*5}).set_index("feature")
    assert C.x2_compound(d, t=50, nfb=nfb) == 0
