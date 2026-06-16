import importlib.util
import pathlib
import numpy as np
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
C = _load("V11_ALT_heuristics_compound")


def test_votes_in_window_counts_bad_crossings():
    win = pd.Series({
        "vsi_ceiling": 26.0,
        "vsi_resid_mean": -0.9,
        "crank_recovery_t": 5.0,
        "resting_vsi_mean": 26.0,
        "ged_churn": 1.0,
    })
    nfb = pd.DataFrame({
        "feature": cfg.VOTE_CHANNELS,
        "nf_p05": [27.0, -0.2, 0.0, 25.0, 0.0],
        "nf_p95": [29.0, 0.2, 12.0, 27.0, 5.0],
    }).set_index("feature")
    n = C.count_votes(win, nfb, cfg)
    assert n == 2
