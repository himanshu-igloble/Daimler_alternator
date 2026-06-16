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
CP = _load("V11_ALT_heuristics_changepoint")


def test_cusum_detects_downward_shift():
    x = np.r_[np.zeros(100), np.full(50, -3.0)]
    idx = CP.cusum_changepoint(x, direction="down", k=cfg.CUSUM_K, h=cfg.CUSUM_H)
    assert 100 <= idx <= 110


def test_cusum_no_shift_returns_none():
    rng = np.sin(np.arange(200) * 0.1) * 0.1
    idx = CP.cusum_changepoint(rng, direction="down", k=cfg.CUSUM_K, h=cfg.CUSUM_H)
    assert idx is None


def test_knee_on_convex_dose():
    cum = np.r_[np.zeros(50), np.arange(1, 51) ** 2]
    k = CP.knee_index(cum)
    assert 45 <= k <= 70
