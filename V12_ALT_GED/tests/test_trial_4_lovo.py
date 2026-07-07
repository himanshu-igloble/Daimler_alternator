"""TDD test for trial_4_lovo.py — validates LOVO harness on separable synthetic."""
import importlib.util
import pathlib
import numpy as np
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


L = _load("trial_4_lovo")


def test_lovo_auroc_perfect_on_separable():
    vins = [f"V{i}" for i in range(10)]
    X = pd.DataFrame({
        "vin":    vins,
        "failed": [1] * 5 + [0] * 5,
        "f":      [10, 11, 12, 13, 14, 0, 1, 2, 3, 4],
    })
    auroc = L.lovo_auroc(X, ["f"], label_col="failed", vin_col="vin")
    assert auroc > 0.99, f"Expected AUROC > 0.99 on separable data, got {auroc:.4f}"
