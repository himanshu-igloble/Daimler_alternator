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
FOR = _load("V11_ALT_heuristics_forensic")


def test_gate_one_vin_detects_planted_drop():
    rng = np.arange(0, 400)
    dtf = 400 - rng
    duty = np.where(dtf <= 7, 0.2, 0.9)
    d = pd.DataFrame({"day": rng, "n_eo": 500, "dtf": dtf, "reg_duty_frac": duty})
    for c in cfg.FEAT_COLS:
        if c not in d.columns:
            d[c] = 0.9 if c not in cfg.BAD_HIGH else 0.0
    d["vin_label"] = "VINX_F_ALT"
    nfb = pd.DataFrame({"feature": cfg.FEAT_COLS}).set_index("feature")
    nfb["nf_p05"] = 0.5
    nfb["nf_p95"] = 1.5
    best_h, best_feat, best_z, _ = FOR._gate_one_vin(d, nfb)
    assert best_feat == "reg_duty_frac"
    assert best_h == "7"
