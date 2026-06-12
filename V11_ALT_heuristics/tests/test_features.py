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
F = _load("V11_ALT_heuristics_features")


def _ts(n, step=5):
    return pd.to_datetime(np.arange(n) * step, unit="s")


def test_prepare_validates_and_adds_columns():
    df = pd.DataFrame({
        "RPM": [0, 800, 4000], "CSP": [0, 2, 50], "ANR": [0.0, 200.0, 65535.0],
        "VSI": [25.0, 28.0, 200.0], "GED": [0, 0, 0], "SMA": [0, 0, 0],
        "DATETIME": _ts(3), "DAYS_SINCE_SALE": [1, 1, 1], "DAYS_TO_FAILURE": [10, 10, 10],
    })
    p = F.prepare(df, cfg)
    assert p["eo"].tolist() == [False, True, False]
    assert p["off"].iloc[0]
    assert np.isnan(p["anr"].iloc[2])
    assert np.isnan(p["vsi"].iloc[2])


def test_reg_duty():
    eo = pd.DataFrame({"vsi": [28.0, 28.5, 24.0, 30.0], "day": [1, 1, 1, 1]})
    out = F.reg_duty(eo, cfg)
    assert abs(out.loc[1] - 0.5) < 1e-9


def test_crank_effort():
    df = pd.DataFrame({
        "SMA": [0, 1, 1, 0, 0], "day": [1, 1, 1, 1, 1],
    })
    eo = pd.DataFrame({"day": [1] * 360})
    out = F.crank_effort(df, eo, cfg)
    assert abs(out.loc[1, "crank_dur_mean"] - 10.0) < 1e-9
    assert abs(out.loc[1, "cranks_per_ehr"] - 2.0) < 1e-6


def test_ged_states():
    df = pd.DataFrame({
        "GED": [0, 2, 0, 3, 1, 1], "day": [1, 1, 1, 1, 1, 1],
    })
    out = F.ged_states(df, cfg)
    assert abs(out.loc[1, "ged1_frac"] - 2 / 6) < 1e-9
    assert abs(out.loc[1, "ged3_frac"] - 1 / 6) < 1e-9
    assert out.loc[1, "ged_churn"] == 2


def test_vsi_rpm_curve():
    rpm = np.linspace(600, 1500, 40)
    vsi = 20.0 + 0.005 * rpm
    plateau_rpm = np.full(20, 2000.0)
    plateau_vsi = np.full(20, 28.0)
    eo = pd.DataFrame({
        "RPM": np.r_[rpm, plateau_rpm],
        "vsi": np.r_[vsi, plateau_vsi],
        "day": np.r_[np.ones(40), np.ones(20)].astype(int),
    })
    out = F.vsi_rpm_curve(eo, cfg)
    assert abs(out.loc[1, "vsi_rpm_slope"] - 0.005) < 1e-4
    assert abs(out.loc[1, "vsi_ceiling"] - 28.0) < 1e-6
    assert abs(out.loc[1, "vsi_onset_rpm"] - 1400.0) < 25.0


def test_load_residual():
    nf = pd.DataFrame({
        "RPM": np.full(100, 800.0), "anr": np.full(100, 200.0),
        "CSP": np.full(100, 0.0), "vsi": np.full(100, 28.0), "day": np.ones(100),
    })
    ref = F.build_load_reference(nf, cfg)
    fl = pd.DataFrame({
        "RPM": np.full(10, 800.0), "anr": np.full(10, 200.0),
        "CSP": np.full(10, 0.0), "vsi": np.full(10, 27.0), "day": np.ones(10),
    })
    out = F.load_residual(fl, ref, cfg)
    assert abs(out.loc[1.0, "vsi_resid_mean"] - (-1.0)) < 1e-6
    assert abs(out.loc[1.0, "vsi_resid_negfrac"] - 1.0) < 1e-9


def test_crank_recovery():
    df = pd.DataFrame({
        "SMA":   [1,   1,   0,    0,    0,    0],
        "vsi":   [22., 22., 22.0, 24.5, 27.0, 28.0],
        "t_s":   [0.,  5.,  10.,  15.,  20.,  25.],
        "day":   [1,   1,   1,    1,    1,    1],
    })
    out = F.crank_recovery(df, cfg)
    assert abs(out.loc[1, "crank_recovery_t"] - 10.0) < 1e-6
    assert out.loc[1, "crank_recovery_slope"] > 0


def test_idle_hunting():
    n = 40
    eo = pd.DataFrame({
        "RPM": np.full(n, 700.0), "CSP": np.full(n, 0.0),
        "vsi": np.where(np.arange(n) % 2 == 0, 28.5, 27.5),
        "t_s": np.arange(n) * 5.0, "day": np.ones(n, dtype=int),
    })
    out = F.idle_hunting(eo, cfg)
    assert out.loc[1, "idle_vsi_var"] > 0
    assert out.loc[1, "idle_vsi_zcr"] > 0.8
    assert out.loc[1, "idle_vsi_acf1"] < 0


def test_sag_typing():
    eo = pd.DataFrame({
        "anr": [500.0, 500.0, 10.0, 10.0],
        "vsi": [23.0, 28.0, 23.0, 23.0],
        "day": [1, 1, 1, 1],
    })
    out = F.sag_typing(eo, cfg)
    assert abs(out.loc[1, "sag_highload_frac"] - 0.5) < 1e-9
    assert abs(out.loc[1, "sag_idle_frac"] - 1.0) < 1e-9


def test_uv_dose_daily():
    eo = pd.DataFrame({
        "vsi": [28.0, 22.0, 23.0], "t_s": [0.0, 5.0, 10.0], "day": [1, 1, 1],
    })
    out = F.uv_dose_daily(eo, cfg)
    assert abs(out.loc[1] - (2.0 * 5 + 1.0 * 5)) < 1e-6
