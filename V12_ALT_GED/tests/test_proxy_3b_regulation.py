import importlib.util, pathlib
import polars as pl, numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


P = _load("proxy_3b_regulation")


def test_residual_zero_when_matches_reference():
    rng = np.random.default_rng(0)
    rpm = rng.uniform(700, 1800, 5000)
    anr = rng.uniform(0, 500, 5000)
    csp = rng.uniform(0, 80, 5000)
    vsi = 28.0 + 0.0 * rpm          # constant healthy voltage
    ref = pl.DataFrame({"RPM": rpm, "ANR": anr, "CSP": csp, "VSI": vsi})
    surf = P.fit_reference(ref)
    r = P.residual(ref, surf)
    assert abs(float(np.nanmean(r["resid"].to_numpy()))) < 0.5


def test_fit_reference_returns_expected_columns():
    rng = np.random.default_rng(1)
    rpm = rng.uniform(700, 1800, 1000)
    anr = rng.uniform(0, 500, 1000)
    csp = rng.uniform(0, 80, 1000)
    vsi = 27.5 + rng.normal(0, 0.3, 1000)
    df = pl.DataFrame({"RPM": rpm, "ANR": anr, "CSP": csp, "VSI": vsi})
    surf = P.fit_reference(df)
    assert set(surf.columns) >= {"rb", "ab", "cb", "vsi_exp"}
    assert surf.height > 0


def test_residual_negative_when_low_voltage():
    """Depressed voltage should produce systematically negative residuals."""
    rng = np.random.default_rng(2)
    n = 3000
    rpm = rng.uniform(700, 1800, n)
    anr = rng.uniform(0, 500, n)
    csp = rng.uniform(0, 80, n)
    vsi_ref = 28.0 + np.zeros(n)
    ref = pl.DataFrame({"RPM": rpm, "ANR": anr, "CSP": csp, "VSI": vsi_ref})
    surf = P.fit_reference(ref)
    # Low-voltage truck: 2 V below healthy
    vsi_low = 26.0 + np.zeros(n)
    low_df = pl.DataFrame({"RPM": rpm, "ANR": anr, "CSP": csp, "VSI": vsi_low})
    r = P.residual(low_df, surf)
    mean_resid = float(np.nanmean(r["resid"].to_numpy()))
    assert mean_resid < -1.0, f"Expected mean_resid < -1.0, got {mean_resid}"


def test_engine_on_filters_zero_rpm():
    df = pl.DataFrame({
        "RPM": [0.0, None, 800.0, 1200.0, 0.0],
        "VSI": [27.0, 27.5, 28.0, 28.1, 27.2],
        "ANR": [0.0, 100.0, 200.0, 300.0, 0.0],
        "CSP": [0.0, 0.0, 60.0, 80.0, 0.0],
    })
    eo = P.engine_on(df)
    assert eo.height == 2
    assert float(eo["RPM"].min()) > 0
