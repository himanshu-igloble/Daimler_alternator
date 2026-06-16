import importlib.util, pathlib
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V = _load("V11_1_ALT_verify")

def test_g_beta_ship_rules():
    m0 = {"mae_model": 140.0, "pi_coverage": 0.87, "mean_pi_width": 400.0}
    good = {"mae_model": 45.0, "pi_coverage": 0.85, "mean_pi_width": 380.0,
            "wilcoxon_p_vs_dummy": 0.01, "wilcoxon_p_vs_m0": 0.01}
    bad = {"mae_model": 130.0, "pi_coverage": 0.85, "mean_pi_width": 395.0,
           "wilcoxon_p_vs_dummy": 0.4, "wilcoxon_p_vs_m0": 0.3}
    sharp = {"mae_model": 138.0, "pi_coverage": 0.83, "mean_pi_width": 320.0,
             "wilcoxon_p_vs_dummy": 0.4, "wilcoxon_p_vs_m0": 0.4}
    assert V.g_beta_ships(good, m0, dummy_mae=49.7) is True
    assert V.g_beta_ships(bad, m0, dummy_mae=49.7) is False
    assert V.g_beta_ships(sharp, m0, dummy_mae=49.7) is True

def test_scan_forbidden():
    assert V.scan_forbidden("import sklearn") == ["sklearn"]
    assert V.scan_forbidden("import numpy") == []
