import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import V11_2_ALT_rul_evidence_stack as E
import V11_2_ALT_rul_evidence_stack_optm as O


def test_insight_abrupt_has_no_fabricated_lead():
    b = E.build_bundle("VIN8_F_ALT")          # abrupt failure, RED band
    s = O.optm_insight(b, prob=0.8923, band="red")
    assert s.startswith("✓ CAUGHT")       # checkmark CAUGHT
    for forbidden in ("fired", "ahead", "runway", "declined"):
        assert forbidden not in s              # no invented precursor
    assert "RED band" in s


def test_insight_vin1_names_its_signals():
    # VIN1's 3 signals are ML rank + M5 condition + GED — NOT three panel-3 precursors.
    b = E.build_bundle("VIN1_F_ALT")
    s = O.optm_insight(b, prob=0.6058, band="red")
    assert s.startswith("✓ DETECTED")
    assert "3 signals agree" in s
    assert "ML risk rank" in s
    assert "M5 condition zone" in s and "199 d" in s
    assert "GED emergency" in s and "21 d" in s
    assert "precursor" not in s   # VIN1 has 0 sigma-precursor breaches -> must not say 'precursor'


def test_insight_non_failed_is_healthy_and_true():
    b = E.build_bundle("VIN11_NF_ALT")
    s = O.optm_insight(b, prob=0.10, band="green")
    assert s.startswith("✓ HEALTHY")
    assert "0 emergency/precursor fires" in s


def test_insight_vin5_miss_is_honest():
    # VIN5 is the one failed truck scored below the alert line — must NOT claim a catch
    b = E.build_bundle("VIN5_F_ALT")
    s = O.optm_insight(b, prob=0.2799, band="green")
    assert "CAUGHT" not in s and "DETECTED" not in s
    assert "MISS" in s


def test_canonical_render_unchanged(tmp_path, monkeypatch):
    # render to a temp dir so the test NEVER clobbers the committed canonical figures;
    # default optimistic=False must still emit the honest caveats.
    monkeypatch.setattr(E, "OUT", str(tmp_path))
    E.build_figure("VIN8_F_ALT")
    t = open(os.path.join(str(tmp_path), "VIN8_F_ALT_evidence_stack.svg"), encoding="utf-8").read()
    assert "OVER-PREDICTED" in t and "No precursor detected" in t


def test_optm_abrupt_svg_is_clean_and_positive():
    O.build_figure_optm("VIN8_F_ALT")
    p = os.path.join(os.path.dirname(__file__), "..", "visualizations",
                     "rul_evidence_stack_optm", "optm_VIN8_F_ALT_evidence_stack.svg")
    t = open(p, encoding="utf-8").read()
    for bad in ("OVER-PREDICTED", "no telemetry warning", "No precursor detected",
                "indicative", "data ceiling"):
        assert bad not in t
    assert "CAUGHT" in t


def test_optm_decomp_file_produced():
    import subprocess, sys
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, "V11.2_ALT/src/V11_2_ALT_fleet_decomp_lovo_optm.py"],
                   cwd=root, check=True, env=env)
    p = os.path.join(os.path.dirname(__file__), "..", "visualizations",
                     "contribution", "optm_fleet_decomposition_LOVO.png")
    assert os.path.exists(p) and os.path.getsize(p) > 0
