"""
V11.2_ALT_rul_smoketest.py — smoke test for the frozen ALT RUL / survival bundle
(Part A) and the legacy V5.2 lifelines loaders (Part B).

Asserts:
  * bundle model shape/scale == frozen (5.1658 / 771.36)
  * median_ttf() ~= 718.5 d
  * survival_function([0, 718.5]) ~= [1.0, ~0.5]
  * ALT_rul_predict.py CLI exits 0
  * rul_band(365, 0.8) reproduces committed ci80 [677.3, 774.4] minus age
  * legacy_predict.load_all() returns 1x KaplanMeierFitter + 3x WeibullAFTFitter;
    KM median may be inf (only 40% events -> curve never crosses 0.5, valid) but
    not NaN; the 3 WeibullAFT medians are finite (needs lifelines==0.30.0)

Run: py -3 "V11.2_ALT/src/V11.2_ALT_rul_smoketest.py"
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Daimler-starter_motor_alternator_battery")
MODEL_DIR = ROOT / "V11.2_ALT" / "models" / "rul_survival_frozen"
LEGACY_DIR = MODEL_DIR / "legacy_v5.2"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # ── Part A: bundle + loader ──────────────────────────────────────────────
    rul = _load_module("alt_rul_predict", MODEL_DIR / "ALT_rul_predict.py")
    b = rul.load_bundle()

    assert b["model"]["shape"] == 5.1658, f"shape drift: {b['model']['shape']}"
    assert b["model"]["scale"] == 771.36, f"scale drift: {b['model']['scale']}"
    assert b["model"]["variant"] == "M0_fleet_shipped"

    med = rul.median_ttf()
    assert abs(med - 718.5) < 0.5, f"median_ttf drift: {med}"

    S = np.asarray(rul.survival_function([0.0, 718.5]))
    assert abs(S[0] - 1.0) < 1e-9, f"S(0) != 1.0: {S[0]}"
    assert abs(S[1] - 0.5) < 0.02, f"S(median) not ~0.5: {S[1]}"

    band = rul.rul_band(age_days=365, ci=0.8)
    assert abs(band["lower"] - (677.3 - 365)) < 5.0, f"band lower off: {band}"
    assert abs(band["upper"] - (774.4 - 365)) < 5.0, f"band upper off: {band}"

    # percentile(0.5) == median
    assert abs(rul.percentile(0.5) - med) < 1e-6, "percentile(0.5) != median"

    # posterior array embedded and correctly shaped
    arr = np.asarray(b["posterior_samples_M0"])
    assert arr.shape == (10000, 2), f"posterior array shape {arr.shape}"

    # CLI exits 0
    r = subprocess.run([sys.executable, str(MODEL_DIR / "ALT_rul_predict.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"CLI exit {r.returncode}\n{r.stderr}"
    print("  [A] bundle + loader + CLI OK "
          f"(shape={b['model']['shape']} scale={b['model']['scale']} "
          f"median={med:.2f} S(median)={S[1]:.4f})")

    # ── Part B: legacy lifelines fitters ─────────────────────────────────────
    from lifelines import KaplanMeierFitter, WeibullAFTFitter
    legacy = _load_module("legacy_predict", LEGACY_DIR / "legacy_predict.py")
    fitters = legacy.load_all()
    assert len(fitters) == 4, f"expected 4 legacy fitters, got {len(fitters)}"
    kms = [f for f in fitters.values() if isinstance(f, KaplanMeierFitter)]
    afts = [f for f in fitters.values() if isinstance(f, WeibullAFTFitter)]
    assert len(kms) == 1, f"expected 1 KaplanMeierFitter, got {len(kms)}"
    assert len(afts) == 3, f"expected 3 WeibullAFTFitter, got {len(afts)}"
    # KM median is legitimately inf (only 10/25 = 40% events -> curve never crosses
    # 0.5); require non-NaN for KM, finite for the AFT models.
    km_med = float(kms[0].median_survival_time_)
    assert not np.isnan(km_med), "KM median is NaN"
    for f in afts:
        assert np.isfinite(float(f.median_survival_time_)), "WeibullAFT median not finite"
    print(f"  [B] legacy OK (1x KM median={km_med} (inf=undefined, <50% events, valid) "
          f"+ 3x WeibullAFT medians finite; lifelines present)")

    print("RUL SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
