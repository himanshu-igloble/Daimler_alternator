"""
V11.2_ALT_package_rul.py — package the SHIPPED ALT RUL / survival model into a
loadable joblib bundle under V11.2_ALT/models/rul_survival_frozen/.

The current ALT RUL is a Bayesian Weibull / AFT posterior fit in V11.1_ALT
(reproducing the V10.6.2_ALT honest-baseline analysis). Only the frozen PARAMS
persist; this script re-wraps them into a plain-dict artifact + parity gates.

Parametrization (Weibull):
  S(t)   = exp(-(t/scale)**shape)
  median = scale * (ln 2)**(1/shape)
  AFT covariate link (M1/M2 only, SHELVED): scale_i = scale0 * exp(beta . x)

Parity gates (script ABORTS on any failure — tolerances are NOT loosened):
  P1  reconstruct S(t)=exp(-(t/771.36)**5.1658) at every t in
      fleet_survival_curve.csv  ->  max|S_recon - S_t| < 1e-3
  P2  median = 771.36*(ln2)**(1/5.1658) == committed 718.5 within 0.5 d, and
      == fleet_weibull_params.json median
  P3  posterior CI: per draw median_i = scale0_i*(ln2)**(1/shape_i); the
      10th/90th percentiles (80% band) ~= committed ci [677.3,774.4] within +-5 d
  P4  joblib round-trip returns a bit-identical reconstructed S(t) vector

Outputs -> V11.2_ALT/models/rul_survival_frozen/:
  ALT_rul_survival_bundle.joblib
  aft_params_M0.json / aft_params_M1.json / aft_params_M2.json      (provenance)
  fleet_weibull_params.json                                          (provenance)
  fleet_survival_curve.csv / posterior_samples_M0.csv                (provenance)
  ALT_rul_verification.json
  ALT_rul_MANIFEST.json
  legacy_v5.2/  (4 lifelines joblibs copied + legacy_MANIFEST.json)

Run: py -3 "V11.2_ALT/src/V11.2_ALT_package_rul.py"
"""
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lifelines
    LIFELINES_VER = lifelines.__version__
except Exception:  # pragma: no cover - lifelines only needed for legacy
    LIFELINES_VER = "not-installed"

ROOT = Path(r"D:\Daimler-starter_motor_alternator_battery")
WBL = ROOT / "V11.1_ALT" / "cache" / "weibull"
AFT_M0 = WBL / "aft_params_M0.json"
AFT_M1 = WBL / "aft_params_M1.json"
AFT_M2 = WBL / "aft_params_M2.json"
FLEET_PARAMS = WBL / "fleet_weibull_params.json"
FLEET_CURVE = WBL / "fleet_survival_curve.csv"
POST_M0 = WBL / "posterior_samples_M0.csv"

LEGACY_SRC = ROOT / "V5.2_ALT" / "models" / "rul"
LEGACY_JOBLIBS = [
    "V5.2_20_5_ALT_kaplan_meier.joblib",
    "V5.2_20_5_ALT_weibull_aft.joblib",
    "V5.2_20_5_ALT_weibull_aft_silent_failure.joblib",
    "V5.2_20_5_ALT_weibull_aft_volatility_explosion.joblib",
]

OUT = ROOT / "V11.2_ALT" / "models" / "rul_survival_frozen"
LEGACY_OUT = OUT / "legacy_v5.2"

# frozen committed constants (M0 = shipped fleet model)
FROZEN_SHAPE = 5.1658
FROZEN_SCALE = 771.36
FROZEN_MEDIAN = 718.5
FROZEN_CI80 = [677.3, 774.4]
LN2 = np.log(2.0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def weibull_S(t, shape, scale):
    t = np.asarray(t, dtype=float)
    return np.exp(-(t / scale) ** shape)


def weibull_median(shape, scale):
    return scale * LN2 ** (1.0 / shape)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LEGACY_OUT.mkdir(parents=True, exist_ok=True)

    # ── load frozen inputs ───────────────────────────────────────────────────
    m0 = json.loads(AFT_M0.read_text())
    m1 = json.loads(AFT_M1.read_text())
    m2 = json.loads(AFT_M2.read_text())
    fleet = json.loads(FLEET_PARAMS.read_text())
    shape = float(m0["map_shape"])
    scale = float(m0["map_scale0"])
    assert abs(shape - FROZEN_SHAPE) < 1e-9 and abs(scale - FROZEN_SCALE) < 1e-9, \
        f"M0 params drifted: shape={shape} scale={scale}"

    # ── P1: reconstruct S(t) over the committed curve grid ───────────────────
    curve = pd.read_csv(FLEET_CURVE)
    S_recon = weibull_S(curve["t_days"].values, shape, scale)
    p1_max = float(np.max(np.abs(S_recon - curve["S_t"].values)))
    print(f"[P1] max|S_recon - S_t| = {p1_max:.3e} (tol 1e-3) over {len(curve)} rows")
    assert p1_max < 1e-3, f"P1 FAIL: {p1_max:.3e}"

    # ── P2: median formula vs committed ──────────────────────────────────────
    median = float(weibull_median(shape, scale))
    p2_diff = abs(median - FROZEN_MEDIAN)
    p2_json = abs(median - float(fleet["median"]))
    print(f"[P2] median = {median:.4f} d | committed {FROZEN_MEDIAN} "
          f"(diff {p2_diff:.4f}) | fleet_json {fleet['median']} (diff {p2_json:.4f})")
    assert p2_diff < 0.5, f"P2 FAIL vs 718.5: {p2_diff:.4f}"
    assert p2_json < 0.5, f"P2 FAIL vs fleet_json: {p2_json:.4f}"

    # ── P3: 80% posterior CI on the median ───────────────────────────────────
    post = pd.read_csv(POST_M0)
    assert list(post.columns) == ["shape", "scale0", "beta1"], \
        f"unexpected posterior columns: {list(post.columns)}"
    assert len(post) == 10000, f"expected 10000 draws, got {len(post)}"
    med_draws = post["scale0"].values * LN2 ** (1.0 / post["shape"].values)
    ci80 = np.percentile(med_draws, [10, 90])
    ci95 = np.percentile(med_draws, [2.5, 97.5])
    d_lo = abs(ci80[0] - FROZEN_CI80[0])
    d_hi = abs(ci80[1] - FROZEN_CI80[1])
    print(f"[P3] 80% CI (10/90 pct) = [{ci80[0]:.2f}, {ci80[1]:.2f}] | "
          f"committed [{FROZEN_CI80[0]}, {FROZEN_CI80[1]}] | "
          f"|dlo|={d_lo:.2f} |dhi|={d_hi:.2f} (tol 5 d)")
    print(f"     (95% CI (2.5/97.5) = [{ci95[0]:.2f}, {ci95[1]:.2f}] -> "
          f"does NOT match committed; 80% band is the shipped CI)")
    assert d_lo <= 5.0 and d_hi <= 5.0, f"P3 FAIL: dlo={d_lo:.2f} dhi={d_hi:.2f}"

    # embed posterior (shape, scale0) as a numpy array for CI at load time
    post_arr = np.column_stack([post["shape"].values,
                                post["scale0"].values]).astype(np.float64)
    assert post_arr.shape == (10000, 2)

    # ── build bundle ─────────────────────────────────────────────────────────
    env = {"python": platform.python_version(), "sklearn": __import__("sklearn").__version__,
           "numpy": np.__version__, "pandas": pd.__version__,
           "joblib": joblib.__version__, "lifelines": LIFELINES_VER,
           "platform": platform.platform()}
    bundle = {
        "component": "alternator",
        "artifact": "fleet_weibull_survival_rul",
        "champion_version": "V11.1_ALT (M0; reproduces V10.6.2)",
        "created": date.today().isoformat(),
        "model": {"family": "weibull", "shape": shape, "scale": scale,
                  "variant": "M0_fleet_shipped"},
        "covariate_variants_shelved": {
            "M1": {"shape": float(m1["map_shape"]), "scale0": float(m1["map_scale0"]),
                   "beta": [float(b) for b in m1["map_beta"]]},
            "M2": {"shape": float(m2["map_shape"]), "scale0": float(m2["map_scale0"]),
                   "beta": [float(b) for b in m2["map_beta"]]},
            "note": "M1/M2 are AFT covariate variants (scale_i = scale0*exp(beta.x)). "
                    "They are EXPOSURE-CONFOUNDED (negative beta reflects usage/exposure, "
                    "not a health signal) and are NOT recommended. Shipped model is M0.",
        },
        "posterior_samples_M0": post_arr,               # 10000 x (shape, scale0)
        "posterior_samples_M0_columns": ["shape", "scale0"],
        "fleet": {"median_ttf_days": FROZEN_MEDIAN, "ci80": FROZEN_CI80,
                  "n_cohort": int(fleet["n_cohort"]), "n_events": int(fleet["n_events"])},
        "score_mapping": (
            "S(t) = exp(-(t/scale)**shape);  "
            "median = scale*(ln2)**(1/shape);  "
            "quantile(p) [time by which fraction p have failed] = "
            "scale*(-ln(1-p))**(1/shape);  "
            "rul_band(age) point = median - age, band from 80% posterior CI on median;  "
            "AFT (SHELVED, M1/M2): scale_i = scale0*exp(beta.x)."
        ),
        "honest_caveat": (
            "Fleet-LEVEL survival curve. M0 is the SHIPPED model; M1/M2 covariate "
            "variants are SHELVED (exposure-confounded). This is NOT a per-truck "
            "days-to-failure clock: per-truck RUL was closed at n=25 (V10.6.2 showed "
            "a per-truck model cannot beat the fleet clock; MAE ~142 d vs ~50 d). "
            "Use it as a cohort prior / runway band, not a per-VIN countdown."
        ),
        "environment": env,
    }
    bundle_path = OUT / "ALT_rul_survival_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    # ── P4: joblib round-trip bit-identical S(t) vector ──────────────────────
    b2 = joblib.load(bundle_path)
    S_after = weibull_S(curve["t_days"].values, b2["model"]["shape"], b2["model"]["scale"])
    p4_ok = bool(np.array_equal(S_recon, S_after))
    print(f"[P4] joblib round-trip S(t) bit-identical: {p4_ok}")
    assert p4_ok, "P4 FAIL: round-trip S(t) drift"

    # ── provenance copies ────────────────────────────────────────────────────
    for src in (AFT_M0, AFT_M1, AFT_M2, FLEET_PARAMS, FLEET_CURVE, POST_M0):
        shutil.copy2(src, OUT / src.name)

    # ── verification.json ────────────────────────────────────────────────────
    verification = {
        "created": date.today().isoformat(),
        "P1_survival_reconstruction": {
            "max_abs_diff": p1_max, "tol": 1e-3, "n_rows": int(len(curve)), "pass": True},
        "P2_median": {
            "value_days": median, "committed": FROZEN_MEDIAN,
            "diff_vs_committed": p2_diff, "diff_vs_fleet_json": p2_json,
            "tol_days": 0.5, "pass": True},
        "P3_ci80": {
            "ci80_10_90": [float(ci80[0]), float(ci80[1])],
            "committed": FROZEN_CI80,
            "abs_diff_lower": d_lo, "abs_diff_upper": d_hi, "tol_days": 5.0,
            "ci95_2p5_97p5_for_reference": [float(ci95[0]), float(ci95[1])],
            "matched_level": "80% (10th/90th percentile of posterior median)",
            "pass": True},
        "P4_roundtrip_survival_vector": {"bit_identical": p4_ok, "pass": True},
        "environment": env,
    }
    (OUT / "ALT_rul_verification.json").write_text(json.dumps(verification, indent=2))

    # ── legacy V5.2 lifelines copies + manifest (Part B) ─────────────────────
    for name in LEGACY_JOBLIBS:
        shutil.copy2(LEGACY_SRC / name, LEGACY_OUT / name)
    legacy_files = sorted(p for p in LEGACY_OUT.iterdir()
                          if p.is_file() and p.name != "legacy_MANIFEST.json")
    legacy_manifest = {
        "artifact": "SUPERSEDED V5.2 ALT lifelines survival models (reference only)",
        "superseded_by": "V10.6.2 / V11.1 fleet Weibull (M0) — see ../ALT_rul_survival_bundle.joblib",
        "requires": "lifelines==0.30.0 to unpickle",
        "created": date.today().isoformat(),
        "git_head": git_head(),
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in legacy_files],
    }
    (LEGACY_OUT / "legacy_MANIFEST.json").write_text(json.dumps(legacy_manifest, indent=2))

    # ── top-level MANIFEST.json (hashes every top-level file) ────────────────
    files = sorted(p for p in OUT.iterdir()
                   if p.is_file() and p.name != "ALT_rul_MANIFEST.json")
    manifest = {
        "artifact": "ALT RUL / survival: shipped fleet Weibull M0 (V11.1_ALT; reproduces V10.6.2)",
        "created": date.today().isoformat(),
        "git_head": git_head(),
        "environment": env,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in files],
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                   for p in (AFT_M0, AFT_M1, AFT_M2, FLEET_PARAMS, FLEET_CURVE, POST_M0)],
        "legacy": {"folder": "legacy_v5.2", "manifest": "legacy_v5.2/legacy_MANIFEST.json",
                   "n_joblibs": len(LEGACY_JOBLIBS)},
    }
    (OUT / "ALT_rul_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nPACKAGED OK -> {OUT}")
    for p in sorted(OUT.iterdir()):
        tag = "/" if p.is_dir() else ""
        print(f"  {p.name}{tag}")
    for p in sorted(LEGACY_OUT.iterdir()):
        print(f"  legacy_v5.2/{p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
