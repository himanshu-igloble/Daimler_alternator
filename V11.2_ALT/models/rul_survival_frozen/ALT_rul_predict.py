"""
ALT_rul_predict.py — self-contained loader for the frozen ALT RUL / survival
model (shipped fleet Weibull M0; fit in V11.1_ALT, reproduces V10.6.2_ALT).

Library use:
    from ALT_rul_predict import (load_bundle, survival_function, median_ttf,
                                 percentile, rul_band, covariate_scale)
    b   = load_bundle()
    S   = survival_function([180, 365, 601, 718])      # fleet S(t)
    m   = median_ttf()                                 # ~718.5 d
    band = rul_band(age_days=365, ci=0.8)              # runway band

CLI:
    py -3 ALT_rul_predict.py            # prints fleet summary + S(t) + a rul_band

Requires only: numpy, pandas, joblib.

HONEST CAVEAT: this is a FLEET-LEVEL survival curve (M0 shipped; M1/M2 covariate
variants are SHELVED / exposure-confounded). It is NOT a per-truck days-to-failure
clock — per-truck RUL was closed at n=25 (V10.6.2). Use it as a cohort prior /
runway band, not a per-VIN countdown.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd  # noqa: F401  (kept for API parity / downstream convenience)

HERE = Path(__file__).resolve().parent
BUNDLE_PATH = HERE / "ALT_rul_survival_bundle.joblib"
LN2 = np.log(2.0)

_CACHE = {}


def load_bundle(path=BUNDLE_PATH):
    """Load (and memoize) the frozen RUL bundle dict."""
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def _params(bundle=None):
    b = bundle if bundle is not None else load_bundle()
    return float(b["model"]["shape"]), float(b["model"]["scale"]), b


def survival_function(t, bundle=None):
    """Fleet survival S(t) = exp(-(t/scale)**shape). Accepts scalar or array."""
    shape, scale, _ = _params(bundle)
    t = np.asarray(t, dtype=float)
    return np.exp(-(t / scale) ** shape)


def median_ttf(bundle=None):
    """Fleet median time-to-failure = scale*(ln2)**(1/shape) [days]."""
    shape, scale, _ = _params(bundle)
    return float(scale * LN2 ** (1.0 / shape))


def percentile(p, bundle=None):
    """Time (days) by which a fraction p of the cohort has failed.
    Weibull quantile: t = scale*(-ln(1-p))**(1/shape). p=0.5 -> median."""
    shape, scale, _ = _params(bundle)
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("percentile p must be in (0, 1)")
    out = scale * (-np.log(1.0 - p)) ** (1.0 / shape)
    return float(out) if out.ndim == 0 else out


def _posterior_median_ci(bundle, ci):
    """(lower, upper) days for the fleet median from the M0 posterior draws."""
    arr = np.asarray(bundle["posterior_samples_M0"], dtype=float)   # (N, [shape, scale0])
    med_draws = arr[:, 1] * LN2 ** (1.0 / arr[:, 0])
    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    lo, hi = np.percentile(med_draws, [lo_q, hi_q])
    return float(lo), float(hi)


def rul_band(age_days, ci=0.8, bundle=None):
    """Remaining-useful-life runway band for a truck of the given age.
    point = fleet_median - age; lower/upper from the posterior CI on the median
    (default 80% -> reproduces committed ci80 [677.3, 774.4] minus age)."""
    b = bundle if bundle is not None else load_bundle()
    med = median_ttf(b)
    lo, hi = _posterior_median_ci(b, ci)
    age = float(age_days)
    return {
        "age_days": age,
        "ci": ci,
        "point": med - age,
        "lower": lo - age,
        "upper": hi - age,
        "median_ttf_days": med,
        "median_ci": [lo, hi],
        "note": "fleet-level runway band, NOT a per-truck clock",
    }


def covariate_scale(x, variant="M1", bundle=None):
    """SHELVED AFT scale for covariate vector x: scale_i = scale0*exp(beta.x).
    variant in {"M1","M2"}. M1/M2 are EXPOSURE-CONFOUNDED and NOT recommended for
    deployment — provided for reference/reproduction only."""
    b = bundle if bundle is not None else load_bundle()
    shelved = b["covariate_variants_shelved"]
    if variant not in ("M1", "M2"):
        raise ValueError("variant must be 'M1' or 'M2' (both SHELVED)")
    beta = np.asarray(shelved[variant]["beta"], dtype=float)
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != beta.shape[0]:
        raise ValueError(f"{variant} expects {beta.shape[0]} covariate(s), got {x.shape[-1]}")
    scale0 = float(shelved[variant]["scale0"])
    return float(scale0 * np.exp(np.dot(x, beta)))


def main():
    b = load_bundle()
    print("=" * 66)
    print("ALT RUL / SURVIVAL — frozen fleet Weibull (M0, shipped)")
    print(f"  provenance : {b['champion_version']}   created {b['created']}")
    print(f"  model      : Weibull  shape={b['model']['shape']}  "
          f"scale={b['model']['scale']}  ({b['model']['variant']})")
    fl = b["fleet"]
    print(f"  fleet      : median TTF {fl['median_ttf_days']} d   "
          f"ci80 {fl['ci80']}   n={fl['n_cohort']} ({fl['n_events']} events)")
    print(f"  median_ttf(): {median_ttf(b):.2f} d")
    print("-" * 66)
    print("  S(t) fleet survival:")
    for t in (180, 365, 601, 718):
        print(f"    t={t:>4} d   S={float(survival_function(t, b)):.4f}")
    print("-" * 66)
    band = rul_band(age_days=365, ci=0.8, bundle=b)
    print(f"  rul_band(age=365 d, ci=0.8): point={band['point']:.1f} d  "
          f"[{band['lower']:.1f}, {band['upper']:.1f}]")
    print(f"  percentile(0.1)={percentile(0.1, b):.1f} d   "
          f"percentile(0.5)={percentile(0.5, b):.1f} d   "
          f"percentile(0.9)={percentile(0.9, b):.1f} d")
    print("-" * 66)
    print(f"  CAVEAT: {b['honest_caveat']}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
