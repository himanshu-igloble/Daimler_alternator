"""
legacy_predict.py — loader + smoke test for the SUPERSEDED V5.2 ALT lifelines
survival models (kept for reference / comparison only).

>>> SUPERSEDED by V10.6.2 / V11.1 fleet Weibull (M0). <<<
The shipped ALT RUL is ../ALT_rul_survival_bundle.joblib. These four fitted
lifelines objects predate that and are retained only for provenance.

There are exactly 4 objects here:
  V5.2_20_5_ALT_kaplan_meier.joblib                    -> KaplanMeierFitter
  V5.2_20_5_ALT_weibull_aft.joblib                     -> WeibullAFTFitter
  V5.2_20_5_ALT_weibull_aft_silent_failure.joblib      -> WeibullAFTFitter
  V5.2_20_5_ALT_weibull_aft_volatility_explosion.joblib-> WeibullAFTFitter

REQUIRES: pip install lifelines==0.30.0  (to unpickle the fitters).

CLI:
    py -3 legacy_predict.py     # loads all 4, prints medians / summaries, smoke test
"""
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent

KM_FILE = "V5.2_20_5_ALT_kaplan_meier.joblib"
AFT_FILES = [
    "V5.2_20_5_ALT_weibull_aft.joblib",
    "V5.2_20_5_ALT_weibull_aft_silent_failure.joblib",
    "V5.2_20_5_ALT_weibull_aft_volatility_explosion.joblib",
]


def load_all(folder=HERE):
    """Load all 4 fitters. Returns {name: fitter}. Needs lifelines==0.30.0."""
    folder = Path(folder)
    out = {}
    for name in [KM_FILE] + AFT_FILES:
        out[name] = joblib.load(folder / name)
    return out


def _median(fitter):
    """median_survival_time_ works for both KM and WeibullAFT in lifelines 0.30."""
    try:
        return float(fitter.median_survival_time_)
    except Exception:
        return float("nan")


def main():
    from lifelines import KaplanMeierFitter, WeibullAFTFitter

    fitters = load_all()

    km = fitters[KM_FILE]
    print("=" * 66)
    print("LEGACY V5.2 ALT lifelines survival models  (SUPERSEDED — reference only)")
    print("-" * 66)
    print(f"[KM] {KM_FILE}")
    print(f"     type = {type(km).__name__}   median_survival_time_ = {_median(km):.2f}")

    for name in AFT_FILES:
        aft = fitters[name]
        med = _median(aft)
        print(f"[AFT] {name}")
        print(f"      type = {type(aft).__name__}   median_survival_time_ = {med:.2f}")
        try:
            summ = aft.summary[["coef", "p"]]
            print(summ.to_string())
        except Exception as e:  # pragma: no cover
            print(f"      (.summary unavailable: {e})")

    # ── smoke assertions ─────────────────────────────────────────────────────
    # NOTE: the KM median is legitimately `inf` here — only 10/25 (40%) of the
    # cohort failed, so the KM survival curve never crosses 0.5 and the median is
    # undefined by construction (a valid fitted property, NOT a load failure).
    # We therefore require the KM median to be non-NaN (finite OR inf) and require
    # the 3 WeibullAFT medians to be finite.
    assert isinstance(km, KaplanMeierFitter), "KM file is not a KaplanMeierFitter"
    km_med = _median(km)
    assert not np.isnan(km_med), "KM median is NaN (fitter failed to load/compute)"
    for name in AFT_FILES:
        aft = fitters[name]
        assert isinstance(aft, WeibullAFTFitter), f"{name} is not a WeibullAFTFitter"
        assert np.isfinite(_median(aft)), f"{name} median not finite"
    print("-" * 66)
    print("LEGACY SMOKE PASS: 1x KaplanMeierFitter (median=inf, <50% events, valid) "
          "+ 3x WeibullAFTFitter (medians finite)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
