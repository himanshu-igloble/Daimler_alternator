# Legacy V5.2 ALT Survival Models — SUPERSEDED (reference only)

> **SUPERSEDED by V10.6.2 / V11.1.** The shipped ALT RUL is the fleet Weibull
> **M0** in the parent folder (`../ALT_rul_survival_bundle.joblib`). These four
> `lifelines` objects are the earlier V5.2 survival experiments, kept here only
> for provenance / comparison. Do **not** deploy them.

## Why superseded
V5.2 fit per-covariate `WeibullAFTFitter` models (silent-failure and
volatility-explosion strata) plus a Kaplan–Meier baseline. The V10.6.2 honest
baseline showed per-truck / covariate RUL cannot beat the fleet clock at n=25
(MAE ~142 d vs a ~50 d fleet-median prior), so the program shipped the
fleet-level Weibull posterior (M0) instead. See parent `README.md`.

## Files
| File | Object |
|---|---|
| `V5.2_20_5_ALT_kaplan_meier.joblib` | `KaplanMeierFitter` |
| `V5.2_20_5_ALT_weibull_aft.joblib` | `WeibullAFTFitter` (base) |
| `V5.2_20_5_ALT_weibull_aft_silent_failure.joblib` | `WeibullAFTFitter` (silent-failure stratum) |
| `V5.2_20_5_ALT_weibull_aft_volatility_explosion.joblib` | `WeibullAFTFitter` (volatility-explosion stratum) |
| `legacy_predict.py` | loader (`load_all()`) + CLI + smoke test |
| `legacy_MANIFEST.json` | SHA256 of the 4 joblibs + loader |

## Requirements
```
pip install lifelines==0.30.0
```
The `.joblib` files are pickled `lifelines` fitters — unpickling requires
`lifelines==0.30.0` (the version they were fit under). Without it, `joblib.load`
will fail to reconstruct the classes.

## Quick start
```
py -3 legacy_predict.py
```
Prints `median_survival_time_` for the KM fitter and `.summary` +
`median_survival_time_` for each of the 3 WeibullAFT fitters, then asserts the
types (1× KM + 3× WeibullAFT) and that every median is finite
(`LEGACY SMOKE PASS`).
