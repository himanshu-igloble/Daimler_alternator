# Model Registry — Daimler / BharatBenz Alternator

Every **deployable, verified** model artifact shipped in this repo, in one place. Each artifact is
self-contained — a `joblib` bundle + a `predict.py` loader/CLI + provenance copies + a
`verification.json` (parity gates, real numbers) + a SHA256 `MANIFEST.json`. Bundles load with only
`joblib` + `numpy` + `pandas` + `scikit-learn 1.8.x`; the legacy survival fitters additionally need
`lifelines 0.30.0`.

> **Honest-engineering note.** Metrics below are the frozen, report-backed validation numbers. The
> classifier AUROC is a **LOVO ranking** metric at n=25 (not field accuracy); the RUL is a **fleet-level**
> survival window, **not** a per-truck days-to-failure clock (per-truck RUL is mathematically closed at
> n=25). Every artifact folder carries its own README with the full caveats.

## Deployable models

| # | Model | Answers | Type | Path | Headline |
|---|---|---|---|---|---|
| 1 | **Failure classifier** (frozen champion) | WHICH trucks are at risk | `RidgeClassifier(α=1.0)` in an sklearn `Pipeline` (impute→scale→ridge) | [`V11.2_ALT/models/V10.5.3_ridge_frozen_champion/`](./V11.2_ALT/models/V10.5.3_ridge_frozen_champion) | LOVO **AUROC 0.9267**, threshold 0.4456; tiers red≥0.55 / amber≥0.35 |
| 2 | **Fleet RUL / survival** | WHEN, at fleet level | Weibull, params-driven (V11.1 M0; reproduces V10.6.2) | [`V11.2_ALT/models/rul_survival_frozen/`](./V11.2_ALT/models/rul_survival_frozen) | median TTF **718.5 d**, 80% CI [677.3, 774.4] |
| 3 | **Legacy survival** (reference, superseded) | — | `lifelines` KaplanMeier + 3× WeibullAFT | [`.../rul_survival_frozen/legacy_v5.2/`](./V11.2_ALT/models/rul_survival_frozen/legacy_v5.2) | archived V5.2 fitters |

## Quick start

```bash
# 1. Failure classifier — score a feature CSV (bundle imputes NaNs with training medians)
py -3 V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_predict.py \
      V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_training_matrix.csv

# 2. Fleet RUL / survival — fleet S(t), median TTF, runway band
py -3 V11.2_ALT/models/rul_survival_frozen/ALT_rul_predict.py

# 3. Legacy survival fitters (needs lifelines 0.30.0)
py -3 V11.2_ALT/models/rul_survival_frozen/legacy_v5.2/legacy_predict.py
```

## What is (and isn't) inside a bundle

- **Inside** each classifier bundle: the fitted `SimpleImputer` (medians), `StandardScaler` (mean/scale),
  and `RidgeClassifier` (coefficients) — all serialized in one `Pipeline` — plus the exact feature-column
  list, threshold, tier bands, frozen metrics, and build environment. Nothing else is required to score
  an already-engineered feature matrix.
- **Not inside**: the **feature engineering** that turns raw CAN telemetry into the model's input columns.
  That lives in each version's `*/src/` feature pipeline. The provenance `*_training_matrix.csv` in every
  artifact folder documents the exact expected input schema.

## Provenance & verification

Each folder's `MANIFEST.json` records a SHA256 for every file and every input; `verification.json` records
the packaging parity gates (e.g. classifier LOVO AUROC reproduced to 0.926667, per-VIN prob diff < 6e-5;
RUL reconstructed S(t) vs the committed curve to 8.45e-06). Rebuild scripts live in `V11.2_ALT/src/`
(`*_package_champion.py`, `*_package_rul.py`, and their smoke tests).
