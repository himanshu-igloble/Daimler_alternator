# ALT RUL / Survival — Frozen Fleet Weibull (M0, shipped) — Deployable Artifact

Loadable packaging of the **shipped alternator RUL model**: a Bayesian
**fleet-level Weibull** survival curve, `S(t) = exp(-(t/scale)**shape)` with
**shape = 5.1658, scale = 771.36** (variant **M0**), giving a fleet
**median time-to-failure of 718.5 days** and an **80% credible interval of
[677.3, 774.4] days** across the 25-truck cohort (10 failed / 15 non-failed).

**Provenance (why the version numbers):** the Weibull posterior was fit in
**V11.1_ALT** (`cache/weibull/`), reproducing the **V10.6.2_ALT** honest-baseline
RUL analysis. This folder lives under `V11.2_ALT/` because V11.2 is the ALT
model-hub / deliverable home; the params inside are the V11.1 (M0) frozen values,
unchanged. Packaging never re-fits — it re-wraps the frozen params and verifies
they reproduce the committed artifacts (parity gates P1–P4).

## What is (and isn't) shipped
- **M0 (shipped):** fleet Weibull, no covariate (`beta = 0`). This is the
  deliverable.
- **M1 / M2 (SHELVED):** AFT covariate variants, `scale_i = scale0 * exp(beta·x)`.
  Their negative `beta` reflects **exposure/usage**, not a health signal — they
  are exposure-confounded and **not recommended**. Kept in the bundle for
  reference only (`covariate_variants_shelved`).

## Files
| File | What it is |
|---|---|
| `ALT_rul_survival_bundle.joblib` | plain dict: model params (shape/scale/variant), shelved M1/M2, embedded 10000×(shape,scale0) posterior array (for CI), fleet summary, score-mapping formulas, honest caveat, env. No custom classes — loads anywhere with numpy/joblib. |
| `ALT_rul_predict.py` | loader + CLI (`load_bundle`, `survival_function`, `median_ttf`, `percentile`, `rul_band`, `covariate_scale`) |
| `aft_params_M0.json` / `M1` / `M2` | provenance copies of the frozen AFT params |
| `fleet_weibull_params.json` | provenance copy of the fleet median + CI |
| `fleet_survival_curve.csv` | provenance copy of the committed S(t) curve (t=0..1200) |
| `posterior_samples_M0.csv` | provenance copy of the 10000 posterior draws (CI source) |
| `ALT_rul_verification.json` | parity-gate results P1–P4 (real numbers) |
| `ALT_rul_MANIFEST.json` | SHA256 of every top-level file + inputs + env |
| `legacy_v5.2/` | **SUPERSEDED** V5.2 lifelines fitters (KM + 3× WeibullAFT), separate README + manifest |

## Quick start
```
py -3 ALT_rul_predict.py
```
Prints the fleet summary, `S(t)` at t = 180 / 365 / 601 / 718 d, and a sample
`rul_band`. Library use:
```python
from ALT_rul_predict import load_bundle, survival_function, median_ttf, rul_band
median_ttf()                      # ~718.5 d
survival_function([180, 365, 601, 718])
rul_band(age_days=365, ci=0.8)    # {'point': ~353.5, 'lower': ..., 'upper': ...}
```

## Parametrization
```
S(t)        = exp(-(t/scale)**shape)
median      = scale*(ln 2)**(1/shape)
quantile(p) = scale*(-ln(1-p))**(1/shape)          # p=0.5 -> median
AFT (SHELVED, M1/M2): scale_i = scale0*exp(beta·x)
```

## Parity gates (P1–P4, in `ALT_rul_verification.json`)
- **P1** reconstruct `S(t)` at every t in `fleet_survival_curve.csv` →
  `max|S_recon − S_t| < 1e-3` (actual ~8.5e-6).
- **P2** `median = 771.36*(ln2)**(1/5.1658)` = 718.53 d = committed 718.5 within 0.5 d.
- **P3** posterior CI: per draw `median_i = scale0_i*(ln2)**(1/shape_i)`; the
  **10th/90th percentiles (80% band)** = [677.3, 774.4] = committed within ±5 d.
  (The 95% / 2.5–97.5 band = [654.8, 805.7] and does **not** match — the shipped
  CI is the 80% band.)
- **P4** joblib round-trip returns a bit-identical reconstructed `S(t)` vector.

## Honesty notes
- This is a **FLEET-LEVEL** survival curve — a cohort prior / runway band, **not
  a per-truck days-to-failure clock**. Per-truck RUL was closed at n=25: V10.6.2
  showed a per-truck model cannot beat the fleet clock (MAE ~142 d vs a ~50 d
  fleet-median prior).
- `rul_band(age)` subtracts truck age from the fleet median and its posterior CI.
  It is a fleet expectation minus age, not a learned per-VIN remaining life.
- M1/M2 are exposure-confounded; do not use their covariate `beta` as a health
  signal.
- Rebuild + re-verify:
  `py -3 "V11.2_ALT/src/V11.2_ALT_package_rul.py"`, then
  `py -3 "V11.2_ALT/src/V11.2_ALT_rul_smoketest.py"`. Rebuilding refreshes the
  `created` date / `git_head`, so the `.joblib` bytes and MANIFEST SHA256 change
  across rebuilds; the MANIFEST verifies the specific committed bytes
  (tamper-detection), not cross-run byte reproducibility.
