# RUL-PREDICTION_ALTERNATOR — V11.2

**V11.2 ALT** — DICV management validation dossier for the frozen V11.1 covariate-RUL model.

Five validation analyses, technical + management reports, the DICV management-review deck, a
management Q&A, and per-VIN RUL + physics evidence-stack graphs. No model change from V11.1 —
this iteration is validation/evidence only.

- Deliverable: `V11.2_ALT/`
- Prior-version lineage: `VERSION_COMPARISON_V10.6.2_vs_V11_vs_V11.1.md`

## Loadable model artifact (deployable)

The frozen champion classifier is packaged as a load-and-predict joblib bundle:

- **Bundle:** `V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_champion_bundle.joblib`
- **Loader / CLI:** `V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_predict.py`
- **Provenance + verification + manifest:** alongside the bundle in the same folder
- **Build / verify scripts:** `V11.2_ALT/src/V11.2_ALT_package_champion.py` · `..._iteration_comparison.py` · `..._bundle_smoketest.py`

Model: `RidgeClassifier(alpha=1.0)`, 6 features, **LOVO AUROC 0.9267**, Youden threshold 0.4456; tiers on raw prob red ≥ 0.55 / amber ≥ 0.35.

```bash
py -3 V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_predict.py \
      V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_training_matrix.csv
```

### RUL / survival model (WHEN, fleet)

Params-driven fleet survival bundle in `V11.2_ALT/models/rul_survival_frozen/` (params fit in V11.1_ALT / V10.6.2_ALT):

- **Bundle:** `ALT_rul_survival_bundle.joblib` — Weibull `S(t)=exp(-(t/771.36)^5.1658)` + 10k posterior draws + shelved M1/M2
- **Loader / CLI:** `ALT_rul_predict.py` — `survival_function(t)`, `median_ttf()`, `rul_band(age_days)`
- **Legacy (superseded):** `legacy_v5.2/` — V5.2 lifelines fitters (KaplanMeier + 3× WeibullAFT), reference only (needs `lifelines==0.30.0`)

Fleet Weibull: **median TTF 718.5 d, 80% CI [677.3, 774.4]** (25 trucks / 10 events). A fleet-level runway band, **not** a per-truck clock; M0 shipped, M1/M2 shelved.
