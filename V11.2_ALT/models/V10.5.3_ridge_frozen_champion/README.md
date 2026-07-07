# ALT Frozen Champion — V10.5.3 Ridge (V11.2-validated) — Deployable Artifact

Loadable packaging of the frozen alternator champion: `RidgeClassifier(alpha=1.0)`
on 6 features, LOVO AUROC **0.9267** (25 trucks: 10 failed / 15 non-failed),
frozen Youden threshold **0.4456**, tiers on raw prob: red ≥ 0.55, amber ≥ 0.35.

**Version lineage (why two version numbers):** the model was trained and frozen as
**V10.5.3** (2026-06-01). **V11.1** froze it for leadership review; **V11.2**
produced its deepest validation (metric suite: 139/150 concordant pairs, PR-AUC
0.94 — copied here); **V12**'s GED challenge failed to beat it (all 7 features
rejected). This folder lives under `V11.2_ALT/` because V11.2 holds the best
validated results; the artifact inside is the V10.5.3 model, unchanged.

## Files
| File | What it is |
|---|---|
| `V10.5.3_20_5_ALT_champion_bundle.joblib` | dict: fitted sklearn Pipeline (median impute → scaler → ridge, fit on all 25 trucks) + threshold + tier bands + metadata. No custom classes — loads anywhere sklearn 1.8.x is installed. |
| `V10.5.3_20_5_ALT_predict.py` | loader + CLI (`py -3 ..._predict.py <features_csv>`) |
| `V10.5.3_20_5_ALT_training_matrix.csv` | provenance copy of the 25-truck feature matrix |
| `V10.5.3_20_5_ALT_ridge_spec.json` | provenance copy of the frozen model spec |
| `V10.5.3_20_5_ALT_lovo_predictions.csv` | provenance copy of the archived LOVO predictions |
| `V11.2_ALT_metric_suite.json` | provenance copy of the V11.2 validation metric suite |
| `ALT_last5_iteration_comparison.csv` / `.md` | evidence table: why this model is the champion |
| `V10.5.3_20_5_ALT_verification.json` | parity-gate results from packaging (P1–P4) |
| `V10.5.3_20_5_ALT_MANIFEST.json` | SHA256 of every file + inputs + env + git commit |

## Quick start
```
py -3 V10.5.3_20_5_ALT_predict.py V10.5.3_20_5_ALT_training_matrix.csv
```

## Honesty notes
- 0.9267 is a **LOVO ranking AUROC** at n=25, not field accuracy. The production
  pipeline here is fit on **all 25 trucks**; its resubstitution scores on the
  training matrix are optimistic by construction and are NOT the validation numbers.
- The 0.4456 threshold and 0.35/0.55 tier bands are frozen from the original run
  (2026-06-01). Packaging never recomputes them.
- Loading under a different sklearn minor version may emit
  `InconsistentVersionWarning`; re-run the packaging script to rebuild in-place.
- Rebuilding regenerates the bundle with a fresh `created` date and the current
  `git_head`, so the `.joblib` bytes and its MANIFEST SHA256 will differ across
  rebuilds. The MANIFEST SHA256 verifies the specific committed bytes
  (tamper-detection), not cross-run byte reproducibility.
- Rebuild + re-verify: `py -3 "V11.2_ALT/src/V11.2_ALT_package_champion.py"`,
  then `py -3 "V11.2_ALT/src/V11.2_ALT_bundle_smoketest.py"`.
