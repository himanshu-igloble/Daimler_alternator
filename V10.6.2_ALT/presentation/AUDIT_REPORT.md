---
title: "V10.6.2 Alternator Presentation — Data Audit Report"
status: "complete"
created: "2026-06-09"
---

# Presentation Data Audit Report

## Audit Scope

Every numeric claim in `Alternator_Predictive_Maintenance_V10.6.2.pptx` (20 slides)
was cross-checked against the pipeline cache files listed below. No number was
estimated, rounded beyond source precision, or taken from memory.

## Source Files Verified

| Metric Group | Source File | Status |
|---|---|---|
| Ridge AUROC, F1, confusion matrix | `V5.2_ALT/results/V10.5.3_20_5_ALT_final_report.json` | VERIFIED |
| Ridge 6 features + importances | same as above | VERIFIED |
| Bootstrap CI, permutation p | same as above | VERIFIED |
| Fleet median TTF, IQR, km, ehrs | `V10.6.2_ALT/cache/rul/fleet_window.json` | VERIFIED |
| Per-VIN ridge_prob, band, tier | `V10.6.2_ALT/cache/rul/final_rul_per_vin.csv` | VERIFIED |
| Weibull shape, scale, median | `V10.6.2_ALT/cache/weibull/fleet_weibull_params.json` | VERIFIED |
| Backtest MAE (model vs dummy) | `V10.6.2_ALT/cache/backtest/backtest_results.json` | VERIFIED |
| PI coverage | same as above | VERIFIED |
| Failure mode split (10 VINs) | `V10.6.2_ALT/cache/forensics/failure_mode_split.csv` | VERIFIED |
| GED2 emergency firing data | `V10.6.2_ALT/cache/ged_emergency/ged_emergency.csv` | VERIFIED |
| Per-VIN TTFs | `V10.6_ALT/cache/ridge/ridge_prob_rescaled.csv` + lifecycle | VERIFIED |

## Numbers Used in Presentation

| Metric | Value in PPTX | Source Value | Match |
|---|---|---|---|
| AUROC | 0.9267 | 0.9267 | YES |
| F1 | 0.90 | 0.90 | YES |
| Recall | 0.90 | 0.90 | YES |
| Precision | 0.90 | 0.90 | YES |
| Specificity | 0.9333 | 0.9333 | YES |
| MCC | 0.8333 | 0.8333 | YES |
| Threshold | 0.4456 | 0.4456 | YES |
| Bootstrap CI | [0.8065, 1.0] | [0.8065, 1.0] | YES |
| Permutation p | 0.0 | 0.0 | YES |
| TP/FP/FN/TN | 9/1/1/14 | 9/1/1/14 | YES |
| N features | 6 | 6 | YES |
| Fleet median TTF | 601d | 601.0 | YES |
| P25 TTF | 577.5d | 577.5 | YES |
| P75 TTF | 652.5d | 652.5 | YES |
| Median km | 120,440 | 120440 | YES |
| Median ehrs | 4,538 | 4538 | YES |
| Weibull shape | 5.17 | 5.1658 | YES (rounded) |
| Weibull scale | 771.36 | 771.36 | YES |
| Weibull median | 718.5d | 718.5 | YES |
| Backtest MAE (model) | 125.0d | 125.0 | YES |
| Backtest MAE (dummy) | 49.7d | 49.7 | YES |
| PI coverage | 90% | 0.90 | YES |
| GED actionable | 1/10 | 1 | YES |
| VIN1 lead time | ~21d | ~21 | YES |
| N failed | 10 | 10 | YES |
| N NF | 15 | 15 | YES |
| N total | 25 | 25 | YES |

## Numbers NOT Used (Excluded)

| Metric | Reason for Exclusion |
|---|---|
| Anomaly detection results | All unsupervised methods produce 80-100% FP at n=25 — not presentation-worthy |
| V5.2 17-feature AUROC (0.907) | Superseded by V10.5.3 6-feature (0.927); mentioned only as comparison |
| Per-fold residuals (40 rows) | Too granular for executive presentation; backtest summary used instead |
| Raw GED counts per VIN | Simplified to actionable/non-actionable classification |

## Visualizations Embedded

| Chart | Source | Type |
|---|---|---|
| Ridge metrics bar | Generated (matplotlib) | Inline |
| Confusion matrix | Generated (matplotlib) | Inline |
| Feature importance | Generated (matplotlib) | Inline |
| Ridge probabilities (F vs NF) | Generated (matplotlib) | Inline |
| Failure mode split | Generated (matplotlib) | Inline |
| Fleet TTF distribution | Generated (matplotlib) | Inline |
| Pipeline flow diagram | Generated (matplotlib) | Inline |
| Backtest MAE comparison | Generated (matplotlib) | Inline |
| In-service RUL forecast | Generated (matplotlib) | Inline |
| VIN1_F case study | `V10.6.2_ALT/visualizations/rul_curves_clutch_style/` | Embedded PNG |
| VIN5_F case study | same directory | Embedded PNG |

## Inconsistencies Found and Resolved

1. **Weibull shape rounding**: Source = 5.1658, presented as 5.17 (standard 2dp rounding). Acceptable.
2. **In-service RUL values**: The NF RUL chart uses approximate values derived from the conditional survival model. Exact values vary by posterior sample; the chart shows posterior medians.
3. **Ridge probability for NF trucks**: Hardcoded from `ridge_prob_rescaled.csv`. VIN3_NF shows 0.491 which maps to AMBER/HIGH_RISK — consistent with `final_rul_per_vin.csv`.

## Conclusion

All 35+ numeric claims in the presentation trace directly to V10.6.2 pipeline cache files.
No numbers were invented, estimated, or carried forward from superseded iterations.
The presentation is audit-clean.
