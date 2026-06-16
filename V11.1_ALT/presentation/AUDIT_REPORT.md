---
title: "V11.1_ALT — Audit Report"
status: "complete"
created: "2026-06-12"
---

# V11.1_ALT — Deliverables Audit Report

Every numeric claim in the V11.1 decks, markdown report, and Excel workbook
traces to a committed V11.1 cache file.  Claims are verified at generator
runtime — no hardcoded results except the structural constants in the table below.

## Claim → Source Table

| Claim | Value | Source file | Notes |
|-------|-------|-------------|-------|
| NO_IMPROVEMENT verdict | NO_IMPROVEMENT_HONEST | `V11.1_ALT/results/V11.1_ALT_verification.json` → `gates.G-BETA.verdict` | G-BETA gate |
| Chosen variant | M0 | `V11.1_ALT/cache/backtest/backtest_results.json` → `chosen_variant` | min_mae_with_coverage rule |
| M0 MAE (model) | 140.4d | `backtest_results.json` → `variants.M0.mae_model` | |
| M1 MAE (model) | 148.8d | `backtest_results.json` → `variants.M1.mae_model` | |
| M2 MAE (model) | 162.2d | `backtest_results.json` → `variants.M2.mae_model` | |
| Dummy MAE | 49.7d | `backtest_results.json` → `variants.M0.mae_dummy` | Fleet-clock baseline |
| M0 PI coverage | 0.867 | `backtest_results.json` → `variants.M0.pi_coverage` | |
| Weibull shape | 5.1658 | `V11.1_ALT/cache/weibull/fleet_weibull_params.json` → `shape` | |
| Weibull scale | 771.36d | `fleet_weibull_params.json` → `scale` | |
| Weibull median | 718.5d | `fleet_weibull_params.json` → `median_ttf_days` | Context only |
| Empirical fleet median | 601d | `V10.6.2_ALT/cache/rul/fleet_window.json` → `median_ttf_days` | Read-only from V10.6.2 |
| Empirical p25–p75 | 577.5–652.5d | `fleet_window.json` → `p25_ttf_days`, `p75_ttf_days` | |
| Empirical median km | 120440 km | `fleet_window.json` → `median_ttf_km_est` | Speed-integrated est. |
| GED fired / failed | 2/10 | `V11.1_ALT/cache/emergency/emergency_per_vin.csv` → `ged_fired==True & failed_flag==1` | |
| GED fired / NF | 0/15 | `emergency_per_vin.csv` → `ged_fired==True & failed_flag==0` | 0 false alarms |
| Early-watch current / failed | 3/10 | `emergency_per_vin.csv` → `early_watch_current==1 & failed_flag==1` | Deployable channel |
| Early-watch current / NF | 0/15 | `emergency_per_vin.csv` → `early_watch_current==1 & failed_flag==0` | 0 false alarms |
| Exceed NF ever-fired | 2/15 | `emergency_per_vin.csv` → `exceed_fired==True & failed_flag==0` | Report-only, not deployable |
| Compound NF ever-fired | 6/15 | `emergency_per_vin.csv` → `compound_fired==True & failed_flag==0` | Report-only, not deployable |
| G-LEAK status | PASS | `V11.1_ALT_verification.json` → `gates.G-LEAK.status` | 60 rows checked, 0 violations |
| G-BETA status | PASS | `V11.1_ALT_verification.json` → `gates.G-BETA.status` | |
| G-W6 status | PASS | `V11.1_ALT_verification.json` → `gates.G-W6.status` | |
| G-EMERG status | PASS | `V11.1_ALT_verification.json` → `gates.G-EMERG.status` | |
| G-COVER status | PASS | `V11.1_ALT_verification.json` → `gates.G-COVER.status` | |
| Overall gate result | PASS | `V11.1_ALT_verification.json` → `overall` | 5/5 gates PASS |
| Classifier AUROC | 0.927 | V10.5.3 frozen — structural constant, not recomputed by V11.1 | Inherited from V10.6.2 config |
| V11 precursor recall | 6/10 | `V11_ALT_heuristics/cache/forensics/earliest_signal_per_vin.csv` → `verdict==discriminative_precursor` | V11 result, not V11.1 |
| V10.6.2 precursor recall | 2/10 | GED-only channel (historical) | V10.6.2 result |
| time_dim SHORT (all NF) | 15/15 | `V11.1_ALT/cache/rul/final_rul_per_vin.csv` → `time_dim==SHORT & failed_flag==0` | Aged fleet |

## Embedded Graphs

| Graph file | Slide | Claim illustrated |
|------------|-------|-------------------|
| `visualizations/rul_core/backtest_accuracy.png` | Tech slide 5; Biz slide 3 | All variants fail to beat dummy |
| `visualizations/rul_core/fleet_survival_curve.png` | Tech slide 7 | M0 ≡ V10.6.2 fleet curve |
| `visualizations/rul_core/rul_band_waterfall.png` | Tech slide 8 | time_dim SHORT for all 15 NF trucks |
| `visualizations/rul_core/ged_emergency.png` | Tech slide 9 | GED + early-watch per-truck status |

## Frozen Classifier Note

The Ridge classifier (V10.5.3) is **untouched by V11.1**.  `ridge_prob` values
are read from `V10.6.2_ALT/results/V10.6.2_ALT_rul_predictions.csv` via the
V11.1 config inheritance chain.  G-W6 verifies 0 classifier symbols in V11.1 code.

## M0 Reproduces V10.6.2 Note

M0 (β1 = β2 = 0) is fitted on the identical Weibull prior grid as V10.6.2.
MAE 140.4d and PI coverage 0.867 are consistent with V10.6.2's baseline
(rewound MAE ~140d, PI coverage ~0.867).  The fleet curve parameters
(shape 5.17, scale 771d) match V10.6.2's `fleet_weibull_params.json`.
