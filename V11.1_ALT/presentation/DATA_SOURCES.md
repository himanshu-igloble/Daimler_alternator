---
title: "V11.1_ALT — Data Sources Reference"
status: "complete"
created: "2026-06-12"
---

# V11.1_ALT — Data Sources Reference

All numeric claims in the V11.1 deliverables (markdown report, Excel workbook,
technical deck, business deck) are loaded at runtime from the files listed below.
No claim is hardcoded in the generator scripts except structural constants
(thresholds, gate names).

## Primary cache files

| File | Contents | Used by |
|------|----------|---------|
| `V11.1_ALT/cache/backtest/backtest_results.json` | MAE / coverage / PI-width for M0, M1, M2; dummy MAE; chosen variant; Wilcoxon p-values | All generators |
| `V11.1_ALT/cache/weibull/fleet_weibull_params.json` | Weibull shape 5.1658, scale 771.36d, median 718.5d, CI [677.3, 774.4] | All generators |
| `V10.6.2_ALT/cache/rul/fleet_window.json` | Empirical window: median 601d, p25–p75 577.5–652.5d, km 120440, ehrs 4538 — **read-only** | All generators |
| `V11.1_ALT/cache/rul/final_rul_per_vin.csv` | Per-truck RUL band (25 × 30 cols); ridge_prob; risk_band; time_dim; emergency_state | Markdown, Excel |
| `V11.1_ALT/cache/rul/decisions_per_vin.csv` | Per-truck risk decision (25 rows) | Markdown, Excel |
| `V11.1_ALT/cache/emergency/emergency_per_vin.csv` | GED / exceed / compound / early-watch per-truck status | All generators |
| `V11.1_ALT/cache/covariates/covariates_fit.csv` | x1 / x2 per truck at truncation date | Markdown, Excel |
| `V11.1_ALT/results/V11.1_ALT_verification.json` | Five gate statuses (G-LEAK, G-BETA, G-W6, G-EMERG, G-COVER) | All generators |

## Embedded graph files

| File | Used in |
|------|---------|
| `V11.1_ALT/visualizations/rul_core/backtest_accuracy.png` | Technical deck slide 5; business deck slide 3 |
| `V11.1_ALT/visualizations/rul_core/fleet_survival_curve.png` | Technical deck slide 7 |
| `V11.1_ALT/visualizations/rul_core/rul_band_waterfall.png` | Technical deck slide 8 |
| `V11.1_ALT/visualizations/rul_core/ged_emergency.png` | Technical deck slide 9 |

## Frozen classifier note

The Ridge classifier (V10.5.3) is **not touched by V11.1**. The `ridge_prob`
values in `final_rul_per_vin.csv` and `decisions_per_vin.csv` are read directly
from the V10.6.2 results artefact (`V10.6.2_ALT/results/V10.6.2_ALT_rul_predictions.csv`
via the config inheritance chain). The gate G-W6 verifies that no classifier
symbol appears in any V11.1 source file.

## M0 ≡ V10.6.2 note

M0 is the AFT variant with no covariates (β1 = β2 = 0), fitted on the same
Weibull prior grid as V10.6.2. Its MAE (140.4d) and PI coverage (0.867) reproduce
the V10.6.2 baseline within sampling variance. The fleet curve parameters
(shape 5.17, scale 771d) match V10.6.2's Weibull output — confirmed by comparing
`V11.1_ALT/cache/weibull/fleet_weibull_params.json` against
`V10.6.2_ALT/cache/weibull/fleet_weibull_params.json`.
