---
title: "V10.6.2 Alternator Presentation — Data Sources & Metrics Derivation"
status: "complete"
created: "2026-06-09"
---

# Data Sources & Metrics Derivation

## Pipeline Architecture

```
CAN Bus (204M rows) → Sentinel Cleaning → Weekly Aggregation → Feature Engineering
  → Ridge Classifier (Layer 1) → Weibull Survival (Layer 2) → GED Emergency (Layer 3)
  → Risk Tier Assignment → RUL Forecast → Fleet Maintenance Board
```

## Data Files Used

### Ridge Classifier (V10.5.3 Final)
- **Source**: `V5.2_ALT/results/V10.5.3_20_5_ALT_final_report.json`
- **Method**: L2-regularized logistic regression, 25-fold LOVO
- **Selection**: Exhaustive backward elimination from 20 candidates → 6 features
- **Key result**: AUROC 0.9267 (6 features) > 0.907 (17 features)

### Fleet Window
- **Source**: `V10.6.2_ALT/cache/rul/fleet_window.json`
- **Derivation**: Empirical statistics from 10 failed truck lifetimes
- **Fields**: median_ttf (601d), p25 (577.5d), p75 (652.5d), median_km (120,440), median_ehrs (4,538)

### Weibull Survival
- **Source**: `V10.6.2_ALT/cache/weibull/fleet_weibull_params.json`
- **Method**: Bayesian posterior via MCMC, right-censoring for 15 NF trucks
- **Fields**: shape (5.1658), scale (771.36), posterior_median (718.5d), 80% CI (677.3–774.4d)

### Backtest
- **Source**: `V10.6.2_ALT/cache/backtest/backtest_results.json`
- **Method**: LOVO on 10 failed VINs, 4 rewind points each (total TTF, 270d, 180d, 90d)
- **Key result**: Model MAE 125.0d vs fleet-clock MAE 49.7d → fleet-clock deployed

### Per-VIN RUL
- **Source**: `V10.6.2_ALT/cache/rul/final_rul_per_vin.csv`
- **Fields**: vin, ridge_prob, ridge_band, risk_tier, median_rul, p10, p90, rul_band

### Failure Mode Split
- **Source**: `V10.6.2_ALT/cache/forensics/failure_mode_split.csv`
- **Classification**: gradual-electrical (2), abrupt (5), inconclusive (3)

### GED Emergency
- **Source**: `V10.6.2_ALT/cache/ged_emergency/ged_emergency.csv`
- **Threshold**: ≥200 GED2 counts per engine-on day
- **Result**: VIN1_F (21d lead), VIN10_F (1d lead), 0 false alarms on 15 NF

### Ridge Probabilities (Rescaled)
- **Source**: `V10.6_ALT/cache/ridge/ridge_prob_rescaled.csv`
- **Tier mapping**: GREEN < 0.35, AMBER 0.35–0.55, RED ≥ 0.55

## Feature Definitions

| Feature | Derivation |
|---|---|
| vsi_std_ratio_30d | Ratio of rolling 30d VSI std to lifetime VSI std |
| vsi_dominant_freq | Dominant spectral frequency from FFT of VSI signal |
| vsi_range_trend_last30d | Linear slope of (max−min VSI) over trailing 30 days |
| vsi_spectral_entropy | Shannon entropy of VSI power spectrum (0=pure tone, high=noise) |
| progressive_drift | Cumulative signed drift of weekly VSI mean from baseline |
| bat_charge_delta_trend_right | Asymmetry between charge/discharge voltage deltas, right-tail trend |

All features derived from VSI (Power Supply Voltage) signal on the 24V bus.

## Validation Checks Performed

1. AUROC matched between V10.5.3 report and presentation
2. Confusion matrix sums: TP+FP+FN+TN = 9+1+1+14 = 25 = N_VINS
3. F1 = 2×TP/(2×TP+FP+FN) = 18/20 = 0.90 ✓
4. Fleet-clock MAE < Weibull MAE confirmed (49.7 < 125.0)
5. PI coverage 0.90 > 0.80 target ✓
6. GED2 false alarm rate on NF = 0/15 = 0% ✓
7. VIN counts: 10F + 15NF = 25 ✓
