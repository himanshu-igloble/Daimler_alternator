# V11 Deliverables — Data Sources

All V11 artifacts are sourced from the committed V11 precursor pipeline. No RUL /
Weibull / backtest data is used — V11 is the focused precursor fork; for RUL /
risk-ranking / service-schedule content see the V10.6.2_ALT deliverables.

## Pipeline
raw CAN (CSP, RPM, ANR, GED, VSI, SMA) → forensic honest gate (35 features) →
compound alarm (#11) + change-point (#12) → V11-vs-V10.6.2 comparison.

## Files consumed by the decks / graphs / report
- results/V11_ALT_heuristics_comparison.csv — per-truck V10.6.2-vs-V11 lead-time.
- cache/forensics/earliest_signal_per_vin.csv — V11 discriminative recall (6/10).
- cache/forensics/nf_self_test.csv — healthy-truck false-alarm test (0/15).
- cache/forensics/failed_window_deviations.csv — per-feature discriminative hits.
- cache/forensics/nf_baseline.csv — healthy-fleet p05/p50/p95 per feature.
- cache/forensics/compound_alarm_lovo.csv — compound early-watch (4/10, 0/15 NF).
- cache/forensics/changepoint_per_vin.csv — change-point / dose-knee / resting-slope.
- cache/forensics/<VIN>_daily.csv — per-truck daily feature panels (G5 trajectories).

## Out of scope (no V11 equivalent)
RUL bands, Weibull survival, backtest MAE, fleet replacement window, service
schedule — unchanged from V10.6.2 and not reproduced here.
