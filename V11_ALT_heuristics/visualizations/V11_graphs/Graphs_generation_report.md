# V11 Graphs - generation report

_All graphs sourced from committed V11 caches; no pipeline re-runs._

| graph | file | source |
|---|---|---|
| G1 recall | G1_recall_comparison.png | comparison.csv, nf_self_test, compound_alarm_lovo |
| G2 feature generalization | G2_feature_generalization.png | failed_window_deviations, nf_self_test |
| G3 lead-time dumbbell | G3_leadtime_dumbbell.png | comparison.csv |
| G4 compound leads | G4_compound_alarm_leads.png | compound_alarm_lovo |
| G5 crank-recovery (MVP) | G5_crank_recovery_trajectories.png | <VIN>_daily.csv, nf_baseline |
| G6 changepoint/resting | G6_changepoint_resting.png | changepoint_per_vin |

## Honesty notes

- G5: crank_recovery_t within-truck z-scores are inflated by near-zero baseline variance + the 30s censor; the trustworthy guard is 0/15 NF false alarms.

- G6: change-point fires far too early (220-599d) to be actionable - exploratory only.

- No RUL/Weibull content: V11 has no RUL data (referenced to V10.6.2).
