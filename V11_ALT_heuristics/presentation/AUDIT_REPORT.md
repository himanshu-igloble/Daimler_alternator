# V11 Deliverables — Audit Report

Every numeric claim in the V11 decks/report traces to a committed V11 cache file.

| Claim | Value | Source file |
|---|---|---|
| V11 discriminative recall | 6/10 | cache/forensics/earliest_signal_per_vin.csv (verdict==discriminative_precursor) |
| V10.6.2 recall (baseline) | 5/10 | results/V11_ALT_heuristics_comparison.csv (v1062_horizon != none) |
| NF self-test false alarms | 0/15 | cache/forensics/nf_self_test.csv (verdict==FALSE_ALARM) |
| Compound early-watch recall | 4/10 | cache/forensics/compound_alarm_lovo.csv (group==FAILED & fired) |
| Compound NF false alarms | 0/15 | cache/forensics/compound_alarm_lovo.csv (group==NF & fired) |
| MVP feature | crank_recovery_t | cache/forensics/failed_window_deviations.csv (6 failed VINs discriminative) |
| New detection | VIN9 @30d | results/V11_ALT_heuristics_comparison.csv (new_in_v11) |
| Earlier lead | VIN1 30d→60d | results/V11_ALT_heuristics_comparison.csv (earlier) |

## Embedded graphs
G1 recall, G2 feature generalization, G3 lead-time dumbbell, G4 compound leads,
G5 crank-recovery trajectories, G6 change-point/resting — all in
visualizations/V11_graphs/ (PNG + _hd.png).

## Honesty notes carried into the decks
- crank_recovery_t z-magnitudes inflated by a near-zero healthy baseline (~0.05s);
  the trustworthy guard is the 0/15 NF self-test, not the z value.
- The recovery signal is episodic (isolated spikes), not a smooth trend; VIN9's
  events span its whole life.
- Change-point (#12) fires too early (220–599d) to be actionable — exploratory.
- No RUL/Weibull numbers appear anywhere in the V11 deliverables.
