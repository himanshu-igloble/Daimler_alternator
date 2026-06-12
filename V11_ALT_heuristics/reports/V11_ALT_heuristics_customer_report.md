# Alternator Lead-Time Heuristics — Customer Report (V11)

## 1. Executive Summary

V11 tests 12 new lead-time heuristics on the same six raw CAN channels used by the frozen classifier. It raises discriminative-precursor recall to **6/10** (V10.6.2: 5/10) with **0/15** false alarms on healthy trucks, and produces an earlier first warning for one truck. This is a real but modest gain; it does NOT change the WHICH-truck classifier (AUROC 0.927) or the fleet replacement window — those remain as delivered in V10.6.2. V11 makes the WHEN-emergency warning fire earlier and for one additional truck.

## 2. What V11 Adds

- **VIN9** newly detected (30-day horizon) via post-crank recovery time.
- **VIN1** earliest warning moved 30d → 60d (earlier lead).
- **`crank_recovery_t` (post-crank voltage recovery, heuristic #3) is the MVP** — discriminative on 6/10 failed trucks, 0/15 false alarms.

## 3. Per-Truck Lead-Time (V10.6.2 vs V11)

| vin_label   | v1062_horizon   | v1062_feature    | v11_horizon   | v11_feature      | earlier   | new_in_v11   |
|:------------|:----------------|:-----------------|:--------------|:-----------------|:----------|:-------------|
| VIN4_F_ALT  | none            | nan              | none          | nan              | False     | False        |
| VIN5_F_ALT  | none            | nan              | none          | nan              | False     | False        |
| VIN9_F_ALT  | none            | nan              | 30            | crank_recovery_t | True      | True         |
| VIN6_F_ALT  | 60              | vsi_sag_frac     | 60            | vsi_sag_frac     | False     | False        |
| VIN10_F_ALT | 7               | vsi_sag_frac     | 7             | vsi_sag_frac     | False     | False        |
| VIN8_F_ALT  | 7               | resting_vsi_mean | 7             | crank_recovery_t | False     | False        |
| VIN1_F_ALT  | 30              | ged2_frac        | 60            | crank_recovery_t | True      | False        |
| VIN7_F_ALT  | none            | nan              | none          | nan              | False     | False        |
| VIN2_F_ALT  | 14              | crank_vsi_min    | 14            | crank_vsi_min    | False     | False        |
| VIN3_F_ALT  | none            | nan              | none          | nan              | False     | False        |

## 4. Heuristic Catalog & Verdict

| feature              |   failed_discriminative |   nf_false | class       |
|:---------------------|------------------------:|-----------:|:------------|
| vsi_rpm_slope        |                       1 |          0 | anecdotal   |
| vsi_resid_negfrac    |                       1 |          0 | anecdotal   |
| crank_recovery_slope |                       1 |          0 | anecdotal   |
| cranks_per_ehr       |                       1 |          0 | anecdotal   |
| idle_vsi_acf1        |                       1 |          0 | anecdotal   |
| idle_vsi_zcr         |                       1 |          0 | anecdotal   |
| uv_dose_day          |                       1 |          0 | anecdotal   |
| crank_recovery_t     |                       6 |          0 | generalizes |
| sag_highload_frac    |                       4 |          0 | generalizes |
| reg_duty_frac        |                       3 |          0 | generalizes |
| crank_dur_mean       |                       3 |          0 | generalizes |
| idle_vsi_var         |                       3 |          0 | generalizes |
| vsi_ceiling          |                       2 |          0 | generalizes |
| vsi_resid_mean       |                       2 |          0 | generalizes |
| ged_churn            |                       2 |          0 | generalizes |
| sag_idle_frac        |                       2 |          0 | generalizes |
| vsi_onset_rpm        |                       0 |          0 | no_signal   |
| ged1_frac            |                       0 |          0 | no_signal   |
| ged3_frac            |                       0 |          0 | no_signal   |

_Generalizing new features: ['crank_recovery_t', 'sag_highload_frac', 'reg_duty_frac', 'crank_dur_mean', 'idle_vsi_var', 'vsi_ceiling', 'vsi_resid_mean', 'ged_churn', 'sag_idle_frac']_

## 5. Compound Early-Watch Alarm (#11)

A 2-of-5 weak vote across orthogonal channels fires on **4/10** failed trucks with **0/15** false alarms, giving the earliest first-trigger for some trucks (e.g. VIN8 at 90 days). It is an orthogonal 'early-watch' tier; the GED=2 storm stays as the separate high-precision emergency.

## 6. Change-point & Resting-Voltage (exploratory)

Per-truck CUSUM change-point and cumulative under-voltage dose fire far too early (220–599 day leads) to be actionable — exploratory only. Resting-voltage decay slope is discriminative on 3/10 (VIN1/4/10).

## 7. Honest Limitations

- `crank_recovery_t` within-truck z-scores are inflated by a near-zero healthy baseline (NF p95 ≈ 0.05 s — healthy trucks recover essentially instantly) plus the 30 s recovery censor; trust the 0/15 NF false-alarm guard, not the z-magnitude.
- The crank_recovery_t signal is **episodic** — isolated slow-recovery events, not a smooth degradation trend. For VIN9 the events span the truck's whole life (spikes at ~590d, ~420d and ~20d before failure), so the '30-day lead' for VIN9 should be read cautiously rather than as a clean precursor.
- VIN8's strict-gate hit rests on only 3 trusted days (fragile); the compound alarm catches it more robustly at 90d.
- 4/10 failed trucks (VIN3/4/5/7) remain undetectable by any channel — consistent with abrupt/silent electrical failure with no precursor.
- n=10 events: treat per-feature results as leads, not calibrated rates.
- No per-truck daily RUL is produced or implied.

## 8. Deployment Recommendation

Add `crank_recovery_t` to the emergency channel alongside the GED=2 storm signal, and surface the compound 2-of-5 vote as an 'early-watch' tier (0 false alarms, earlier first-triggers). The load-split sag features (`sag_highload_frac` vs `sag_idle_frac`) and `reg_duty_frac` add repair-direction guidance even where they do not change recall.
