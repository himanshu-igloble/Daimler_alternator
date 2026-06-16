---
title: "V11_ALT_heuristics — New Lead-Time Precursor Heuristics (Design Spec)"
status: complete
created: 2026-06-12
updated: 2026-06-12
---

# V11_ALT_heuristics — Design Spec

## 0. Purpose & honest framing

V10.6.2 left one box weak: the **WHEN-emergency** precursor channel. It proved:

- The frozen V10.5.3 Ridge classifier tells you **WHICH** truck will fail
  (AUROC 0.927) but is a static whole-life score — **no timing**.
- Per-truck survival RUL does **not** beat a fleet-clock dummy on day accuracy
  (rewound MAE 142d vs 50d); it only yields a wide band.
- The only genuine short-horizon precursor across the 6 raw channels is the
  **GED==2 excitation-fault storm**, firing for just **2/10** failures
  (VIN1 ~21d, VIN10 ~1d), with 0/15 false alarms.

V11 implements the 12 candidate heuristics from
`V10.6.2_ALT/reports/V10.6.2_ALT_candidate_heuristics_for_lead_time.txt` to try
to extract **earlier, higher-recall** precursors from the **same six raw CAN
channels** (CSP, RPM, ANR, GED, VSI, SMA). The goal is to make the WHEN-emergency
box fire **earlier and for more trucks** — NOT to manufacture daily RUL.

**Honest constraint (n = 10 failure events).** Every new heuristic must pass the
exact V10.6.2 discriminative gate (within-truck `z ≥ 2` AND outside the healthy
NF p05–p95 envelope, `MIN_EO_SAMPLES = 200`), or it is curve-fitting to 10 trucks.

## 1. Scope decisions (locked)

- **Pipeline scope:** Focused forensic + alarm fork. New `V11_ALT_heuristics/`
  directory implementing ONLY the precursor/lead-time channel. The frozen
  classifier (Ridge, AUROC 0.927) and Weibull fleet window are **reused by
  reference** (read from existing caches), never recomputed.
- **Heuristic set:** **All 12** candidate heuristics (#1–#12).
- **Change-point method (#12):** CUSUM (deterministic, no new dependencies),
  not Bayesian.
- **#9 idle hunting:** variance / lag-1 ACF / zero-crossing rate; the optional
  FFT/PSD peak is deferred unless variance shows promise.

## 2. Verified preconditions

- Source parquets: `V5.2_ALT/features/parquets/V5.2_20_5_ALT_<VIN>.parquet`
  (25 files: 10 F + 15 NF). Verified schema:
  `DATETIME, VIN, VIN_LABEL, FAILED, CSP, RPM, ANR, GED, VSI, SMA, SALEDATE,
  JCOPENDATE, DAYS_SINCE_SALE, DAYS_TO_FAILURE`.
- **ANR (torque) and DATETIME both present.** The V10.6.2 forensic loader read
  only `RPM, CSP, VSI, GED, SMA, DAYS_SINCE_SALE, DAYS_TO_FAILURE` — V11 must add
  **ANR** (for #2 load-residual, #10 sag-typing) and **DATETIME** (for #3 crank
  recovery, #5 volt-second dose, #12 change-point ordering).
- Sampling ≈ 5 s (V5.2_20_5; `MIN_EO_SAMPLES=200` ≈ 17 min engine-on). Use real
  `DATETIME` deltas for any time-integral, not an assumed constant.
- `JCOPENDATE` = failure-open date, used for first-trigger lead validation.

## 3. Architecture

New directory mirrors V10.6.2 layout but only the precursor channel:

```
V11_ALT_heuristics/
  src/
    V11_ALT_heuristics_config.py        # extends V10.6.2 config
    V11_ALT_heuristics_forensic.py      # CORE: build_daily() + 12 feature families + honest gate
    V11_ALT_heuristics_compound.py      # #11 weak-vote alarm + LOVO
    V11_ALT_heuristics_changepoint.py   # #12 CUSUM, #5 dose-knee, #6 resting-slope
    V11_ALT_heuristics_compare.py       # V11 vs V10.6.2 comparison report
    V11_ALT_heuristics_verify.py        # honest gates; exits(1) on regression
    V11_ALT_heuristics_orchestrator.py  # runs all via py -3
  cache/forensics/   results/   reports/   visualizations/
```

Reference baseline source files (read-only inputs):

- `V10.6.2_ALT/src/V10.6.2_ALT_forensic_features.py` — the harness to extend.
- `V10.6.2_ALT/src/V10.6.2_ALT_config.py` — base config.
- `V10.6.2_ALT/cache/forensics/earliest_signal_per_vin.csv`,
  `failed_window_deviations.csv`, `nf_baseline.csv` — V10.6.2 results to compare against.

## 4. Inherited honest gate (unchanged, for comparability)

- Within-truck baseline: healthy mid-life (`DAYS_TO_FAILURE` 120–365), fallback
  earliest 40% of days.
- z-score: `(window_mean − baseline_mean) / baseline_std`, discriminative if
  `|z| ≥ 2`.
- NF envelope: outside fleet `p05–p95` (BAD_HIGH features → `> p95`; BAD_LOW
  features → `< p05`).
- Trust filter: `MIN_EO_SAMPLES = 200` on both NF baseline rows and failed-window rows.
- Horizon bins (`days_to_failure`): 90, 60, 45, 30, 14, 7.

## 5. The 12 heuristics → features

### Group A — daily-aggregate features (standard horizon-bin gate)

| # | New column(s) | Computation | Direction |
|---|---|---|---|
| 1 | `vsi_rpm_slope`, `vsi_ceiling`, `vsi_onset_rpm` | per-day OLS VSI~RPM in 600–1500 rpm band; plateau VSI mean at RPM>1500; min RPM where VSI first ≥27V | slope/ceiling LOW-bad; onset HIGH-bad |
| 2 | `vsi_resid_mean`, `vsi_resid_negfrac` | NF reference surface `E[VSI\|RPM,ANR,CSP]` (binned lookup over 15 NF trucks); daily residual = obs − expected at actual operating point | resid_mean LOW-bad; negfrac HIGH-bad |
| 3 | `crank_recovery_t`, `crank_recovery_slope` | per SMA 1→0 edge: time (s, via DATETIME) to reach ≥27V, slope V/s over first ~30s; daily mean across starts | time HIGH-bad; slope LOW-bad |
| 4 | `reg_duty_frac` | fraction of engine-on samples with 27.0 ≤ VSI ≤ 29.0 | LOW-bad |
| 7 | `cranks_per_ehr`, `crank_dur_mean` | SMA rising edges per engine-hour; mean run-length of SMA==1 (crank duration) | both HIGH-bad |
| 8 | `ged1_frac`, `ged3_frac`, `ged_churn` | daily rate of GED==1, GED==3; count of 0→2 and 0→3 transitions per day | all HIGH-bad |
| 9 | `idle_vsi_var`, `idle_vsi_acf1`, `idle_vsi_zcr` | within idle band (550–950 rpm, CSP<5): VSI variance, lag-1 autocorr, mean-crossing rate | var/zcr HIGH-bad |
| 10 | `sag_highload_frac`, `sag_idle_frac` | VSI<24 events split by ANR: high-torque (stator/diode) vs idle/low-load (regulator) fractions | both HIGH-bad |

All Group A columns are appended to the daily panel in `build_daily()`, registered
in `feat_cols`, classified in `BAD_HIGH`/`BAD_LOW`, and (the physically strongest)
added to `KEY_FEATURES` for earliest-precursor detection. They flow through the
existing gate with no change to the gate logic.

### Group B — series / trend diagnostics (separate evaluation)

- **#5 UV dose** — cumulative `∫(24 − VSI)·dt` while engine-on & VSI<24
  (volt-seconds, real DATETIME deltas). Knee/elbow detection → lead time. Daily
  increment `uv_dose_day` stored in the panel; cumulative curve in changepoint module.
- **#6 resting decay** — life-long slope of `resting_vsi_mean` + overnight hold
  (last reading pre-key-off vs first at next key-on). Discriminative if slope < NF p05.
- **#12 change-point** — CUSUM on each truck's own normalized series (primary:
  `vsi_resid_mean`, `resting_vsi_mean`); change-point timestamp **is** the lead time.

### #11 Compound voting alarm

Fire `early-watch` when **≥ 2 of** {dVSI/dRPM ceiling drop, load-residual drift,
slow post-crank recovery, resting-voltage decay, GED churn} cross their NF p05/p95
in the same window. Evaluated via LOVO over the 10 failed VINs. The existing GED==2
storm remains a **separate** high-precision "days-left" emergency, not part of the vote.

## 6. Validation, honesty guards & comparison (the deliverable)

1. **Reuse the exact gate** so V11 numbers are directly comparable to V10.6.2.
2. **NF self-test** — run the identical gate on each of the 15 NF trucks as if it
   were failing; any feature that fires on NF trucks is a false-alarm generator,
   reported per-feature in `nf_self_test.csv`.
3. **Anecdotal flag** — any feature firing on ≤ 1 of 10 failed VINs is labeled
   anecdotal (curve-fit), not counted as a win.
4. **Primary metric** — recall = # failed VINs with a discriminative precursor at
   ≥ 7d. Baseline to beat: V10.6.2 = 2/10 genuine (GED==2), 4/10 borderline.
5. **Secondary metrics** — first-trigger **lead distribution** vs `JCOPENDATE`
   (must beat fleet-clock); **NF false-alarm rate** (target 0).

### Output artifacts

- `cache/forensics/<VIN>_daily.csv` (extended panel, all new columns)
- `cache/forensics/nf_baseline.csv` (extended)
- `cache/forensics/failed_window_deviations.csv` (extended)
- `cache/forensics/earliest_signal_per_vin.csv` (extended)
- `cache/forensics/nf_self_test.csv` (new honesty guard)
- `cache/forensics/compound_alarm_lovo.csv` (#11 recall / NF false-alarm / lead)
- `cache/forensics/changepoint_per_vin.csv` (#12 / #5 / #6 lead times)
- `results/V11_ALT_heuristics_comparison.csv` (V11 vs V10.6.2 head-to-head)
- `reports/V11_ALT_heuristics_report.md` (full comparison + honest verdict on
  which of the 12 generalize)

## 7. Execution

```
cd D:\Daimler-starter_motor_alternator_battery\V11_ALT_heuristics\src
py -3 V11_ALT_heuristics_orchestrator.py
```

Order: forensic → compound → changepoint → verify → compare. `verify` exits(1) if
recall regresses below the V10.6.2 baseline OR if any "win" is anecdotal-only with
NF false-alarms — forcing an honest report rather than a silent overclaim.

## 8. Out of scope

- No retraining / re-fitting of the frozen Ridge classifier (W6 forbidden-pattern
  audit still applies: no `Ridge(`, `sklearn`, `IsotonicRegression`).
- No recompute of Weibull / survival / predictive-RUL / backtest stages — those
  inputs are frozen; their numbers would not change.
- No presentation (.pptx) or Excel rebuild this iteration; the markdown comparison
  report is the deliverable.
- No cross-dataset (SM) work; ALT only.

## 9. Success criteria

- All 12 heuristics implemented and run on real data (no stubs).
- A committed `V11_ALT_heuristics_report.md` with the real V11-vs-V10.6.2 table.
- Each of the 12 explicitly classified as: **generalizes** (≥ raise recall,
  0 NF false-alarms), **anecdotal** (≤1 VIN), or **no signal**.
- The honest verdict states whether recall rose above 2/10 genuine and whether any
  new channel beats the GED==2 lead for any truck.
