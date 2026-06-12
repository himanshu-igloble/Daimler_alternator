---
title: "V11_ALT_heuristics — Presentation, Customer Report & Graphs (V11-native)"
status: complete
created: 2026-06-12
---

# V11 Deliverables Design Spec (V11-native lead-time)

## 0. Purpose & hard constraint

Produce the same artifact TYPES V10.6.2 shipped — professional graphs, a customer
report (markdown + xlsx), and two python-pptx decks (technical + business) — but
with **V11 data only**. V11 is the focused precursor fork: it has NO RUL / Weibull
/ backtest / per-VIN-RUL-band data. Therefore these deliverables are
**precursor/lead-time centric**, not RUL centric. RUL / risk-ranking / service-
schedule content is explicitly OUT (referenced to V10.6.2, never reproduced).

All inputs already exist under `V11_ALT_heuristics/cache/forensics/`,
`results/`, and the daily panels — no pipeline re-runs are required.

## 1. Verified V11 data sources (column contracts)

- `cache/forensics/earliest_signal_per_vin.csv`: vin_label,
  earliest_discriminative_horizon_days, feature, z, ged2_total,
  n_days_final_30d, verdict
- `cache/forensics/nf_self_test.csv`: vin_label, false_precursor_horizon_days,
  feature, z, verdict ("FALSE_ALARM"/"clean")
- `cache/forensics/failed_window_deviations.csv`: vin_label, horizon_days,
  n_days, feature, window_mean, baseline_mean, z_vs_baseline, nf_p05, nf_p95,
  discriminative (True/False)
- `cache/forensics/nf_baseline.csv`: feature, nf_p05, nf_p50, nf_p95, nf_mean, nf_std
- `cache/forensics/compound_alarm_lovo.csv`: group (FAILED/NF), vin_label,
  early_watch_horizon_days, n_votes, fired
- `cache/forensics/changepoint_per_vin.csv`: vin_label, cp_resid_lead_days,
  cp_resting_lead_days, dose_knee_lead_days, resting_slope,
  resting_slope_nf_p05, resting_slope_disc
- `cache/forensics/<VIN>_daily.csv`: day, n_eo, <35 features incl crank_recovery_t>, dtf, vin_label
- `results/V11_ALT_heuristics_comparison.csv`: vin_label, v1062_horizon,
  v1062_feature, v11_horizon, v11_feature, earlier, new_in_v11
- config exposes: FAILED_VIN_SET, ALL_VINS, NEW_FEATS, BAD_HIGH, BAD_LOW,
  FORENSICS, RESULTS_DIR, REPORTS_DIR, MIN_EO_SAMPLES.

Verified headline (do not recompute by hand): V11 recall 6/10 vs V10.6.2 5/10;
NF self-test 0/15; compound 4/10, 0/15 NF; MVP feature crank_recovery_t.

## 2. House style (mirror V10.6.2)

matplotlib, 150 dpi PNG + 300 dpi `_hd.png`; navy `#0D1B2A` text/axes, gold
`#C58B1F` headers, green `#27AE60` good, red `#C0392B` bad, blue `#2980B9`;
grid alpha ~0.25; legends OUTSIDE the plot (below or right); titles 12–14pt bold;
honest annotations (no hidden truncation). Each graph writes PNG + HD PNG.

## 3. Component A — Graphs

Module: `V11_ALT_heuristics/src/V11_ALT_heuristics_graphs.py`, output to
`V11_ALT_heuristics/visualizations/V11_graphs/`. Pure-ish (matplotlib Agg
backend, no display). One function per graph + `main()` that writes all and a
`Graphs_generation_report.md`.

- **G1 `recall_comparison`** — grouped bars: discriminative recall V11 6/10 vs
  V10.6.2 5/10; annotate compound 4/10 and NF false alarms 0/15. Source:
  comparison.csv (count verdicts), nf_self_test, compound_alarm_lovo.
- **G2 `feature_generalization`** — horizontal bars of the 19 NEW_FEATS by number
  of failed VINs discriminative (nunique vin_label where discriminative==True in
  failed_window_deviations), bar color by class (generalizes ≥2/0–1NF green;
  anecdotal ==1 amber; no_signal ==0 grey); `crank_recovery_t` annotated as MVP;
  NF false-alarm count per feature shown as a red marker if >0. Source:
  failed_window_deviations + nf_self_test.
- **G3 `leadtime_dumbbell`** — one row per failed VIN; dot at V10.6.2 horizon,
  dot at V11 horizon, connector; "none"→0. Highlight VIN9 (new_in_v11) and VIN1
  (earlier). Source: comparison.csv.
- **G4 `compound_alarm_leads`** — horizontal bars of the 4 fired FAILED VINs'
  early_watch_horizon_days (sorted), annotate n_votes; footnote 0/15 NF false
  alarms. Source: compound_alarm_lovo.csv.
- **G5 `crank_recovery_trajectories`** — the MVP physics graph. For VIN1_F,
  VIN8_F, VIN9_F: plot crank_recovery_t (y) vs dtf (x, reversed so failure at
  right) for trusted days (n_eo≥MIN_EO_SAMPLES); horizontal NF p95 band from
  nf_baseline (crank_recovery_t row); mark the earliest-detection horizon.
  Source: <VIN>_daily.csv + nf_baseline.csv.
- **G6 `changepoint_resting`** — two-panel: (left) per-VIN change-point lead days
  (cp_resid/cp_resting/dose_knee) as grouped bars with an "actionable <90d" line
  to show most fire too early; (right) resting_slope vs NF p05 with disc flag.
  Source: changepoint_per_vin.csv. Labeled "exploratory".

`Graphs_generation_report.md`: list of graphs, source file per graph, honesty
notes (G5 z-magnitude caveat, G6 too-early caveat).

## 4. Component B — Customer report

Module: `V11_ALT_heuristics/src/V11_ALT_heuristics_customer_report.py`.
Outputs `reports/V11_ALT_heuristics_customer_report.md` and
`reports/V11_ALT_heuristics_fleet_report.xlsx` (openpyxl/pandas).

Markdown sections (precursor-centric):
1. Executive Summary — verified headline + honest framing (modest, real gain).
2. What V11 Adds — VIN9 new, VIN1 earlier (30→60d), crank_recovery_t MVP.
3. Per-Truck Lead-Time Table — V10.6.2 vs V11 (from comparison.csv).
4. Heuristic Catalog & Verdict — 12 heuristics → generalizes/anecdotal/no_signal.
5. Compound Early-Watch Alarm — 4/10, 0/15 NF, earliest first-triggers.
6. Change-point & Resting-slope — exploratory, leads too early.
7. Honest Limitations — inflated z, VIN8 3-day coverage, 4/10 undetectable,
   NO RUL change, n=10 multiple-comparison caveat.
8. Deployment Recommendation — add crank_recovery_t + compound 2-of-5 vote to the
   emergency channel alongside GED=2; fault-typing features for repair guidance.

Xlsx sheets: Summary, LeadTime_Comparison, Heuristic_Verdict, Compound_Alarm,
Changepoint, NF_SelfTest.

## 5. Component C — Presentation

Modules under `V11_ALT_heuristics/presentation/`:
`build_technical_presentation.py`, `build_business_presentation.py` (python-pptx,
13.33"x7.5", Calibri, navy/gold/green/red). Embed graphs G1–G6 as images.

Technical deck (~13 slides): Title · Problem (V10.6.2 emergency box fires 2/10,
no timing) · Approach (12 heuristics on 6 raw CAN channels) · Honest-gate method
(z≥2 AND outside NF p05-p95) · Results headline (G1) · Per-VIN head-to-head (G3)
· MVP crank_recovery_t (G5) · Feature generalization (G2) · Compound alarm (G4) ·
Change-point exploratory (G6) · Limitations · Deployment recommendation · Appendix.

Business deck (5 slides): Headline · What we did · Results & impact (G1) ·
Limitations & data gaps · Next steps.

Also `presentation/DATA_SOURCES.md` and `presentation/AUDIT_REPORT.md` — every
numeric claim traced to a V11 cache file; explicit note that RUL/Weibull content
is out of scope for V11.

## 6. Out of scope

- No RUL / Weibull / backtest / risk-ranking / service-schedule content
  (V11 has none; reference V10.6.2 for those).
- No re-run of the V11 forensic pipeline (inputs already committed).
- No new classifier training (W6 forbidden patterns still apply: no sklearn/Ridge( in src).

## 7. Success criteria

- 6 graphs (PNG + HD) + generation report, all from V11 data.
- Customer report md + xlsx, precursor-centric, honest caveats present.
- Two pptx decks embedding the graphs, + DATA_SOURCES.md + AUDIT_REPORT.md.
- Every number traceable to a V11 cache file; no fabricated RUL figures.
- Build order: graphs → report → decks, each committed.
