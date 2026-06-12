---
title: "V11.1_ALT — Covariate-Informed RUL, Full Pipeline Redo (Design Spec)"
status: complete
created: 2026-06-12
---

# V11.1_ALT Design Spec

## 0. Purpose & frozen surface

V11.1 redoes every phase of the alternator pipeline EXCEPT classification, asking
one new scientific question: **can the V11 heuristic channels (built from the 12
candidates in `V10.6.2_ALT/reports/V10.6.2_ALT_candidate_heuristics_for_lead_time.txt`)
individualize per-truck RUL via a covariate survival model?**

- **Frozen (only thing):** classification results — `V10.6_ALT/cache/ridge/ridge_prob_rescaled.csv`
  (AUROC 0.927) + decision threshold 0.4456, read by reference. W6 forbidden-pattern
  audit still applies (no `Ridge(`, `sklearn`, `IsotonicRegression` in V11.1 src).
- **Redone:** survival, predictive RUL, backtest, emergency, decisions, assembly,
  verification, graphs, reports, presentation — all under `V11.1_ALT/` with
  `V11.1_ALT_` file prefix, mirroring V10.6.2's stage structure.
- **Reused by reference (deterministic inputs):** V11's committed forensic outputs
  (`V11_ALT_heuristics/cache/forensics/` — 25 daily panels, nf_baseline,
  compound_alarm_lovo, earliest_signal) and V10.6's lifecycle parquet
  (`V10.6_ALT/cache/lifecycle/vin_lifecycle.parquet`, read-only) for TTF/censor times.

**Honest expectation:** V10.6.2 shelved covariate-β once (per-truck RUL lost to the
fleet-clock dummy, MAE 125d vs 49.7d). V11.1 re-asks with channels that did not
exist then. The gate decides; the β=0 fallback ships as a first-class outcome.

## 1. Stage map

| # | Module (V11.1_ALT_*) | Function |
|---|---|---|
| 0 | config | paths, VIN registry, grids, priors, thresholds, frozen refs |
| 1 | covariates | leakage-safe per-truck covariates x(t) from V11 daily panels |
| 2 | survival_aft | Bayesian Weibull AFT grid posterior (variants: β=0, x1, x1+x2) |
| 3 | predictive_rul | conditional-on-survival PI per VIN from chosen posterior |
| 4 | backtest | time-rewound LOVO with truncated covariates vs dummy + β=0 |
| 5 | emergency | 3-channel: GED=2 storm + crank-recovery exceedance + compound vote |
| 6 | decisions | risk band × time window × emergency state (adds early-watch tier) |
| 7 | assemble_rul + narratives | final per-VIN table + human-readable narratives |
| 8 | verify | hard gates G-LEAK / G-BETA / G-W6 / G-EMERG / G-COVER (exit 1) |
| 9 | rul_graphs + fleet/per-VIN graphs | survival curve, backtest bars, RUL waterfall, per-VIN conditional-RUL curves, fleet overlays (V11.1 owns posterior samples again) |
| 10 | markdown_report + excel_report + decks | customer md + xlsx, technical + business pptx, DATA_SOURCES, AUDIT_REPORT |

Orchestrator runs 1→10 with GATE_FAIL halt semantics (V10.6.2 pattern).

## 2. Covariates (leakage-critical)

Computed from V11 trusted daily rows (`n_eo >= 200`) only:

- **x1(t)** = `log(1 + #days{day <= t AND crank_recovery_t > NF_p95}`) — cumulative
  exceedance count of the MVP channel; monotone, robust to the episodic-spike nature.
- **x2(t)** = 1 if the compound 2-of-5 vote fired in the trailing 90 days up to t, else 0.

Rules (each enforced by gate G-LEAK):
1. At rewind time t, covariates use rows with `day <= t` only.
2. NF p95 reference computed from training-fold NF trucks only (LOO inside LOVO).
3. Training trucks (already failed) use full observed life — legitimately observable.
4. Held-out truck's covariate never sees post-rewind data.
5. For the whole-fleet fit (non-backtest), failed trucks use x at failure, censored
   trucks x at censor time — standard time-fixed AFT exposure summary; stated
   plainly in reports that the fit covariate on failed trucks contains pre-failure
   signal *by construction*; the rewound LOVO is the honest validation.

## 3. Survival model

Weibull AFT: `T_i ~ Weibull(shape k, scale λ_i)`, `λ_i = λ0 · exp(−β·x_i)`
(positive β·x → shorter life). Bayesian grid posterior; reduces exactly to
V10.6.2's model at β=0.

- Grids: k ∈ [2,12] (100 pts), λ0 ∈ [500,1100] (100 pts), β1 ∈ [−0.2,1.0] (25 pts),
  β2 ∈ [−0.2,1.5] (18 pts). Priors: k~N(3.5,1.5), λ0~N(650,100) (V10.6.2 values),
  β~N(0,0.5) each. Likelihood: event → pdf, censored → survival; vectorized numpy.
- Three variants fit: M0 (β=0 — V10.6.2-equivalent), M1 (x1 only), M2 (x1+x2).
  Model selection by rewound-LOVO day-MAE; G-BETA decides what ships.
- Cohort: all 25 VINs (10 events + 15 censored), TTF/censor from lifecycle parquet.
- Outputs: posterior grids + samples per variant, fleet survival curve, params json.

## 4. Backtest (the decisive validation)

Time-rewound LOVO over the 10 failed VINs at T−270/T−180/T−90:
fit the variant on the 24 others, truncate the held-out truck's covariate to the
rewind date, predict conditional median RUL given survival-to-rewind, error vs the
true h. Metrics: day-MAE vs **dummy-A fleet-clock (49.7d)** and vs **M0 (125d/141.8d)**,
PI coverage, signed-rank p. Also report mean 80%-PI width per variant.

## 5. Emergency layer (3 channels, all kept separate in reporting)

1. **GED=2 storm** — daily count ≥ 200 (V10.6.2, unchanged logic, recomputed).
2. **Crank-recovery exceedance** — ≥ k trusted days with `crank_recovery_t > NF p95`
   within any trailing 30-day window; k calibrated as the smallest integer giving
   **0/15 NF fires** (start k=2; raise if needed; report the calibration table).
3. **Compound 2-of-5 vote** — from V11's committed `compound_alarm_lovo.csv` logic,
   re-evaluated as a daily/weekly state for the decision engine.

## 6. Decision engine

Extends V10.6.2's 2×2 to **risk band (frozen ridge: green/amber/red) ×
time window (short if PI p10 < 90d) × emergency state (none / early-watch /
emergency)** → tiered recommendation per VIN. Early-watch = compound or exceedance
channel active; emergency = GED=2 storm.

## 7. Hard gates (verify exits 1)

- **G-LEAK** — recompute x1/x2 at every rewind point from truncated panels inside
  verify; any mismatch with backtest's values → fail.
- **G-BETA** — ship a covariate variant only if (a) rewound day-MAE beats BOTH
  dummy-A and M0 with signed-rank p<0.05, OR (b) mean PI width shrinks ≥15% vs M0
  at maintained ≥80% coverage. Else verdict `NO_IMPROVEMENT`: shipped tables use
  M0 (≡V10.6.2 numbers) and every report states the negative result plainly.
- **G-W6** — forbidden-pattern scan over `V11.1_ALT/src` (config excluded).
- **G-EMERG** — expanded emergency: 0/15 NF fires across ALL channels; failed-truck
  recall ≥ each individual channel's V11 recall (GED 2/10, compound 4/10).
- **G-COVER** — chosen variant's LOVO PI coverage ≥ 80% (8/10).

## 8. Deliverables

- Full artifact suite mirroring V10.6.2: 4 core RUL graphs, 25 per-VIN
  conditional-RUL curves, 3 fleet overlays (+stats csv/xlsx), customer report
  (md + xlsx), technical + business decks, DATA_SOURCES.md, AUDIT_REPORT.md —
  all from V11.1's own caches (legitimate again, since V11.1 owns posterior samples).
- `results/V11.1_ALT_verification.json` + a V11.1-vs-V10.6.2-vs-V11 comparison table.

## 9. Out of scope

- No classifier re-training or re-scoring (frozen surface).
- No re-run of V11 forensic feature extraction (reused by reference; a config flag
  documents how to re-run it if ever needed).
- No time-varying-covariate AFT (n=10 events; time-fixed exposure summaries only).
- No SM-fleet work.

## 10. Success criteria

- All 11 stages run green end-to-end via the orchestrator; gates pass or fail honestly.
- The covariate question answered with evidence either way; fallback path exercised
  in code (unit-tested), not just promised.
- Every reported number traceable to a V11.1 cache file; TDD on all new math
  (likelihood, truncation, exceedance windows, decision matrix).
