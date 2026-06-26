---
title: "Alternator GED Signal — Deep Investigation & Prognostic Reverse-Engineering"
status: "draft"
created: "2026-06-26"
updated: "2026-06-26"
---

# Alternator GED Investigation — Design Spec

## 1. Motivation & Origin

A deep-research request asks us to (1) reverse-engineer the physical meaning of the
alternator **GED** signal (observed states 0/1/2), (2) determine whether GED is a
quantized representation of a continuous excitation variable, (3) reconstruct that
continuous excitation `E(t)` from neighbouring signals, and (4) mine it for failure
prognostics / RUL — validated throughout against the project's KT (Knowledge Transfer)
documents.

A context sweep (3 parallel agents over KT docs, prior GED code/reports, and the actual
parquet schemas) materially reframes the request. This spec captures that reframing and
defines a **comprehensive, honest, evidence-first investigation** delivered as a
**technical R&D dossier** (executed analyses + new feature code + graphs + cited
literature validation). Management packaging (DOCX/slides) is deliberately deferred until
findings justify it.

## 2. Reframing — what is already settled vs. genuinely open

### 2.1 GED is a KT-defined STATE ENUM, not a duty cycle (high confidence)
KT docs (`KT_daimler/KT_startermotor_alternator.md`, `docs/column_dictionary.md`)
authoritatively define GED = **"State Alternator Excitation"**, discrete `{0,1,2,3}`:

| State | KT meaning | Observed in fleet |
|---|---|---|
| 0 | No disturbance (normal excitation) | > 99.7% |
| 1 | Not allowed (ECU inhibited) | **Never observed** across all 59 VINs |
| 2 | Disturbance (active excitation fault) | 0.31% failed vs 0.0006% healthy → **485× enrichment** |
| 3 | Signal not available (CAN comms fault) | Up to 12–29% in some *non-failed* VINs — **not** a failure signal |

The "Generator Excitation **Duty** / PWM duty-cycle" reading exists in the project **only**
as an external research hypothesis (`docs/2026-06-09-...-alternator-domain-research-brief.md`),
explicitly framed as inferred physics, and even there describes GED as a *flag reporting*
duty + fault flags — not a quantized continuous duty signal. **Treat enum-not-duty-cycle
as the working truth; still test it empirically (Phase 2A) and against external literature
(Phase 1).**

### 2.2 Continuous-excitation reconstruction is sensor-blocked (high confidence)
The alternator dataset contains exactly **6 CAN signals**: `CSP, RPM, ANR, GED, VSI, SMA`
(verified against parquet schemas). The reconstruction recipe in the original prompt
requires **battery current, alternator output current, rotor/field current, alternator
temperature, electrical load** — **all absent**. Only **VSI (voltage)** is a usable
continuous analog channel. True physics/regression/HMM/Kalman reconstruction of `E(t)`
cannot be validated against any ground-truth current and is therefore **infeasible**.
The honest, feasible substitute is a **VSI regulation-effort proxy** (Phase 3).

### 2.3 Prior GED work vs. open gaps
**Already done** (do not repeat, cite as baseline): 485× row-level enrichment; the VIN1
**T-21d GED=2 storm**; deployed high-precision/low-recall **emergency channel** (2/10
failed fire, 0/15 NF); GED as the M5 health-zone component C3 (weight 0.294) + transition
driver labels; the full GED feature family in the classifier — **none ever helped the
0.927 Ridge model** (standalone AUROC ~0.43–0.54, because the signal lives in only 2 trucks).

**Never attempted (this investigation's novel core):** empirical GED ordinality/quantization
test; **Markov transition structure** (matrices + dwell times, healthy vs failed);
data-driven reverse-engineering of *what triggers* GED states; **null-informativeness**
(21–44% missing); recommended-but-unbuilt prognostic features (`ged2_acceleration`,
onset-slope, regime-conditioned rates); the **VSI regulation-effort proxy** as a continuous
health indicator.

## 3. Scope & Deliverable
- **In scope:** all 12 research deliverables of the original prompt, reframed around what is
  feasible and novel; external literature validation; executed empirical analyses on the
  25 ALT VINs; an honest offline prognostic-feature trial; a sensor-gap recommendation.
- **Deliverable form:** technical R&D dossier — a markdown report (in the style of the V11
  fleet reports) + committed, tested analysis/feature code + professional graphs.
- **Out of scope (now):** management DOCX/slides; touching the frozen V11.1 production
  model; any cross-dataset (ALT↔SM) VIN analysis (forbidden by the VIN-independence rule).

## 4. Verified ground-truth (anchors for the work)
- **Fleet:** 25 ALT trucks = 10 failed + 15 non-failed. (SM fleet is a separate set of
  trucks; not used here.)
- **Data:** `Data/processed/alternator_complete/2026-03-06-12-38-15-alternator_failed.parquet`
  (36.67M rows, 11 cols, 2024-01-31→2025-12-27) and `...-alternator_non_failed.parquet`
  (60.08M rows, 8 cols, 2023-12-31→2026-02-18). Cadence ≈ 5 s; `timestamp` (µs). Failed
  files add `SALEDATE`, `JCOPENDATE` (= failure date), `Failure_type`.
- **GED nulls:** 21–44%; OEM guideline imputes nulls → GED=3.
- **GED=2 distribution:** failed — VIN1_F ≈ 82,357 events (~2.0%), VIN10_F ≈ 2,897, other
  8 failed = **0**; non-failed — all trace (≤315 events), **0/15 fire the emergency**.
- **VIN1 storm onset:** T-22d=2 → T-21d=765 → T-19d=6,692 → T-15d=12,367 (peak).
- **Baseline model:** Ridge, 6 features
  (`vsi_std_ratio_30d, vsi_dominant_freq, vsi_spectral_entropy, bat_charge_delta_trend_right,
  vsi_range_trend_last30d, progressive_drift`), **LOVO AUROC 0.9267**, zero GED features.
- **Reusable code:** `V5.2_20_5_ALT_feature_engineering.py` (GED feature family),
  `V11_ALT_heuristics_features.py` (`ged_states`, `vsi_rpm_curve`, `load_residual`),
  `V10.5.3_20_5_ALT_hourly_ged_monitor.py`, `V10.6.2_ALT_ged_emergency.py`,
  `V11_1_ALT_emergency.py`, `V11_2_ALT_rul_evidence_stack.py` (M5 zones/transitions).

## 5. Design — six phases

### Phase 0 — Data substrate (foundation; must complete first)
Build or locate a **per-VIN daily GED aggregate cache** for all 25 ALT VINs: per day, the
count/rate of each GED state, null rate, and co-occurring distributions of
VSI/RPM/ANR/CSP/SMA (means, key percentiles), plus days-to-failure for failed VINs. One
verified pass over the parquet (reuse existing daily-cache builders if present). All
downstream phases read this cache, never the raw 96M rows. **Verification:** row counts and
GED=2 totals must reconcile with the Section-4 anchors above.

### Phase 1 — External literature validation (deep-research skill, web)
Reverse-engineer GED's meaning across Bosch / Valeo / Denso / Prestolite / Mitsubishi,
Daimler-Mercedes charging architectures, LIN smart-alternator nodes, and **SAE J1939 / ISO**.
Priority lead: GED's `{0=No disturbance, 1=Not allowed, 2=Disturbance, 3=Not available}`
pattern strongly resembles a **J1939 2-bit measured-status SPN** (11→"not available",
10→"error/disturbance", 01→"not allowed/command-disabled", 00→"normal") — if confirmed this
*settles* enum-not-duty-cycle and may pin the exact SPN/PGN. Output: a cited interpretation
of each state with explicit confidence levels, cross-validated against KT, with any
discrepancies flagged. Adversarial verification of load-bearing claims.

### Phase 2 — Empirical GED characterization (novel data work; five independent analyses)
- **2A — Quantization / ordinality test.** Is 0/1/2/3 an ordered continuum or a categorical
  status enum? Test for monotonic relationships of state vs VSI/load; check whether state 2
  sits "between" 0 and 3 in any signal space. Expected: refutes quantization, confirms enum.
- **2B — Occupancy & per-VIN distribution.** Clean state-occupancy %, per VIN, healthy vs
  failed; reconcile with anchors. Foundation table for the report.
- **2C — Markov transition structure.** 3×3 (states {0,2,3}) transition-probability matrices,
  dwell-time distributions, per healthy vs failed; compare transition structure between
  populations. Never built before.
- **2D — Data-driven trigger reverse-engineering.** Model `P(GED=2 onset | VSI, RPM, ANR,
  CSP, SMA, operating-regime)` (regime = idle/cruise/load/crank derived from RPM/CSP/ANR/SMA)
  to learn what GED empirically responds to. Repeat for GED=3 onset. Reveals the real
  trigger conditions behind the enum.
- **2E — Null-informativeness.** Is the 21–44% GED missingness itself a signal (correlated
  with state, regime, VIN, or failure proximity)? Validate / challenge the "nulls→3" rule.

### Phase 3 — The continuous-excitation question, answered honestly
- **3A — Infeasibility documentation.** Precisely state why physics/regression/HMM/Kalman
  reconstruction of `E(t)` (original Objectives 3A–3D) is sensor-blocked: no output current,
  no field current, no alternator temperature; only VSI is continuous.
- **3B — VSI regulation-effort proxy (feasible substitute).** Fit a healthy-fleet conditional
  surface `E[VSI | RPM, ANR, CSP]`; residual = "regulation effort." Extend the existing
  `vsi_rpm_curve` / `load_residual` heuristics into a **continuous health indicator** with
  statistical / dynamic / stability / drift features. Label clearly as a *regulation* proxy,
  **not** true excitation/field current.
- **3C — (stretch) Exploratory latent-state model.** A small HMM over VSI-regime + GED to see
  whether a hidden "excitation-health" state emerges. Exploratory only; not a reconstruction
  claim. Drop if 3B is sufficient.

### Phase 4 — Prognostic trial & RUL applicability (honest offline experiment)
Build the never-built GED-derived features (`ged2_acceleration`, onset-slope,
regime-conditioned GED2 rate, Markov transition-rate, dwell-time, and Phase-3 regulation-
residual features) and test them **strictly** under the project's LOVO protocol against:
(a) the **AUROC 0.927** Ridge classifier ceiling, and (b) the **2-of-10 lead-time recall**
of the emergency channel. Report what beats the baseline and what does not. **Do not modify
the production model;** this is an offline verdict.

### Phase 5 — Synthesis & deliverables
Technical report answering **all 12 deliverables** with evidence + confidence levels;
professional graphs (state timelines, transition heatmaps, regulation-residual overlays,
trigger-model summaries); and a concrete **sensor-gap recommendation** — which CAN
signals / J1939 SPNs to add (field-current shunt, output-current sensor, alt-temperature,
generator-speed) to actually enable excitation reconstruction — tied to the 500-truck
scale-up plan.

## 6. Mapping to the original 12 deliverables
1. Physical interpretation of GED states → Phase 1 + 2D.
2. Evidence per interpretation → Phase 1 (literature) + Phase 2 (data).
3. Confidence levels → assigned per finding in Phases 1–2.
4. Validation against KT → Phase 1 cross-validation; discrepancy log.
5. Whether GED is quantized → Phase 2A + Phase 1 (verdict: expected enum).
6. Industry reconstruction approaches → Phase 1 (literature) + Phase 3A (mapping).
7. Recommended reverse-engineering methodology → Phase 2D + Phase 3B.
8. Potential failure signatures → Phase 2B/2C + existing storm baseline.
9. Prognostic applications → Phase 4.
10. Novel excitation-derived health indicators → Phase 3B (regulation proxy) + 2C features.
11. Feature-engineering recommendations → Phase 4 feature catalog + verdicts.
12. Applicability for RUL → Phase 4 verdict vs 0.927 / lead-time recall + Phase 5 sensor gap.

## 7. Execution approach — phased parallel waves
Orchestrator (me) + subagents, per the superpowers model. Phase 1 runs on the
`deep-research` skill; Phase 2's five analyses are independent and fan out to parallel
data-analysis agents; Phase 4 features build/test in parallel. **Findings from subagents are
treated as leads and verified** (key citations, numbers, and any claim that drives a verdict)
before entering the report. Respect the ≤3-parallel default and run a usage check between
waves; pause if any active window ≥95%. Phases are sequential (0→5) with verification gates;
parallelism is *within* phases.

## 8. Output structure
```
V12_ALT_GED/
  src/        # data-substrate builder, analyses (2A–2E), proxy (3B), features+trial (4)
  results/    # cached aggregates, transition matrices, trial metrics (json/csv)
  reports/    # V12_ALT_GED_investigation_report.md (the dossier)
  graphs/     # timelines, transition heatmaps, regulation-residual overlays, trigger plots
docs/superpowers/specs/2026-06-26-alternator-ged-investigation-design.md   # this spec
```

## 9. Success criteria
- Every one of the 12 deliverables answered with cited evidence and an explicit confidence
  level (no hand-waving; "infeasible" is an acceptable, documented answer).
- Phase-2 numbers reconcile with the Section-4 anchors (or discrepancies are explained).
- Phase-4 verdict is a clean pass/fail against the 0.927 ceiling and the 2/10 lead-time
  recall, under the existing LOVO protocol.
- New code is committed and has at least smoke/unit tests; analyses are reproducible from
  the daily cache.
- The report is honest: it states what GED *is*, what cannot be reconstructed and why, what
  is genuinely new, and what (if anything) is worth deploying.

## 10. Honest expectations
Literature likely **confirms** the enum (probably a J1939 status SPN); quantization
**refuted**; reconstruction **documented infeasible**; Markov + trigger analyses yield real
new *understanding*; the feature trial **probably does not beat 0.927**. Realistic wins:
(a) definitively closing the duty-cycle / reconstruction questions with evidence;
(b) the VSI **regulation-effort proxy** as a new continuous health indicator;
(c) a possible **lead-time / emergency-channel** extension (where GED actually lives).

## 11. Non-goals
- No production-model change (V11.1 stays frozen).
- No management DOCX/slides in this pass.
- No cross-dataset (ALT↔SM) VIN analysis.
- No claim of true excitation/field-current reconstruction from the current 6-signal data.

## 12. Risks & mitigations
- **Re-deriving known results.** Mitigation: Section-4 anchors + prior-work map; new work is
  scoped to the explicit gap list (§2.3).
- **Over-interpreting a 2-truck signal.** Mitigation: every GED=2 claim reports n-VINs and
  guards against single-VIN dominance; per-VIN breakdowns mandatory.
- **Literature over-reach (claiming the OEM).** Mitigation: OEM make/model is an unverified
  gap; report only what sources support, with confidence levels; J1939-SPN claim must be
  adversarially verified.
- **Compute on 96M rows.** Mitigation: Phase 0 cache; all later phases read aggregates.
- **Null handling distorting rates.** Mitigation: Phase 2E quantifies missingness before any
  rate-based conclusion; report rates both raw and null-adjusted.

## 13. Open questions
None blocking. Management packaging is intentionally deferred (revisit after Phase 5).
