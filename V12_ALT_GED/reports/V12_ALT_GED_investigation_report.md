# V12 — Alternator GED Signal: Deep Investigation & Prognostic Reverse-Engineering

**Iteration:** V12 (`V12_ALT_GED/`) · **Date:** 2026-06-26 · **Fleet:** 25 ALT trucks (10 failed + 15 non-failed) · **Baseline:** frozen V11.1 Ridge, LOVO AUROC 0.9267
**Status:** Complete. Honest, evidence-first. Every interpretive claim carries a confidence level.

> **One-paragraph verdict.** GED is **not** an unknown signal and **not** a quantized duty cycle — it is a **discrete 4-state status enum** ("State of Alternator Excitation"), confirmed by both the KT and external OEM/standards literature (High confidence), and confirmed empirically on the fleet data (Spearman ρ = 0.03 vs voltage; state-3 sits *above* state-2, so the values are not an ordered continuum). The continuous excitation variable the original brief hoped to reconstruct **does exist** — but only on the **LIN bus inside the alternator** (regulator duty-cycle/current registers); it is never broadcast to the CAN telematics feed, so true reconstruction is **sensor-blocked**, not method-limited. The feasible substitute we built — a VSI regulation-effort residual — and a battery of novel GED features (acceleration, onset-slope, regime-conditioned rate, Markov transition-rate, dwell) were tested honestly against the 0.9267 ceiling under the frozen LOVO protocol: **none add lift (best Δ −0.0067; all-combined −0.16)**. GED's real, deployable value remains exactly what V11.1 already ships: a **high-precision binary emergency channel** (GED2 ≥ 200/day → fires 2/10 failed, 0/15 non-failed). The genuinely new understanding from V12 is *structural*: in failed trucks GED=2 is far "stickier" (P(2→2)=0.992 vs 0.860; dwell 31.4 vs 4.1 samples), and the data-quality reality that several "GED-silent" failures are actually **GED-absent** (up to 99.9% null), not GED-healthy.

---

## How to read this dossier
Each of the 12 commissioned deliverables is answered below with its evidence and a confidence level. Supporting artifacts live in `V12_ALT_GED/results/` (JSON/CSV), `V12_ALT_GED/literature/` (cited external research), and `V12_ALT_GED/graphs/` (figures). The data substrate is `results/ged_daily_cache.parquet` (per-VIN-per-day GED counts, regimes, co-occurring signal stats; reconciles exactly to known anchors).

---

## Deliverable 1–3 — Physical interpretation of GED states, evidence, confidence

GED is the ECU's discrete rollup of the alternator excitation circuit's health, synthesised from the voltage-regulator fault flags (LIN) and the WL/D+ hardware pins, then broadcast on the vehicle CAN bus. Per-state interpretation (cross-validated KT ↔ external literature; see `literature/V12_ALT_GED_literature_findings.md`):

| State | Meaning | Physical interpretation | Evidence | Confidence |
|---|---|---|---|---|
| **0** | No disturbance | Excitation normal; regulator actively driving rotor field | KT (>99.7% of rows); J1939 2-bit `00b` = normal | **High** |
| **1** | Not allowed | ECU-commanded excitation inhibit (load-shed / protection) | KT; J1939 `01b`; Bosch CR665 / NXP AR6000 support programmatic suppression. **Never observed in any of the 59 fleet trucks** | **Medium** (functionally plausible; empirically absent) |
| **2** | Disturbance | Active excitation fault (brush/rotor wear, rectifier, regulator instability, open D+/WL) | KT; J1939 `10b` = error; **485× enrichment** in failed (0.31% vs 0.0006%); 2D shows it co-occurs with depressed voltage | **High** |
| **3** | Signal not available | CAN/sensor comms gap — **not** a component fault | KT; J1939 `11b` = not-available; appears in healthy trucks (up to ~29%) | **High** |

**Deliverable 4 — Validation against KT.** Every external finding **agrees** with the KT enum; no contradiction found. The previously-flagged KT caveat that **GED=3 is not failure-exclusive** is re-confirmed empirically (it appears across healthy VINs). Confidence: **High**.

---

## Deliverable 5 — Is GED quantized? (the core hypothesis)

**No — GED is a status enum, not a quantized continuous variable. Confidence: High.**
- **Empirical (Phase 2A, `results/2a_ordinality.json`):** treating each VIN-day's dominant state as an ordinal and correlating with co-occurring voltage gives Spearman **ρ = 0.030** (n=13,720) — no monotonic relationship. Per-state median VSI: state-0 = 27.97 V, state-2 = 27.81 V, **state-3 = 28.04 V**. State-3 sits *above* state-2; a quantized "more-excitation" ladder would be monotonic. The ordering is categorical, not ordinal.
- **Structural (Phase 1):** the four values map exactly onto the SAE J1939 2-bit status convention (`00`=normal, `01`=second-state, `10`=error, `11`=not-available). A duty cycle would take fractional values across a range; GED takes only integers {0,1,2,3}.

---

## Deliverable 6 — Industry reconstruction approaches, and why they don't apply here

The continuous excitation the brief targeted is real but lives one bus away from our data (Phase 1, High confidence):

| Layer | Signal | Form | Captured in our feed? |
|---|---|---|---|
| LIN (regulator ↔ ECU) | excitation **duty cycle** (`RDC`, 5-bit), excitation **current** (`RMC`, 8-bit), temperature | continuous | **No** |
| CAN (ECU → telematics) | excitation **status** (= GED) | discrete enum | **Yes** |

Industry methods to obtain continuous excitation (multi-signal regression, HMM/Kalman state estimation, physics back-calculation) all require at least one of: alternator **output current** (J1939 SPN 115, exists but typically not broadcast), rotor **field current** (LIN `RMC`, never on CAN), or **alternator temperature**. **All three are absent** from our 6-signal feed (`CSP, RPM, ANR, GED, VSI, SMA`). **Deliverable 6 verdict: continuous-excitation reconstruction is sensor-blocked, not method-limited.** Confidence: High.

---

## Deliverable 7 — Recommended reverse-engineering methodology (what we *can* do)

Two feasible, data-driven substitutes, both built and run:
1. **Empirical trigger model (Phase 2D, `results/2d_trigger_*`).** Logistic + gradient-boosted model of `P(GED=2 | VSI,RPM,ANR,CSP,SMA)` on a balanced sample (n=45,792 from the two GED-bearing failed VINs). **VSI is the dominant trigger** (logit coef **−0.348**; permutation importance 0.185 ≫ RPM 0.068 ≫ ANR 0.030 ≫ CSP 0.007 ≫ SMA 0). GED=2 fires when bus voltage is depressed and engine speed elevated — empirically reverse-engineering the fault condition the ECU is flagging. Confidence: High (within the 2-VIN signal base).
2. **VSI regulation-effort proxy (Phase 3B, `results/3b_regulation_features.csv`).** Fit the healthy-fleet conditional surface `E[VSI | RPM,ANR,CSP]` (binned medians over 15 non-failed VINs, 7.5M sampled engine-on rows, 1,959 bins); the residual `VSI_obs − VSI_expected` is a *regulation-effort* proxy. This is the closest feasible stand-in for "excitation effort," explicitly **not** field current.

---

## Deliverable 8 — Failure signatures

| Signature | Finding | Source | Confidence |
|---|---|---|---|
| **GED=2 storm** | VIN1_F: count ramps 2→765→6,692→12,367 starting **T-21 d** — textbook brush/rectifier signature | prior + `2b_occupancy.csv` | High |
| **Markov stickiness** (NEW) | Once in GED=2, failed trucks **stay**: P(2→2) = **0.992** vs **0.860** non-failed; entry P(0→2) ~8× higher in failed | `2c_transitions_*.csv` | High |
| **Dwell length** (NEW) | Mean GED=2 run = **31.4 samples (failed)** vs **4.1 (non-failed)** — 7.6× longer episodes | `2c_dwell.json` | High |
| **Regulation oscillation** (NEW) | VIN8_F shows `resid_oscillation` = **0.793** vs fleet median ~0.09 — erratic voltage regulation, a distinct electrical-degradation signature | `3b_regulation_features.csv` | Medium (single VIN) |

**Critical data-quality caveat (Phase 2E, NEW & important).** The "8/10 failed have zero GED=2" headline conflates two very different situations: trucks that ran healthy then failed abruptly **vs** trucks whose **GED channel is essentially absent**. VIN3_F (99.89% null) and VIN4_F (99.88% null) had almost no GED telemetry at all — GED *could not* have warned because it was barely transmitted. Fleet GED-null ranges **0.96%–99.9%**; missingness does **not** rise toward failure (near-failure 23.5% vs baseline 21.1%, +2.4 pp — `2e_null_report.json`), so nulls are a telemetry-coverage artifact, not a precursor. Confidence: High.

---

## Deliverable 9 — Prognostic applications

- **Deployable today (unchanged from V11.1):** the **GED=2 emergency channel** — `ged2_cnt ≥ 200/day` fires **2/10 failed** (VIN1 lead 21 d, VIN10 lead 1 d), **0/15 non-failed**. High precision, low recall, event-driven; kept separate from the RUL/classifier accuracy claim. Confidence: High.
- **Classifier:** GED contributes nothing beyond voltage features (Deliverable 12).

---

## Deliverable 10 — Novel excitation-derived health indicators

Three are genuinely new and physically meaningful, even though none beats the classifier ceiling: **(a) Markov GED=2 persistence** `P(2→2)` and **dwell length** (cleanly separate failed vs non-failed populations); **(b) regulation-effort oscillation** `resid_oscillation` (flags erratic regulation, e.g. VIN8_F); **(c) regime-conditioned GED=2 rate** (idle vs cruise). These are best used as **supplemental/triage signals on the emergency channel**, not as classifier inputs.

---

## Deliverable 11 — Feature-engineering recommendations

All seven candidate features were materialised per-VIN (`results/4_ged_features.csv`) and tested. Recommendation: **do not add any to the production classifier.** Reasoning is empirical (Deliverable 12) and structural — GED-derived features live in only 2/10 failed trucks, so as per-VIN aggregates at n=25 they are near-constant and add variance, not signal. Keep `resid_oscillation` and `P(2→2)`/dwell as **monitored diagnostics** feeding the emergency/triage layer.

---

## Deliverable 12 — Applicability for RUL / classification

**Honest verdict: no GED-derived feature improves the model. Confidence: High.** Under the exact frozen LOVO protocol (RidgeClassifier α=1.0, leave-one-VIN-out, train-median impute → StandardScaler → pooled-OOF AUROC), reproduced **exactly at 0.9267** as a gate (`results/4_lovo_trial.csv`):

| Feature set | LOVO AUROC | Δ vs baseline |
|---|---|---|
| **FAMILY_A (baseline 6)** | **0.9267** | — |
| + ged2_acceleration / onset_slope / rate_idle / rate_cruise | 0.9200 | −0.0067 each |
| + resid_oscillation | 0.9133 | −0.0133 |
| + resid_mean | 0.9067 | −0.0200 |
| + resid_slope_30d | 0.8800 | −0.0467 |
| + ALL 7 GED features | 0.7667 | −0.1600 |

Every candidate *hurts*. This re-confirms the **data ceiling** (not a method ceiling): at n=25 with GED signal concentrated in 2 trucks, voltage-shape features (FAMILY_A) already capture everything separable. **RUL applicability of GED = the binary emergency channel only.**

---

## Sensor-gap recommendation (to unlock true excitation prognostics)

The investigation's most actionable output. To convert "sensor-blocked" into "feasible," the next-gen / 500-truck fleet feed should add (in priority order):
1. **LIN regulator registers** `RMC` (excitation current) + `RDC` (excitation duty cycle) — the actual continuous excitation, today trapped on the in-alternator LIN bus. **Highest value.**
2. **J1939 SPN 115 (alternator output current)** — already defined in PGN 65271/VEP1; usually just not broadcast. Enabling it is low-cost.
3. **Alternator temperature** — required for any physics-based excitation model.
4. **Generator speed (`W` pin)** — decouples alternator speed from engine RPM (the new ASAM/CPC platform exposes it).

With (1)+(3) the brief's original reconstruction (regression/HMM/Kalman on excitation) becomes genuinely feasible and could target the *gradual-electrical* failure class directly. This feeds the **500-truck scale-up plan** as a concrete instrumentation ask. Confidence: High that these signals are necessary; Medium on how much prognostic lift they would yield (the gradual-electrical mode is only a fraction of failures).

---

## Confidence & limitations summary
- **High confidence:** GED is a status enum (not a duty cycle); reconstruction is sensor-blocked; no GED feature beats 0.9267; Markov stickiness & dwell differences; null-coverage caveat.
- **Medium confidence:** exact J1939 SPN (proprietary, unconfirmed — PGN 64934 is a name-match only); `resid_oscillation` as a signature (single VIN); prognostic lift achievable from added sensors.
- **Hard limits:** n=25, GED=2 present in only 2 failed trucks, 6-signal feed, no current/field/temperature channels.

## Artifacts
- Data substrate: `results/ged_daily_cache.parquet` · Phase outputs: `results/2a_ordinality.json`, `2b_occupancy.csv`, `2c_transitions_{failed,nonfailed}.csv`, `2c_dwell.json`, `2d_trigger_importance.csv`, `2d_trigger_report.json`, `2e_null_*`, `3b_regulation_features.csv`, `4_ged_features.csv`, `4_lovo_trial.csv`
- Literature: `literature/V12_ALT_GED_literature_findings.md` (14 sources)
- Figures: `graphs/ged_state_timeline_VIN1_F_ALT.png`, `transition_heatmap_{failed,nonfailed}.png`, `regulation_residual_overlay.png`, `trigger_importance.png`
- Code: `src/*.py` · Tests: `tests/*.py` (all green) · Spec/plan: `docs/superpowers/{specs,plans}/2026-06-26-alternator-ged-investigation*.md`
