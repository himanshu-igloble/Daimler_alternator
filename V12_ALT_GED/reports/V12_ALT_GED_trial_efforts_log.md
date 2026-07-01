# V12 / V12.1 — Trial Efforts Log (Excitation & GED Reverse-Engineering)

**Scope:** every experiment we ran in the V12 GED deep-investigation + the V12.1 engine-torque proxy + the internet methods survey — with hypothesis, method, result, and honest verdict. This is the **internal, full-detail** record (no redaction). It documents what we tried to squeeze more alternator signal / reconstruct excitation, and exactly why each did or did not move the needle.
**Date:** 2026-07-01 · **Branch:** `v12-alt-ged` · **Baseline under test:** frozen V11.1 Ridge, 6 features (FAMILY_A), **LOVO AUROC 0.9267**, 25 ALT VINs (10 failed + 15 non-failed), 6 CAN signals (`CSP, RPM, ANR, GED, VSI, SMA`) at ~5 s.

> **One-line summary:** across **8 trials** — GED enum/Markov/trigger/null analyses, the VSI regulation proxy, the GED prognostic-feature LOVO trial, the excitation-reconstruction methods survey, and the ANR engine-torque proxy — **nothing beats the frozen 0.9267 or extends lead-time.** The consistent verdict is a **data/sensor ceiling, not a method gap.** The by-products (new GED-Markov understanding, the telemetry-coverage cluster, the honest reconstruction verdict + sensor-gap ask) are the real value.

---

## Trial 1 — GED quantization / ordinality test (Phase 2A)
- **Hypothesis:** the GED states `{0,1,2,3}` are a *quantized continuum* of an underlying excitation variable (i.e., an ordinal ladder), so they could be de-quantized.
- **Method:** per VIN-day dominant GED state vs co-occurring daily-mean voltage; Spearman ρ of state-rank against VSI. `src/analysis_2a_quantization.py` → `results/2a_ordinality.json`.
- **Result:** ρ = **0.030** (n=13,720). Per-state median VSI: state-0 = 27.97 V, state-2 = 27.81 V, **state-3 = 28.04 V** — state-3 sits *above* state-2.
- **Verdict:** **REFUTED.** Non-monotonic → GED is a categorical **status enum**, not an ordinal continuum. (Corroborated by the literature: maps exactly onto the SAE J1939 2-bit status convention.) No de-quantization possible.

## Trial 2 — GED Markov transition structure + dwell (Phase 2C)
- **Hypothesis:** the *transition dynamics* of GED (not just counts) differ between failed and healthy trucks.
- **Method:** gap-aware (≤60 s) sample-to-sample transitions, null→3; 4×4 row-normalised transition matrices + mean dwell run-length, per population. `src/analysis_2c_markov.py` → `results/2c_transitions_{failed,nonfailed}.csv`, `2c_dwell.json`.
- **Result:** **P(2→2) = 0.992 (failed) vs 0.860 (non-failed)**; P(0→2) = 6.81e-6 (F) vs 8.2e-7 (NF) (~8× higher entry in failed); mean GED=2 dwell **31.4 samples (F) vs 4.1 (NF)** (7.6×).
- **Verdict:** **NEW UNDERSTANDING (positive), but not a classifier win.** GED=2 is far "stickier" in failed trucks — a real structural difference never characterised before. Because GED=2 lives in only 2/10 failed VINs, it does not generalise as a per-VIN classifier feature (see Trial 6); its value is diagnostic / emergency-channel context.

## Trial 3 — GED=2 trigger reverse-engineering (Phase 2D)
- **Hypothesis:** we can empirically learn *what conditions trigger* a GED=2 disturbance.
- **Method:** logistic + gradient-boosted model of `P(GED=2 | VSI,RPM,ANR,CSP,SMA)` on a balanced sample (n = **45,792**, drawn from VIN1_F + VIN10_F — the only two GED=2-bearing VINs). `src/analysis_2d_triggers.py` → `results/2d_trigger_*`.
- **Result:** **VSI is the dominant trigger** (logit coef **−0.348**; permutation importance 0.185 ≫ RPM 0.068 ≫ ANR 0.030 ≫ CSP 0.007 ≫ SMA 0). RPM positive (+0.703).
- **Verdict:** **UNDERSTANDING (confirms KT).** GED=2 fires when bus voltage is depressed — empirically reverse-engineers the fault condition the ECU flags. Not a new predictive feature; explanatory only.

## Trial 4 — Null-informativeness (Phase 2E)
- **Hypothesis:** GED *missingness* is itself informative (e.g., rises as a truck approaches failure).
- **Method:** null fraction by population / operating-regime / days-to-failure. `src/analysis_2e_nulls.py` → `results/2e_null_*`.
- **Result:** NF null 33.1%, failed null 21.2%; near-failure (dtf≤30 d) **23.5% vs baseline 21.1% (+2.4 pp)**. Per-VIN null 0.96%–**99.9%**.
- **Verdict:** **REJECTED as a precursor** (missingness does not meaningfully rise toward failure). **BUT** surfaced the key data-quality finding: VIN3_F (99.89%) and VIN4_F (99.88%) are ~totally GED-null → several "GED-silent" failures are actually **GED-absent**, not GED-healthy.

## Trial 5 — VSI regulation-effort proxy (Phase 3B)
- **Hypothesis:** the residual of voltage against a healthy-fleet conditional surface is a usable "regulation-effort" health proxy (feasible stand-in for the sensor-blocked excitation).
- **Method:** fit `E[VSI | RPM,ANR,CSP]` (binned medians over 15 NF VINs, 7.5M sampled engine-on rows, 1,959 bins); per-VIN residual features `resid_mean, resid_neg_frac, resid_oscillation, resid_slope_30d`. `src/proxy_3b_regulation.py` → `results/3b_regulation_features.csv`.
- **Result:** `resid_mean` does **not** separate (fleet means +0.015 F vs +0.036 NF — negligible). `resid_oscillation` flags **VIN8_F = 0.793 vs fleet median ~0.09** (erratic-regulation signature; single VIN).
- **Verdict:** **WEAK.** No fleet-wide separation; `resid_oscillation` is a single-VIN diagnostic, carried into Trial 6 as a candidate.

## Trial 6 — GED / regulation prognostic-feature LOVO trial (Phase 4)
- **Hypothesis:** the never-built GED + regulation features add lift over the 0.9267 classifier ceiling.
- **Features tested:** `ged2_acceleration, ged2_onset_slope, ged2_rate_idle, ged2_rate_cruise, resid_mean, resid_slope_30d, resid_oscillation`.
- **Method:** self-contained reproduction of the frozen LOVO protocol (RidgeClassifier α=1.0, leave-one-VIN-out, train-median impute → StandardScaler → pooled-OOF `roc_auc_score`); **gate: FAMILY_A reproduced exactly at 0.9267**; then FAMILY_A + each candidate. `src/features_4_ged.py`, `src/trial_4_lovo.py` → `results/4_lovo_trial.csv`.
- **Result:** every candidate **hurts** — `ged2_acceleration/onset_slope/rate_idle/rate_cruise` −0.0067 each; `resid_oscillation` −0.0133; `resid_mean` −0.0200; `resid_slope_30d` −0.0467; **+ALL 7 → 0.7667 (−0.16)**.
- **Verdict:** **ALL REJECTED.** GED signal lives in 2/10 failed VINs, so as per-VIN aggregates at n=25 the features are near-constant and add variance, not signal. Data ceiling.

## Trial 7 — Continuous-excitation reconstruction (internet methods survey)
- **Hypothesis:** the continuous field-excitation current / PWM duty cycle can be reverse-engineered from the 6 CAN signals.
- **Method:** deep-research harness — 6 angles, 25 sources, **25 claims 3-vote adversarially verified (24 confirmed, 1 killed)**. `literature/V12_ALT_GED_excitation_reconstruction_methods.md` (+ downloaded source PDFs in `literature/papers/`).
- **Result:** (1) **State observers** (Kalman/Luenberger/SMO) can recover field current but need **≥1 real current sensor + kHz sampling** → infeasible at 0.2 Hz (Eull, IEEE ITEC 2022). (2) **Analytical VSI→field inversion** is double-blocked — regulator flat-lines VSI ~28 V + magnetic saturation (IntechOpen ch.38166; Ostovic MEC). (3) **Ripple / load-dump / coast-down** are kHz signatures → fully aliased at 0.2 Hz. (4) Only feasible routes are **honest load proxies** (engine-torque or grey-box), never the true value.
- **Verdict:** **TRUE RECONSTRUCTION INFEASIBLE (sensor-blocked, not method-limited).** The duty cycle + field current live on the in-alternator **LIN bus** (`RMC`/`RDC` registers), never broadcast to CAN. Unlock = added instrumentation (LIN `RMC`/`RDC`, J1939 SPN 115, alt temp). This *pointed to Trial 8*.

## Trial 8 — V12.1 idle-ANR engine-torque load proxy (LOVO)
- **Hypothesis:** the survey's #1 feasible avenue — an idle-conditioned engine-torque residual — is a usable alternator electrical-**load** proxy. Physics: `T = V·i/(η·ω)`; at idle, propulsion torque ≈ 0 so ANR ≈ accessory load (Remy US7,283,899; Ford US9,126,580; CNH US10,752,188).
- **Features:** `anr_idle_mean, anr_idle_resid_mean, anr_idle_resid_slope_30d, anr_idle_resid_oscillation` (idle = RPM∈[500,800], CSP<3, SMA=0; residual vs healthy per-RPM-bin baseline). `src/proxy_anr_torque.py`, `src/trial_v121_anr.py` → `results/v121_anr_*.csv`.
- **Result:** baseline 0.9267 reproduced as gate; **single-feature AUROC 0.11–0.16 (anti-predictive)**; additive Δ −0.0533 / −0.0533 / −0.0333 / −0.0467; **+ALL 4 → 0.8733 (−0.0533)**. Failed trucks show slightly *lower* idle-ANR (18.5 vs 24.8 Nm). The **5 idle-sparse VINs (≤2.8k idle rows)** with inflated 40–81 Nm residuals are **exactly the 5 near-total-GED-null VINs** (VIN3_F, VIN4_F, VIN8_NF, VIN9_NF, VIN15_NF).
- **Verdict:** **REJECTED.** The ~1–5 Nm alternator drag is swamped by other idle accessories (AC/cooling fan) that vary by route/climate; the sparse-telemetry cluster confounds further. Confirms the survey's separability caveat empirically.

---

## Summary table

| # | Trial | What was tested | Verdict | Headline number |
|---|-------|-----------------|---------|-----------------|
| 1 | GED quantization (2A) | is GED an ordinal continuum? | REFUTED | ρ=0.030; state-3 VSI > state-2 |
| 2 | GED Markov structure (2C) | transition dynamics F vs NF | NEW UNDERSTANDING | P(2→2) 0.992 F vs 0.860 NF; dwell 31.4 vs 4.1 |
| 3 | GED=2 trigger (2D) | what triggers GED=2 | UNDERSTANDING | VSI top trigger, logit −0.348 |
| 4 | Null-informativeness (2E) | is missingness a precursor? | REJECTED (+ data-quality find) | near-fail 23.5% vs base 21.1% (+2.4pp) |
| 5 | VSI regulation proxy (3B) | residual = health proxy? | WEAK | resid_mean no separation; osc VIN8_F 0.793 |
| 6 | GED feature LOVO trial (P4) | 7 features beat 0.9267? | ALL REJECTED | best −0.0067; all-7 −0.16 |
| 7 | Excitation reconstruction | recover field/duty from 6 sig | INFEASIBLE (sensor-blocked) | observers need sensor+kHz; VSI-inv blocked |
| 8 | ANR torque proxy (V12.1) | idle-ANR = load proxy? | REJECTED | single-feat AUROC 0.11–0.16 (anti-predictive) |

## Overarching conclusion
- **Nothing beat the frozen 0.9267 or extended lead-time.** Every predictive trial (6, 8, and the feature families in 5) was rejected under the identical honest LOVO protocol, with the baseline reproduced exactly as a gate each time.
- **The ceiling is data + sensors, not method.** Two structural walls: (a) **n = 25 / 10 events** — signals that live in a handful of trucks (GED=2 in 2/10) can't generalise as per-VIN aggregates; (b) **the 6-signal, 0.2 Hz CAN feed** physically lacks the channels (output/field current, alt temperature, kHz sampling) that true excitation methods require.
- **Genuine by-products (the value):** the **GED Markov stickiness** (new characterisation), the **telemetry-coverage cluster** (the same 5 VINs are degenerate across GED-null *and* idle-sparse analyses — a data-quality signature worth flagging at fleet onboarding), the honest **reconstruction-infeasibility verdict**, and the concrete **sensor-gap recommendation** (LIN `RMC`/`RDC` + SPN 115 + alt temp + generator-speed W-pin, transferred from a few reference trucks).
- **What would move it:** more trucks (narrows the CI, lets sparse signals generalise) and added instrumentation — not another feature squeezed from the current feed.

## Provenance
Code under `V12_ALT_GED/src/`, results under `V12_ALT_GED/results/`, literature under `V12_ALT_GED/literature/`. All LOVO numbers reproduce the frozen 0.9267 baseline as a gate. Earlier-iteration negative results (V2.1 SM feature hunt, unsupervised anomaly detection, usage-clock RUL, per-truck covariate RUL) live in their own iteration folders and can be consolidated into this log on request.
