# V12.2 — Fable-5 Re-Run Addendum (GED Investigation)

**Date:** 2026-07-03 · **Trigger:** full re-run of the original GED deep-research brief under the Fable 5 model · **Method:** (R1) complete deterministic re-execution of the V12/V12.1 empirical suite; (R2) adversarial red-team of every V12 conclusion; (R3) independent recomputation + bounded new experiments on the red-team's hypotheses; (R4) targeted literature gap-fill. This addendum is the authoritative errata + extension layer over `V12_ALT_GED_investigation_report.md`, `V12_ALT_GED_trial_efforts_log.md`, and the two literature files — the originals are preserved unchanged as the historical record.

> **Verdict in one paragraph.** All three headline verdicts **survive adversarial re-review**: GED is a status enum; continuous-excitation reconstruction is sensor-blocked; nothing beats the 0.9267 LOVO ceiling. The re-run's real yield is fourfold: (1) a **reproducibility stamp** — all 12 scripts reproduce the original anchors exactly, 25/25 tests green; (2) **evidence-hygiene errata** — several V12 evidence sentences over-claimed and are corrected below (the decisions they supported all stand); (3) **two deployment improvements** — a verified two-tier (count × voltage-depth) emergency rule that hardens the 0-false-alarm property for the 500-truck scale-up, and an intra-day tally that converts VIN10-type storm alarms from end-of-day to ~52 minutes after onset; (4) **literature upgrades** — the best public SPN match is now **PGN 65237 / SPN 23904 "Alternator Excitation Status"**, the OAD/decoupler confounder is moot for this fleet, and a maritime transfer-learning precedent (arXiv:2510.03003) now backs the reference-truck instrumentation proposal.

---

## 1. Reproducibility stamp (R1)

Every V12/V12.1 script re-executed on the unchanged data: **all key numbers match the original anchors exactly** (2A ρ=0.030275; 2B VIN1 c2=82,357; 2C P(2→2)=0.991941/0.859729; 2D n=45,792, VSI logit −0.348395; 2E 23.5%/21.1%; 3B VIN8_F osc 0.792994; LOVO gate 0.9267 with the identical delta table; V12.1 singles 0.16/0.1467/0.1067/0.1067). Full pytest: **25/25 passed**. The R3 per-day GED2 table independently reconciles to the occupancy anchor to the event (its VIN1 rows sum to exactly 82,357).

## 2. Red-team verdicts (R2) — the five attacked conclusions

| Conclusion | Verdict | Net effect |
|---|---|---|
| A. GED = status enum | **CONFIRMED** — but the 2A empirical test is weak (see errata E1); the verdict rests on structural evidence | unchanged |
| B. Reconstruction sensor-blocked | **CONFIRMED** — observability + Nyquist arguments airtight; the one unsurveyed method class (estimation from quantized observations) dies on the same facts | unchanged |
| C. All 7 GED/regulation features rejected | **Decision CONFIRMED, evidence WEAKENED** (errata E2, E3) | reject stands, language corrected |
| D. Idle-ANR torque proxy rejected | **Decision CONFIRMED, evidence corrected** (errata E4) | reject stands, language corrected |
| E. Emergency channel (≥200/day) | **CONFIRMED decisively** — any threshold in [143, 765] yields the identical operating point; plus a scale-up hardening found (§4.1) | improved |

## 3. Errata (evidence-hygiene corrections; no decision changes)

**E1 — The 2A ordinality test is structurally weak; the enum verdict is re-based on its structural leg.** The "dominant state" labeling marks a day state-2 if it has even one GED2 sample; the Spearman is computed on a tie-saturated series (13,643/13,720 state-0 days); the state-2 vs state-3 voltage comparison is a between-truck confound; and the test is circular — the regulator flat-lines VSI, so under the duty-cycle hypothesis VSI would not track excitation anyway. **The enum verdict is unchanged** but now rests where it belongs: four integer values in a 2-bit CAN field (a quantized 5-bit RDC duty would show 32 levels), the LIN-continuous/CAN-discrete two-layer architecture, and the KT + J1939 convention agreement.

**E2 — LOVO noise-null calibration (new, R3): "anti-predictive" and "every candidate hurts" over-claimed.** Under the frozen pooled-OOF LOVO protocol at n=25, a pure-noise single feature scores **mean AUROC 0.303** (bimodal; p5=0.00, p50=0.34, p95=0.69), and adding a useless feature to FAMILY_A costs **mean −0.019** (delta band p5 −0.047 … p95 +0.007). Corrected reading: the candidates' single-feature AUROCs (0.11–0.16) are *at or below the noise-null* — "no signal," not "reversed physics"; the additive deltas (−0.0067…−0.053) are *within the null band of adding any useless feature*. The correct claim is **"no evidence of value at n=25 under this protocol"** — which fully supports the REJECT decisions. (Also: even a +0.0067 "gain" would have been inside the null; the protocol could not have crowned a weak winner either.) `results/r3_lovo_null.json`.

**E3 — Why the ceiling holds, precisely.** The baseline's 11 misranked LOVO pairs involve exactly the GED-blind trucks — VIN5_F (zero GED2 ever) below 10 NF trucks, and VIN4_F (99.88% GED-null) below VIN3_NF. GED-derived features are identically zero for every truck whose rank could improve; they were mathematically unable to fix the pairs that matter. The n=25 verdict therefore says nothing about a hurdle/two-stage design at n=500 — a fair caveat for the scale-up. (Bookkeeping: `resid_neg_frac` was built in 3B but not carried into the trial; raw AUROC 0.48 — no loss.)

**E4 — V12.1 group-mean contrast was a telemetry artifact.** The trial log's "failed trucks show lower idle-ANR (18.5 vs 24.8 Nm)" is driven entirely by the 5 idle-sparse VINs; excluding them, clean-fleet means are **10.2 vs 10.9 Nm** — no contrast. The rejection stands on the calibrated no-signal result (E2) plus the physics (~1–5 Nm drag swamped by other accessories), not on a direction reversal.

**E5 — The 2D trigger story needs a two-regime split.** "GED=2 co-occurs with depressed voltage" is true for the pooled row-level model but is driven by VIN10's terminal sag and within-day transients. **VIN1's storm days fire at healthy absolute bus voltage (27.83–28.59 V — above the NF GED2 maximum of 27.20 V)**, and its `vsi_when_ged2` is usually (not always; dtf=20 is −0.13 V) above its own day-mean. Two distinct GED2 regimes: *excitation-fault-at-healthy-bus* (VIN1 storm) vs *voltage-sag transients* (VIN10 terminal day at 23.1 V, and all 9 NF GED2 days at 20.8–27.2 V). This split is what powers the two-tier rule (§4.1).

**E6 — Markov population language downgraded.** P(2→2)=0.992 vs 0.860 and dwell 31.4 vs 4.1 are real but effectively single-truck-dominated contrasts (VIN1_F carries ~97% of failed GED2 mass; VIN11_NF ~71% of NF). Keep the finding; report it as per-truck-dominated, not fleet-population, evidence.

## 4. New results (R3 — independently recomputed, not red-team hearsay)

### 4.1 Two-tier emergency rule — ADOPT AT SCALE-UP (deployment hardening)
From the full per-day GED2 table (22 VIN-days; `results/r3_emergency_2tier.csv`): VIN1 storm days (cnt≥50) have `vsi_when_ged2` ∈ **[27.83, 28.59] V**; every NF GED2 day is ≤ **27.20 V** → voltage margin **+0.63 V**, count margin 765 vs 142 (**+623**). Rule: **Tier-1 = cnt ≥ 50 AND vsi_when_ged2 ≥ 27.0 V** (catches VIN1-type excitation storms; 0/15 NF across the full sweep C1∈{10..100}×V1∈{26.5..27.5}); **Tier-2 = cnt ≥ 200** unchanged (catches VIN10-type low-voltage terminal bursts, at 23.1 V). Combined: 2/10 F (leads 21 d / 1 d), 0/15 NF — identical operating point to today, **but with a second qualifying axis**. Honest framing: at n=25 the voltage gate adds *no incremental filtering* (count alone suffices); its value is robustness at 500 trucks, where the count-only margin (NF ceiling 142 vs threshold 200, 1.4×) is thin and every observed NF GED2 episode is a low-voltage sag the voltage gate would reject.

### 4.2 Intra-day tally — same-day alarms for terminal storms (ops improvement)
VIN10's single storm day (dtf=1, onset 10:00): cumulative GED2 crossed 50 events in **~4 minutes** and 200 in **~52 minutes**. An intra-day running tally (vs end-of-day batch) turns the "1-day lead" into an actionable **same-day alarm within the hour** — enough for a stand-down instruction. VIN1's storm is diffuse (15–24 episodes/day spanning the clock) and fires under any aggregation. `results/r3_storm_subdaily.json`.

### 4.3 Honestly killed (tested, negative)
- **Pre-storm VSI-sag burst** (self-normalized trailing-30d): VIN1's pre-storm max burst = **6.0**, NF ceiling = **7.0** (VIN7_NF) → inside NF range. **KILL** — no storm early-warning extension. `results/r3_sag_burst.json`.
- **Crank-recovery-time trend** (last-60d self-normalized slope from the V11 heuristics daily panels): Mann-Whitney AUROC **0.633, p=0.279** — not significant; VIN8_F (+2.97) and VIN2_F (+1.60) are suggestive but sit amid comparable NF noise. **Not deployable.** `results/r3_crank_trend.json`.
- Red-team pre-killed (probed dead, not re-run): GED-null blackout channel (NF trucks post identical acute null rises when parked), GED=3 burst channel (lives in the telemetry-degenerate cluster), time-varying-GED survival/AFT (10 events, covariate nonzero in 2 → re-reports VIN1).

## 5. Literature updates (R4 — targeted gap-fill, cited)

1. **SPN identification upgraded:** best public match is **PGN 65237 "Alternator Information 1" with SPN 23904 "Alternator Excitation Status"** (+ SPNs 3353–3356 "Alternator 1–4 Status"; Detroit Diesel SB-10055076-3172 surfaces SPN 3353/FMI 2 as "Generator D+ Terminal Failure"). This **replaces PGN 64934** (J1939-75, genset scope) from the first pass. Exact state enumerations remain paywalled in the J1939 Digital Annex; the Daimler-proprietary-SPN hypothesis stays open. *Medium confidence.* (isobus.net; hemdata.com; oemdtc SB PDF.)
2. **OAD/decoupler confounder resolved as moot:** overrunning decouplers do transmit ~zero torque during rotor overrun (Litens/Gates documentation — the mechanically-true half of the killed claim), **but** HD 24V truck alternators standardly ship with **solid pulleys** (Leece-Neville/Delco evidence); OADs are premium upgrades. The V12.1 torque-proxy rejection needs no OAD carve-out. *Medium confidence.*
3. **Reference-vehicle transfer precedent found:** Sharma et al., **arXiv:2510.03003** — high-frequency-instrumented reference vessel → low-rate fleet reports via transfer learning (10.6% MAPE cut on sister vessels). Structurally identical to our LIN-tap reference-truck proposal; the automotive version is confirmed **research whitespace** → frames our sensor-gap/Phase-3 pitch as both precedented (by analogue) and novel (in domain). *High confidence.*
4. **Low-rate signatures:** kHz-ripple infeasibility re-confirmed (US6862504 needs 20 kHz). Commercial fleet products (Intangles, Oxmaint) practice **day-scale VSI drift monitoring** (~0.15 V/30 d thresholds, claimed 18–42 d warnings) — no peer-reviewed benchmark, but it commercially corroborates the day-scale-VSI feature approach FAMILY_A already deploys. Cranking-event timing patents (US11473545 et al.) are event-based cousins of our existing `crank_recovery_t`. *Nothing new to build; validation of the existing design.*

## 6. Bottom line — what the Fable-5 re-run changed

| Layer | Outcome |
|---|---|
| Headline verdicts (enum · sensor-blocked · 0.9267 ceiling) | **Unchanged — survived adversarial attack** |
| Empirical numbers | **Reproduced exactly** (stamp) |
| Evidence hygiene | **6 errata** (E1–E6) — sentences corrected, decisions intact |
| Deployment | **+2 improvements**: two-tier (count × voltage-depth) emergency rule for scale-up robustness; intra-day tally → same-day storm alarms |
| New channels | **0** — sag-burst, crank-trend, GED-null, GED3, time-varying AFT all honestly killed |
| Literature | SPN 23904/PGN 65237; OAD moot; maritime transfer precedent; commercial drift corroboration |

The data ceiling stands: at n=25 with 6 signals, the deployed system (0.9267 ranking + 601-day window + emergency channel) is what this data supports. The re-run made the *evidence* bulletproof and the *deployment* slightly better — it did not, and could not, conjure new signal.

## 7. Artifacts
- R3 code: `src/analysis_r3_emergency_2tier.py`, `analysis_r3_lovo_null.py`, `analysis_r3_sag_burst.py`, `analysis_r3_storm_subdaily.py`, `analysis_r3_crank_trend.py`
- R3 results: `results/r3_emergency_2tier.csv`, `r3_lovo_null.json`, `r3_sag_burst.json`, `r3_storm_subdaily.json`, `r3_crank_trend.json`
- Superseded-in-part (originals preserved): `reports/V12_ALT_GED_investigation_report.md` (§ noted in E1/E5/E6), `reports/V12_ALT_GED_trial_efforts_log.md` (Trials 1, 6, 8 language per E1/E2/E4), `literature/V12_ALT_GED_literature_findings.md` (PGN 64934 → PGN 65237/SPN 23904), `literature/V12_ALT_GED_excitation_reconstruction_methods.md` (OAD open question → resolved moot)
