# Alternator RUL — Version Comparison: V10.6.2 vs V11

> This document is **identical** on the `v10.6.2-alt` and `v11-alt` branches so either branch
> tells the full story. It is sourced from the shipped reports (see *Provenance* at the bottom);
> no numbers are from memory.

Both versions share the **same validated classifier** (V10.5.3 frozen, LOVO AUROC **0.927**) and the
**same fleet replacement curve** (Weibull, empirical median **601 d**). They differ only in the
question each was asked and in the **lead-time / emergency channel**.

---

## TL;DR

| Dimension | **V10.6.2** — Honest Baseline | **V11** — Lead-Time Heuristics |
|---|---|---|
| Question asked | Does *per-truck RUL* beat the fleet clock? | Can new heuristics improve *lead-time recall*? |
| Classifier — *WHICH* trucks | V10.5.3 frozen, LOVO AUROC **0.927** | **Same** 0.927 (frozen, untouched) |
| Fleet window — *WHEN (fleet)* | Weibull, empirical median **601 d** (p25–p75 578–652 d) ≈ 120 440 km / 4 538 eng-h | **Same** curve (untouched) |
| Per-truck RUL | backtest MAE **142 d** vs fleet-clock dummy **50 d** → **no_improvement**; intervals calibrated (9/10 coverage) | **Not addressed** (V11 is precursor-only) |
| Discriminative-precursor recall — *WHEN (emergency)* | **5/10** forensic · **2/10** GED-only deployable | **6/10** forensic · **0/15** NF false alarms |
| Deployable emergency channel | GED=2 storm only (2/10) | GED=2 **+ post-crank recovery** (`crank_recovery_t`) |
| New features evaluated | — | **12 candidate heuristics** on the same 6 raw CAN channels |
| Verdict | **NO_IMPROVEMENT** — ship fleet curve + GED monitor | Real but **modest** gain: +1 truck, +1 earlier; no structural change |

**One-line takeaway:** V11 keeps everything V10.6.2 validated and adds *one* genuinely useful new
emergency feature — post-crank voltage recovery — which detects **one extra failed truck (VIN9)** and
gives an **earlier warning on a second (VIN1)**, all with **zero false alarms** on the 15 healthy trucks.
It does **not** overturn the standing finding that most alternator failures are abrupt/silent.

---

## What V11 actually changed (only 2 of 10 failed trucks moved)

| VIN | V10.6.2 lead | V10.6.2 feature | V11 lead | V11 feature | Earlier? | New in V11? |
|---|---|---|---|---|---|---|
| **VIN9_F_ALT** | none | — | **30 d** | `crank_recovery_t` | ✅ | ✅ newly detected |
| **VIN1_F_ALT** | 30 d | `ged2_frac` | **60 d** | `crank_recovery_t` | ✅ | — (earlier lead) |
| VIN2_F_ALT | 14 d | `crank_vsi_min` | 14 d | `crank_vsi_min` | — | — |
| VIN6_F_ALT | 60 d | `vsi_sag_frac` | 60 d | `vsi_sag_frac` | — | — |
| VIN8_F_ALT | 7 d | `resting_vsi_mean` | 7 d | `crank_recovery_t` | — | — |
| VIN10_F_ALT | 7 d | `vsi_sag_frac` | 7 d | `vsi_sag_frac` | — | — |
| VIN3_F_ALT | none | — | none | — | — | — |
| VIN4_F_ALT | none | — | none | — | — | — |
| VIN5_F_ALT | none | — | none | — | — | — |
| VIN7_F_ALT | none | — | none | — | — | — |

- **4 trucks already detected** (VIN2/6/8/10) — unchanged.
- **4 trucks still undetected** by any channel (VIN3/4/5/7) — the abrupt/silent failures.
- **2 trucks improved** (VIN9 new, VIN1 earlier) — both driven by the single MVP heuristic below.

---

## The MVP heuristic — #3 post-crank recovery (`crank_recovery_t`)

The only one of the 12 new features that moved the headline recall.

- Discriminative on **6/10** failed VINs with **0/15** NF false alarms.
- Physical story: *every key-on is a free stress test.* A weakening charging/battery interface
  recovers to ~27 V **more slowly** after the crank load is removed.
- Caveat (do not over-read): within-truck z-scores are huge (VIN8 ~1288, VIN9 ~155) because the
  healthy-fleet baseline recovery-time variance is near zero plus a 30 s censor — read them as
  "recovery clearly slowed vs the healthy fleet," **not** a calibrated effect size. The trustworthy
  guard is the **0/15 NF self-test**, which it passes.

**Orthogonal early-watch tier — compound 2-of-5 weak-vote alarm (#11):** fires on **4/10** failed
trucks, **0/15** NF false alarms, and gives the *earliest* first-trigger for some (VIN8 @90 d, VIN1 @60 d).

Of the 12 candidates, 9 "generalize" (≥2 failed VINs, 0–1 NF false: `crank_recovery_t`,
`sag_highload_frac`, `reg_duty_frac`, `crank_dur_mean`, `idle_vsi_var`, `vsi_ceiling`, `vsi_resid_mean`,
`ged_churn`, `sag_idle_frac`) — useful for **fault-typing / repair guidance** — but only
`crank_recovery_t` changed the earliest-precursor verdict.

---

## What did NOT change (the honest part)

- **Classifier 0.927** (WHICH trucks) — identical, frozen.
- **Fleet Weibull curve** (WHEN, fleet-level) — identical: median 601 d, shape ≈ 5.17.
- **Structural limit** — still **no per-truck daily RUL**; V10.6.2's backtest (MAE 142 d vs 50 d
  fleet clock) showed covariates can't beat the fleet clock at n=25, and V11 did not revisit RUL.
- **Deliverable shape** — unchanged: **WHICH** (classifier) + **WHEN-fleet** (Weibull) +
  **WHEN-emergency** (now firing earlier for VIN1 and for one additional truck, VIN9).
- Most alternator failures remain **abrupt/silent** with no electrical precursor.

---

## Methodology note — why you'll see "2/10", "5/10" and "6/10"

These are not contradictions; they count different things:

| Number | What it counts | Source |
|---|---|---|
| **2/10** | V10.6.2 **deployable GED-only** emergency channel (GED=2 storm) | V10.6.2 customer report §3 |
| **5/10** | V10.6.2 **broad forensic** discriminative-precursor recall (any channel) | V11 head-to-head |
| **6/10** | V11 **broad forensic** discriminative-precursor recall (any channel) | V11 report headline |

So the apples-to-apples lift is **5/10 → 6/10** (broad forensic, +1 truck), and the deployable
emergency channel grows from **GED-only (2/10)** to **GED + post-crank recovery**.

---

## Provenance (where every number comes from)

- **V10.6.2** — `V10.6.2_ALT/reports/V10.6.2_ALT_customer_report.md`
  (AUROC 0.927; median 601 d; RUL MAE 142 d vs 50 d, no_improvement; GED 2/10, VIN1 21-day lead).
- **V11** — `V11_ALT_heuristics/reports/V11_ALT_heuristics_report.md` and
  `V11_ALT_heuristics/results/V11_ALT_heuristics_comparison.csv`
  (recall 6/10 vs 5/10; VIN9 new @30 d, VIN1 @30→60 d; 0/15 NF false alarms; MVP `crank_recovery_t`).
- **3-way context (incl. V11.1)** — `V11.1_ALT/reports/V11.1_ALT_customer_report.md` §8
  "Three-Way Comparison V10.6.2 → V11 → V11.1" (lives on the `v11.1-alt` branch, not here).
