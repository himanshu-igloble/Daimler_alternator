# Alternator RUL — Three-Way Version Comparison: V10.6.2 vs V11 vs V11.1

> This is the curated **V11.1** branch (alternator only). It mirrors the `v10.6.2-alt` and `v11-alt`
> branches. The authoritative 3-way table lives in `V11.1_ALT/reports/V11.1_ALT_customer_report.md` §8;
> this file restates it for quick reference. No numbers are from memory.

All three iterations share the **same validated classifier** (V10.5.3 frozen, LOVO AUROC **0.927**) and
the **same fleet replacement curve** (Weibull, empirical median **601 d**, shape ≈ 5.17, scale 771 d).
Each asked a *different* question on top of that frozen base.

---

## TL;DR — what each iteration asked, and answered

| Iteration | Question | Classifier AUROC | Precursor recall | RUL day-MAE | Dummy MAE | Verdict |
|---|---|---|---|---|---|---|
| **V10.6.2** | Does *per-truck RUL* beat the fleet clock? | 0.927 | **2/10** (GED-only) | ~140 d | 49.7 d | **NO_IMPROVEMENT** — M0 fleet curve only |
| **V11** | Can 12 new heuristics improve *lead-time recall*? | 0.927 (frozen) | **6/10** (compound + `crank_recovery_t`) | N/A (precursor-only) | N/A | +1 truck (VIN9), +1 earlier (VIN1); 0/15 FP |
| **V11.1** | Can **AFT covariates** from V11 channels *individualize RUL*? | 0.927 (frozen) | **3/10** current-state early-watch | **140.4** (M0) / 148.8 (M1) / 162.2 (M2) | 49.7 d | **NO_IMPROVEMENT_HONEST** — covariates exposure-confounded |

**One-line takeaway:** the classifier (WHICH trucks) is solid at **0.927** and unchanged across all three.
Lead-time recall improved **2/10 → 6/10** between V10.6.2 and V11. **V11.1 confirms the third structural
barrier:** no covariate formulation can extract per-truck *timing* from these channels at n=25 — the
per-truck RUL MAE (~140 d) is unchanged and the fleet-clock dummy (49.7 d) still beats any per-truck
model until the sample is much larger.

---

## What V11.1 specifically tried (and why it didn't beat the clock)

- **Approach:** an Accelerated Failure Time (AFT) survival model with covariates `x1`/`x2` derived from
  the V11 channels (crank-recovery exceedance, compound trailing votes), fit as M1/M2 against the
  covariate-free baseline **M0** (which reduces exactly to the V10.6.2 Weibull at β=0).
- **Result:** out-of-sample (rewound LOVO with truncated covariates), **M0 140.4 d < M1 148.8 d <
  M2 162.2 d** — adding covariates made per-truck RUL *worse*, and all three are far above the naive
  **fleet-clock dummy (49.7 d)**. So **M0 is selected and β is shelved.**
- **Why:** the covariates are **exposure-confounded** — they mostly encode how much a truck has been
  driven/aged rather than independent degradation, and `time_dim = SHORT` for **all 15** non-failed
  trucks (the fleet is aged; current ages sit within/past the empirical failure band, so the time
  dimension is currently non-discriminating).

## The deliverable V11.1 actually ships

The trustworthy outputs are unchanged in shape — **WHICH** (classifier 0.927) + **WHEN-fleet**
(Weibull, M0 ≡ V10.6.2 curve) + **WHEN-emergency** — now surfaced as **3 alert channels**:
1. **GED=2 storm** — 2/10 failed, 0/15 NF false alarms (deployable, high-precision).
2. **Current-state compound early-watch** — 3/10 currently-active failed trucks, 0/15 NF false alarms.
3. Per-truck RUL is shipped as a **survival-conditioned 80% band** (point estimate explicitly
   non-actionable: MAE 140.4 d vs 49.7 d).

---

## The honest through-line across all three versions

- **Unchanged:** classifier **0.927** and the fleet Weibull curve (median **601 d**) — frozen in all three.
- **V10.6.2** proved per-truck RUL doesn't beat the fleet clock (MAE 142 d vs 50 d).
- **V11** added the only feature that moved lead-time recall — post-crank recovery (`crank_recovery_t`):
  recall 5/10 → 6/10 forensic, 0/15 false alarms; VIN9 newly detected, VIN1 earlier.
- **V11.1** closed the per-truck-RUL question for good at this n: covariates can't individualize timing
  (M0 wins, β shelved). The structural limit stands — **ship rank + window + emergency, not a date.**

---

## Provenance

- **V10.6.2** — see the `v10.6.2-alt` branch: `V10.6.2_ALT/reports/V10.6.2_ALT_customer_report.md`.
- **V11** — see the `v11-alt` branch: `V11_ALT_heuristics/reports/V11_ALT_heuristics_report.md`.
- **V11.1** — this branch: `V11.1_ALT/reports/V11.1_ALT_customer_report.md` (§8 = the authoritative
  3-way table) and `V11.1_ALT/reports/Alternator_3Version_Summary.docx`.
