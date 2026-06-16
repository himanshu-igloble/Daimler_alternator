# Requirements — Daimler / BharatBenz Alternator RUL & Risk

This document captures the **functional**, **data**, **technical**, and **acceptance** requirements for
the alternator predictive-maintenance project. It applies to both versions in this repo
(`V10.6.2_ALT/` and `V11_ALT_heuristics/`); the [`README.md`](./README.md) covers navigation and the
[`VERSION_COMPARISON_V10.6.2_vs_V11.md`](./VERSION_COMPARISON_V10.6.2_vs_V11.md) covers what differs.

---

## 1. Problem statement

Predict and prioritise alternator maintenance for a fleet of BharatBenz 5528T heavy-duty trucks from
on-board CAN-bus telemetry, so a service operation can (a) inspect the right trucks first, (b) plan a
fleet replacement window, and (c) be paged when a rare early electrical precursor appears — **without
over-promising a per-truck failure date that the data cannot support**.

---

## 2. Functional requirements

| # | Requirement | Delivered as | Status |
|---|---|---|---|
| FR-1 | Rank in-service trucks by failure risk (**WHICH**) | Frozen Ridge classifier, LOVO AUROC **0.927** | ✅ both versions |
| FR-2 | Provide a **fleet-level** replacement window (**WHEN, fleet**) | Weibull survival fit, empirical median **601 d** (p25–p75 578–652 d) ≈ 120 440 km / 4 538 eng-h | ✅ both versions |
| FR-3 | Emit an **emergency precursor** alert for trucks that show one (**WHEN, truck**) | GED=2 storm monitor (V10.6.2); GED=2 **+ post-crank recovery** (V11) | ✅ both versions |
| FR-4 | Provide a per-truck RUL **band** (interval), clearly caveated | Survival-conditioned 80% interval (point estimate flagged as non-actionable) | ✅ V10.6.2 |
| FR-5 | Improve lead-time recall with new heuristics, **only if it survives an honest gate** | 12 candidate heuristics; recall 5/10 → **6/10**, 0/15 false alarms | ✅ V11 |
| FR-6 | Ship human- and machine-readable outputs | `reports/` (md + xlsx), `results/` (csv + json), `service/` tables, `presentation/` decks | ✅ both versions |
| FR-7 | Be **reproducible end-to-end** and self-gating | Orchestrator + `verify` stage (exits non-zero on gate failure) | ✅ both versions |

**Explicit non-goal (out of scope):** a trustworthy per-truck *days-to-failure* number. V10.6.2's
out-of-sample backtest shows the survival RUL point estimate (MAE **142 d**) does **not** beat a naive
fleet-clock baseline (**50 d**) at n=25 — so per-truck RUL is shipped as a wide band, not a date, and
V11 does not attempt to individualise it further.

---

## 3. Data requirements

- **Source:** ~204 M rows / ~14.5 GB across 4 CSVs (failed/non-failed × alternator/starter-motor).
  **Not stored in this repo** (`.gitignore`d). Downstream stages consume the committed per-VIN daily
  aggregates in `*/cache/forensics/VIN*_daily.csv`, so all reports and figures reproduce offline.
- **Alternator fleet:** **25 trucks = 10 failed + 15 non-failed.** All labels carry an `_ALT` suffix.
- **VIN independence (hard rule):** alternator and starter-motor VINs are **different physical trucks**
  that reuse the same numbering. `VIN1_ALT ≠ VIN1_SM`. **No cross-dataset VIN-level analysis is valid.**
- **Canonical signals** (validated dictionary):

  | Column | Meaning | Valid range | Sentinels |
  |---|---|---|---|
  | `CSP` | Vehicle speed (km/h) | 0–100 | 65535 |
  | `RPM` | Engine speed (rev/min) | 0–3500 | 65535 |
  | `ANR` | Engine torque (Nm) | −400…1300 | 65535, −5000 |
  | `GED` | Alternator-excitation state | {0,1,2,3} | — (0 normal, 2 disturbance, 3 unavailable) |
  | `VSI` | Supply voltage (V) | 0–36 | 0, 255 (scale ×0.2 if raw is large) |
  | `SMA` | Starter-motor active | {0,1} | — |

---

## 4. Technical / runtime requirements

- **Python** 3.11+.
- **Libraries:** `numpy`, `pandas`, `scipy`, `matplotlib`, `lifelines` (survival fits).
  Reporting/decks additionally use `openpyxl` (xlsx) and `python-pptx` (pptx). `pytest` for V11 tests.
  ```bash
  pip install numpy pandas scipy matplotlib lifelines openpyxl python-pptx pytest
  ```
- **Run:**
  ```bash
  python V10.6.2_ALT/src/V10.6.2_ALT_orchestrator.py
  python V11_ALT_heuristics/src/V11_ALT_heuristics_orchestrator.py
  ```
- **Determinism:** the orchestrators rebuild every artefact in dependency order from the cached daily
  CSVs; the `verify` stage runs last and **fails the build** if any honest gate is violated.

---

## 5. Constraints & honest-assessment principles

- **No over-promising.** Claims must be backed by a shipped report/results file; conservative verdicts
  (e.g. `NO_IMPROVEMENT`) are preferred over flattering ones that the sample size cannot support.
- **Small-n discipline.** n=25 (10 events). New features must pass a within-truck z-gate **and** an
  out-of-fleet check, plus a 0/15 non-failed self-test, before they count.
- **Frozen reference.** The classifier (V10.5.3) and the fleet Weibull are frozen across versions so
  version-to-version differences are attributable only to the new lead-time work.
- **Physical plausibility.** Precursor features must have a mechanism (e.g. "every key-on is a free
  stress test" for post-crank recovery), not just a correlation.

---

## 6. Acceptance criteria (honest gates)

A version is acceptable when its `verify` stage passes and:

- FR-1 classifier reproduces **AUROC 0.927** (LOVO).
- FR-2 fleet window reproduces the **601 d** empirical anchor (Weibull shape ≈ 5.17).
- FR-3 emergency channel fires with **0/15** non-failed false alarms.
- FR-5 (V11) any new feature that changes the headline recall also passes the **0/15** self-test;
  recall moved **5/10 → 6/10** with the gate intact.
- FR-7 the full orchestrator runs green end-to-end from the committed cache.
