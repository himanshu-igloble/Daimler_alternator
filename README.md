# Daimler / BharatBenz Alternator — RUL & Risk Prediction

**Branches:**
[![main](https://img.shields.io/badge/main-all_3_versions-2ea44f?logo=github)](https://github.com/himanshu-igloble/Daimler_alternator/tree/main)
[![v10.6.2-alt](https://img.shields.io/badge/v10.6.2--alt-honest_baseline-1f6feb?logo=github)](https://github.com/himanshu-igloble/Daimler_alternator/tree/v10.6.2-alt)
[![v11-alt](https://img.shields.io/badge/v11--alt-lead--time_heuristics-1f6feb?logo=github)](https://github.com/himanshu-igloble/Daimler_alternator/tree/v11-alt)
[![v11.1-alt-curated](https://img.shields.io/badge/v11.1--alt--curated-covariate_RUL-1f6feb?logo=github)](https://github.com/himanshu-igloble/Daimler_alternator/tree/v11.1-alt-curated)
[![v11.1-alt](https://img.shields.io/badge/v11.1--alt-full_dev_branch-9e9e9e?logo=github)](https://github.com/himanshu-igloble/Daimler_alternator/tree/v11.1-alt)

Predictive-maintenance pipeline for the **alternator** of the BharatBenz 5528T heavy-duty truck.
The project answers three operational questions from on-board CAN-bus telemetry:

- **WHICH** trucks are most likely to fail (risk ranking — validated classifier).
- **WHEN**, at fleet level, alternators wear out (replacement window — survival/Weibull).
- **WHEN**, for the few trucks that show it, an early electrical precursor fires (emergency channel).

> **Honest-engineering project.** Every metric in this repo is backed by a shipped report or results
> file (no numbers from memory), and the headline finding is deliberately conservative: most alternator
> failures in this fleet are **abrupt / silent** with no usable electrical precursor, so we ship a
> *risk ranking + fleet window + emergency monitor* rather than a per-truck "days-to-failure" gauge.

---

## 1. Repository & branch map

This repo keeps **each version on its own branch** so you can return to any version at any time, and
`main` carries **all three curated alternator versions** + `docs/` merged together. The **starter-motor**
work lives in its own repo: <https://github.com/himanshu-igloble/Daimler_Starter-motor>.

| Branch | What's on it | Use it to… |
|---|---|---|
| **`main`** | The three curated versions (`V10.6.2_ALT/` + `V11_ALT_heuristics/` + `V11.1_ALT/`) + `docs/` + this README, requirements, and comparisons | Browse everything in one place |
| **`v10.6.2-alt`** | **Only** the V10.6.2 deliverable (honest baseline) | Check out / roll back to V10.6.2 in isolation |
| **`v11-alt`** | **Only** the V11 deliverable (lead-time heuristics) | Check out / roll back to V11 in isolation |
| **`v11.1-alt-curated`** | **Only** the V11.1 deliverable (covariate RUL) | Check out / roll back to V11.1 in isolation |
| **`v11.1-alt`** | Full dev branch — V11.1 plus the complete `docs/` history. (The starter-motor work it once carried now lives in [its own repo](https://github.com/himanshu-igloble/Daimler_Starter-motor) — removed from this branch's tip, retained only in older history.) | Inspect the full development history |

```bash
git clone https://github.com/himanshu-igloble/Daimler_alternator
cd Daimler_alternator
git switch v10.6.2-alt        # see ONLY V10.6.2
git switch v11-alt            # see ONLY V11
git switch v11.1-alt-curated  # see ONLY V11.1 (curated)
git switch main              # see all three alternator versions (+ docs)
```

---

## 2. Version comparison

Full metric-by-metric comparison of all three curated versions. The classifier and fleet curve are
**frozen and identical** across all three; the differences are entirely in the emergency channel and in
the (negative) per-truck-RUL findings.

| Dimension | **V10.6.2** — Honest Baseline | **V11** — Lead-Time Heuristics | **V11.1** — Covariate RUL |
|---|---|---|---|
| Branch | [`v10.6.2-alt`](https://github.com/himanshu-igloble/Daimler_alternator/tree/v10.6.2-alt) | [`v11-alt`](https://github.com/himanshu-igloble/Daimler_alternator/tree/v11-alt) | [`v11.1-alt-curated`](https://github.com/himanshu-igloble/Daimler_alternator/tree/v11.1-alt-curated) |
| Question asked | Does *per-truck RUL* beat the fleet clock? | Can new heuristics improve *lead-time recall*? | Can *AFT covariates* individualize RUL? |
| Classifier — WHICH (AUROC) | **0.927** (V10.5.3 frozen, LOVO) | 0.927 (frozen, unchanged) | 0.927 (frozen, unchanged) |
| Fleet window — WHEN, fleet | Weibull median **601 d** (≈120 440 km / 4 538 eng-h) | same curve | same curve (M0 ≡ V10.6.2) |
| Per-truck RUL — point (MAE) | **142 d** | not addressed (precursor-only) | M0 **140.4** / M1 148.8 / M2 162.2 d |
| Fleet-clock dummy (MAE) | 50 d | — | 49.7 d |
| Per-truck RUL verdict | worse than dummy → `no_improvement` | — | covariates worse → M0 wins, β shelved |
| RUL interval | 80% band, 9/10 coverage | — | 80% band (currently age-driven) |
| Precursor recall — forensic | 5/10 | **6/10** | inherits V11 (6/10) |
| Deployable emergency channel | GED=2 storm (**2/10**) | GED=2 **+ post-crank recovery** | 3 channels incl. current-state watch (**3/10**) |
| NF false alarms | **0/15** | **0/15** | **0/15** |
| New work | — | **12 heuristics**; MVP `crank_recovery_t` | AFT covariates `x1`/`x2` |
| Headline change | baseline established | +VIN9 detected, VIN1 earlier (30→60 d) | per-truck RUL closed at n=25 |
| Deliverable shape | WHICH + WHEN-fleet + WHEN-emergency | same (emergency fires earlier) | same (3 alert channels) |
| **Verdict** | **NO_IMPROVEMENT** | **modest gain** | **NO_IMPROVEMENT_HONEST** |

**Head-to-head docs:** [`VERSION_COMPARISON_V10.6.2_vs_V11.md`](./VERSION_COMPARISON_V10.6.2_vs_V11.md) (2-way) and
[`VERSION_COMPARISON_V10.6.2_vs_V11_vs_V11.1.md`](./VERSION_COMPARISON_V10.6.2_vs_V11_vs_V11.1.md) (full 3-way).

---

## 3. How to navigate the project

All three versions follow the **same directory grammar**, so once you can read one you can read the others.

```
V10.6.2_ALT/  (and  V11_ALT_heuristics/,  V11.1_ALT/)
├── src/              # the pipeline — config + one module per stage + an orchestrator
├── tests/            # unit tests (V11 and V11.1)
├── cache/            # committed intermediates → reports reproduce WITHOUT the 14.5 GB raw data
│   ├── forensics/    #   per-VIN daily aggregates (VIN*_daily.csv) — the real pipeline input here
│   ├── weibull/      #   fleet survival fit (params, posterior, curve)
│   ├── rul/          #   per-VIN RUL bands + decisions
│   └── ged_emergency/#   daily GED=2 alert table
├── results/          # machine-readable outputs (predictions.csv, verification.json, narratives)
├── service/          # operations-facing tables (alerts, replacement schedule, per-VIN service)
├── reports/          # human-facing deliverables: *_customer_report.md + *_fleet_report.xlsx
├── visualizations/   # PNG/PDF/SVG figures (fleet overlays, per-VIN RUL/precursor curves)
└── presentation/     # PPTX decks + AUDIT_REPORT.md + DATA_SOURCES.md
```

**Where to start reading:**
1. `*/reports/*_customer_report.md` — the plain-language result for each version.
2. `VERSION_COMPARISON_V10.6.2_vs_V11.md` — what changed between the two.
3. `*/presentation/AUDIT_REPORT.md` — how the numbers were produced and gated.
4. `*/src/*_config.py` then `*_orchestrator.py` — the code, in execution order.

**Pipeline order**
- **V10.6.2:** `weibull_fleet → predictive_rul → backtest → ged_emergency → decisions → assemble_rul → rul_graphs → narrative_rul → markdown_report → excel_report → verify` (verify is the honest gate, runs last).
- **V11:** `forensic → changepoint → compound → verify → compare` (classifier & Weibull are reused by reference, not re-fit).
- **V11.1:** `weibull_fleet → covariates → survival (AFT M0/M1/M2) → predictive_rul → backtest → emergency → decisions → assemble_rul → rul_graphs → narrative_rul → markdown_report → excel_report → summary_docx → verify`.

---

## 4. Requirements & how to run

See [`REQUIREMENTS.md`](./REQUIREMENTS.md) for the full functional + data + technical requirements.
Quick start:

```bash
# Python 3.11+ recommended
python -m venv .venv && . .venv/Scripts/activate        # Windows; use bin/activate on *nix
pip install numpy pandas scipy matplotlib lifelines openpyxl python-pptx

# Reproduce a version end-to-end (runs from the committed cache/ — no raw data needed):
python V10.6.2_ALT/src/V10.6.2_ALT_orchestrator.py
python V11_ALT_heuristics/src/V11_ALT_heuristics_orchestrator.py
python V11.1_ALT/src/V11_1_ALT_orchestrator.py

# unit tests
pytest V11_ALT_heuristics/tests
pytest V11.1_ALT/tests
```

> On Windows the project was developed with `py -3 <script>` (the repo `.venv` historically lacked
> pandas). Either invocation works as long as the dependencies above are installed.

---

## 5. Data & domain notes (read before interpreting results)

- **Raw data is NOT in this repo.** The source is ~204 M rows / ~14.5 GB across 4 CSVs (failed/non-failed
  × alternator/starter-motor). It is `.gitignore`d. The committed `cache/forensics/*_daily.csv` are the
  per-VIN daily aggregates the downstream stages actually consume, so reports/figures reproduce offline.
- **ALT fleet = 25 trucks** (10 failed + 15 non-failed). All VIN labels carry an `_ALT` suffix.
- **VIN independence:** alternator and starter-motor VINs are **different physical trucks** that happen
  to reuse the same numbering. `VIN1_ALT ≠ VIN1_SM`. No cross-dataset VIN-level analysis is valid.
- **Key CAN signals:** `CSP` vehicle speed (0–100 km/h), `RPM` engine speed, `ANR` engine torque
  (−400…1300 Nm), `GED` alternator-excitation state (0 normal, 2 disturbance, 3 unavailable),
  `VSI` supply voltage (0–36 V), `SMA` starter-motor active {0,1}. Sentinels: 65535 (CSP/RPM/ANR),
  −5000 (ANR), 0 & 255 (VSI).

---

## 6. Bottom line

The **classifier (AUROC 0.927)** and the **fleet replacement window (median 601 d)** are the trustworthy,
deployable deliverables and are identical across **all three** versions. **V11** adds one genuinely useful
early signal — post-crank voltage recovery — that catches **one extra failing truck** and gives an
**earlier warning on another**, with **zero false alarms**. **V11.1** then closes the per-truck-RUL
question for good at this sample size: AFT covariates are exposure-confounded, so the covariate-free M0
(≡ the V10.6.2 fleet curve) wins and β is shelved. None of the three change the structural limit: at n=25
there is no reliable per-truck "days-to-failure" number. Ship the **rank + window + emergency monitor**.

---

## 7. Loadable model artifact (deployable)

The frozen champion classifier is packaged as a **load-and-predict** joblib bundle — no re-fit needed.

**Path (this branch):** [`V11.2_ALT/models/V10.5.3_ridge_frozen_champion/`](./V11.2_ALT/models/V10.5.3_ridge_frozen_champion)

| File | What it is |
|---|---|
| [`V10.5.3_20_5_ALT_champion_bundle.joblib`](./V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_champion_bundle.joblib) | fitted sklearn `Pipeline` (median-impute → `StandardScaler` → `RidgeClassifier(α=1.0)`, fit on all 25 trucks) + frozen threshold 0.4456 + tier bands + metadata. Plain dict of standard sklearn objects — loads anywhere `scikit-learn 1.8.x` is installed. |
| [`V10.5.3_20_5_ALT_predict.py`](./V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_predict.py) | loader + CLI — `py -3 V10.5.3_20_5_ALT_predict.py <features_csv>` |
| `V10.5.3_20_5_ALT_training_matrix.csv` | provenance: the 25-truck feature matrix |
| `V10.5.3_20_5_ALT_ridge_spec.json` | provenance: the frozen model spec |
| `V10.5.3_20_5_ALT_lovo_predictions.csv` | provenance: archived LOVO predictions |
| `V11.2_ALT_metric_suite.json` | provenance: the V11.2 validation metric suite |
| `V10.5.3_20_5_ALT_verification.json` | packaging parity gates P1–P4 (real numbers) |
| `V10.5.3_20_5_ALT_MANIFEST.json` | SHA256 of every file + inputs + build env |
| `README.md` | artifact usage + honesty notes |

**Model:** `RidgeClassifier(alpha=1.0)`, 6 features, **LOVO AUROC 0.9267** (recall 9/10, specificity 14/15),
Youden threshold **0.4456**; alert tiers on raw prob `red ≥ 0.55 / amber ≥ 0.35 / green`.
Build + verify scripts: [`V11.2_ALT/src/V11.2_ALT_package_champion.py`](./V11.2_ALT/src/V11.2_ALT_package_champion.py),
`V11.2_ALT_iteration_comparison.py`, `V11.2_ALT_bundle_smoketest.py`.

```bash
# score trucks from a features CSV (the bundle imputes NaNs with training medians)
py -3 V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_predict.py \
      V11.2_ALT/models/V10.5.3_ridge_frozen_champion/V10.5.3_20_5_ALT_training_matrix.csv
```

> 0.9267 is a **LOVO ranking AUROC** at n=25, not field accuracy; the shipped pipeline is fit on all 25
> trucks (its resubstitution scores are optimistic by design). Threshold and tier bands are frozen from
> the 2026-06-01 run.
