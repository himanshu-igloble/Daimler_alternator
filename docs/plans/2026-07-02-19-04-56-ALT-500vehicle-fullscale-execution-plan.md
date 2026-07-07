---
title: "Alternator Vertical — 500+ Vehicle Full-Scale Execution Plan (HLD + LLD, Phase-Wise)"
status: "complete"
created: "2026-07-02"
updated: "2026-07-02"
---

# ALTERNATOR — 500+ Vehicle Full-Scale Execution Plan

**Program:** DICV / BharatBenz 5528T Predictive Maintenance — Alternator vertical
**Baseline frozen model:** V10.5.3/V11.1 Ridge (6 features), LOVO AUROC 0.9267, validated in V11.2 dossier, GED deep-dive V12
**Pilot fleet:** 25 trucks (10 failed + 15 non-failed) → **Target fleet: 500+ trucks (~250 failed + ~250 healthy retrospective cohort, then prospective)**
**Companion document:** `2026-07-02-19-04-56-SM-500vehicle-fullscale-execution-plan.md` (Starter Motor vertical — independent fleet, independent VIN namespace, shared platform)

> **VIN Independence rule carries forward:** ALT and SM fleets are disjoint physical trucks. All IDs suffixed `_ALT`. No cross-vertical VIN-level analysis is valid.

---

## 0. Executive Summary

The 25-truck pilot is **feature-complete and honest**: a 6-feature Ridge classifier ranks failed above healthy trucks with LOVO AUROC 0.9267 (139/150 correct pairs), a GED=2 storm channel gives a zero-false-alarm emergency wire (2/10 failures caught, one with a 21-day lead), and per-truck point-RUL was proven **worse than the fleet clock** at n=10 failures (MAE ~125–155 d vs ~50 d) — so the pilot ships risk bands + fleet maintenance windows + emergency channel instead. Four iterations of feature hunting confirmed a **data ceiling, not a method ceiling**.

At **500 vehicles (~250 failure events)** the constraints that shaped every pilot decision are removed one by one:

| Pilot constraint (n=25, 10 failures) | At 500 vehicles (~250 failures) | Consequence for this plan |
|---|---|---|
| Only linear models survive LOVO; 6 features beat 17 | Trees/GBMs become estimable (SCANIA 23k-truck evidence: LightGBM beat Bi-LSTM) | Champion–challenger program, Ridge stays champion until beaten |
| Anomaly detection: 80–100 % FP, unusable | Fleet-relative baselines from ~250 healthy trucks | Re-open anomaly as a budgeted watchlist channel (Phase 7) |
| Per-truck RUL loses to fleet clock (p=0.0) | Survival/hazard models estimable at ~250 events | Calibrated P10–P90 *windows* (never points), must beat fleet-clock gate (Phase 8) |
| Youden threshold from 25 trucks | Thresholds re-derived on out-of-fold n=500 + field-prevalence correction | Alert economics re-costed; FP budget ≤ 2 alerts/100 trucks/month |
| LOVO (25 folds) validation | Grouped stratified 10×5 CV + temporal holdout + prequential replay | Same leakage discipline, scalable protocol |
| GED features rejected (best −0.0067 at n=25) | Markov P(2→2) 0.992 F vs 0.860 NF re-tested with power | Graduation re-test list (Phase 6) |
| 5s/6-signal CAN feed; excitation trapped on LIN | 500-truck instrumentation window is open | Sensor-gap workstream: LIN RMC/RDC, SPN 115, temp, W-terminal (Phase 11) |

**The honest physics constraint that does NOT go away with n:** in the pilot, only 2/10 alternator failures were gradual-electrical (warnable from voltage/excitation); 5 were electrically flat to failure; 3 inconclusive. More trucks sharpen statistics but do not create precursors where the physics emits none on a 6-signal, 5-second feed. Recall targets in this plan are therefore stated **per failure-mode class**, and the biggest single unlock is instrumentation (Section 11), not modeling.

**Deliverable stack at production:** daily batch scoring of 500+ trucks on Azure (single CPU VM class, no Spark), parquet lakehouse (measured 7.6–10:1 compression vs CSV), champion Ridge + challenger GBM/survival/anomaly program behind promotion gates, three alert channels feeding TruckConnect, label-triggered retraining on Spot A10 GPU (~$41/mo), full MLOps loop. Time to production: **~16 weeks with 2 engineers** (Section 13).

---

## 1. Starting Point — The Frozen 25-Truck Baseline (what we port, verbatim)

Everything in this section is the frozen, evidence-backed pilot state. It is the contract for Phase 4 ("champion port") — the scale-up must reproduce these numbers on pilot data byte-for-byte before anything new is trained.

### 1.1 Champion model — exact frozen spec

- **Model:** `sklearn.linear_model.RidgeClassifier(alpha=1.0, random_state=42)`, labels failed=+1 / non-failed=−1, probability `p = 1/(1+exp(−decision_function))`.
- **Pipeline per fold:** train-median NaN impute → `StandardScaler` (train-fit) → Ridge. **No winsorization/clipping in the frozen spec** (V5.2.1 had 1–99 % clip; V10.5.3 dropped it).
- **Validation:** 25-fold LOVO (leave-one-VIN-out), pooled out-of-fold probabilities.
- **Headline:** AUROC **0.9267** = 139/150 concordant (failed, healthy) pairs — a *ranking* statistic, not accuracy. PR-AUC 0.9400; at Youden threshold **0.4456**: recall 9/10, specificity 14/15, MCC 0.8333, Brier 0.1433; bootstrap AUROC 0.9234 (95 % CI [0.8065, 1.0000]); permutation p < 0.01.
- **Known errors:** FN = VIN5_F_ALT (score 0.2799, sits in healthy cluster); FP = VIN3_NF_ALT (0.4906).
- **Risk tiers:** GREEN < 0.35, AMBER 0.35–0.55, RED ≥ 0.55. RED band is pure on pilot data (7 F, 0 NF).
- **No calibration layer** in frozen ALT spec (raw sigmoid). (Contrast: SM vertical ships per-fold Platt — at 500 vehicles ALT will adopt calibration too; Section 7.3.)

### 1.2 The six frozen features (daily-cache based; VSI-dominated by design)

All computed from a **per-(VIN, calendar-day) aggregation cache**, RPM-state masks: engine-off RPM=0, idle 0<RPM<700, cruise 700–1800, heavy >1800.

| # | Feature | Formula (verbatim) | Coef (std.) | Perm. imp. |
|---|---|---|---|---|
| 1 | `vsi_std_ratio_30d` | `mean(VSI_daily_std[last 30 d]) / mean(VSI_daily_std[first 60 d])` | +0.44257 | 0.1547 (#1) |
| 2 | `vsi_dominant_freq` | `argmax_freq(rFFT(VSI_daily_mean − mean))`, DC zeroed, ≥14 d required | +0.42647 | 0.1053 (#2) |
| 3 | `vsi_spectral_entropy` | normalized Shannon entropy of rFFT PSD of daily VSI mean, ≥14 d | +0.24470 | 0.0573 (#4) |
| 4 | `bat_charge_delta_trend_right` | OLS slope over last 90 d of `(VSI_cruise_mean − VSI_engineoff_mean)` daily series, ≥10 valid d | +0.22399 | 0.0340 (#6) |
| 5 | `vsi_range_trend_last30d` | OLS slope of daily `(VSI_max − VSI_min)` vs day index, last 30 d, ≥5 d | +0.22990 | 0.0613 (#3) |
| 6 | `progressive_drift` | `cumsum(VSI_daily_mean − mean(first 60 d)) [-1] / n_days` | **−0.20551** | 0.0480 (#5) |

⚠ Known artifact: `progressive_drift`'s negative coefficient is an **exposure artifact** (longer-observed healthy trucks accumulate more drift). At n=500 this feature is a mandatory re-audit item — expect replacement by an exposure-normalized variant or removal (Section 6.3).

### 1.3 Preprocessing chain (frozen order)

1. Cast float64, parse ISO8601 timestamps, sort, drop duplicate timestamps (keep-first).
2. **VSI ×0.2 scaling** if raw > 36 (applied BEFORE sentinel filter).
3. Sentinel nulling: CSP/RPM/ANR 65535→null; ANR −5000→null; VSI 0,255→null.
4. Range validation to null: CSP (0,100), RPM (0,3500), ANR (−400,1300), VSI (0,36), GED ∈{0,1,2,3}, SMA ∈{0,1}.
5. Calendar-day aggregation with state masks → daily cache.
6. Per-fold: train-median impute → StandardScaler.

Note (V3.1-era correction, verified on parquets): the delivered parquet data is **already in volts** and the 65535/0/255 sentinels do not actually occur (missingness is NaN) — steps 2–3 are retained as a **harmless no-op contract** so the pipeline is robust to a rawer future feed. Keep them at 500-vehicle scale; the data contract (§2.3) does not guarantee pre-scaled inputs.

### 1.4 Alert stack (three boxes)

| Box | Channel | Rule | Pilot evidence |
|---|---|---|---|
| WHICH | Ridge risk ranking | tiers GREEN/AMBER/RED, Youden 0.4456 | 9/10 F caught, 1/15 NF FP |
| WHEN-fleet | Fleet wear-out window | median **601 d** (P25–P75 577.5–652.5, range 472–673) ≈ **~120,440 km / ~4,538 engine-h** | all 10 failures inside window |
| WHEN-emergency | **GED=2 storm**: `ged2_count ≥ 200/day` (hourly monitor) | binary emergency | 2/10 F (VIN1 21-d lead; VIN10 1-d), **0/15 NF** |
| (supplement) | Compound early-watch: 5 equal-weight heuristic votes (`vsi_ceiling`, `vsi_resid_mean`, `crank_recovery_t`, `resting_vsi_mean`, `ged_churn`), fire at ≥2 | 3/10 F, 0/15 NF | equal weights deliberate at n=10 |
| (supplement) | M5 health zones GREEN<0.15 / YELLOW<0.35 / ORANGE<0.55 / RED≥0.55 with debounce (+0.02 deadband, 14-d dwell) | visual context only | only 3/10 F ever reached ORANGE+; 6/15 NF also did — **not a standalone alert** |

### 1.5 Honest ledger — refuted / bounded claims the plan must not resurrect

- **Per-truck point RUL:** survival RUL MAE 125 (total-TTF) / 130–155 (rewound) vs fleet-clock dummy 49.7 d; signed-rank p=0.0 → `NO_IMPROVEMENT`. Shipped fleet window instead. Any 500-vehicle RUL claim must beat this dummy first (Phase 8 gate).
- **No 3–4 week universal precursor exists** on this feed. Failure-mode split: 2/10 gradual-electrical (VIN1 ~21 d actionable, VIN10 ~1 d), 5/10 flat/abrupt, 3/10 inconclusive (VIN3 telemetry gap, VIN8 under-observed, VIN2 recovered transient).
- **GED is a J1939 2-bit status enum, not a duty-cycle** (Spearman ρ=0.03 vs VSI; state-3 median VSI above state-2). State 1 never observed. State 3 = comms gap, not component fault.
- **GED-absent ≠ GED-silent:** VIN3_F/VIN4_F had 99.88–99.89 % GED nulls — the signal wasn't transmitted; missingness does not rise toward failure (+2.4 pp only). Continuous excitation lives on the internal LIN bus (RMC/RDC registers) — **sensor-blocked, not method-limited**.
- **All 7 novel V12 GED features rejected** against the frozen 0.9267 at n=25 (best −0.0067) — but the underlying discriminative *facts* are real: Markov **P(2→2)=0.992 (F) vs 0.860 (NF)**, entry P(0→2) ~8×, dwell **31.4 vs 4.1 samples**. These graduate to the re-test list at n=500.
- **Anomaly detection (V7 program): all unsupervised methods 80–100 % FP at n=25.** Re-opened only under the Phase-7 FP budget.
- **JCOPENDATE is the failure date**, not last telemetry: 7/10 VINs gap=0, VIN3_F gap **+66 d** (largest), VIN1_F +11 d, VIN9_F +2 d. RUL curves extrapolate (dashed) to RUL=0 at JCOPENDATE.

---

## 2. Scope, Planning Assumptions, Data Contract

### 2.1 Fleet & cohort assumptions (from the DICV 500-vehicle program brief)

| Parameter | Value | Status |
|---|---|---|
| Vehicles (ALT vertical) | ~500 (≈250 failed + ≈250 healthy, ~50/50 curated retrospective cohort) | **Assumption A1** — DICV brief 2026-06-24; re-confirm at data drop |
| Failure events | ~250–500 (1–2 failure dates per failed truck) | Assumption A2 |
| Signal set | **Same 6 signals**: CSP, RPM, ANR, GED, VSI, SMA + VIN, timestamp (+SALEDATE, JCOPENDATE, Failure_type on failed) | Confirmed by user 2026-07-02 |
| Cadence | ~5 s CAN burst (expect the pilot's bimodal 5 s / 900 s heartbeat mix to persist) | Assumption A3 |
| Delivery | Daily batch files (CSV) from TruckConnect → Azure Blob | Per DICV Azure plan v2 |
| History depth | ≥ 18 months per truck retrospective at program start | Assumption A4 |
| Platform | Azure, Central India region | Locked (prior costing) |
| Currency | ₹94/USD (per commercial Q&A 2026-06-26) | Locked |

**Prevalence warning (design-critical):** the curated cohort is ~50/50; the field is ~1–5 %/yr. Every PPV/alert-volume number must be reported at **field prevalence** (default 10 % program-window prevalence per FM-pilot spec; FA budget ≤ 2/100 trucks/month), not cohort prevalence. All thresholds get a prior-shift logit correction at deployment (Section 7.7).

### 2.2 Exact signal schema (unchanged from pilot — the contract)

| Column | Meaning | dtype (parquet) | Valid range | Sentinels (contract) | States |
|---|---|---|---|---|---|
| VIN | pseudonymized ID | dictionary/large_string | — | — | namespace `_ALT` |
| CSP | vehicle speed km/h | float32 | 0–100 | 65535 | — |
| RPM | engine speed | float32 | 0–3500 | 65535 | — |
| ANR | engine torque Nm | float32 | −400–1300 | 65535, −5000 | negative = engine braking |
| GED | alternator excitation state | float32→int8 (silver) | {0,1,2,3} | — | 0 normal, 1 inhibit (never seen), 2 **disturbance**, 3 signal-unavailable |
| VSI | supply voltage V | float32 | 0–36 | 0, 255; ×0.2 if raw>36 | 24 V system, healthy ~27.9–28.2 V |
| SMA | starter active | float32→int8 | {0,1} | — | — |
| timestamp | reading time | timestamp[us], tz-naive | 5 s nominal | — | ISO8601 `…Z` in CSV; **policy: treat as IST vehicle-local, store tz-naive** |
| SALEDATE | in-service date | date32 | — | — | failed-only today → **contract ask: all VINs** |
| JCOPENDATE | job-card open (failure) date | date32 | — | — | failed-only |
| Failure_type | component | large_string | `"Alternator"` | — | failed-only |

Failed files: 11 columns; healthy files: 8. **Column ORDER differs between cohorts (GED/VSI/SMA vs VSI/SMA/GED) — all readers are name-based, never positional.** Leading all-null sensor blocks occur at file heads (observed in raw CSVs) — handled by the cleaning suite, not by skipping rows blindly.

### 2.3 Data-contract asks to DICV (negotiate before data drop; each has a fallback)

| # | Ask | Why | Fallback if refused |
|---|---|---|---|
| D1 | **SALEDATE for ALL vehicles** (incl. healthy) | survival left-truncation; pilot proxied NF age from first telemetry | keep first-telemetry proxy, document bias |
| D2 | JCOPENDATE + job-card failure code/text + **part-replacement confirmation** | label quality; distinguish alternator swap vs rewire vs belt | manual label review queue (Phase 2) |
| D3 | **ODO (odometer)** and IGN (ignition) | km-based RUL axes; true engine-off detection | integrate CSP for km (pilot method ~120,440 km estimate), RPM=0 for off |
| D4 | Telematics **firmware/config version per VIN** | pilot found continuous-transmit vs 900 s-heartbeat families; GED-absent VINs (99.9 % null) are config artifacts | infer families from dt/null fingerprints (pilot method) |
| D5 | **One export pipeline for failed AND healthy cohorts** | pilot NF files carried NaT timestamps & different column order — cohort-correlated artifacts are leakage risks | asymmetry detectors in DQ suite (§5.4 R15) |
| D6 | Stable VIN pseudonymization, per-vertical namespace | joins across drops; VIN-independence | maintain own mapping table |
| D7 | Sensor adds: **LIN RMC/RDC, SPN 115, alternator temp, W-terminal** (Section 11) | the real unlock for gradual-electrical detection | plan works without; recall ceiling stays |
| D8 | Cadence & heartbeat policy documented per config | dt-aware features | infer empirically |

---

## 3. Volume Math & Sizing (measured pilot densities → 500-vehicle projections)

Measured pilot (ALT): 96,759,243 raw rows / 25 trucks / ~20–26 months; 13.6 GB across 4 CSVs for both verticals; parquet compression measured **9.93:1 (ALT-F), 6.40:1 (ALT-NF)** (snappy-era files; avg both verticals 7.61:1). Row density ≈ **5–6 k rows/truck/day** (bimodal 5 s burst + 900 s heartbeat). CSV ≈ 71 bytes/row.

| Quantity | Basis | 500-truck ALT projection |
|---|---|---|
| Rows/day | 500 × 5.5 k | **~2.75 M rows/day** |
| Rows/year | ×365 | **~1.0 B rows/yr** |
| Upper bound (24 h continuous 5 s) | 500 × 17,280 | 8.6 M/day, 3.2 B/yr — sizing headroom 3× |
| Raw CSV/yr | 71 B/row | **~66–71 GB/yr** (upper bound ~225 GB/yr) |
| Parquet (zstd, sorted)/yr | ~8–12:1 | **~6–9 GB/yr** (upper ~25 GB/yr) |
| Retrospective backfill (one-time) | 500 trucks × ~600 d | ~1.65 B rows ≈ 115 GB CSV ≈ **~12–15 GB parquet** |
| Daily cache | 500 rows/day × ~40 stats | trivial (<100 MB/yr) |
| Feature matrix | 500 × ~50 features | KBs |

**Sizing verdict:** this is a **single-node Polars problem end-to-end** (the pilot already processed 204 M rows on a workstation). One `D16ds_v5` (16 vCPU/64 GB) handles daily incrementals in minutes and full-history rebuilds in ~1–2 h with Polars streaming. Spark/Databricks is explicitly **rejected** at this scale (ADR-2, §4.1) and revisited only ≥10 k vehicles.

---

## 4. High-Level Design (HLD)

### 4.1 Architecture Decision Records (the choices, alternatives, and why)

| ADR | Decision | Alternatives considered | Rationale |
|---|---|---|---|
| **ADR-1** | **Daily batch** pipeline; hourly micro-batch only for the GED emergency scan | Real-time streaming (Event Hub/Kafka + Flink) | Failure horizons are weeks–months; TruckConnect delivers daily files; pilot's only sub-daily need is the GED≥200/day storm counter — an hourly incremental scan of one column covers it. Streaming raises cost/complexity ~10× with zero lead-time benefit at these physics. |
| **ADR-2** | **Single-node Polars** on Azure VM/Container Apps Jobs | Spark/Databricks; DuckDB; pandas | 1 B rows/yr, ~9 GB parquet/yr is far below Spark's break-even; Polars streaming is the team's proven stack (204 M-row pilot); DuckDB kept as ad-hoc SQL sidecar. Escape hatch documented at ≥10 k vehicles or ≥50 GB/yr parquet. |
| **ADR-3** | **Champion–challenger**: port frozen Ridge as champion; nothing replaces it without beating promotion gates | Greenfield retrain; jump to GBM/deep | Preserves 2 years of validated honesty; SCANIA evidence says GBM will be competitive at n≈250 events, but it must *prove* it on this fleet under leakage-safe gates. |
| **ADR-4** | **Grouped stratified 10-fold ×5 CV + temporal holdout + prequential replay** replaces LOVO | 500-fold LOVO; single random split | LOVO at 500 is 20× compute for no statistical gain; grouping by VIN preserves the pilot's leakage discipline; temporal holdout adds the prospective dimension the pilot lacked. |
| **ADR-5** | **Parquet lakehouse (bronze/silver/gold), zstd** | Keep CSV; Delta Lake; database | Measured 7.6–10:1 compression; columnar predicate pushdown per VIN/date; Delta's ACID is unneeded for append-only daily batches (a `_manifest` table covers idempotency); zstd over snappy for ~20–35 % smaller files at negligible scan cost. |
| **ADR-6** | Scoring cadence: **daily** risk scores; weekly roll-up report; hourly GED scan | Weekly only (pilot cadence) | Daily is nearly free at this scale and halves alert latency; weekly aggregation remains the *feature* granularity where the frozen features demand it (they are daily-cache based already). |
| **ADR-7** | **MLflow registry on Azure ML**; frozen-spec JSONs (pilot pattern `*_ridge_spec.json`) remain the deployment contract | Bare git tags | Auditable promotions, artifact lineage, DICV-facing model cards. |
| **ADR-8** | Anomaly & RUL ship as **budgeted advisory channels**, never autonomous work orders | Full autonomy | Pilot FP history; workshop trust is earned via shadow mode first. |

### 4.2 System diagram

```
                            ┌──────────────────────────── AZURE (Central India) ────────────────────────────┐
 TruckConnect               │                                                                                │
 (DICV telematics)          │  ┌─────────────┐   ┌──────────────────────────────────────────────┐           │
   500+ trucks ──daily──▶   │  │ ADLS Gen2   │   │  COMPUTE: Container Apps Job / Azure ML       │           │
   CSV batch drops          │  │ landing/    │──▶│  (single D16ds_v5-class, Polars)              │           │
   (per-day, per-cohort)    │  │ (immutable) │   │                                               │           │
                            │  └─────────────┘   │  01_ingest_validate  (pandera schema gate)    │           │
   hourly increment ──────▶ │        │           │  02_csv_to_parquet   (bronze, zstd)           │           │
   (optional, GED scan)     │        ▼           │  03_clean_conform    (silver: R1–R15 rules)   │           │
                            │  ┌─────────────┐   │  04_daily_cache      (gold: per-VIN-day agg)  │           │
                            │  │ QUARANTINE/ │◀──│  05_features         (feature store parquet)  │           │
                            │  │ (DQ rejects)│   │  06_score_champion   (Ridge → tiers)          │           │
                            │  └─────────────┘   │  07_channels         (GED storm, early-watch, │           │
                            │                    │                       window matrix)          │           │
                            │  ┌─────────────┐   │  08_alerts_publish   (dedupe, debounce)       │           │
                            │  │ MLflow /    │◀──│  09_monitors         (PSI, calib, alert SPC)  │           │
                            │  │ Azure ML    │   └──────────────┬───────────────────────────────┘           │
                            │  │ registry    │                  │                                            │
                            │  └─────────────┘                  ▼                                            │
                            │  ┌─────────────┐   ┌──────────────────────────┐    ┌──────────────────┐        │
                            │  │ Spot A10 GPU│   │ Serving store (scores,   │───▶│ TruckConnect API │──▶ workshops,
                            │  │ retrain jobs│   │ alerts, windows; parquet │    │ + webhook alerts │    dashboards,
                            │  │ (scale-to-0)│   │ + Postgres flexible srv) │    │ Power BI         │    job cards
                            │  └─────────────┘   └──────────────────────────┘    └──────────────────┘        │
                            └────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Component inventory

| Component | Azure service | SKU / posture | Cost posture (₹94/USD) |
|---|---|---|---|
| Landing + lake | ADLS Gen2 (hot→cool lifecycle 90 d) | LRS, versioned, immutable landing | ~$13–35/mo at this scale |
| Batch compute | Container Apps Jobs or Azure ML pipelines | D16ds_v5 (jobs), D8ds_v5 (scoring) — scale-to-zero | ~$50–150/mo |
| Training GPU | Azure ML compute, `Standard_NV36ads_A10_v5` **Spot** ($0.828/h) | min 0 / max 2, idle-scale-down 300 s | ~$17–41/mo; ~$2–3 per retrain |
| Heavy sweeps (rare) | `NC24ads_A100_v4` Spot ($0.950/h) | on demand only | burst only |
| Registry/orchestration | Azure ML + MLflow; Log Analytics | — | ~$80–250/mo |
| Serving store | Postgres Flexible Server (B-series) + gold parquet | scores/alerts only | ~$30/mo |
| Secrets/identity | Key Vault + Entra ID managed identities | no keys in code | — |
| Dashboards | Power BI + TruckConnect embed | signed licensed container for API push | — |

Reference envelope (both verticals + clutch share this platform): total batch production run-rate **~$300–560/mo at 10,000 vehicles**; at 500 vehicles the ALT share is a rounding error (≲ ₹10–15 k/mo). Training bundle measured: **~$77/mo all-in** (GPU $41 + disk $22 + blob $11 + workspace $3).

### 4.4 Environments, security, tenancy

- `dev` → `uat` → `prod` resource groups; identical IaC (Bicep/Terraform), data only in `uat/prod`.
- Managed identity between jobs–storage–registry; no SAS keys in code; Key Vault for TruckConnect API credentials.
- VINs arrive pseudonymized (D6); no PII in the lake; job-card free-text stripped to coded fields at ingestion.
- Immutable landing zone + 35-day soft delete = replay/DR story; every derived layer reproducible from landing by manifest.

---

## 5. Low-Level Design — Data Engineering

### 5.1 Landing & ingestion contract

```
abfss://dicv-alt@<account>.dfs.core.windows.net/
  landing/alternator/ingest_date=YYYY-MM-DD/           # immutable, as-delivered CSV (or gz)
  bronze/alternator/cohort={failed|nonfailed}/          # parquet, 1:1 with landing content
         ingest_date=YYYY-MM-DD/part-*.parquet
  silver/alternator/vin=VIN0001_ALT/month=YYYY-MM/      # cleaned, conformed, deduped, typed
         part-*.parquet
  gold/alternator/daily_cache/vin=.../                  # per-(VIN, day) aggregates
  gold/alternator/features/snapshot_date=YYYY-MM-DD/    # scoring-ready feature matrix
  quarantine/alternator/ingest_date=.../                # DQ rejects + reason codes
  manifests/alternator/ingest_log.parquet               # file → rows_in/out, hashes, idempotency
```

- **Idempotency:** ingest keyed by (file hash, ingest_date); reruns overwrite bronze partitions atomically (write-to-temp + rename).
- **Late/duplicate drops:** manifest dedupe; silver upsert is per (VIN, month) partition rebuild — cheap at these sizes.
- Landing accepts the pilot's reality: two cohort file shapes (11 vs 8 columns), differing column order, possible all-null leading blocks.

### 5.2 CSV → Parquet conversion spec (the "make the data light" step)

**Why parquet (measured on this exact data):** ALT-failed 3,272 MB CSV → 330 MB (9.93:1); ALT-NF 3,441 → 538 MB (6.40:1). Columnar reads mean a GED-only hourly scan touches ~2 % of bytes; row-group statistics + partition pruning make per-VIN feature builds ~50–100× faster than CSV re-parsing.

**Conversion rules (bronze):**

| Aspect | Spec |
|---|---|
| Engine | Polars `scan_csv(...).sink_parquet(...)` streaming (constant memory) |
| Read | **name-based column mapping** (never positional); `schema_overrides` per §2.2; `null_values=[""]`; ISO8601 parse to `timestamp[us]`, tz-naive after UTC-marker strip (IST policy) |
| dtypes | VIN → categorical/dictionary; sensors float32; GED/SMA kept float32 in bronze (cast int8 in silver after enum validation); dates date32 |
| Compression | **zstd level 3** (expect ~20–35 % smaller than the pilot's snappy files; verify ≥8:1 on first drop) |
| Row groups | target ~1 M rows (~10–15 MB) — pilot files averaged 123 k rows/group; larger groups improve scan speed at our file sizes |
| Sort | within-partition sort by (VIN, timestamp) — enables run-length VIN encoding + timestamp delta stats |
| Statistics | enabled (min/max per column per row group) for predicate pushdown |
| Partitioning | bronze by `ingest_date`; silver by `vin`/`month` (500 VINs × ~24 mo ≈ 12 k partitions — fine) |

**Reference implementation sketch (bronze step):**

```python
import polars as pl

SCHEMA = {"VIN": pl.Utf8, "CSP": pl.Float32, "RPM": pl.Float32, "ANR": pl.Float32,
          "GED": pl.Float32, "VSI": pl.Float32, "SMA": pl.Float32,
          "timestamp": pl.Utf8, "SALEDATE": pl.Utf8, "JCOPENDATE": pl.Utf8,
          "Failure_type": pl.Utf8}          # missing cols tolerated for 8-col cohort

lf = (pl.scan_csv(src, schema_overrides=SCHEMA, null_values=[""], ignore_errors=False)
        .with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ",
                                                          time_unit="us", strict=False))
        .with_columns([pl.col(c).cast(pl.Date, strict=False)
                       for c in ("SALEDATE", "JCOPENDATE") if c in cols]))
lf.sink_parquet(dst, compression="zstd", compression_level=3,
                row_group_size=1_000_000, statistics=True)
```

**Acceptance gate for the converter:** row count parity CSV↔parquet per file; per-column null-count parity; min/max timestamp parity; ≥8:1 size ratio; re-run idempotent (byte-identical row counts).

### 5.3 Cleaning & conformance suite (silver) — rules R1–R15, exact

Every rule emits per-(VIN, ingest_date) counters to the DQ ledger; nothing is silently dropped.

| Rule | Name | Exact behavior | Pilot evidence motivating it |
|---|---|---|---|
| R1 | Schema/name gate | pandera-polars schema: required columns present by NAME, dtypes coerced, unknown columns logged | cohort column-order differs |
| R2 | Timestamp parse + tz policy | ISO8601 `…Z` → tz-naive IST-local `timestamp[us]`; unparseable → quarantine | pilot convention (V3.1 P0-4) |
| R3 | **NaT drop** | null timestamps dropped BEFORE sort/dedupe; counter per VIN | SM cohort had up to 689 k NaT rows/VIN, NF-only — cohort-correlated artifact; ALT must guard identically |
| R4 | Sort + exact-duplicate drop | sort (VIN, timestamp); drop full-row duplicates | pilot removed 5.6 M dup rows (5.8 %) |
| R5 | Duplicate-timestamp resolve | same (VIN, timestamp), differing values → keep-first + counter (frozen pilot behavior) | frozen spec |
| R6 | VSI ×0.2 conditional scaling | if raw VSI > 36 → ×0.2; count scaling events | contract no-op on current feed; guards rawer feeds |
| R7 | Sentinel masking | 65535 (CSP/RPM/ANR), −5000 (ANR), 0/255 (VSI) → null | contract no-op today |
| R8 | Range validation | out-of-range → null (§2.2 ranges); GED∉{0,1,2,3}→null; SMA∉{0,1}→null | frozen spec |
| R9 | All-null-sensor rows | rows with all 6 sensors null: KEEP in silver (they carry cadence/heartbeat info), EXCLUDE from sensor aggregates via masks | CSV heads showed long null blocks |
| R10 | Cadence classifier | per row: `dt` to previous; label `burst` (≤15 s), `heartbeat` (600–1200 s), `gap` (>1200 s); per-VIN daily shares stored | 80–93 % @5 s, 7–20 % @900 s bimodal |
| R11 | Firmware-family tag | per VIN: `continuous_transmit` (dt_p99 ≤ 6 s) vs `rest_heartbeat` (dt_p99 ≈ 900 s); joined from D4 contract if provided | pilot found two hardware families |
| R12 | Silent-gap / end-truncation detector | flag VINs whose telemetry ends > 14 d before max(fleet date) or JCOPENDATE; drives label handling (§5.6) | VIN3_F 66-d gap |
| R13 | GED-coverage tag | per VIN monthly GED null-rate; `ged_absent` if >95 % null | VIN3/4_F 99.9 % null; 4 NF VINs similar |
| R14 | Physical-plausibility screens | VSI daily median outside 20–32 V → flag (24 V system); RPM>0 while CSP>60 & ANR null streaks → sensor-fault flag | analyst screens from pilot audits |
| R15 | **Cohort-asymmetry sentinel** | weekly job compares failed vs healthy cohorts on: null rates, NaT counts, dup rates, cadence mix, column order — alert if distributions diverge (KS test p<0.01) | pilot NF-only NaT + column-order differences = leakage trap |

**Leakage doctrine for cleaning:** any cleaning statistic that differs systematically by cohort is a *label leak* through the data plumbing. R15 exists to catch exporter-induced separability before a model "learns" it. This was a real pilot phenomenon and is the most under-appreciated risk at 500 vehicles.

### 5.4 DQ gates & monitors

- **Hard gates (fail the day's run):** schema mismatch; >5 % unparseable timestamps; row-count parity failure; duplicate-file hash.
- **Soft gates (quarantine + continue):** per-VIN anomalies (NaT spike, null spike >3σ vs trailing 28 d, cadence-mix shift).
- **Weekly DQ report:** per-VIN completeness heatmap (the pilot's daily-coverage audit, scaled), sentinel/scaling counters, R15 asymmetry verdict, GED-coverage roster.
- Tooling: **pandera (polars backend)** for schemas; custom Polars DQ jobs for the rest (Great Expectations rejected: heavier, pandas-centric).

### 5.5 Label pipeline & curation (Phase 2's core)

1. **Ingest job-card table** (per D2): VIN, JCOPENDATE, failure code, part-replacement flag, free-text→coded.
2. **Failure date = JCOPENDATE** (frozen rule). Telemetry-end vs JCOPENDATE gap computed per VIN (pilot: 7/10 zero-gap, max +66 d); gaps > 30 d → manual review queue; gaps never silently shift labels.
3. **Event windows:** for each failure event, define pre-failure horizon windows (e.g., last-30/60/90-d states) used by classification framing; multi-failure trucks (A2 says 1–2 events) contribute independent right-censored spells with post-repair reset (repair = new alternator = new life; requires D2 replacement confirmation).
4. **Censoring table (survival-ready):** per spell: `t_start` (SALEDATE or post-repair date), `t_end` (JCOPENDATE or last telemetry), `event∈{1,0}`, `left_truncated` flag (healthy trucks without SALEDATE), usage axes (cum engine-h via RPM>0 integration ×dt; cum km via CSP integration ×dt — pilot method).
5. **Label QA:** distribution of TTF (pilot: 472–673 d, median 601 d) vs new cohort; outliers (<180 d or >1200 d) reviewed; failure-code mix tabulated → failure-mode stratification (gradual vs abrupt) becomes a *labeled* attribute at scale (pilot had to infer it from signals).
6. **Label ledger versioned** (labels change as job cards arrive; every training run pins a label-ledger version).

### 5.6 Daily aggregation cache (gold) — LLD

One row per (VIN, calendar-day). Exact pilot aggregation, plus dt-awareness:

- Masks: `engine_off` RPM=0; `idle` 0<RPM<700; `cruise` 700≤RPM≤1800; `heavy` RPM>1800; all masks require sensor non-null.
- VSI stats per mask + overall: mean, std, min, max, median, p05, p95, skew; `vsi_cruise_mean`, `vsi_resting_mean` (=engine_off), counts.
- GED: counts per state {0,1,2,3}, `ged2_cnt`, `ged2_maxrun` (longest consecutive state-2 run, in samples), transition counts (00,02,20,22,03,30…) for Markov features, null share.
- SMA: crank-event count/day (rising edges), for `crank_recovery_t` heuristic (seconds to VSI≥27 V post-crank).
- Exposure: `n_rows`, `burst_share`, `heartbeat_share`, `active_hours` (Σdt where RPM>0, clipped dt ≤ 900 s), `est_km` (Σ CSP×dt/3600, burst rows only).
- **dt-weighting rule (new at scale, pilot-consistent):** aggregates computed on burst rows only OR dt-weighted — decided once in Phase 3 via A/B against pilot reproduction; heartbeat rows never dominate daily means ("a 100-row window spans ~8 min dense but ~25 h in heartbeat").

Cache is append-only per day; full rebuild ~1–2 h single-node; incremental daily update seconds–minutes.

### 5.7 Feature store

- `gold/features/snapshot_date=.../features.parquet`: one row per VIN per scoring day: the 6 frozen features + challenger candidates + exposure covariates + DQ flags (`ged_absent`, firmware family, silent-gap).
- Point-in-time correctness: features at snapshot D use data ≤ D only (walk-forward safe); enforced by the cache being date-indexed.
- Feature definitions live in ONE registry module (pilot pattern `FEATURE_REGISTRY` with formula strings) — the registry string is what appears in model cards and this plan.

---

## 6. Low-Level Design — Feature Engineering

### 6.1 Frozen six — ported verbatim (§1.2 formulas; registry entries unchanged)

Reproduction gate (Phase 4): on the 25 pilot trucks, the scale-up pipeline must reproduce each feature value to ≤1e-9 relative error and the LOVO AUROC to 0.9267 exactly (the V12 gate pattern — it reproduced the frozen number exactly before judging candidates).

### 6.2 Leak-guard protocol (carried from pilot, hardened)

- **Banned feature families:** anything monotone in observation length or start date (`n_weeks`, `t_start` proxies — SM measured leak ceilings 0.952/0.893; ALT audit passed 8/8 but the ban is program-wide). Fixed-window (L-window) controls mandatory: any candidate whose AUROC drops >0.05 under fixed-window evaluation OR with |Spearman| >0.5 vs an observation-length proxy → REJECT.
- `progressive_drift` **re-audit** (known exposure artifact, negative coef): test exposure-normalized variant `progressive_drift_rate` (drift per active-hour); if the artifact persists at n=500, drop from champion successor.
- Cohort-plumbing leaks: model trained with R15 asymmetry features must show no lift (negative control).
- **Negative-control battery:** permuted-label AUROC ≈ 0.5; year-shuffled temporal control; exporter-fingerprint-only model ≈ 0.5.

### 6.3 Expansion candidates & graduation re-tests (Phase 6 backlog, pre-registered)

Rejections at n=25 were mostly **power-limited, not physics-refuted**. Pre-registered re-test list with pilot stats:

| Candidate | Pilot evidence | Pilot verdict | Re-test hypothesis at n=500 |
|---|---|---|---|
| `ged_markov_p22` (P(2→2) stickiness) | 0.992 F vs 0.860 NF | REJECT vs frozen (−0.0067 best of 7) | strong prior; needs GED-coverage mask |
| `ged2_dwell_mean` | 31.4 vs 4.1 samples | REJECT (same family) | ditto |
| `ged_entry_rate_p02` | ~8× higher in F | REJECT | ditto |
| `ged_churn` (heuristic #5) | early-watch voter | heuristic only | promote to feature |
| `vsi_ceiling`, `vsi_resid_mean`, `resting_vsi_mean`, `crank_recovery_t` | early-watch voters, 3/10 F 0/15 NF | heuristics only | promote to features; `vsi_resid_mean` gets a proper healthy-fleet expected surface (n=250 NF) |
| Fleet-relative percentile features | impossible at n=25 (no stable fleet reference) | n/a | per-day VSI percentile vs healthy cohort of same age/usage band |
| Seasonality (monsoon), usage-intensity interactions | SM leads p≈0.02 | vertical-specific | test ALT analogues |
| Failure-mode-conditional models | 2 gradual vs 5 abrupt — unlearnable at n=10 | n/a | with D2 failure codes: two-headed model (gradual-electrical detector + all-mode ranker) |

Selection discipline: pilot's nested screen (MW p<0.10 → AUROC≥0.60 → dedup |ρ|<0.85 → stability across folds) with pool cap raised (10 → 25) and **BH-FDR across the candidate battery** (V3.1 pattern). "Fewer features better" was an n=25 lesson — expect the optimum to move from 6 to 10–20 features at n=500, but let nested CV say so, never assume.

---

## 7. Modeling Program

### 7.1 Validation harness at n=500 (the referee — build once, Phase 4, before any challenger)

- **Primary:** stratified **grouped 10-fold CV, 5 repeats** (group = VIN; stratify by label × failure-mode class × firmware family). All preprocessing (impute/scale/select/calibrate) inside folds — the pilot's nested doctrine.
- **Temporal:** rolling-origin backtest — train ≤ month M, evaluate months (M, M+3] on trucks unseen in training; plus a **locked prospective holdout**: the final 3 months of the retrospective drop + all post-drop data are untouched until Phase 10 sign-off.
- **Prequential replay (pilot k-week rewind):** score every truck at T−k weeks for k=0..26; report AUROC(k) curve and the sustained-≥0.75 horizon (pilot metric).
- **Metrics contract:** AUROC (with bootstrap CI), PR-AUC, recall @ FP-budget, **lead-time-at-precision** (median days from first sustained RED to JCOPENDATE at precision ≥0.7), calibration (Brier, CITL, slope ∈[0.5,2]), **alerts/100 trucks/month at field prevalence** (budget ≤2), and per-failure-mode recall (gradual vs abrupt vs inconclusive — from D2 codes).
- **Prevalence correction:** cohort 50/50 → field π via logit offset `log(π/(1−π)) − log(π_train/(1−π_train))`; all PPV/alert volumes reported at π ∈ {2 %, 5 %, 10 %}/yr scenarios.
- **Statistical referee:** champion vs challenger compared on paired fold AUROCs (Wilcoxon) + DeLong on pooled OOF; promotion needs Δ ≥ +0.02 AND p<0.05 AND no degradation in lead-time-at-precision or calibration (§7.7).

### 7.2 Champion port (Phase 4)

Ridge(α=1.0) + 6 features, retrained on the 500-vehicle cohort under §7.1. Expected outcome: AUROC in 0.85–0.93 band (pilot CI was [0.81, 1.00]; regression to mean expected). Champion defines the bar; its card documents tier thresholds re-derived from OOF scores (Youden AND cost-optimal thresholds; see §8).

### 7.3 Classification challengers (Phase 6) — model choices and why

| Family | Config (initial) | Why this model for THIS dataset | Risk / mitigation |
|---|---|---|---|
| **Elastic-net logistic** | `LogisticRegression(penalty='elasticnet', solver='saga', C∈[0.01,10], l1_ratio∈[0.1,0.9])` | Direct Ridge upgrade: sparsity for the widened feature pool; calibrated-ish scores; interpretable coefficients for DICV | minor gain expected; cheap to run |
| **LightGBM** (primary challenger) | `num_leaves≤15, max_depth 3–4, min_child_samples≥25, feature_fraction 0.7, bagging 0.8, monotone_constraints` where physics is known (e.g., ged2 dwell ↑risk) | Tabular weekly/daily aggregates with interactions + missingness (GED-absent VINs) — GBMs handle NaN natively; SCANIA fleet evidence (LightGBM ≥ Bi-LSTM at 2,272 failures); fast enough for nested CV | overfit at ~250 events → shallow trees, heavy regularization, nested selection |
| **XGBoost** | `max_depth 3, eta 0.05, subsample 0.8, colsample 0.7, scale_pos_weight=1` (cohort balanced) | cross-check LightGBM (implementation-diversity) | redundant if LightGBM wins clearly |
| **Random Forest** | 500 trees, `max_features='sqrt'`, `min_samples_leaf≥10` | robustness baseline; OOB sanity | usually dominated by GBM |
| **TabPFN v2** | as-is (≤10 k rows, ≤100 features — our matrix fits) | "cheap first swing" per FM-pilot verdict; zero tuning | context-size limits; advisory only |
| **Calibrated ensemble** | soft-vote / stacking (logistic meta) over {Ridge, EN, LGBM} | variance reduction at modest n | only if components are diverse (ρ<0.9 on OOF) |
| Calibration layer | **isotonic** (n now sufficient; pilot's SM used Platt at n=34) on OOF decision values; validate CITL/slope | probability quality is a shipping requirement (tiers, economics) | isotonic overfits small n — verify with reliability curves per fold |

Class imbalance: none in-cohort (~50/50); **do NOT oversample**. The imbalance problem here is *deployment prevalence*, handled by prior-shift correction + FP-budget thresholds, not by SMOTE (which fabricates trucks and violates the physics-honesty doctrine).

### 7.4 Anomaly-detection program (Phase 7) — re-opened with a budget

History: V7 pilot program — IForest, LOF, AE, one-class SVM all produced 80–100 % FP at n=25. Cause: no stable "healthy" reference from 15 trucks with heterogeneous usage/firmware. At n≈250 healthy trucks the reference exists. **Stance: watchlist/triage channel with a hard FP budget (≤0.2 episodes/truck-year, the SM H2 bar), never an autonomous work order.**

| Method | Design | Why it fits this data | Kill criterion |
|---|---|---|---|
| **Fleet-relative residuals** (primary) | healthy-fleet expected VSI surface: quantile regression of VSI on (RPM band, ANR band, age, firmware family) from ~250 NF trucks → per-truck daily residual → robust EWMA z; alarm on sustained z>3 for ≥14 d (pilot debounce pattern) | operationalizes the pilot's `vsi_resid_mean` heuristic with a real reference surface; fully interpretable (volts below expected) | FP budget breach on NF holdout |
| **IsolationForest** | weekly feature vectors (frozen 6 + candidates), per-firmware-family fit, `contamination='auto'`, score→percentile vs healthy | cheap, nonparametric; catches multivariate weirdness | ranking must enrich failures ≥3× lift @ top decile |
| **Autoencoder (dense, 6→3→6 on daily agg; conv1d on 90-d windows later)** | reconstruction error percentile vs healthy | only if IForest shows signal; deep AE deferred to Phase 9 hardware | same lift gate |
| **Matrix Profile (STUMPY)** | daily VSI series discords per truck (window ~7 d) | regime-break detector for abrupt precursors the aggregates smooth over | discord dates must precede JCOPENDATE >7 d at ≥2× chance rate |
| **HMM / Markov drift** | GED transition-matrix per rolling 30 d; alarm on P(2→2) crossing 0.95 (pilot: F 0.992 vs NF 0.860) | direct productization of the V12 finding | GED-coverage mask; ≥95 %-null VINs exempt |
| Control charts (EWMA/CUSUM) | per-channel (resting VSI, ged2_cnt) with per-truck baselines | free, interpretable, workshop-explainable | subsumed by residual method if redundant |

Evaluation frame: anomaly channels are scored as **event detectors** (episodes, precision at episode level, median lead) on the prequential replay — the same frame as the pilot's alert-channel table, so numbers are directly comparable to GED-storm (2/10 @ 0 FP) and early-watch (3/10 @ 0 FP).

### 7.5 RUL / forecasting program (Phase 8) — windows, never points

**Gate zero (carried from V10.6.2):** any RUL model must beat the fleet-clock dummy (median-TTF constant; pilot MAE 49.7 d) on grouped OOF, Wilcoxon p<0.05 — otherwise ship the fleet clock. The pilot's dummy is embarrassingly strong because TTF concentrated at 472–673 d; expect the 500-truck TTF spread to be wider (more duty-cycle diversity), which is exactly what gives covariates room to work.

| Model | Config | Why for this dataset | Output |
|---|---|---|---|
| **Fleet clock v2** (baseline, ships day 1) | Kaplan–Meier on 250 events; strata: duty-cycle band (est_km/day tercile), firmware family, region if available | pilot's 601-d median generalized with real survival machinery + censoring | median + P10–P90 band per stratum, on 3 axes (days / est-km / engine-h) |
| **Discrete-time hazard** (primary challenger) | weekly person-period logistic/GBM: P(fail in week t | covariates_t, age splines); covariates = frozen features + GED Markov + usage | natural fit to the weekly panel; handles time-varying covariates cleanly; calibrated by construction; ~250 events × ~80 weeks = ~2 M person-periods — trivial compute | per-truck hazard curve → survival curve → calibrated window |
| **Cox PH** (lifelines) | baseline + time-varying covariates variant | interpretable HRs for DICV; standard | risk-adjusted windows |
| **Random Survival Forest** (scikit-survival) | 500 trees, `min_samples_leaf≥15` | nonlinear interactions without proportionality assumption | C-index cross-check |
| **AFT (Weibull)** | `WeibullAFTFitter`; also gradient-boosted AFT (XGBoost `survival:aft`) | direct TTF distribution; pilot's Weibull machinery exists (`weibull_log_sf`) | parametric windows |
| Quantile GBM | LightGBM `objective=quantile` α∈{0.1,0.5,0.9} on TTF of failed spells | simple, strong baseline for window width | P10/P50/P90 directly |

**Evaluation:** C-index (target ≥0.65 to bother), calibrated **interval coverage** (P10–P90 must contain 80±10 % of actual failures on OOF), MAE vs dummy, and **usefulness**: fraction of failures where the P10 bound gave ≥30 d actionable notice. Left-truncation handled per D1; competing risks not material for ALT (single component), but repair-reset spells are.

**Shipping rule:** customer-facing output is a **maintenance window** (banded runway on the V11.2 evidence-stack graph pattern), tier-gated: windows shown only for AMBER+ trucks; GREEN trucks show fleet-stratum window.

### 7.6 Deep / TS-foundation-model challengers (Phase 9 — gated)

Per the FM-pilot go/no-go (2026-06-24, spec `docs/superpowers/specs/2026-06-24-clutch-first-fm-pilot-design.md`): Path A (text-LLM fine-tune) is **permanently dead**; Path B (MOMENT-style TS-FM as feature extractor + head) and Path C (PatchTST / 1D-CNN / TCN from scratch) are pilot-worthy **only after** the classical program plateaus, and clutch pilots first (real 63-d precursor = cleanest testbed). ALT enters this phase only if: (a) clutch pilot shows Δ≥+0.02 for B/C, and (b) ALT classical program has plateaued, and (c) ≥200 ALT failure events with ≥60 d dense pre-failure telemetry exist.

- Inputs: multichannel weekly (or daily) sequences per truck (VSI stats, GED counts, usage), NOT raw 5 s rows (5 s × 600 d = 10 M steps/truck — pointless for weeks-scale physics).
- Kill criteria (frozen before training): beat champion GBM on §7.1 harness at Δ≥+0.02 AUROC AND ≥ equal lead-time-at-precision AND FA ≤2/100/mo at field prevalence; else document and stop. Compute: A10 Spot; A100 only for sweeps.

### 7.7 Promotion gates (all model families, one contract)

```
PROMOTE challenger → champion iff ALL:
  G1  ΔAUROC ≥ +0.02 (grouped OOF, DeLong p<0.05) vs current champion
  G2  paired-fold Wilcoxon p < 0.05 across 50 fold-repeats
  G3  calibration slope ∈ [0.5, 2.0], CITL ∈ [−0.1, +0.1], Brier ≤ champion
  G4  lead-time-at-precision(0.7) ≥ champion − 0 d
  G5  alerts/100 trucks/mo @ field prevalence ≤ 2.0 (or ≤ champion)
  G6  negative controls pass (permuted ≈0.5; exporter-fingerprint ≈0.5)
  G7  temporal holdout AUROC within 0.05 of CV estimate
  G8  model card + frozen spec JSON + reproduction script committed
DEMOTE (rollback) iff in production: calibration drift (slope outside [0.4,2.5])
  OR alert-rate SPC breach for 2 consecutive weeks OR PSI>0.25 on ≥2 features
```

---

## 8. Alerting & Decision Layer (port + re-cost)

### 8.1 Channels at 500 vehicles

| Channel | Port | Changes at scale |
|---|---|---|
| **C1 Risk tiers** (champion scores) | GREEN/AMBER/RED | thresholds re-derived: Youden on OOF AND cost-optimal (₹ economics below); tier debounce adopted from V11.2 (+0.02 deadband, 14-d dwell) as default, re-tuned |
| **C2 GED=2 storm** | `ged2_cnt ≥ 200/day`, hourly scan | keep 200/day as anchor; re-fit threshold on 250-NF distribution to hold 0-FP posture; add `ged_markov_p22>0.95 sustained` as a v2 variant behind shadow flag; GED-absent VINs excluded (roster from R13) |
| **C3 Early-watch compound** | 5 heuristics, ≥2 votes | re-fit each heuristic's cut on NF quantiles (n=250 gives real p95s); consider learned weights ONLY if they beat equal weights on §7.1 (pilot's equal-weights honesty stays the default) |
| **C4 Maintenance windows** | fleet clock → §7.5 survival windows | tier-gated exposure |
| C5 (new, shadow) | anomaly watchlist (§7.4) | shadow-only until FP budget proven 2 consecutive quarters |

### 8.2 Alert economics (thresholds priced, not just ROC-optimal)

Using the SM-vertical cost frame as the program anchor (inspection ₹1,500; breakdown ₹46,000; R≈31; p_convert 0.70 — ALT-specific costs to be confirmed with DICV in Phase 0): expected-cost thresholding on OOF scores → the operating point that minimizes ₹/truck/yr; report both Youden and cost-optimal points; the **43 % cost-saving framework from the SM Youden-queue analysis is replicated for ALT** with its own numbers at field prevalence. Alert volume forecast: at 500 trucks and budget 2/100/mo → ≤10 alerts/mo fleet-wide → workshop-capacity sanity confirmed with DICV.

### 8.3 Alert plumbing

Dedupe key (VIN, channel, week); hysteresis: tier downgrade requires 14-d dwell below boundary; escalation: C2 fires → same-day webhook; C1 RED sustained 3 weeks → job-card recommendation w/ evidence-stack graph attached (V11.2 per-VIN 5-panel pattern is the artifact contract); all alerts logged with model version + feature snapshot for post-hoc audit (the "alert replay" pilot pattern).

---

## 9. Deployment LLD (Azure jobs & TruckConnect)

### 9.1 Job DAG (daily, Container Apps Jobs / Azure ML pipeline)

```
00:30 IST  ingest_validate     landing → bronze     (pandera gate, manifest)
01:00      clean_conform       bronze → silver      (R1–R15, DQ ledger)
02:00      daily_cache_update  silver → gold        (incremental day append)
02:30      feature_snapshot    gold → features      (point-in-time matrix)
03:00      score_champion      features → scores    (Ridge + calibration + tiers)
03:15      channels            scores+cache → alerts (C1–C4, debounce, dedupe)
03:30      publish             alerts → Postgres + TruckConnect API + webhooks
04:00      monitors            PSI, calibration, alert SPC, DQ roll-up → Log Analytics
hourly     ged_scan            today's increment → C2 evaluation (GED,VSI columns only)
weekly     dq_asymmetry (R15), fleet report, evidence-stack graph refresh (AMBER+)
```

Runtime budget: full daily chain <30 min on D8ds_v5 at 500 trucks (pilot processed 97 M rows on a workstation; daily increment is ~2.75 M rows).

### 9.2 Serving contract to TruckConnect

- `GET /v1/alt/vehicles/{vin}/risk` → {score, tier, tier_since, window{p10,p50,p90, axis}, channels[], model_version, snapshot_date}
- Webhook on C2 fire and tier upgrade; payload includes evidence summary (top features + GED counts).
- Signed licensed container; per-tenant isolation per commercial Q&A open item Q17 (resolve in Phase 0).

### 9.3 Retraining loop (label-triggered, from the Azure v2 plan)

Trigger = **new confirmed failure labels** (≥10 new events OR quarterly, whichever first — label-count gates cadence, "labels not frames"). DAG: feature pull (<1 GB) → Spot A10 spin-up → retrain champion + challengers (~1–3 h wall) → §7.7 gates vs live champion → promote/keep → auto-deallocate (300 s idle). Cost ~$2–3 GPU per cycle; classical models train on CPU anyway (GPU reserved for Phase-9 deep work — keep the A10 config, use CPU SKU for classical retrains to cut even that).

---

## 10. MLOps

| Concern | Mechanism |
|---|---|
| Registry | MLflow on Azure ML: model + frozen-spec JSON (pilot `*_ridge_spec.json` pattern: features, coefs, thresholds, CV protocol, label-ledger version, data manifest hash) |
| Reproducibility | every artifact traces to (git commit, label ledger vN, lake manifest hash, seed); pilot's byte-exact reproduction gate is the standard |
| Drift — features | monthly PSI per feature vs training reference; >0.25 on ≥2 features → investigate; >0.25 on ≥4 → retrain trigger |
| Drift — calibration | rolling CITL/slope on realized outcomes (as labels mature); slope outside [0.4, 2.5] → demote to previous champion |
| Drift — alerts | SPC chart on alerts/100 trucks/week per channel; 2-week breach → freeze thresholds, investigate |
| Shadow mode | every challenger and every threshold change runs ≥4 weeks in shadow (scores logged, no customer exposure) before flip |
| Audit | alert replay tooling (pilot pattern): re-score any historical date from pinned artifacts |
| Model cards | DICV-facing card per promotion (pilot V1.1 SM card is the template: features, formulas, validation, known errors, artifacts list, honest-limits section) |
| Runbooks | ingest failure, DQ hard-gate, Spot eviction (checkpoint+resume), TruckConnect API outage (queue+backfill), rollback |

---

## 11. Instrumentation Workstream (the real unlock — runs parallel to everything)

Priority-ordered sensor asks (V12 evidence), negotiated with DICV as part of the 500-vehicle contract:

| Priority | Signal | Where it lives today | What it unlocks |
|---|---|---|---|
| 1 | **LIN regulator registers RMC (excitation current, 8-bit) + RDC (duty cycle, 5-bit)** | inside-alternator LIN bus (sensor-blocked from CAN feed) | continuous excitation → direct gradual-electrical degradation tracking; regression/HMM/Kalman reconstruction becomes feasible; targets the failure class that produced VIN1's 21-d lead |
| 2 | **J1939 SPN 115 — alternator output current** (PGN 65271/VEP1) | defined but not broadcast; "enabling is low-cost" | load-normalized voltage analysis; charging-capability trend |
| 3 | **Alternator temperature** | not instrumented | physics-based excitation model prerequisite; thermal stress cycling |
| 4 | **W-terminal (generator speed)** | new ASAM/CPC platform exposes it | decouples alternator speed from engine RPM (belt slip detection) |
| 5 | GED coverage fix + heartbeat policy + IGN + ODO (D3/D4/D8) | config | removes GED-absent blind spot (2/10 pilot F trucks); exact km axes |

Plan impact if granted: Phase 6 gains an "excitation-feature" candidate family with first-class priors; the recall ceiling from the 5-abrupt/10 mode mix is expected to move — this is stated as the **only** credible path to catching the flat-to-failure class, and the plan carries NO recall promises contingent on it.

---

## 12. Phase-Wise Execution Plan

Durations assume 2 engineers (E1 data/platform, E2 ML) with the shared platform being built once for both verticals (ALT leads, SM follows 2 weeks behind — see companion plan §12). Solo-engineer durations ×1.8. Every phase has an entry gate (EG), tasks, deliverables, and an exit gate (XG) with verification.

### P0 — Program setup & contracts (Week 1)
- EG: signed 500-vehicle program brief.
- Tasks: Azure RG/IaC (dev/uat/prod); identities/Key Vault; data-contract negotiation (D1–D8, sensor asks §11); ALT-specific cost parameters from DICV (inspection/breakdown ₹, workshop capacity); TruckConnect API contract + tenancy answers (Q17); repo scaffold (`ALT_500/` mirroring pilot conventions, config-as-code, frozen-spec JSONs vendored).
- Deliverables: signed data contract; IaC; runbook skeletons; risk register v1.
- XG: test file round-trips landing→bronze in dev; contract asks dispositioned (granted/refused+fallback).

### P1 — Ingestion & parquet lakehouse (Weeks 2–3)
- EG: P0 XG; first sample drop from DICV (even 5 trucks suffices).
- Tasks: converter (§5.2) + schema gate; medallion layout; manifest/idempotency; quarantine path; backfill runner for the retrospective drop; DQ ledger schema.
- Deliverables: bronze/silver populated for sample; converter acceptance report (row/null/minmax parity, ratio ≥8:1).
- XG: converter acceptance gate green on all delivered files; rerun idempotency proven.

### P2 — Label pipeline & fleet registry (Weeks 3–4, overlaps P1)
- Tasks: job-card ingest; JCOPENDATE reconciliation + gap table (pilot §1.5 rule); censoring/spell table; label-QA report (TTF distribution vs pilot's 472–673 d; failure-code mix; review queue for gaps >30 d); fleet registry (VIN, firmware family, GED coverage, SALEDATE, usage axes).
- XG: label ledger v1 signed off by DICV service-data owner; every failed VIN has a resolved failure date + mode code (or documented `mode=unknown`).

### P3 — Cleaning suite & daily cache (Weeks 4–6)
- Tasks: R1–R15 implemented with counters; cadence/firmware classifiers; daily cache builder (incremental + full-rebuild); dt-weighting A/B; **pilot-reproduction harness**: run the scale pipeline on the 25 pilot trucks.
- XG: **byte-exact reproduction** of pilot daily cache and the six feature values (≤1e-9); R15 asymmetry report on the 500-truck drop reviewed; DQ weekly report live.

### P4 — Champion port & validation harness (Weeks 6–8)
- Tasks: §7.1 harness (grouped CV, temporal splits, prequential replay, metrics contract, negative controls); Ridge champion retrained on 500; calibration layer (isotonic); threshold derivation (Youden + cost-optimal); champion model card v1.
- XG: pilot LOVO 0.9267 reproduced exactly on pilot data; 500-cohort champion card published with CI; negative controls ≈0.5; prospective holdout LOCKED (hash-sealed).

### P5 — Alert channels & economics (Weeks 8–10)
- Tasks: C1–C4 ported (§8); GED storm threshold re-fit on 250-NF; early-watch heuristic re-cut; debounce tuning; ₹ economics + alert-volume forecast at field prevalence; evidence-stack graph generator ported (V11.2 5-panel per-VIN).
- XG: replay on retrospective data meets FP budget (≤2/100/mo at field prevalence); DICV sign-off on alert taxonomy + volumes.

### P6 — Challenger program: features & classifiers (Weeks 10–13)
- Tasks: graduation re-tests (§6.3, pre-registered, BH-FDR); challenger battery (§7.3) under §7.7 gates; failure-mode-conditional analysis (with D2 codes); ensemble evaluation; update champion iff gates pass.
- XG: challenger report (every candidate: PROMOTE/REJECT + stats); registry updated; if promotion → shadow-mode start.

### P7 — Anomaly program (Weeks 12–14, overlaps P6)
- Tasks: healthy-fleet expected surface; residual channel; IForest/MatrixProfile/HMM battery (§7.4) on prequential replay; watchlist channel in shadow.
- XG: anomaly verdict vs FP budget; SHIP-shadow / PARK decision per method (PARK is an acceptable outcome — pilot precedent).

### P8 — RUL & survival (Weeks 13–16)
- Tasks: KM fleet clock v2 (strata, 3 axes); discrete-time hazard + Cox/RSF/AFT battery; interval-coverage calibration; beat-the-dummy referee; window UX (tier-gated banded runway).
- XG: RUL verdict: ship windows (coverage 80±10 %, beats dummy p<0.05) or ship fleet-clock v2 only (honest fallback, pilot precedent).

### P9 — Deep/TS-FM challengers (gated; scheduled only if §7.6 entry conditions met — earliest Weeks 17+)
- Tasks per FM-pilot spec P3 replication: PatchTST/1D-CNN/TCN, MOMENT-extractor+head, frozen kill criteria.
- XG: beat-GBM verdict documented either way.

### P10 — Production deployment (Weeks 14–16, overlaps P8)
- Tasks: job DAG to prod (§9.1); TruckConnect API + webhooks; Power BI; monitors (§10); shadow → soft-launch (alerts to DICV analysts only) → full launch.
- XG: 2 clean weeks of soft-launch (zero hard-gate failures, alert volumes in budget); prospective-holdout evaluation unsealed and within G7; DICV launch sign-off.

### P11 — Operate & improve (ongoing from Week 17)
- Label-triggered retrains; quarterly model-risk review; drift dashboards; instrumentation-signal onboarding when §11 sensors land (new bronze schema version, feature families, targeted gradual-electrical detector).

---

## 13. Timeline, Staffing, Cost

```
Wk:        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17+
P0 setup   ██
P1 lake       ████
P2 labels        ████
P3 clean            ██████
P4 champ                  ██████
P5 alerts                       ██████
P6 chall                              █████████
P7 anom                                     ██████
P8 RUL                                         █████████
P10 deploy                                        ████████
P9 deep(gated)                                             ██████…
P11 operate                                                ██████…
```

| Scenario | Duration to production (P10 XG) | Effort |
|---|---|---|
| 2 engineers (E1 platform, E2 ML) | **~16 weeks** | ~150 eng-days |
| 1 engineer | ~28–30 weeks | ~150 eng-days serialized |
| Both verticals together (shared platform; SM offset +2 wk) | ~18–20 weeks combined | ~230 eng-days total (platform amortized) |

Cost (₹94/USD): build compute ≤ ₹1.2 L (CPU-dominant; A10 Spot bursts); production run-rate at 500 trucks ≲ ₹10–15 k/mo (storage + ~1 h/day CPU + monitors — the $300–560/mo envelope is the 10 k-vehicle figure); retrains ~₹450–1,300/yr classical (A10 Spot 1–3 GPU-h/quarter), Phase-9 transformer tier if triggered ~₹1,700–3,800/yr per vertical. These slot inside the commercial Q&A per-vehicle economics (compute 3–7 % of ₹150–600/veh/yr price points).

---

## 14. Risks & Mitigations (honest register)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | **Failure-mode mix caps recall** (pilot: 5/10 electrically flat) — 500 trucks won't warn where physics is silent | High | High (expectation mgmt) | per-mode recall reporting (D2 codes); instrumentation workstream §11; DICV comms anchored to "ranking + windows + emergency", never "predict every failure" |
| R-2 | Curated 50/50 cohort ≠ field → optimistic PPV | Certain | High | prevalence-corrected reporting everywhere (§7.1); field-prevalence FP budget governs thresholds |
| R-3 | Cohort export asymmetry (pilot NaT/column-order precedent) → plumbing leakage | Medium | High (silent inflation) | D5 contract ask; R15 sentinel; exporter-fingerprint negative control (G6) |
| R-4 | Label noise: JCOPENDATE lag, non-replacement job cards, silent gaps (pilot VIN3 +66 d) | High | Medium | Phase-2 curation + review queue; gap table published; sensitivity analysis (labels ±14 d) |
| R-5 | GED-absent trucks (pilot 2/10 F + 4/15 NF ~99 % null) blind the emergency channel | Medium | Medium | R13 roster; C2 exemption list; D4/D8 config fix ask; report C2 coverage explicitly |
| R-6 | Heartbeat/firmware heterogeneity confounds features | Medium | Medium | R10/R11 tags as covariates + stratification in CV; dt-aware aggregation |
| R-7 | Anomaly program repeats V7 (FP flood) | Medium | Low (budgeted) | shadow-only; hard FP budget; PARK is a sanctioned outcome |
| R-8 | RUL fails to beat fleet clock again | Medium | Low (fallback ready) | fleet-clock v2 ships regardless; windows tier-gated |
| R-9 | Single-VM bottleneck if DICV delivers 24 h continuous 5 s (3× volume) | Low | Low | Polars streaming headroom ~10×; escape hatch: partitioned parallel jobs, then Spark at ≥10 k vehicles |
| R-10 | Spot A10 eviction mid-retrain | Medium | Trivial | checkpointing; retry; PAYG fallback flag |
| R-11 | Workshop follow-through (p_convert < 0.70) erodes ROI | Medium | Medium | economics re-fit quarterly from realized job-card outcomes; alert volume kept ≤ workshop capacity |
| R-12 | Threshold transfer: pilot Youden 0.4456 invalid at 500 | Certain (by design) | Low | re-derived in P4/P5; never reuse pilot thresholds |

---

## 15. Build Inventory — What We Are Going To Code

Everything below is a deliverable code artifact with an owner, a phase, and a Definition of Done (DoD). DoD for every module = code + unit tests green + docstring header with formula/contract + entry in the module README + (where marked ⓖ) golden-reproduction test against pilot artifacts.

### 15.1 Repository layout (`ALT_500/`)

```
ALT_500/
  config/            champion.yaml, thresholds.yaml, schema_contracts/, cost_params.yaml
  src/
    ingest/          i01_landing_sync  i02_schema_gate  i03_csv_to_parquet  i04_manifest
    clean/           c01_conform(R1–R9)  c02_cadence_firmware(R10–R11)  c03_flags(R12–R14)
                     c04_asymmetry_sentinel(R15)  c05_dq_ledger
    labels/          l01_jobcard_ingest  l02_label_ledger  l03_spells  l04_label_qa
    cache/           d01_daily_cache  d02_usage_axes
    features/        f01_registry  f02_frozen_six ⓖ  f03_candidates  f04_snapshot
                     f05_leak_gates
    models/          m01_harness  m02_champion ⓖ  m03_challengers  m04_anomaly
                     m05_survival  m06_negative_controls  m07_thresholds  m08_promotion
    alerts/          a01_risk_tiers  a02_ged_storm ⓖ  a03_early_watch ⓖ  a04_windows
                     a05_debounce_dedupe  a06_evidence_stack  a07_replay ⓖ
    serve/           s01_score_job  s02_publish_api  s03_webhooks  s04_monitors
    ops/             o01_retrain_trigger  o02_registry_io  o03_report_weekly  o04_dq_report
  tests/             unit/  integration/  reproduction/  property/  perf/  fixtures/
  infra/             bicep|terraform/  pipelines/  Dockerfile  container-policy/
  docs/              runbooks/  model_cards/  review_records/
```

### 15.2 Module inventory (module → purpose → key contents → I/O → phase)

**Data plane**

| Module | Purpose | Key functions / classes | Inputs → Outputs | Phase |
|---|---|---|---|---|
| `i01_landing_sync` | pull/verify TruckConnect drops | `sync()`, hash check, gz handling | Blob landing → verified file list | P1 |
| `i02_schema_gate` | pandera-polars contracts, name-based | `FailedFileSchema`, `HealthyFileSchema`, `validate()` | CSV header/sample → pass/quarantine | P1 |
| `i03_csv_to_parquet` | bronze converter (§5.2) | `convert(src,dst)`, streaming sink, zstd-3, 1M row groups | CSV → bronze parquet + parity report | P1 |
| `i04_manifest` | idempotency ledger | `register()`, `is_ingested()`, atomic rename | file hash → manifest row | P1 |
| `c01_conform` | rules R1–R9 | one function per rule, `apply_all()`, counters | bronze → silver + counters | P3 |
| `c02_cadence_firmware` | R10 dt classifier, R11 family tag | `classify_dt()`, `tag_family()` | silver → cadence cols, VIN registry update | P3 |
| `c03_flags` | R12 silent-gap, R13 GED-coverage, R14 plausibility | `flag_*()` per rule | silver → per-VIN flag table | P3 |
| `c04_asymmetry_sentinel` | R15 cohort-plumbing leak detector | KS tests over DQ stats by cohort | DQ ledger → weekly verdict | P3 |
| `c05_dq_ledger` | counters store + reconciliation | `write()`, `reconcile()` (rows_in = rows_out + drops) | all clean steps → ledger parquet | P3 |
| `l01_jobcard_ingest` | job-card table ingest (D2) | code mapping, free-text strip | DICV extract → labels bronze | P2 |
| `l02_label_ledger` | JCOPENDATE reconciliation + gap table + versioning | `build_gap_table()`, `bump_version()` | labels + telemetry ends → ledger vN | P2 |
| `l03_spells` | survival spells, censoring, repair reset | `build_spells()`, left-truncation flags | ledger + registry → spell table | P2 |
| `l04_label_qa` | TTF distribution, outlier review queue | `qa_report()` | spells → QA md report | P2 |
| `d01_daily_cache` | §5.6 per-(VIN,day) aggregates | `build_full()`, `append_day()`, state masks, GED transitions | silver → gold daily cache | P3 |
| `d02_usage_axes` | est_km, engine-hours integration | dt-clipped integrators | silver → per-VIN cumulative axes | P3 |

**Feature & model plane**

| Module | Purpose | Key contents | Phase |
|---|---|---|---|
| `f01_registry` | single source of truth: name → formula string → callable → ban-list check | `FEATURE_REGISTRY`, `assert_not_banned()` | P3 |
| `f02_frozen_six` ⓖ | §1.2 features, verbatim | 6 functions, exact pilot semantics | P3 |
| `f03_candidates` | §6.3 graduation battery | GED-Markov set, heuristic-promotions, fleet-relative set | P6 |
| `f04_snapshot` | point-in-time feature matrix | `snapshot(date)` walk-forward safe | P3 |
| `f05_leak_gates` | L-window control, proxy-ρ gate, spectral-grid audit | `fixed_window_control()`, `proxy_gate()` | P4 |
| `m01_harness` | §7.1 referee: grouped CV, temporal, prequential, metrics contract, DeLong/Wilcoxon | `run_cv()`, `prequential()`, `referee()` | P4 |
| `m02_champion` ⓖ | Ridge port + isotonic + tiers | `fit()`, `predict_proba()`, spec JSON I/O | P4 |
| `m03_challengers` | EN / LightGBM / XGB / RF / TabPFN / ensemble | one class per family, common interface | P6 |
| `m04_anomaly` | §7.4 battery | expected-surface fit, IForest, MatrixProfile, GED-HMM, episode scorer | P7 |
| `m05_survival` | §7.5 battery | KM strata, person-period expander, Cox/RSF/AFT wrappers, coverage metric | P8 |
| `m06_negative_controls` | permuted / year-shuffled / exporter-fingerprint models | `run_controls()` | P4 |
| `m07_thresholds` | Youden + cost-optimal + prior-shift correction | `derive()`, `shift_prior()` | P4–P5 |
| `m08_promotion` | G1–G8 gate evaluator + promotion record | `evaluate(challenger, champion)` | P6+ |

**Alert & serving plane**

| Module | Purpose | Key contents | Phase |
|---|---|---|---|
| `a01_risk_tiers` | tier machine + debounce (deadband 0.02, dwell 14 d) | `assign_tier()`, hysteresis | P5 |
| `a02_ged_storm` ⓖ | C2: ged2_cnt ≥200/day, hourly scan, ged_absent exemption | `scan_hourly()` | P5 |
| `a03_early_watch` ⓖ | C3: 5 heuristics, ≥2 votes | one function per heuristic, `vote()` | P5 |
| `a04_windows` | C4: fleet-clock v2 + survival windows, tier-gating | `window_for(vin)` | P5/P8 |
| `a05_debounce_dedupe` | (VIN, channel, week) dedupe, escalation rules | `publish_gate()` | P5 |
| `a06_evidence_stack` | per-VIN 5-panel graph (V11.2 design port) | matplotlib builders | P5 |
| `a07_replay` ⓖ | historical alert replay from pinned artifacts | `replay(range, registry_version)` | P5 |
| `s01_score_job` | daily DAG steps 05–08 orchestration | CLI entrypoints per step | P10 |
| `s02_publish_api` / `s03_webhooks` | §9.2 contract | FastAPI app, webhook sender w/ idempotency keys | P10 |
| `s04_monitors` | PSI, calibration drift, alert SPC → Log Analytics | `run_monitors()` | P10 |
| `o01_retrain_trigger` | label-count ≥10 OR quarterly | counter + trigger emit | P11 |
| `o02_registry_io` | MLflow + frozen-spec JSON round-trip | `save_spec()`, `load_spec()` | P4 |
| `o03_report_weekly` / `o04_dq_report` | fleet md/PDF report, DQ heatmap | report builders | P5/P3 |

**Infra & tooling:** IaC (RGs, ADLS, jobs, AML compute incl. Spot A10 scale-to-zero, Key Vault, Postgres, Log Analytics); CI pipeline (lint + unit on PR; nightly reproduction suite; weekly replay suite); `fixtures/` synthetic-fleet generator (`make_truck(profile=…)` — parameterized failure signatures, heartbeat mixes, GED storms); container build + signing.

**Estimated volume:** ~38 source modules, ~25 infra/CI files, ~120 test files. Reuse note: `f02`, `a02`, `a03`, `a06`, `a07` are ports of existing pilot code (V5.2_ALT/V10.5.3/V11.2 sources), not greenfield — porting + hardening, with the pilot artifacts as oracles.

---

## 16. Review Plan — What We Are Going To Review

### 16.1 Engineering reviews (every PR; second-engineer approval mandatory)

Per-code-class checklists applied by the reviewer:

| Code class | Review checklist (blocking items) |
|---|---|
| Data plane | name-based column access only (no positional); nulls propagate (never silently filled); counters for every dropped row; deterministic (seeded, stable sort); partition writes atomic (temp+rename); no cohort-conditional logic without an R15 justification comment |
| Feature code | formula string in registry matches implementation; walk-forward safety (no future rows reachable); NaN semantics documented; ban-list check invoked; unit fixture + golden test present |
| Model/harness code | all transformers fit inside folds only; seeds pinned and logged; metrics computed on pooled OOF only; no test-fold access anywhere (grep for `X_test` misuse); spec JSON round-trip test |
| Alert code | causality at cut (uses data ≤ cut); thresholds read from config, never literals; dedupe/debounce covered by tests; replay determinism |
| Serving/infra | least-privilege identities; secrets only via Key Vault; API schema versioned; idempotent webhooks; rollback path stated in PR description |

### 16.2 Design & statistical review gates (named, scheduled, with written records in `docs/review_records/`)

| Review | When | Reviewed artifact | Pass criteria |
|---|---|---|---|
| Data-contract review (w/ DICV) | P0 | D1–D8 dispositions | every ask granted or fallback accepted in writing |
| Lakehouse design review | P1 | §5 LLD vs implementation | layout, idempotency, quarantine paths as specified |
| **Label review board** | P2, then monthly | label ledger diff, gap table, review queue | every failed VIN dispositioned; gaps >30 d resolved or flagged |
| **Leakage review board** | P4, then per candidate batch | negative-control report, leak-gate outputs, R15 verdicts | permuted ≈0.5; exporter-fingerprint ≈0.5; all candidates gated |
| Harness design review | P4 | §7.1 implementation, referee stats | fold hygiene demonstrated; CI on synthetic known-answer sets |
| Calibration review | P4/P6 | reliability curves, CITL/slope per fold | slope ∈[0.5,2], CITL ∈±0.1 |
| **Threshold & economics review (w/ DICV)** | P5 | cost curves, alert-volume forecast at field prevalence | budget ≤2/100/mo; workshop capacity confirmed |
| Challenger verdict review | P6/P7/P8 | per-candidate PROMOTE/REJECT evidence packs | G1–G8 table complete, no missing cells |
| Model-card review | every promotion | card vs V11.2-dossier honesty standard | known-errors + limits sections present; claims traceable |
| Security review | P10 | identities, API authZ, container signing, PII scan | zero criticals; tenancy isolation tested |
| **Launch readiness review (w/ DICV)** | P10 | soft-launch report, holdout unseal, runbooks | 2 clean weeks; G7 met; sign-off minuted |
| Quarterly model-risk review | P11 | drift dashboards, realized-outcome calibration, alert SPC | demote/retrain decisions minuted |

### 16.3 Recurring data reviews

Weekly: DQ report (completeness heatmap, counters, quarantine digest) — engineer sign-off; R15 asymmetry verdict — **any breach freezes modeling work until dispositioned**. Monthly: GED-coverage roster, silent-gap watchlist, firmware-family census, label-ledger diff.

### 16.4 Deliverable/report reviews

Every DICV-facing artifact (fleet report, evidence-stack graphs, model cards) passes the pilot's QA-render review: render → PNG → visual inspection checklist (axis labels, units, VIN suffixes, no internal-only fields, no IP-sensitive feature formulas in external variants).

---

## 17. Test Plan — What We Are Going To Test (Test-Case Catalog)

### 17.0 Strategy

- **Pyramid:** unit (per rule/feature/function) → component (module I/O) → integration (multi-stage) → reproduction/golden (pilot oracles ⓖ) → model-validation (statistical) → replay (channels) → deployment/API → performance → operational drills → shadow/UAT.
- **Fixtures:** (a) `fixtures/synthetic_fleet.py` — parameterized truck generator (healthy flat, GED-storm, voltage-sag, heartbeat-only, GED-absent, silent-gap profiles); (b) **pilot goldens** — the 25-truck frozen dataset and its published numbers are immutable oracles; (c) hand-computed micro-fixtures (≤20 rows) for every formula.
- **Tolerances:** features ≤1e-9 relative; AUROC exact to 4 dp (integer pair counts must match exactly); monetary ₹ to the rupee.
- **CI cadence:** PR = lint + unit + component (<10 min). Nightly = full reproduction + integration on pilot data. Weekly = alert replay + performance suite. Pre-promotion = everything + model-validation battery.
- **Coverage bars:** every cleaning rule ≥1 dedicated test; every registry feature has fixture + golden + walk-forward tests; every alert channel has fire/no-fire/boundary/replay tests; line coverage ≥85 % on `src/` (informational, not a substitute for the above).

### 17.1 Suite T-ING — Ingestion & conversion

| ID | Case | Setup / input | Expected |
|---|---|---|---|
| ING-01 | Column-order permutation | failed-file order vs healthy order (GED/VSI/SMA ↔ VSI/SMA/GED) | identical silver values per column (name-based read proven) |
| ING-02 | 8-column healthy file | no SALEDATE/JCOPENDATE/Failure_type | columns created as null; no crash; schema gate passes healthy contract |
| ING-03 | Row parity | each CSV → bronze | row counts equal, per file |
| ING-04 | Null parity | per column | CSV empty-string count == bronze null count |
| ING-05 | Timestamp span parity | per file | min/max timestamp equal CSV↔bronze |
| ING-06 | Compression gate | pilot-scale file | ratio ≥8:1 (failed cohort), ≥6:1 (healthy); measured baseline 9.93:1 / 6.40:1 |
| ING-07 | Idempotent rerun | same file, second run | manifest skip; bronze byte-identical row counts |
| ING-08 | Duplicate delivery | same content, new filename | hash dedupe; single bronze copy; manifest logs both |
| ING-09 | Truncated CSV | cut mid-row | hard gate fail; file quarantined; nonzero exit; nothing partial in bronze |
| ING-10 | Bad timestamps | 6 % unparseable | hard gate; 4 % → rows quarantined + soft gate pass |
| ING-11 | Empty / header-only file | — | graceful no-op; manifest entry with rows=0 |
| ING-12 | Unknown extra column | feed adds `NEWSIG` | ingested to bronze, logged as schema drift; silver unaffected |
| ING-13 | Timestamp format | `2024-01-31T00:07:46.000Z` | parsed to tz-naive `timestamp[us]` per IST policy |
| ING-14 | Leading all-null block | pilot head pattern (sensors empty, metadata present) | rows preserved in bronze; no row skipped |
| ING-15 | gz delivery | same file gzipped | identical bronze output |
| ING-16 | VIN namespacing | raw `VIN7` in ALT feed | stored as `VIN7_ALT`; cross-vertical path access fails (see E2E-07) |

### 17.2 Suite T-CLN — Cleaning rules (R1–R15)

| ID | Rule | Case | Expected |
|---|---|---|---|
| CLN-01 | R2 | mixed good/bad timestamps | bad → quarantine with reason code; good parsed |
| CLN-02 | R3 | inject 500 NaT rows | dropped BEFORE sort; counter == 500 |
| CLN-03 | R4 | 10 exact duplicate rows | 9 dropped; keep-first stable |
| CLN-04 | R5 | same (VIN,ts), different VSI | keep-first (frozen); counter increments |
| CLN-05 | R6 | raw VSI 140 / 28 / null | 28.0 / 28.0 unchanged / null |
| CLN-06 | R6 boundary | raw VSI 36.0 / 36.1 | unchanged / ×0.2 (contract behavior, boundary pinned) |
| CLN-07 | R7 | 65535 in CSP/RPM/ANR; −5000 ANR; 0,255 VSI | all → null; per-sentinel counters |
| CLN-08 | R8 | boundary battery for all 6 signals from config table | exact frozen inclusion/exclusion pinned to pilot outputs |
| CLN-09 | R8 | GED=5, SMA=2 | → null + counter |
| CLN-10 | R9 | all-null-sensor rows | kept in silver; excluded from aggregate masks |
| CLN-11 | R10 | dt = 5 s / 900 s / 2000 s / boundaries 15/600/1200 s | burst / heartbeat / gap; boundaries as configured |
| CLN-12 | R11 | synthetic VINs with dt_p99 = 6 s vs 900 s | families `continuous_transmit` / `rest_heartbeat` |
| CLN-13 | R12 | truck ends 15 d before fleet max / 13 d | flagged / not flagged |
| CLN-14 | R13 | GED null 96 % / 94 % | `ged_absent` / not |
| CLN-15 | R14 | VSI daily median 19 V | plausibility flag raised |
| CLN-16 | R15 | inject NaT only into healthy batch | asymmetry sentinel fires (KS p<0.01); modeling freeze flag set |
| CLN-17 | — | determinism | same input twice → byte-identical silver |
| CLN-18 | — | ledger arithmetic | rows_in == rows_out + Σ(drop counters), per (VIN, ingest_date) |
| CLN-19 | — | rule order | scaling (R6) before sentinel (R7) before range (R8) — pipeline order test with a value that resolves differently if reordered |

### 17.3 Suite T-LBL — Labels & spells

| ID | Case | Expected |
|---|---|---|
| LBL-01 | failed VIN without job card | excluded from failed cohort; report entry |
| LBL-02 ⓖ | pilot gap table | VIN3_F_ALT gap = 66 d; VIN1_F = 11 d; VIN9_F = 2 d; 7/10 zero |
| LBL-03 | gap 31 d | review-queue entry created; label not silently shifted |
| LBL-04 | spell build | failed: SALEDATE→JCOPENDATE event=1; healthy: censored at last telemetry |
| LBL-05 | healthy without SALEDATE | left-truncation flag + first-telemetry proxy |
| LBL-06 | two failure events, replacement confirmed / unconfirmed | two spells with reset / one spell + warning |
| LBL-07 | usage integration | 2 h @ 50 km/h synthetic → est_km 100 ±1 %; heartbeat dt clipped at 900 s in engine-h |
| LBL-08 | ledger versioning | new job card bumps vN; training pinned to vN−1 reproduces old outputs |
| LBL-09 | TTF outlier 150 d | review queue; QA report lists it |
| LBL-10 | JCOPENDATE < SALEDATE | hard error, run fails |
| LBL-11 ⓖ | pilot TTF stats | median 601 d, range 472–673 reproduced from ledger |

### 17.4 Suite T-CAC — Daily cache (ALT)

| ID | Case | Expected |
|---|---|---|
| CAC-01 | state masks at RPM 0/500/700/1000/1800/2000 | engine_off/idle/cruise(700)/cruise/cruise(1800)/heavy — boundaries pinned to pilot |
| CAC-02 | daily VSI stats on 20-row fixture | mean/std/min/max/median/p05/p95/skew match hand-computed values |
| CAC-03 | GED transition counts on synthetic sequence `0,0,2,2,2,0,3,2` | counts (0→0,0→2,2→2,2→0,0→3,3→2) exact; `ged2_maxrun` = 3 |
| CAC-04 | incremental == rebuild | `append_day(D)` result equals full-rebuild row for D |
| CAC-05 | dt-weighting | 12 burst rows @28 V + 96 heartbeat rows @20 V → daily mean per the decided §5.6 rule (test pins the decision) |
| CAC-06 | zero-valid-row day | no cache row emitted (not zero-filled) |
| CAC-07 | crank recovery input | synthetic crank at 09:00, VSI recovers to 27 V after 42 s → `crank_recovery_t` = 42 ±5 s (sampling) |
| CAC-08 ⓖ | pilot cache reproduction | 25 trucks: cache tables byte-equal to pilot daily caches |

### 17.5 Suite T-FEA — Features

| ID | Case | Expected |
|---|---|---|
| FEA-01 | `vsi_std_ratio_30d` fixture | last30 std mean 0.8, first60 0.4 → 2.0; denominator 0 → NaN |
| FEA-02 | `vsi_dominant_freq` on 7-day sine, 56 d | 1/7 c/d; 13 d history → NaN; DC component excluded |
| FEA-03 | `vsi_spectral_entropy` | constant series → ~0; white noise → ~1 (±0.05); <14 d → NaN |
| FEA-04 | `bat_charge_delta_trend_right` | constructed cruise−rest delta ramp slope 0.01 V/d → 0.01; <10 valid d → NaN |
| FEA-05 | `vsi_range_trend_last30d` | linear range ramp → exact OLS slope; <5 d → NaN |
| FEA-06 | `progressive_drift` | constant +0.5 V offset over 100 d vs 60-d baseline → hand value; <10 d → NaN |
| FEA-07 ⓖ | pilot reproduction | 6 features × 25 VINs ≤1e-9 relative error |
| FEA-08 | walk-forward invariance | append 30 future days → snapshot(D) values unchanged |
| FEA-09 | ban-list enforcement | registering `n_weeks`-family feature raises; CI fails |
| FEA-10 | leak-gate meta-test | deliberately leaky feature (=observation length) → auto-REJECT by `f05_leak_gates` |
| FEA-11 | proxy-ρ gate | synthetic feature with ρ=0.6 vs span → REJECT; ρ=0.3 → pass |
| FEA-12 | GED-Markov candidates | fixture transition streams → P(2→2), dwell, entry-rate exact; ged_absent VIN → NaN |
| FEA-13 | NaN policy | feature NaN in cache stays NaN in snapshot (imputation only inside model folds) |
| FEA-14 | registry↔implementation parity | formula strings hash-pinned; changing code without registry update fails CI |

### 17.6 Suite T-MDL — Models & harness

| ID | Case | Expected |
|---|---|---|
| MDL-01 | fold hygiene | spies assert imputer/scaler/calibrator `.fit` called only with train indices, every fold |
| MDL-02 ⓖ | champion reproduction | pilot LOVO AUROC 0.9267 exact; 139/150 concordant pairs; FN=VIN5_F (0.2799), FP=VIN3_NF (0.4906); Youden 0.4456 |
| MDL-03 | grouped CV integrity | no VIN in both train and test of any fold, all repeats |
| MDL-04 | stratification | per-fold failed count within ±2 of expectation per stratum |
| MDL-05 | determinism | same seed → identical folds, identical AUROC |
| MDL-06 | prequential causality | cut-k score unchanged when post-cut data appended |
| MDL-07 | permuted-label control | 20 permutations → AUROC mean ∈[0.45,0.55] |
| MDL-08 | exporter-fingerprint control | model on (NaT count, column order, cadence mix, dup rate) only → AUROC <0.65 else HARD ALARM |
| MDL-09 | year-shuffled control | AUROC ≈0.5 |
| MDL-10 | calibration math | synthetic logits → Platt/isotonic slope & CITL match reference implementation |
| MDL-11 | prior-shift correction | 50 %→5 % prevalence: corrected PPV matches closed-form on fixture confusion matrix |
| MDL-12 | referees | DeLong & Wilcoxon on known score sets → published p-values (reference vectors) |
| MDL-13 | thresholds | Youden on fixture ROC = expected; cost-optimal minimizes ₹ on fixture cost grid |
| MDL-14 | challenger smoke | each family trains on 50-truck synthetic <5 min; card fields complete |
| MDL-15 | survival machinery | KM matches lifelines on textbook data; person-period expansion (10-wk spell + event → 10 rows, last event=1); censored → all 0 |
| MDL-16 | left truncation | late-entry spell contributes no risk before entry week |
| MDL-17 | interval coverage | synthetic P10–P90 with known 80 % coverage → metric 0.80 ±0.01 |
| MDL-18 ⓖ | RUL dummy referee | pilot survival-RUL vs fleet clock reproduces `NO_IMPROVEMENT` (MAE 125–155 vs 49.7; signed-rank p≈0) |
| MDL-19 | anomaly episode scorer | synthetic weekly alarm stream → episodes merged correctly; ep/truck-yr exact |
| MDL-20 | spec round-trip | save→load spec JSON → identical predictions on fixture; required fields enforced |
| MDL-21 | promotion evaluator | synthetic better/worse/equal challengers → G1–G8 truth-table verdicts |

### 17.7 Suite T-ALR — Alert channels (ALT)

| ID | Case | Expected |
|---|---|---|
| ALR-01 | GED storm boundary | ged2_cnt 199/day → silent; 200 → fire; counts accumulate across hourly scans of one day |
| ALR-02 | ged_absent exemption | VIN on R13 roster with 500 counts → excluded + coverage report entry (no fire) |
| ALR-03 ⓖ | storm replay | pilot: exactly VIN1_F (T−21 d) and VIN10_F (T−1 d) fire; 0/15 NF |
| ALR-04 | early-watch voting | 2/5 votes → fire; 1/5 → silent; each heuristic has its own boundary fixture |
| ALR-05 ⓖ | early-watch replay | pilot: 3/10 F, 0/15 NF reproduced |
| ALR-06 | tier debounce | score >0.37 for 13 d → no upgrade; 14 d → upgrade (deadband 0.35+0.02) |
| ALR-07 | tier hysteresis | drop below boundary 13 d → tier held; 14 d → downgrade |
| ALR-08 | dedupe | same (VIN, channel, week) twice → one alert |
| ALR-09 | payload contract | alert JSON schema: model_version, snapshot_date, evidence fields — schema-validated |
| ALR-10 | replay determinism | same registry version + date range → byte-identical alert set |
| ALR-11 | volume gate simulation | synthetic 500-truck score distribution at chosen thresholds → alerts/100/mo ≤ 2.0 |
| ALR-12 | escalation | C2 fire → same-day webhook queued; RED sustained 3 wk → job-card recommendation artifact generated |

### 17.8 Suite T-E2E — Integration

| ID | Case | Expected |
|---|---|---|
| E2E-01 | 3-truck synthetic mini-fleet, 90 d | landing→alerts full DAG; planted GED-storm truck fires C2, sag truck reaches AMBER, healthy stays GREEN |
| E2E-02 ⓖ | pilot end-to-end | raw 96,759,243 rows → clean 91,144,087 (dups removed 5,615,156); features ≤1e-9; AUROC 0.9267; channel replay table reproduced |
| E2E-03 | day-2 increment | one new day → only that day's partitions/caches touched; scores refresh; chain < runtime budget |
| E2E-04 | backfill runner | staged subset backfill completes; manifest complete; quarantine empty or dispositioned |
| E2E-05 | mid-DAG kill | kill converter mid-write → rerun resumes; no partial partitions (temp+rename verified) |
| E2E-06 | schema evolution | add instrumentation column → bronze schema v2 registered; silver v1 outputs unchanged |
| E2E-07 | vertical isolation | ALT job identity cannot read SM paths (permission denied test) |

### 17.9 Suite T-API/DEP — Serving & deployment

| ID | Case | Expected |
|---|---|---|
| API-01 | risk endpoint | schema-complete response; unknown VIN → 404; stale snapshot → `stale=true` flag |
| API-02 | webhook idempotency | 5xx then retry → exactly-once effect via idempotency key |
| API-03 | authZ | unauthenticated → 401; cross-tenant VIN → 403 |
| API-04 | TruckConnect outage | alerts queue; backfill on recovery preserves order |
| DEP-01 | reproducible container | pinned digests; signed image passes admission policy |
| DEP-02 | IaC idempotency | second `apply` = no changes; RBAC: scoring job cannot write landing |
| DEP-03 | secret rotation | rotate API key in Key Vault → next run picks up, no redeploy |
| DEP-04 | Spot eviction drill | evict mid-retrain → checkpoint resume; result equals uninterrupted seeded run |
| DEP-05 | rollback drill | promote → demote → serving reverts; alerts stamped with reverted model_version |

### 17.10 Suite T-PRF — Performance

| ID | Case | Budget |
|---|---|---|
| PRF-01 | daily increment (~2.75 M rows) | ingest→alerts chain < 30 min on D8ds_v5 |
| PRF-02 | full-history rebuild (~1.65 B rows) | < 2 h streaming on D16ds_v5; peak RSS < 48 GB |
| PRF-03 | 3× headroom (24 h continuous 5 s synthetic) | chain < 90 min; no OOM |
| PRF-04 | hourly GED scan | < 2 min; verifies column pruning (reads GED/VSI only — assert scanned-bytes ≪ file size) |
| PRF-05 | harness | full grouped 10×5 CV champion run < 30 min CPU; challenger battery overnight |

### 17.11 Suite T-DQM — Monitors & MLOps

| ID | Case | Expected |
|---|---|---|
| DQM-01 | PSI | synthetic shifted feature → PSI matches reference; trigger at >0.25 on ≥2 features |
| DQM-02 | calibration drift | slope 0.35 on synthetic outcomes → demote signal |
| DQM-03 | alert SPC | 2-week rate breach → freeze flag |
| DQM-04 | retrain trigger | 10th new label → trigger; 9 → none; quarterly timer fires regardless |
| DQM-05 | monitor plumbing | events land in Log Analytics with correct dimensions |
| DQM-06 | GED-coverage roster drift | VIN transitions absent→present → roster update + C2 re-inclusion after burn-in |

### 17.12 Suite T-PBT — Property-based tests (hypothesis)

| ID | Property |
|---|---|
| PBT-01 | ∀ random truck stream: silver timestamps strictly increasing, unique per VIN |
| PBT-02 | ∀ cache row: min ≤ p05 ≤ median ≤ p95 ≤ max; counts consistent with masks |
| PBT-03 | ∀ features: finite or NaN, never ±inf; invariant to row order within a day |
| PBT-04 | ∀ random future-append: past snapshots immutable (walk-forward property) |
| PBT-05 | ∀ score path: raising a week's score never lowers that week's tier |

### 17.13 Suite T-UAT — Shadow & launch

| ID | Case | Expected |
|---|---|---|
| UAT-01 | 4-week shadow | zero customer-visible alerts; internal parity report vs expectations |
| UAT-02 | soft launch | analyst acknowledgment loop works; alert→job-card round trip recorded |
| UAT-03 | holdout unseal | hash verified pre/post; G7 (holdout within 0.05 of CV) evaluated and minuted |
| UAT-04 | runbook drills | one tabletop per runbook (ingest failure, API outage, rollback) before launch |

### 17.14 Out of scope (tested by contract, not by us)

TruckConnect internal UI rendering; DICV job-card upstream correctness (we test ingest + QA, not their ERP); telematics firmware behavior (we classify and flag, we don't certify); sensor physical calibration.

### 17.15 Traceability

ING/CLN → P1/P3 XGs · LBL → P2 XG · CAC/FEA → P3 XG (goldens) · MDL → P4/P6 XGs · ALR → P5 XG · E2E → P3/P4/P10 · API/DEP/PRF/DQM/UAT → P10/P11 XGs. The nightly reproduction suite (all ⓖ cases) is the program's regression backbone: **any red ⓖ test blocks every promotion and every deploy.**

---

## 18. Appendices

### A. Pilot artifacts this plan inherits (contract of record)
- Frozen model: `V5.2_ALT/models/classification/V10.5.3_20_5_ALT_ridge_spec.json`; features `V5.2_ALT/src/V5.2_20_5_ALT_feature_engineering.py` (FEATURE_REGISTRY); pipeline `V5.2_20_5_ALT_data_pipeline.py`.
- Validation dossier: `V11.2_ALT/reports/V11.2_ALT_technical_validation_report.md` (+ metric methodology, debounce note, QnA).
- GED: `V12_ALT_GED/reports/V12_ALT_GED_investigation_report.md` (+ results/2c_*).
- RUL honesty: `V10.6.2_ALT/reports/V10.6.2_ALT_customer_report.md`.
- Failure-mode split: `docs/2026-06-09-06-48-43-alternator-failure-mode-split.md`.
- Azure/costing: `Plan/DICV_Azure_Deployment_Retraining_and_GPU_Plan_v2.pdf`, `Plan/DICV_Azure_GPU_OnDemand_Compute_Costing_v1.pdf`, `deliverables/DICV_TruckConnect_Predictive_Intelligence_Commercial_QA.docx`.
- FM-pilot gate spec: `docs/superpowers/specs/2026-06-24-clutch-first-fm-pilot-design.md`.
- Evidence-stack graph design: `V11.2_ALT` per-VIN 5-panel generators.

### B. Champion config template (`alt500_champion.yaml`, excerpt)
```yaml
model: {type: ridge_classifier, alpha: 1.0, random_state: 42}
features: [vsi_std_ratio_30d, vsi_dominant_freq, vsi_spectral_entropy,
           bat_charge_delta_trend_right, vsi_range_trend_last30d, progressive_drift]
preprocess: {impute: train_median, scale: standard, clip: none}
calibration: {method: isotonic, fit_on: grouped_oof}
validation: {scheme: grouped_stratified_kfold, k: 10, repeats: 5,
             groups: vin, strata: [label, failure_mode, firmware_family],
             temporal_holdout_months: 3, prequential_max_rewind_weeks: 26}
thresholds: {derive: [youden, cost_optimal], field_prevalence: [0.02, 0.05, 0.10],
             fp_budget_per_100_trucks_month: 2.0}
tiers: {green_lt: 0.35, red_ge: 0.55, debounce: {deadband: 0.02, dwell_days: 14}}  # re-derived P4/P5
channels:
  ged_storm: {rule: ged2_cnt_daily >= 200, scan: hourly, exempt: ged_absent_roster}
  early_watch: {votes_required: 2, heuristics: [vsi_ceiling, vsi_resid_mean,
                crank_recovery_t, resting_vsi_mean, ged_churn]}
labels: {failure_date: jcopendate, gap_review_days: 30, ledger_version: pinned}
```

### C. Verification quick-list (used at every XG)
1. `pytest tests/ -k reproduction` → pilot byte-exact gates.
2. DQ ledger: zero hard-gate failures for the evaluated window.
3. Negative-control battery report ≈ 0.5 AUROC (±0.05).
4. Metrics contract table rendered into the model card (no missing cells).
5. Alert replay on retrospective drop within FP budget.

---
*Prepared 2026-07-02. Baseline commit 7b59ba1 (branch v11.1-alt). All pilot numbers cited from the frozen artifacts in Appendix A; all 500-vehicle numbers are targets/assumptions pending the data drop and are labeled as such.*
