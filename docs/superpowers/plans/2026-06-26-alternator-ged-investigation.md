# Alternator GED Investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Iteration name:** **V12** (the complete GED investigation iteration).

**Goal:** Deliver a comprehensive, honest GED investigation (literature + empirical fleet analysis + offline prognostic trial) as a technical R&D dossier, reframed around what is feasible given the 6-signal data.

**Architecture:** A new `V12_ALT_GED/` package reuses the existing labeled per-VIN parquets and the frozen 0.9267 LOVO protocol. Phase 0 builds a per-VIN daily GED cache; Phases 2–4 read that cache (never raw 96M rows); Phase 1 runs the `deep-research` skill for external literature; Phase 3B extends the existing `build_load_reference`/`load_residual` heuristics; Phase 4 reproduces the LOVO baseline in a self-contained harness then tests candidate GED features against it. Phase 5 synthesizes the report + graphs + sensor-gap recommendation.

**Tech Stack:** Python via `py -3` (Polars-first; pandas + scikit-learn for modeling), pytest (dynamic `importlib` module loading, synthetic DataFrames), matplotlib for graphs.

**Spec:** `docs/superpowers/specs/2026-06-26-alternator-ged-investigation-design.md`

---

## Ground-truth anchors (verified — used as reconciliation targets)
- Fleet: `FAILED_VINS = [VIN1_F_ALT … VIN10_F_ALT]` (10), `NONFAILED_VINS = [VIN1_NF_ALT … VIN15_NF_ALT]` (15). Source: `V5.2_ALT/src/V5.2_20_5_ALT_config.py:73-77`.
- GED states `{0,1,2,3}`: 0=No disturbance (>99.7%), 1=Not allowed (**never observed**), 2=Disturbance (485× enrichment), 3=Signal not available (CAN fault; up to 12–29% in some NF VINs).
- GED=2 totals: VIN1_F_ALT ≈ 82,357; VIN10_F_ALT ≈ 2,897; **other 8 failed = 0**. NF all trace (≤315), 0/15 fire.
- Emergency channel: `ged2_cnt >= 200` → fires 2/10 F (VIN1 lead 21d, VIN10 lead 1d), 0/15 NF. Threshold `GED_EMERGENCY_DAILY_COUNT_MIN=200` (`V10.6.2_ALT/src/V10.6.2_ALT_config.py:150`).
- Baseline: RidgeClassifier(alpha=1.0), LOVO **pooled-OOF AUROC = 0.9267**, 6 features `FAMILY_A = [vsi_std_ratio_30d, vsi_dominant_freq, vsi_spectral_entropy, bat_charge_delta_trend_right, vsi_range_trend_last30d, progressive_drift]` (`V11.2_ALT/src/V11_2_ALT_common.py:13-14`; frozen in `V5.2_ALT/models/classification/V10.5.3_20_5_ALT_ridge_spec.json`).
- 25-row feature matrix: `V5.2_ALT/features/V5.2_20_5_ALT_selected_features.csv` (cols `truck_id,VIN,failed,` + 17 features; already contains FAMILY_A + `ged2_daily_rate_log` + `ged2_max_burst_24h`).
- Per-VIN cleaned parquets: `V5.2_ALT/features/parquets/V5.2_20_5_ALT_{VIN_LABEL}.parquet` (sentinel-filtered, VSI ×0.2 applied, GED preserved as-is incl. nulls).
- JCOPENDATE dict + `jcopendate_failure_age(t0, jcopendate)`: `V11.2_ALT/src/V11_2_ALT_common.py:21-44`.
- ⚠️ Reproduce AUROC with sklearn `roc_auc_score` on **pooled** OOF probabilities — NOT `auroc_from_scores` (Mann-Whitney) in `common.py`.
- ⚠️ The data pipeline preserves GED nulls (does NOT impute null→3). The KT/OEM guideline says impute null→3. Phase 2E quantifies this; analyses report rates both raw and null→3.

## File structure
```
V12_ALT_GED/
  src/
    ged_common.py              # paths, VIN lists, per-VIN loader, regime bands, JCOPENDATE re-export
    build_ged_daily_cache.py   # Phase 0: per-VIN-per-day GED cache (counts/regime/co-occur) + transition counts
    analysis_2a_quantization.py# Phase 2A: ordinality / quantization test
    analysis_2b_occupancy.py   # Phase 2B: per-VIN occupancy table (reconciles to anchors)
    analysis_2c_markov.py      # Phase 2C: transition matrices + dwell times, F vs NF
    analysis_2d_triggers.py    # Phase 2D: P(GED=2 onset | signals, regime) trigger model
    analysis_2e_nulls.py       # Phase 2E: null-informativeness
    proxy_3b_regulation.py     # Phase 3B: VSI regulation-effort proxy (extends build_load_reference)
    features_4_ged.py          # Phase 4: new per-VIN scalar GED prognostic features
    trial_4_lovo.py            # Phase 4: reproduce 0.9267 + test candidate features
    graphs_5.py                # Phase 5: figures
  tests/
    test_ged_common.py
    test_build_ged_daily_cache.py
    test_analysis_2c_markov.py
    test_analysis_2d_triggers.py
    test_proxy_3b_regulation.py
    test_features_4_ged.py
    test_trial_4_lovo.py
  results/                     # ged_daily_cache.parquet, *.csv/*.json outputs
  graphs/                      # *.png / *.pdf
  literature/                  # Phase 1 deep-research output
  reports/
    V12_ALT_GED_investigation_report.md
```

## Conventions (all tasks)
- Run scripts: `py -3 V12_ALT_GED/src/<script>.py`. Run tests: `py -3 -m pytest V12_ALT_GED/tests/<file>.py -v`.
- Tests load modules via dynamic `importlib` (no conftest), pass synthetic DataFrames, assert with float tolerance — matching `V11_ALT_heuristics/tests/test_features.py`.
- Commit after each task with `git add <task files> && git commit -m "<msg>"` ending with the project's Co-Authored-By trailer.

---

## Task 0.1: Scaffold package + `ged_common.py`

**Files:**
- Create: `V12_ALT_GED/src/ged_common.py`
- Create: `V12_ALT_GED/tests/test_ged_common.py`
- Create (empty dirs via `.gitkeep`): `V12_ALT_GED/results/.gitkeep`, `V12_ALT_GED/graphs/.gitkeep`, `V12_ALT_GED/literature/.gitkeep`, `V12_ALT_GED/reports/.gitkeep`

- [ ] **Step 1: Inspect the real per-VIN parquet schema** (discovery — do not guess columns)

Run:
```
py -3 -c "import polars as pl; df=pl.read_parquet(r'V5.2_ALT/features/parquets/V5.2_20_5_ALT_VIN1_F_ALT.parquet'); print(df.columns); print(df.head(3))"
```
Record the actual datetime column name (`timestamp` vs `DATETIME`), and confirm presence of `GED,VSI,RPM,ANR,CSP,SMA` and a days-to-failure column. If the filename pattern differs, `ls V5.2_ALT/features/parquets/` and adapt `PARQUET_TMPL` below.

- [ ] **Step 2: Write `ged_common.py`** (adapt `DT_COL`/`PARQUET_TMPL`/`DTF_COL` to Step-1 findings)

```python
"""Shared constants + per-VIN loader for the V12 GED investigation."""
import pathlib
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[2]          # repo root
PARQUET_DIR = ROOT / "V5.2_ALT" / "features" / "parquets"
PARQUET_TMPL = "V5.2_20_5_ALT_{vin}.parquet"                 # confirm in Task 0.1 Step 1
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
DAILY_CACHE = RESULTS / "ged_daily_cache.parquet"

FAILED_VINS = [f"VIN{i}_F_ALT" for i in range(1, 11)]        # 10
NONFAILED_VINS = [f"VIN{i}_NF_ALT" for i in range(1, 16)]    # 15
ALL_VINS = FAILED_VINS + NONFAILED_VINS                      # 25

DT_COL = "timestamp"        # confirm in Task 0.1 Step 1 (else "DATETIME")
DTF_COL = "DAYS_TO_FAILURE" # confirm; may be absent for NF

GED_STATES = [0, 1, 2, 3]

# KT-anchored operating regimes (RPM rev/min; CSP km/h; SMA crank flag)
def regime_expr() -> pl.Expr:
    """Return a polars Expr classifying each row into an operating regime."""
    return (
        pl.when(pl.col("SMA") == 1).then(pl.lit("crank"))
        .when(pl.col("RPM").is_null() | (pl.col("RPM") <= 0)).then(pl.lit("engine_off"))
        .when(pl.col("RPM") <= 700).then(pl.lit("idle"))
        .when(pl.col("RPM") <= 1800).then(pl.lit("cruise"))
        .otherwise(pl.lit("heavy"))
        .alias("regime")
    )

def load_vin(vin: str) -> pl.DataFrame:
    """Load one cleaned per-VIN parquet, add `day` (date) and `regime`."""
    df = pl.read_parquet(PARQUET_DIR / PARQUET_TMPL.format(vin=vin))
    df = df.with_columns(pl.col(DT_COL).cast(pl.Datetime).dt.date().alias("day"))
    df = df.with_columns(regime_expr())
    return df

def is_failed(vin: str) -> bool:
    return vin.endswith("_F_ALT")
```

- [ ] **Step 3: Write the test**

```python
import importlib.util, pathlib
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
C = _load("ged_common")

def test_vin_lists():
    assert len(C.FAILED_VINS) == 10 and len(C.NONFAILED_VINS) == 15
    assert C.ALL_VINS[0] == "VIN1_F_ALT" and C.ALL_VINS[-1] == "VIN15_NF_ALT"
    assert C.is_failed("VIN1_F_ALT") and not C.is_failed("VIN1_NF_ALT")

def test_load_vin_has_day_and_regime():
    df = C.load_vin("VIN1_F_ALT")
    assert "day" in df.columns and "regime" in df.columns
    assert df["regime"].is_in(["crank","engine_off","idle","cruise","heavy"]).all()
    assert {"GED","VSI","RPM","ANR","CSP","SMA"}.issubset(set(df.columns))
```

- [ ] **Step 4: Run tests** — `py -3 -m pytest V12_ALT_GED/tests/test_ged_common.py -v` → Expected: PASS (2 tests).
- [ ] **Step 5: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): scaffold package + per-VIN loader (ged_common)"`

---

## Task 0.2: Build per-VIN daily GED cache

**Files:**
- Create: `V12_ALT_GED/src/build_ged_daily_cache.py`
- Create: `V12_ALT_GED/tests/test_build_ged_daily_cache.py`
- Output: `V12_ALT_GED/results/ged_daily_cache.parquet`

Cache schema (one row per VIN per calendar day): `vin, failed, day, dtf, n_rows, ged_null, ged_cnt_0, ged_cnt_1, ged_cnt_2, ged_cnt_3, ged2_rate, regime_<r>_rows, ged2_in_<regime>, vsi_mean, vsi_p10, vsi_p50, vsi_p90, vsi_when_ged2_mean, rpm_mean, anr_mean, csp_mean, sma_sum`.

- [ ] **Step 1: Write the failing test** (uses a synthetic frame to validate the pure aggregation function `daily_from_df`)

```python
import importlib.util, pathlib, datetime as dt
import polars as pl
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
B = _load("build_ged_daily_cache")

def test_daily_from_df_counts_states_and_rate():
    d = dt.date(2025, 1, 1)
    df = pl.DataFrame({
        "day":[d,d,d,d,d], "GED":[0,2,2,3,None],
        "VSI":[28.0,24.0,25.0,28.0,28.0], "RPM":[800,800,800,0,800],
        "ANR":[100.0]*5, "CSP":[40.0]*5, "SMA":[0,0,0,0,0],
        "regime":["cruise","cruise","cruise","engine_off","cruise"],
    })
    out = B.daily_from_df(df, vin="VIN1_F_ALT", failed=True).sort("day")
    r = out.row(0, named=True)
    assert r["n_rows"] == 5 and r["ged_null"] == 1
    assert r["ged_cnt_2"] == 2 and r["ged_cnt_0"] == 1 and r["ged_cnt_3"] == 1
    # ged2_rate denominator = non-null GED (4) -> 2/4
    assert abs(r["ged2_rate"] - 0.5) < 1e-9
    assert abs(r["vsi_when_ged2_mean"] - 24.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails** — `py -3 -m pytest V12_ALT_GED/tests/test_build_ged_daily_cache.py -v` → Expected: FAIL (`daily_from_df` not defined).

- [ ] **Step 3: Implement `build_ged_daily_cache.py`**

```python
"""Phase 0: per-VIN per-day GED aggregate cache."""
import importlib.util, pathlib
import polars as pl
_SRC = pathlib.Path(__file__).resolve().parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
C = _load("ged_common")

REGIMES = ["crank","engine_off","idle","cruise","heavy"]

def daily_from_df(df: pl.DataFrame, vin: str, failed: bool) -> pl.DataFrame:
    g = df.group_by("day").agg([
        pl.len().alias("n_rows"),
        pl.col("GED").is_null().sum().alias("ged_null"),
        *[(pl.col("GED") == s).sum().alias(f"ged_cnt_{s}") for s in C.GED_STATES],
        pl.col("GED").filter(pl.col("GED").is_not_null()).count().alias("_ged_obs"),
        pl.col("VSI").mean().alias("vsi_mean"),
        pl.col("VSI").quantile(0.10).alias("vsi_p10"),
        pl.col("VSI").quantile(0.50).alias("vsi_p50"),
        pl.col("VSI").quantile(0.90).alias("vsi_p90"),
        pl.col("VSI").filter(pl.col("GED") == 2).mean().alias("vsi_when_ged2_mean"),
        pl.col("RPM").mean().alias("rpm_mean"),
        pl.col("ANR").mean().alias("anr_mean"),
        pl.col("CSP").mean().alias("csp_mean"),
        pl.col("SMA").sum().alias("sma_sum"),
        *[(pl.col("regime") == r).sum().alias(f"regime_{r}_rows") for r in REGIMES],
        *[((pl.col("GED") == 2) & (pl.col("regime") == r)).sum().alias(f"ged2_in_{r}") for r in REGIMES],
    ])
    g = g.with_columns(
        pl.when(pl.col("_ged_obs") > 0).then(pl.col("ged_cnt_2") / pl.col("_ged_obs"))
        .otherwise(None).alias("ged2_rate")
    ).drop("_ged_obs")
    g = g.with_columns([pl.lit(vin).alias("vin"), pl.lit(failed).alias("failed")])
    if C.DTF_COL in df.columns:
        dtf = df.group_by("day").agg(pl.col(C.DTF_COL).first().alias("dtf"))
        g = g.join(dtf, on="day", how="left")
    else:
        g = g.with_columns(pl.lit(None, dtype=pl.Int32).alias("dtf"))
    return g

def main():
    frames = []
    for vin in C.ALL_VINS:
        df = C.load_vin(vin)
        frames.append(daily_from_df(df, vin, C.is_failed(vin)))
        print(f"{vin}: {df.height} rows -> {frames[-1].height} days")
    cache = pl.concat(frames, how="diagonal_relaxed").sort(["vin","day"])
    C.DAILY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache.write_parquet(C.DAILY_CACHE)
    # reconciliation print
    tot = cache.group_by("vin").agg(pl.col("ged_cnt_2").sum().alias("ged2")).sort("vin")
    print(tot)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test to verify pass** — `py -3 -m pytest V12_ALT_GED/tests/test_build_ged_daily_cache.py -v` → Expected: PASS.

- [ ] **Step 5: Build the real cache and reconcile to anchors**

Run: `py -3 V12_ALT_GED/src/build_ged_daily_cache.py`
Expected: prints per-VIN day counts; final table shows **VIN1_F_ALT ged2 ≈ 82,357**, **VIN10_F_ALT ≈ 2,897**, the other 8 failed = **0**, and all NF small (≤~315). If VIN1/VIN10 are within ±2% and the 8 zeros hold, the cache is validated. If not, recheck `DT_COL`/sentinel handling in `ged_common` before proceeding. Record any deviation in the report's data-quality notes.

- [ ] **Step 6: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): per-VIN daily GED cache + anchor reconciliation"`

---

## Task 1.1: Phase 1 — external literature validation (deep-research skill)

**Files:**
- Create: `V12_ALT_GED/literature/V12_ALT_GED_literature_findings.md` (deep-research output)

- [ ] **Step 1: Invoke the `deep-research` skill** with this exact question set (pass as args):

> "For a heavy-duty truck (BharatBenz/Daimler 24V CAN) alternator, what does a CAN signal named 'State of Alternator Excitation' (GED) with discrete values {0=No disturbance, 1=Not allowed, 2=Disturbance, 3=Signal not available} physically represent? Specifically: (1) Is this a standard **SAE J1939** measured-status SPN using the 2-bit encoding convention (00=normal/off, 01=command/enabled, 10=error, 11=not available/not supported)? Identify the candidate SPN/PGN if one matches. (2) How do Bosch / Valeo / Denso / Prestolite / Mitsubishi and Mercedes-Benz/Daimler charging architectures report excitation/field state — as a status enum or a PWM duty-cycle value? (3) In LIN-controlled smart alternators, is excitation reported as a continuous duty cycle? (4) Does any OEM expose alternator field/excitation current as a continuous CAN signal? Provide citations and a confidence level per claim. Flag where external sources contradict the enum interpretation."

- [ ] **Step 2: Adversarially verify the load-bearing claim.** The J1939 2-bit-status hypothesis is decision-relevant — require the deep-research pass to either cite a concrete J1939 SPN/encoding source or explicitly mark it "plausible, unconfirmed." Do NOT let "GED is a duty cycle" enter the findings unless a source directly supports it for a status-named signal.

- [ ] **Step 3: Save** the synthesized, cited findings to `V12_ALT_GED/literature/V12_ALT_GED_literature_findings.md` with a per-claim confidence column and a "discrepancies vs KT enum" section.

- [ ] **Step 4: Commit** — `git add V12_ALT_GED/literature/ && git commit -m "docs(v12-alt-ged): external literature validation of GED semantics"`

---

## Task 2.1: Phase 2A — quantization / ordinality test

**Files:**
- Create: `V12_ALT_GED/src/analysis_2a_quantization.py`
- Output: `V12_ALT_GED/results/2a_ordinality.json`

Logic: if GED were a quantized continuum, the co-occurring VSI distribution would vary **monotonically** with state order 0→2→3. Test the opposite: states are a categorical status code. For each state s∈{0,2,3} compute the distribution of co-occurring per-day `vsi_mean` and `ged2_rate`-day VSI; test monotonicity (Spearman of state-rank vs median VSI) and separation (are state-3 days a CAN-fault cluster, not "more excitation than 2"?).

- [ ] **Step 1: Implement** (reads the daily cache; pure read + stats; writes JSON verdict)

```python
"""Phase 2A: is GED an ordered continuum or a status enum?"""
import importlib.util, pathlib, json
import polars as pl
from scipy.stats import spearmanr
_SRC = pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")

def main():
    df = pl.read_parquet(C.DAILY_CACHE)
    # per (vin,day) dominant state by max count among {0,2,3}
    dom = df.with_columns(
        pl.when((pl.col("ged_cnt_3")>=pl.col("ged_cnt_2")) & (pl.col("ged_cnt_3")>pl.col("ged_cnt_0"))).then(3)
        .when(pl.col("ged_cnt_2")>0).then(2).otherwise(0).alias("dom_state")
    ).filter(pl.col("vsi_mean").is_not_null())
    med = dom.group_by("dom_state").agg(pl.col("vsi_mean").median().alias("vsi_med"),
                                        pl.len().alias("n")).sort("dom_state")
    rho, p = spearmanr(dom["dom_state"].to_list(), dom["vsi_mean"].to_list())
    verdict = ("ENUM (non-monotonic / state-3 is a CAN-fault cluster)"
               if abs(rho) < 0.3 else "ORDINAL signal warrants further test")
    out = {"per_state_median_vsi": med.to_dicts(), "spearman_rho": rho, "spearman_p": p,
           "verdict": verdict}
    (C.RESULTS/"2a_ordinality.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** — `py -3 V12_ALT_GED/src/analysis_2a_quantization.py` → Expected: writes `2a_ordinality.json`; verdict expected to be ENUM (|rho|<0.3). Record the numbers for the report.
- [ ] **Step 3: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): 2A ordinality/quantization test"`

---

## Task 2.2: Phase 2B — per-VIN occupancy table

**Files:**
- Create: `V12_ALT_GED/src/analysis_2b_occupancy.py`
- Output: `V12_ALT_GED/results/2b_occupancy.csv`

- [ ] **Step 1: Implement** (per-VIN state occupancy %, null %, F-vs-NF rollup; reconciles to anchors)

```python
"""Phase 2B: per-VIN GED occupancy."""
import importlib.util, pathlib
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")

def main():
    df = pl.read_parquet(C.DAILY_CACHE)
    occ = df.group_by("vin","failed").agg([
        pl.col("ged_cnt_0").sum().alias("c0"), pl.col("ged_cnt_1").sum().alias("c1"),
        pl.col("ged_cnt_2").sum().alias("c2"), pl.col("ged_cnt_3").sum().alias("c3"),
        pl.col("ged_null").sum().alias("nnull"), pl.col("n_rows").sum().alias("ntot"),
    ]).with_columns([
        (pl.col("c2")/(pl.col("c0")+pl.col("c1")+pl.col("c2")+pl.col("c3"))).alias("ged2_frac_obs"),
        (pl.col("c3")/(pl.col("c0")+pl.col("c1")+pl.col("c2")+pl.col("c3"))).alias("ged3_frac_obs"),
        (pl.col("nnull")/pl.col("ntot")).alias("null_frac"),
    ]).sort(["failed","vin"])
    occ.write_csv(C.RESULTS/"2b_occupancy.csv")
    print(occ)
    # anchor check
    v1 = occ.filter(pl.col("vin")=="VIN1_F_ALT")["c2"].item()
    print("VIN1_F_ALT GED2:", v1, "(expect ~82357)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run + reconcile** — `py -3 V12_ALT_GED/src/analysis_2b_occupancy.py` → Expected: VIN1_F_ALT c2 ≈ 82,357; 8 failed c2 = 0; null_frac in 0.21–0.44 band. Note GED=1 must be 0 across all VINs.
- [ ] **Step 3: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): 2B per-VIN occupancy table"`

---

## Task 2.3: Phase 2C — Markov transition structure + dwell times

**Files:**
- Create: `V12_ALT_GED/src/analysis_2c_markov.py`
- Create: `V12_ALT_GED/tests/test_analysis_2c_markov.py`
- Output: `V12_ALT_GED/results/2c_transitions_failed.csv`, `2c_transitions_nonfailed.csv`, `2c_dwell.json`

Transitions are computed on **raw per-VIN samples** (not the daily cache), only between consecutive samples ≤ `MAX_GAP_S=60` apart (avoid counting across offline gaps); nulls mapped to state 3 (KT/OEM rule) for the transition view. Aggregated into a 4×4 count→row-normalized probability matrix per population.

- [ ] **Step 1: Write the failing test** (pure function `transition_counts` on a synthetic series)

```python
import importlib.util, pathlib, datetime as dt
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=_load("analysis_2c_markov")

def test_transition_counts_respects_gap_and_nulls():
    base = dt.datetime(2025,1,1,0,0,0)
    ts = [base, base+dt.timedelta(seconds=5), base+dt.timedelta(seconds=10),
          base+dt.timedelta(seconds=10000), base+dt.timedelta(seconds=10005)]
    df = pl.DataFrame({"timestamp":ts, "GED":[0,2,None,0,0]})
    cm = M.transition_counts(df, dt_col="timestamp", max_gap_s=60)
    # 0->2 counted; 2->null(=3) counted; null gap (10000s) NOT counted; 0->0 counted
    assert cm[(0,2)] == 1 and cm[(2,3)] == 1 and cm[(0,0)] == 1
    assert sum(cm.values()) == 3   # the 10000s jump is excluded
```

- [ ] **Step 2: Run test → FAIL** — `py -3 -m pytest V12_ALT_GED/tests/test_analysis_2c_markov.py -v` → Expected: FAIL (`transition_counts` missing).

- [ ] **Step 3: Implement**

```python
"""Phase 2C: GED Markov transition structure + dwell times."""
import importlib.util, pathlib, json
from collections import Counter
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")
MAX_GAP_S = 60

def transition_counts(df: pl.DataFrame, dt_col=C.DT_COL, max_gap_s=MAX_GAP_S) -> Counter:
    d = df.sort(dt_col).with_columns(pl.col("GED").fill_null(3).cast(pl.Int64).alias("g"))
    d = d.with_columns([
        (pl.col(dt_col).cast(pl.Datetime).diff().dt.total_seconds()).alias("gap_s"),
        pl.col("g").shift(1).alias("g_prev"),
    ])
    d = d.filter(pl.col("g_prev").is_not_null() & (pl.col("gap_s") <= max_gap_s))
    cm = Counter()
    for prev, cur in zip(d["g_prev"].to_list(), d["g"].to_list()):
        cm[(int(prev), int(cur))] += 1
    return cm

def matrix_from_counts(cm: Counter):
    states = [0,1,2,3]
    rows = {}
    for i in states:
        tot = sum(cm[(i,j)] for j in states)
        rows[i] = {j: (cm[(i,j)]/tot if tot else 0.0) for j in states}
    return rows

def dwell_summary(df, dt_col=C.DT_COL, max_gap_s=MAX_GAP_S):
    """Mean run-length (consecutive same-state samples within gap) per state."""
    d = df.sort(dt_col).with_columns(pl.col("GED").fill_null(3).cast(pl.Int64).alias("g"))
    runs = {s: [] for s in [0,1,2,3]}
    prev=None; rl=0; prev_t=None
    for t, g in zip(d[dt_col].to_list(), d["g"].to_list()):
        gap = (t-prev_t).total_seconds() if prev_t is not None else 0
        if g==prev and gap<=max_gap_s: rl+=1
        else:
            if prev is not None: runs[prev].append(rl)
            rl=1
        prev=g; prev_t=t
    if prev is not None: runs[prev].append(rl)
    return {s: (sum(v)/len(v) if v else 0.0) for s,v in runs.items()}

def main():
    pops = {"failed": C.FAILED_VINS, "nonfailed": C.NONFAILED_VINS}
    dwell_out={}
    for pop, vins in pops.items():
        agg = Counter()
        for vin in vins:
            agg.update(transition_counts(C.load_vin(vin)))
        mat = matrix_from_counts(agg)
        pl.DataFrame([{"from":i, **{f"to_{j}":mat[i][j] for j in [0,1,2,3]}} for i in [0,1,2,3]]) \
          .write_csv(C.RESULTS/f"2c_transitions_{pop}.csv")
        # dwell averaged over the population
        ds=[dwell_summary(C.load_vin(v)) for v in vins]
        dwell_out[pop]={s: sum(x[s] for x in ds)/len(ds) for s in [0,1,2,3]}
        print(pop, "P(0->2)=", mat[0][2], "P(2->2)=", mat[2][2])
    (C.RESULTS/"2c_dwell.json").write_text(json.dumps(dwell_out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test → PASS** — `py -3 -m pytest V12_ALT_GED/tests/test_analysis_2c_markov.py -v`.
- [ ] **Step 5: Run analysis** — `py -3 V12_ALT_GED/src/analysis_2c_markov.py` → Expected: two transition CSVs + dwell JSON. Compare `P(0→2)` and `P(2→2)` (persistence) failed vs non-failed; failed should show higher GED=2 persistence/entry. Record for report.
- [ ] **Step 6: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): 2C Markov transition matrices + dwell times"`

---

## Task 2.4: Phase 2D — data-driven GED=2 trigger model

**Files:**
- Create: `V12_ALT_GED/src/analysis_2d_triggers.py`
- Create: `V12_ALT_GED/tests/test_analysis_2d_triggers.py`
- Output: `V12_ALT_GED/results/2d_trigger_importance.csv`, `2d_trigger_report.json`

Goal: learn what GED=2 rows look like vs GED=0 rows in signal space (VSI, RPM, ANR, CSP, SMA, regime). Sample all GED=2 rows + a matched random GED=0 sample (cap per VIN), fit a logistic regression (standardized) + report coefficients and a GBM permutation importance. This **reverse-engineers the empirical trigger** (expected: low VSI co-occurrence + specific regime), without claiming causation.

- [ ] **Step 1: Write the failing test** (`balanced_sample` returns equal-class frame, capped)

```python
import importlib.util, pathlib
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=_load("analysis_2d_triggers")

def test_balanced_sample_equal_classes():
    df = pl.DataFrame({"GED":[2]*10 + [0]*1000, "VSI":[25.0]*1010})
    bs = T.balanced_sample(df, cap=100, seed=0)
    n2 = (bs["GED"]==2).sum(); n0 = (bs["GED"]==0).sum()
    assert n2 == 10 and n0 == 10        # matched to the minority count
```

- [ ] **Step 2: Run test → FAIL**.

- [ ] **Step 3: Implement**

```python
"""Phase 2D: empirical GED=2 trigger reverse-engineering."""
import importlib.util, pathlib, json
import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")
FEATS=["VSI","RPM","ANR","CSP","SMA"]

def balanced_sample(df: pl.DataFrame, cap=20000, seed=0) -> pl.DataFrame:
    pos = df.filter(pl.col("GED")==2)
    neg = df.filter(pl.col("GED")==0)
    k = min(pos.height, neg.height, cap)
    if k == 0: return df.head(0)
    return pl.concat([pos.sample(k, seed=seed), neg.sample(k, seed=seed)])

def main():
    rows=[]
    for vin in C.FAILED_VINS:           # GED=2 lives in failed VINs; include all for signal
        df=C.load_vin(vin).select(["GED"]+FEATS).drop_nulls()
        bs=balanced_sample(df, seed=0)
        if bs.height: rows.append(bs)
    data=pl.concat(rows)
    X=data.select(FEATS).to_numpy(); y=(data["GED"]==2).to_numpy().astype(int)
    Xs=StandardScaler().fit_transform(X)
    lr=LogisticRegression(max_iter=1000).fit(Xs,y)
    gb=GradientBoostingClassifier(random_state=0).fit(X,y)
    imp=permutation_importance(gb,X,y,n_repeats=5,random_state=0)
    out=pl.DataFrame({"feature":FEATS,
                      "logit_coef":lr.coef_[0].tolist(),
                      "perm_importance":imp.importances_mean.tolist()}).sort("perm_importance",descending=True)
    out.write_csv(C.RESULTS/"2d_trigger_importance.csv")
    rep={"n_samples":int(data.height),"lr_intercept":float(lr.intercept_[0]),
         "top_trigger":out["feature"][0]}
    (C.RESULTS/"2d_trigger_report.json").write_text(json.dumps(rep,indent=2))
    print(out); print(rep)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test → PASS**.
- [ ] **Step 5: Run analysis** — `py -3 V12_ALT_GED/src/analysis_2d_triggers.py` → Expected: VSI has a strong negative logit coefficient (GED=2 co-occurs with depressed voltage), confirming the KT "GED=2 co-occurs with VSI<26V" note empirically. Record top triggers.
- [ ] **Step 6: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): 2D GED=2 trigger reverse-engineering"`

---

## Task 2.5: Phase 2E — null-informativeness

**Files:**
- Create: `V12_ALT_GED/src/analysis_2e_nulls.py`
- Output: `V12_ALT_GED/results/2e_null_structure.csv`, `2e_null_report.json`

- [ ] **Step 1: Implement** (does GED-null fraction differ by regime / by F-vs-NF / near failure?)

```python
"""Phase 2E: is GED missingness informative?"""
import importlib.util, pathlib, json
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")

def main():
    df=pl.read_parquet(C.DAILY_CACHE)
    by_fail=df.group_by("failed").agg((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("null_frac"))
    # near-failure (dtf<=30) vs baseline for failed VINs
    fail=df.filter(pl.col("failed"))
    near=fail.filter(pl.col("dtf")<=30).select((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("nf"))
    base=fail.filter(pl.col("dtf")>30).select((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("bf"))
    per_vin=df.group_by("vin","failed").agg((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("null_frac")).sort("null_frac",descending=True)
    per_vin.write_csv(C.RESULTS/"2e_null_structure.csv")
    rep={"by_failed":by_fail.to_dicts(),
         "failed_near_le30d_null":float(near["nf"][0]) if near.height else None,
         "failed_base_gt30d_null":float(base["bf"][0]) if base.height else None}
    (C.RESULTS/"2e_null_report.json").write_text(json.dumps(rep,indent=2)); print(rep)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** — `py -3 V12_ALT_GED/src/analysis_2e_nulls.py` → Expected: per-VIN null fractions (VIN8_NF/VIN9_NF high per anchors); verdict on whether missingness rises near failure. Record.
- [ ] **Step 3: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): 2E null-informativeness"`

---

## Task 3.1: Phase 3A+3B — infeasibility note + VSI regulation-effort proxy

**Files:**
- Create: `V12_ALT_GED/src/proxy_3b_regulation.py`
- Create: `V12_ALT_GED/tests/test_proxy_3b_regulation.py`
- Output: `V12_ALT_GED/results/3b_regulation_features.csv`

3A (infeasibility) is documented in the Phase-5 report (no code). 3B builds a continuous health indicator: fit the healthy-fleet surface `E[VSI | RPM, ANR, CSP]` from **non-failed** engine-on data, then per-VIN compute the residual `VSI_obs − VSI_expected` and derive features: `resid_mean`, `resid_neg_frac`, `resid_slope_30d` (drift), `resid_oscillation` (std of daily resid). This reuses the existing binning idea from `V11_ALT_heuristics_features.build_load_reference`.

- [ ] **Step 1: Write the failing test** (`fit_reference` + `residual` on synthetic data: a VIN matching the surface → resid ≈ 0)

```python
import importlib.util, pathlib
import polars as pl, numpy as np
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=_load("proxy_3b_regulation")

def test_residual_zero_when_matches_reference():
    rng=np.random.default_rng(0)
    rpm=rng.uniform(700,1800,5000); anr=rng.uniform(0,500,5000); csp=rng.uniform(0,80,5000)
    vsi=28.0 + 0.0*rpm                      # constant healthy voltage
    ref=pl.DataFrame({"RPM":rpm,"ANR":anr,"CSP":csp,"VSI":vsi})
    surf=P.fit_reference(ref)
    r=P.residual(ref, surf)
    assert abs(float(np.nanmean(r["resid"].to_numpy()))) < 0.5
```

- [ ] **Step 2: Run test → FAIL**.

- [ ] **Step 3: Implement**

```python
"""Phase 3B: VSI regulation-effort proxy (feasible substitute for excitation reconstruction)."""
import importlib.util, pathlib
import polars as pl, numpy as np
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")
RPM_BIN, ANR_BIN, CSP_BIN = 100.0, 100.0, 10.0

def _bins(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("RPM")//RPM_BIN).alias("rb"),
        (pl.col("ANR")//ANR_BIN).alias("ab"),
        (pl.col("CSP")//CSP_BIN).alias("cb"),
    ])

def fit_reference(nf_eo: pl.DataFrame) -> pl.DataFrame:
    """Median VSI per (RPM,ANR,CSP) bin over healthy engine-on data."""
    d=_bins(nf_eo.drop_nulls(["RPM","ANR","CSP","VSI"]))
    return d.group_by(["rb","ab","cb"]).agg(pl.col("VSI").median().alias("vsi_exp"))

def residual(df: pl.DataFrame, surf: pl.DataFrame) -> pl.DataFrame:
    d=_bins(df.drop_nulls(["RPM","ANR","CSP","VSI"])).join(surf,on=["rb","ab","cb"],how="inner")
    return d.with_columns((pl.col("VSI")-pl.col("vsi_exp")).alias("resid"))

def engine_on(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("RPM")>0) & pl.col("RPM").is_not_null())

def main():
    nf=pl.concat([engine_on(C.load_vin(v)) for v in C.NONFAILED_VINS])
    surf=fit_reference(nf)
    rows=[]
    for vin in C.ALL_VINS:
        r=residual(engine_on(C.load_vin(vin)), surf)
        if r.height==0: continue
        daily=r.group_by("day").agg(pl.col("resid").mean().alias("rd")).sort("day")
        x=np.arange(daily.height); y=daily["rd"].to_numpy()
        last=min(30,daily.height)
        slope=np.polyfit(x[-last:],y[-last:],1)[0] if last>=2 else float("nan")
        rows.append({"vin":vin,"failed":C.is_failed(vin),
            "resid_mean":float(np.nanmean(r["resid"].to_numpy())),
            "resid_neg_frac":float((r["resid"]<0).mean()),
            "resid_oscillation":float(np.nanstd(y)),
            "resid_slope_30d":float(slope)})
    pl.DataFrame(rows).write_csv(C.RESULTS/"3b_regulation_features.csv")
    print(pl.DataFrame(rows).sort("resid_mean"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test → PASS**.
- [ ] **Step 5: Run** — `py -3 V12_ALT_GED/src/proxy_3b_regulation.py` → Expected: per-VIN regulation-residual features; failed VINs (esp. VIN1) expected to show more negative `resid_mean` / steeper `resid_slope_30d`. These feed Phase 4. Record.
- [ ] **Step 6: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): 3B VSI regulation-effort proxy (excitation reconstruction documented infeasible)"`

---

## Task 4.1: Phase 4 — per-VIN GED prognostic features

**Files:**
- Create: `V12_ALT_GED/src/features_4_ged.py`
- Create: `V12_ALT_GED/tests/test_features_4_ged.py`
- Output: `V12_ALT_GED/results/4_ged_features.csv`

New per-VIN scalar features from the daily cache: `ged2_acceleration` (slope of daily GED2 rate over last 30 obs days), `ged2_onset_slope` (max 7-day rise in ged2_rate), `ged2_rate_idle` / `ged2_rate_cruise` (regime-conditioned), plus the Phase-2C `markov_p_0to2` and Phase-3B residual features joined in. Each pure function takes the per-VIN daily slice (a `pl.DataFrame`) → `float`.

- [ ] **Step 1: Write the failing tests** (synthetic daily slice with a rising GED2 ramp)

```python
import importlib.util, pathlib, datetime as dt
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
F=_load("features_4_ged")

def _ramp():
    days=[dt.date(2025,1,1)+dt.timedelta(days=i) for i in range(40)]
    rate=[0.0]*30+[0.01*i for i in range(10)]   # rises in the last 10 days
    return pl.DataFrame({"day":days,"ged2_rate":rate})

def test_ged2_acceleration_positive_on_ramp():
    assert F.ged2_acceleration(_ramp()) > 0

def test_ged2_acceleration_zero_on_flat():
    flat=pl.DataFrame({"day":_ramp()["day"],"ged2_rate":[0.0]*40})
    assert abs(F.ged2_acceleration(flat)) < 1e-9
```

- [ ] **Step 2: Run tests → FAIL**.

- [ ] **Step 3: Implement**

```python
"""Phase 4: per-VIN scalar GED prognostic features (offline candidates)."""
import polars as pl, numpy as np

def _slope_last(d: pl.DataFrame, col: str, n: int) -> float:
    s=d.sort("day")[col].to_numpy(); s=s[~np.isnan(s)]
    if len(s)<2: return 0.0
    k=min(n,len(s)); x=np.arange(k)
    return float(np.polyfit(x, s[-k:], 1)[0])

def ged2_acceleration(daily: pl.DataFrame) -> float:
    return _slope_last(daily, "ged2_rate", 30)

def ged2_onset_slope(daily: pl.DataFrame) -> float:
    s=daily.sort("day")["ged2_rate"].fill_null(0).to_numpy()
    if len(s)<8: return 0.0
    diffs=[s[i]-s[i-7] for i in range(7,len(s))]
    return float(max(diffs)) if diffs else 0.0

def ged2_rate_regime(daily: pl.DataFrame, regime: str) -> float:
    num=daily[f"ged2_in_{regime}"].sum(); den=daily[f"regime_{regime}_rows"].sum()
    return float(num/den) if den else 0.0

def build_feature_row(daily_vin: pl.DataFrame) -> dict:
    return {
        "ged2_acceleration": ged2_acceleration(daily_vin),
        "ged2_onset_slope": ged2_onset_slope(daily_vin),
        "ged2_rate_idle": ged2_rate_regime(daily_vin, "idle"),
        "ged2_rate_cruise": ged2_rate_regime(daily_vin, "cruise"),
    }
```

- [ ] **Step 4: Run tests → PASS**.

- [ ] **Step 5: Build the per-VIN feature table** (append a small `main()` joining Markov + regulation features)

```python
# append to features_4_ged.py
def main():
    import importlib.util, pathlib
    _SRC=pathlib.Path(__file__).resolve().parent
    def _load(n):
        s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    C=_load("ged_common")
    cache=pl.read_parquet(C.DAILY_CACHE)
    reg=pl.read_csv(C.RESULTS/"3b_regulation_features.csv")
    rows=[]
    for vin in C.ALL_VINS:
        dv=cache.filter(pl.col("vin")==vin)
        row={"VIN":vin, **build_feature_row(dv)}
        rows.append(row)
    feats=pl.DataFrame(rows).join(reg.select(["vin","resid_mean","resid_slope_30d","resid_oscillation"]).rename({"vin":"VIN"}),on="VIN",how="left")
    feats.write_csv(C.RESULTS/"4_ged_features.csv"); print(feats)

if __name__ == "__main__":
    main()
```
Run: `py -3 V12_ALT_GED/src/features_4_ged.py` → Expected: 25-row `4_ged_features.csv`.

- [ ] **Step 6: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): Phase-4 candidate GED prognostic features"`

---

## Task 4.2: Phase 4 — LOVO trial vs the 0.9267 baseline

**Files:**
- Create: `V12_ALT_GED/src/trial_4_lovo.py`
- Create: `V12_ALT_GED/tests/test_trial_4_lovo.py`
- Output: `V12_ALT_GED/results/4_lovo_trial.csv`

Self-contained reproduction of the frozen LOVO protocol (RidgeClassifier alpha=1.0; leave-one-VIN-out; train-median impute; StandardScaler; decision_function→sigmoid; **pooled** OOF `roc_auc_score`). **Gate:** FAMILY_A must reproduce 0.9267 (±0.002) or stop and debug before testing candidates.

- [ ] **Step 1: Write the failing test** (separable synthetic → AUROC = 1.0; validates the harness itself)

```python
import importlib.util, pathlib
import numpy as np, pandas as pd
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
L=_load("trial_4_lovo")

def test_lovo_auroc_perfect_on_separable():
    vins=[f"V{i}" for i in range(10)]
    X=pd.DataFrame({"vin":vins,"failed":[1]*5+[0]*5,"f":[10,11,12,13,14,0,1,2,3,4]})
    auroc=L.lovo_auroc(X, ["f"], label_col="failed", vin_col="vin")
    assert auroc > 0.99
```

- [ ] **Step 2: Run test → FAIL**.

- [ ] **Step 3: Implement**

```python
"""Phase 4: leave-one-VIN-out Ridge trial reproducing the frozen 0.9267 protocol."""
import importlib.util, pathlib
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")
ALPHA=1.0
FAMILY_A=["vsi_std_ratio_30d","vsi_dominant_freq","vsi_spectral_entropy",
          "bat_charge_delta_trend_right","vsi_range_trend_last30d","progressive_drift"]

def lovo_auroc(df: pd.DataFrame, feats, label_col="failed", vin_col="VIN") -> float:
    y=df[label_col].to_numpy().astype(int); probs=np.zeros(len(df))
    for i in range(len(df)):
        tr=df.index!=df.index[i]
        Xtr=df.loc[tr,feats].to_numpy(dtype=float); Xte=df.loc[~tr,feats].to_numpy(dtype=float)
        med=np.nanmedian(Xtr,axis=0)
        Xtr=np.where(np.isnan(Xtr),med,Xtr); Xte=np.where(np.isnan(Xte),med,Xte)
        sc=StandardScaler().fit(Xtr)
        clf=RidgeClassifier(alpha=ALPHA).fit(sc.transform(Xtr), y[tr])
        d=clf.decision_function(sc.transform(Xte))[0]
        probs[i]=1/(1+np.exp(-d))
    return float(roc_auc_score(y, probs))

def main():
    base=pd.read_csv(C.ROOT/"V5.2_ALT"/"features"/"V5.2_20_5_ALT_selected_features.csv")
    cand=pd.read_csv(C.RESULTS/"4_ged_features.csv").rename(columns={"VIN":"VIN"})
    df=base.merge(cand, on="VIN", how="left")
    # GATE: reproduce baseline
    a0=lovo_auroc(df, FAMILY_A)
    print(f"BASELINE FAMILY_A AUROC = {a0:.4f} (expect 0.9267)")
    assert abs(a0-0.9267) < 0.002, "Baseline not reproduced — fix before testing candidates"
    new=["ged2_acceleration","ged2_onset_slope","ged2_rate_idle","ged2_rate_cruise",
         "resid_mean","resid_slope_30d","resid_oscillation"]
    rows=[{"set":"FAMILY_A","features":"baseline-6","auroc":a0,"delta":0.0}]
    for f in new:                                  # each candidate added singly
        a=lovo_auroc(df, FAMILY_A+[f]); rows.append({"set":f"+{f}","features":f,"auroc":a,"delta":a-a0})
    a_all=lovo_auroc(df, FAMILY_A+new); rows.append({"set":"+ALL_GED","features":";".join(new),"auroc":a_all,"delta":a_all-a0})
    out=pd.DataFrame(rows).sort_values("auroc",ascending=False)
    out.to_csv(C.RESULTS/"4_lovo_trial.csv",index=False); print(out.to_string(index=False))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test → PASS** — `py -3 -m pytest V12_ALT_GED/tests/test_trial_4_lovo.py -v`.
- [ ] **Step 5: Run the trial** — `py -3 V12_ALT_GED/src/trial_4_lovo.py`
  Expected: the GATE prints `BASELINE FAMILY_A AUROC ≈ 0.9267` and the assertion passes (if it fails, the matrix/feature names are misaligned — debug before continuing). Then a ranked table of `FAMILY_A + <each GED feature>` with deltas vs 0.9267. **Honest verdict:** record whether any candidate beats baseline by a meaningful margin; expectation is no material gain (prior result), which is itself a reportable finding.
- [ ] **Step 6: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): Phase-4 LOVO trial vs 0.9267 baseline (honest verdict)"`

---

## Task 5.1: Phase 5 — graphs

**Files:**
- Create: `V12_ALT_GED/src/graphs_5.py`
- Output: `V12_ALT_GED/graphs/{ged_state_timeline_VIN1_F_ALT.png, transition_heatmap_failed.png, transition_heatmap_nonfailed.png, regulation_residual_overlay.png, trigger_importance.png}`

- [ ] **Step 1: Implement** the five figures from the result CSVs/cache (matplotlib). Core snippets:

```python
"""Phase 5: investigation figures."""
import importlib.util, pathlib
import polars as pl, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common"); G=C.RESULTS.parent/"graphs"

def state_timeline(vin="VIN1_F_ALT"):
    d=pl.read_parquet(C.DAILY_CACHE).filter(pl.col("vin")==vin).sort("day").to_pandas()
    fig,ax=plt.subplots(figsize=(11,3.2))
    ax.plot(d["day"], d["ged_cnt_2"], color="#c0392b", lw=1.2)
    ax.axhline(200, ls="--", color="gray", label="emergency threshold (200/day)")
    ax.set_title(f"GED=2 daily count — {vin}"); ax.legend(); fig.tight_layout()
    fig.savefig(G/f"ged_state_timeline_{vin}.png", dpi=150); plt.close(fig)

def heatmap(pop):
    m=pd.read_csv(C.RESULTS/f"2c_transitions_{pop}.csv").set_index("from")
    fig,ax=plt.subplots(figsize=(4.5,4))
    im=ax.imshow(m.values, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([0,1,2,3]); ax.set_yticks(range(4)); ax.set_yticklabels([0,1,2,3])
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{m.values[i,j]:.2f}",ha="center",va="center",color="w",fontsize=8)
    ax.set_title(f"GED transition P — {pop}"); fig.colorbar(im); fig.tight_layout()
    fig.savefig(G/f"transition_heatmap_{pop}.png", dpi=150); plt.close(fig)

def main():
    G.mkdir(parents=True, exist_ok=True)
    state_timeline("VIN1_F_ALT"); heatmap("failed"); heatmap("nonfailed")
    # regulation overlay
    r=pd.read_csv(C.RESULTS/"3b_regulation_features.csv")
    fig,ax=plt.subplots(figsize=(7,4))
    ax.scatter(r["resid_mean"], r["resid_slope_30d"], c=r["failed"].map({True:"#c0392b",False:"#2980b9"}))
    ax.set_xlabel("resid_mean"); ax.set_ylabel("resid_slope_30d"); ax.set_title("Regulation-effort proxy (red=failed)")
    fig.tight_layout(); fig.savefig(G/"regulation_residual_overlay.png", dpi=150); plt.close(fig)
    # trigger importance
    t=pd.read_csv(C.RESULTS/"2d_trigger_importance.csv")
    fig,ax=plt.subplots(figsize=(6,3.5)); ax.barh(t["feature"], t["perm_importance"], color="#16a085")
    ax.set_title("GED=2 trigger — permutation importance"); fig.tight_layout()
    fig.savefig(G/"trigger_importance.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** — `py -3 V12_ALT_GED/src/graphs_5.py` → Expected: 5 PNGs in `graphs/`. Eyeball each opens and is non-empty.
- [ ] **Step 3: Commit** — `git add V12_ALT_GED/ && git commit -m "feat(v12-alt-ged): Phase-5 investigation figures"`

---

## Task 5.2: Phase 5 — synthesis report (answers all 12 deliverables)

**Files:**
- Create: `V12_ALT_GED/reports/V12_ALT_GED_investigation_report.md`

- [ ] **Step 1: Write the report** with these sections, each filled from the produced artifacts (cite result files + literature findings; assign a confidence level to every interpretive claim):
  1. Executive summary (what GED is; what can/can't be reconstructed; what's new; deploy verdict).
  2. GED physical interpretation + per-state evidence + confidence (Phase 1 + 2A + 2D) — incl. the J1939-SPN finding (or its explicit non-confirmation).
  3. KT validation + discrepancy log (Phase 1 vs `KT_daimler/`).
  4. Quantization verdict (Phase 2A): enum vs continuum, with `rho`/medians.
  5. Occupancy + Markov structure (Phase 2B/2C): F-vs-NF transition/dwell differences.
  6. Trigger reverse-engineering (Phase 2D): what GED=2 responds to.
  7. Null-informativeness (Phase 2E).
  8. Continuous excitation: **infeasibility** (3A, sensor table) + the **regulation-effort proxy** (3B) as the feasible substitute, with its features.
  9. Prognostic trial (Phase 4): table of deltas vs 0.9267; lead-time/emergency context (2/10, threshold 200); honest verdict.
  10. Novel health indicators + feature-engineering recommendations.
  11. RUL applicability conclusion.
  12. **Sensor-gap recommendation:** the exact signals/SPNs to add (alternator output current, rotor/field current, alternator temperature, generator speed `W`) to enable true excitation reconstruction — tie to the 500-truck scale-up plan.

- [ ] **Step 2: Self-check** the report against the spec's success criteria: every one of the 12 deliverables answered; every interpretive claim carries a confidence level; numbers reconcile to anchors (note any deviation). Fix gaps inline.
- [ ] **Step 3: Commit** — `git add V12_ALT_GED/reports/ && git commit -m "docs(v12-alt-ged): investigation report answering all 12 deliverables"`

---

## Task 5.3: Final verification + memory

- [ ] **Step 1: Run the full test suite** — `py -3 -m pytest V12_ALT_GED/tests/ -v` → Expected: all PASS.
- [ ] **Step 2: Confirm all result artifacts exist** — list `V12_ALT_GED/results/` and `graphs/`; confirm the 0.9267 gate passed and is recorded in `4_lovo_trial.csv`.
- [ ] **Step 3: Write a project memory** capturing the verdict (enum confirmed; reconstruction infeasible/sensor-blocked; regulation proxy built; trial verdict vs 0.927) per the memory format, and add the MEMORY.md pointer.
- [ ] **Step 4: Final commit** — `git add -A V12_ALT_GED/ && git commit -m "chore(v12-alt-ged): final verification + investigation complete"`

---

## Self-Review (completed by plan author)
- **Spec coverage:** all 6 phases mapped to tasks (0.1–0.2, 1.1, 2.1–2.5, 3.1, 4.1–4.2, 5.1–5.3); the 12 deliverables are explicitly enumerated in Task 5.2. ✔
- **Placeholder scan:** no TBD/TODO; every code step contains runnable code; discovery steps (0.1 Step 1) are real verification actions, not placeholders. ✔
- **Type/name consistency:** `lovo_auroc(df, feats, label_col, vin_col)`, `FAMILY_A`, `daily_from_df`, `transition_counts`, `fit_reference`/`residual`, `build_feature_row`, `ged2_acceleration` are referenced consistently across tasks and tests. Cache column names used in 2B/2E/4.1 match those emitted by `daily_from_df` in 0.2. ✔
- **Known risk:** per-VIN parquet column names (`DT_COL`, `DTF_COL`, filename) are verified in Task 0.1 Step 1 and threaded through `ged_common`; all later tasks import from `ged_common`, so a single adjustment propagates. ✔
