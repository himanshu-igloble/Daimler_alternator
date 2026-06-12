# V11_ALT_heuristics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 12 candidate lead-time heuristics from `V10.6.2_ALT/reports/V10.6.2_ALT_candidate_heuristics_for_lead_time.txt` as a focused forensic + alarm fork (`V11_ALT_heuristics/`), run the full precursor pipeline on real data, and produce an honest V11-vs-V10.6.2 comparison.

**Architecture:** New `V11_ALT_heuristics/` directory implementing ONLY the precursor/lead-time channel. Pure, unit-tested feature functions live in `*_features.py`; `*_forensic.py` builds the extended daily panel and runs the unchanged honest gate (within-truck `z≥2` AND outside NF `p05–p95`, `MIN_EO_SAMPLES=200`, horizons 90/60/45/30/14/7). The frozen Ridge classifier (AUROC 0.927) and Weibull window are reused by reference, never recomputed. Compound voting (#11) and CUSUM change-point (#12) are separate modules. A compare module emits the head-to-head report.

**Tech Stack:** Python 3.11 via `py -3`, pandas 2.3, numpy 1.26, scipy 1.17, pyarrow; pytest 9.0 for tests. No sklearn (W6 forbidden-pattern gate).

**Reference source files (read-only, do not edit):**
- `V10.6.2_ALT/src/V10.6.2_ALT_forensic_features.py` — the harness being forked.
- `V10.6.2_ALT/src/V10.6.2_ALT_config.py` — base config (inherits V10.6 → ALL_VINS, FAILED_VIN_SET, V52_PARQUET_DIR, V52_PARQUET_PREFIX).
- `V10.6.2_ALT/cache/forensics/{earliest_signal_per_vin,failed_window_deviations,nf_baseline}.csv` — V10.6.2 results to compare against.

**Verified facts (do not re-verify):**
- Parquet schema: `DATETIME, VIN, VIN_LABEL, FAILED, CSP, RPM, ANR, GED, VSI, SMA, SALEDATE, JCOPENDATE, DAYS_SINCE_SALE, DAYS_TO_FAILURE`. ANR and DATETIME present.
- 25 parquets at `V5.2_ALT/features/parquets/V5.2_20_5_ALT_<VIN>.parquet`. 10 FAILED (`VIN1_F_ALT`..`VIN10_F_ALT`), 15 NF (`VIN1_NF_ALT`..`VIN15_NF_ALT`).
- Sampling ≈ 5 s. ANR sentinels 65535/-5000 (filter to [-400,1300]). VSI valid 10–36 V (×0.2 if >36).
- V10.6.2 baseline recall = 4/10 discriminative at ≥7d (genuine 2/10 via GED==2).

**Working directory for all run commands:** `D:\Daimler-starter_motor_alternator_battery`
**Run interpreter:** `py -3`. Run scripts from `V11_ALT_heuristics\src`. Run tests from repo root.

---

## Task 0: Scaffold directory + config

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_config.py`
- Create (empty dirs via the config self-test mkdir, or git): `V11_ALT_heuristics/cache/forensics/`, `results/`, `reports/`, `visualizations/`, `tests/`

- [ ] **Step 1: Create the config module**

Create `V11_ALT_heuristics/src/V11_ALT_heuristics_config.py`:

```python
"""V11_ALT_heuristics — Config & Constants.

Inherits the V10.6.2 forensic config (which itself inherits V10.6 → V10.5.3 →
V5.2). Adds the constants and feature registry for the 12 new lead-time
heuristics. Frozen classifier / Weibull are reused by reference; nothing here
re-fits them (W6 forbidden-pattern gate still applies).
"""
from __future__ import annotations

import pathlib
from importlib.util import spec_from_file_location, module_from_spec

# --- inherit V10.6.2 config -------------------------------------------------
_v1062_path = (
    pathlib.Path(__file__).resolve().parents[2]
    / "V10.6.2_ALT" / "src" / "V10.6.2_ALT_config.py"
)
_spec = spec_from_file_location("_v1062_cfg", str(_v1062_path))
_v1062 = module_from_spec(_spec)
_spec.loader.exec_module(_v1062)
for _attr in dir(_v1062):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_v1062, _attr)

# --- V11 identity & paths ---------------------------------------------------
VERSION = "V11_ALT_heuristics"
FILE_PREFIX = "V11_ALT_heuristics_"
V11_ROOT = pathlib.Path(__file__).resolve().parents[1]
FORENSICS = V11_ROOT / "cache" / "forensics"
RESULTS_DIR = V11_ROOT / "results"
REPORTS_DIR = V11_ROOT / "reports"

# V10.6.2 forensic outputs to compare against (read-only)
V1062_FORENSICS = pathlib.Path(_v1062.V10_6_2_ROOT) / "cache" / "forensics"

# --- inherited honest-gate constants (re-declared for clarity) --------------
MIN_EO_SAMPLES = 200
HORIZON_BINS = [(60, 90, "90"), (45, 60, "60"), (30, 45, "45"),
                (14, 30, "30"), (7, 14, "14"), (0, 7, "7")]

# --- new heuristic constants ------------------------------------------------
SAMPLE_SEC = 5.0                 # nominal CAN sampling interval
ANR_VALID_LO, ANR_VALID_HI = -400.0, 1300.0
VSI_TARGET = 27.0                # charging-onset / recovery target volts
SAG_V = 24.0                     # under-voltage threshold
REG_BAND_LO, REG_BAND_HI = 27.0, 29.0          # #4 regulation band
RAMP_RPM_LO, RAMP_RPM_HI = 600.0, 1500.0       # #1 charging ramp band
CEILING_RPM_MIN = 1500.0                        # #1 plateau band
IDLE_RPM_LO, IDLE_RPM_HI, IDLE_CSP_MAX = 550.0, 950.0, 5.0   # #9 idle band
HIGH_LOAD_NM, LOW_LOAD_NM = 400.0, 100.0        # #10 sag typing split
RECOVERY_WINDOW_S = 30.0                         # #3 post-crank window
DOSE_DT_CAP_S = 30.0                             # #5 cap per-sample gap
# #2 load-residual reference surface bins
REF_RPM_BIN, REF_ANR_BIN, REF_CSP_BIN = 250.0, 100.0, 10.0
REF_MIN_BIN_COUNT = 50
# #12 CUSUM
CUSUM_K = 0.5                    # slack (in baseline sigma units)
CUSUM_H = 5.0                    # decision threshold (in baseline sigma units)

# --- feature registry -------------------------------------------------------
# Old 16 V10.6.2 features (kept for apples-to-apples comparison + compound votes)
OLD_FEATS = ["vsi_mean", "vsi_std", "vsi_cv", "vsi_min", "vsi_p05", "vsi_range",
             "vsi_entropy", "vsi_sag_frac", "idle_vsi_mean", "cruise_vsi_mean",
             "resting_vsi_mean", "crank_vsi_min", "ged2_cnt", "ged2_frac",
             "sma_starts", "rpm_mean"]
# New heuristic daily features (Group A + #5 daily increment)
NEW_FEATS = ["vsi_rpm_slope", "vsi_ceiling", "vsi_onset_rpm",          # #1
             "vsi_resid_mean", "vsi_resid_negfrac",                    # #2
             "crank_recovery_t", "crank_recovery_slope",              # #3
             "reg_duty_frac",                                          # #4
             "cranks_per_ehr", "crank_dur_mean",                      # #7
             "ged1_frac", "ged3_frac", "ged_churn",                   # #8
             "idle_vsi_var", "idle_vsi_acf1", "idle_vsi_zcr",         # #9
             "sag_highload_frac", "sag_idle_frac",                    # #10
             "uv_dose_day"]                                            # #5 daily
FEAT_COLS = OLD_FEATS + NEW_FEATS

# direction: True if a HIGH value means worse health
BAD_HIGH = {"vsi_std", "vsi_cv", "vsi_range", "vsi_sag_frac", "vsi_entropy",
            "ged2_frac", "ged2_cnt",
            "vsi_onset_rpm", "vsi_resid_negfrac", "crank_recovery_t",
            "cranks_per_ehr", "crank_dur_mean", "ged1_frac", "ged3_frac",
            "ged_churn", "idle_vsi_var", "idle_vsi_zcr", "sag_highload_frac",
            "sag_idle_frac", "uv_dose_day"}
BAD_LOW = {"vsi_mean", "vsi_min", "vsi_p05", "idle_vsi_mean", "cruise_vsi_mean",
           "resting_vsi_mean", "crank_vsi_min",
           "vsi_rpm_slope", "vsi_ceiling", "vsi_resid_mean",
           "crank_recovery_slope", "reg_duty_frac", "idle_vsi_acf1"}
# strong, physically-motivated features that can trigger earliest-precursor verdict
KEY_FEATURES = ["vsi_std", "vsi_cv", "vsi_range", "vsi_sag_frac", "vsi_entropy",
                "ged2_frac", "vsi_min", "idle_vsi_mean", "resting_vsi_mean",
                "crank_vsi_min",
                "vsi_ceiling", "vsi_rpm_slope", "vsi_resid_mean",
                "vsi_resid_negfrac", "crank_recovery_t", "reg_duty_frac",
                "ged_churn", "uv_dose_day"]

# #11 compound voting channels (one representative feature each, orthogonal physics)
VOTE_CHANNELS = ["vsi_ceiling", "vsi_resid_mean", "crank_recovery_t",
                 "resting_vsi_mean", "ged_churn"]
VOTE_MIN = 2   # fire early-watch when >= this many channels cross NF p05/p95

if __name__ == "__main__":
    print(f"{VERSION} config OK")
    print(f"  FEAT_COLS: {len(FEAT_COLS)} ({len(OLD_FEATS)} old + {len(NEW_FEATS)} new)")
    print(f"  ALL_VINS: {len(ALL_VINS)}  FAILED: {len(FAILED_VIN_SET)}")
    missing = [f for f in FEAT_COLS if f not in BAD_HIGH and f not in BAD_LOW]
    print(f"  features missing a direction: {missing}")
    assert not missing, missing
```

- [ ] **Step 2: Run config self-test**

Run: `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_config.py"`
Expected: prints `FEAT_COLS: 35 (16 old + 19 new)`, `ALL_VINS: 25  FAILED: 10`, `features missing a direction: []`, no assertion error.

- [ ] **Step 3: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_config.py
git commit -m "feat(v11-alt): scaffold V11_ALT_heuristics config + feature registry"
```

---

## Task 1: Pure feature functions module + prepare/helpers

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_features.py`
- Test: `V11_ALT_heuristics/tests/test_features.py`

This task creates the module with `prepare()`, `_ols_slope()`, and the simple Group-A features (#4 reg duty, #7 crank effort, #8 GED states). Later tasks append more functions to the same file with their own tests.

- [ ] **Step 1: Write the failing test**

Create `V11_ALT_heuristics/tests/test_features.py`:

```python
import importlib.util
import pathlib
import numpy as np
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
F = _load("V11_ALT_heuristics_features")


def _ts(n, step=5):
    # epoch-second timestamps as a DATETIME column
    return pd.to_datetime(np.arange(n) * step, unit="s")


def test_prepare_validates_and_adds_columns():
    df = pd.DataFrame({
        "RPM": [0, 800, 4000], "CSP": [0, 2, 50], "ANR": [0.0, 200.0, 65535.0],
        "VSI": [25.0, 28.0, 200.0], "GED": [0, 0, 0], "SMA": [0, 0, 0],
        "DATETIME": _ts(3), "DAYS_SINCE_SALE": [1, 1, 1], "DAYS_TO_FAILURE": [10, 10, 10],
    })
    p = F.prepare(df, cfg)
    assert p["eo"].tolist() == [False, True, False]   # RPM 0 off, 4000 > 3500 invalid
    assert p["off"].iloc[0]
    assert np.isnan(p["anr"].iloc[2])                 # 65535 sentinel -> NaN
    assert abs(p["vsi"].iloc[2] - 40.0) > 0 or np.isnan(p["vsi"].iloc[2])  # 200*0.2=40 >36 -> NaN
    assert np.isnan(p["vsi"].iloc[2])


def test_reg_duty():
    eo = pd.DataFrame({"vsi": [28.0, 28.5, 24.0, 30.0], "day": [1, 1, 1, 1]})
    out = F.reg_duty(eo, cfg)
    assert abs(out.loc[1] - 0.5) < 1e-9   # 2 of 4 in [27,29]


def test_crank_effort():
    # 1 start (rising edge at idx1), SMA==1 run length 2 samples = 10s
    df = pd.DataFrame({
        "SMA": [0, 1, 1, 0, 0], "day": [1, 1, 1, 1, 1],
    })
    eo = pd.DataFrame({"day": [1] * 360})   # 360 eo samples -> 0.5 engine-hours
    out = F.crank_effort(df, eo, cfg)
    assert abs(out.loc[1, "crank_dur_mean"] - 10.0) < 1e-9
    assert abs(out.loc[1, "cranks_per_ehr"] - 2.0) < 1e-6   # 1 start / 0.5 ehr


def test_ged_states():
    df = pd.DataFrame({
        "GED": [0, 2, 0, 3, 1, 1], "day": [1, 1, 1, 1, 1, 1],
    })
    out = F.ged_states(df, cfg)
    assert abs(out.loc[1, "ged1_frac"] - 2 / 6) < 1e-9
    assert abs(out.loc[1, "ged3_frac"] - 1 / 6) < 1e-9
    assert out.loc[1, "ged_churn"] == 2   # 0->2 and 0->3 transitions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py -q`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` (features module / functions not defined).

- [ ] **Step 3: Write the module**

Create `V11_ALT_heuristics/src/V11_ALT_heuristics_features.py`:

```python
"""V11_ALT_heuristics — pure daily feature functions (no file IO, no globals).

Each function takes a prepared per-VIN DataFrame (see prepare()) or a subset and
returns a pandas Series/DataFrame indexed by 'day'. Kept pure so the 12 heuristics
are unit-testable on tiny synthetic frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    vsi = df["VSI"].where(df["VSI"] <= 36, df["VSI"] * 0.2)
    df["vsi"] = vsi.where((vsi > 10) & (vsi <= 36))
    df["eo"] = (df["RPM"] > 0) & (df["RPM"] <= 3500)
    df["off"] = (df["RPM"] == 0)
    df["anr"] = df["ANR"].where((df["ANR"] >= cfg.ANR_VALID_LO) & (df["ANR"] <= cfg.ANR_VALID_HI))
    df["t_s"] = pd.to_datetime(df["DATETIME"]).astype("int64").to_numpy() / 1e9
    df["day"] = df["DAYS_SINCE_SALE"]
    return df


def _ols_slope(x, y, min_n: int = 10, min_span: float = 50.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < min_n or np.ptp(x) < min_span:
        return np.nan
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan
    return float(((x - xm) * (y - ym)).sum() / sxx)


def reg_duty(eo: pd.DataFrame, cfg) -> pd.Series:
    """#4 fraction of engine-on samples with VSI in [REG_BAND_LO, REG_BAND_HI]."""
    inb = eo["vsi"].between(cfg.REG_BAND_LO, cfg.REG_BAND_HI)
    return inb.groupby(eo["day"]).mean().rename("reg_duty_frac")


def crank_effort(df: pd.DataFrame, eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#7 cranks per engine-hour and mean crank (SMA==1 run-length) duration."""
    rise = (df["SMA"] == 1) & (df["SMA"].shift(1) == 0)
    starts = rise.groupby(df["day"]).sum()
    ehrs = eo.groupby("day").size() * cfg.SAMPLE_SEC / 3600.0
    cph = (starts / ehrs).replace([np.inf, -np.inf], np.nan).rename("cranks_per_ehr")
    on = (df["SMA"] == 1).to_numpy().astype(int)
    dayv = df["day"].to_numpy()
    runs = []
    i, n = 0, len(on)
    while i < n:
        if on[i] == 1:
            j = i
            while j < n and on[j] == 1:
                j += 1
            runs.append((dayv[i], (j - i) * cfg.SAMPLE_SEC))
            i = j
        else:
            i += 1
    if runs:
        rr = pd.DataFrame(runs, columns=["day", "dur"])
        dur = rr.groupby("day")["dur"].mean()
    else:
        dur = pd.Series(dtype=float)
    return pd.DataFrame({"cranks_per_ehr": cph, "crank_dur_mean": dur})


def ged_states(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """#8 GED==1 and GED==3 daily rates, plus 0->2 / 0->3 transition churn."""
    day = df["day"]
    g1 = (df["GED"] == 1).groupby(day).mean()
    g3 = (df["GED"] == 3).groupby(day).mean()
    churn = ((df["GED"].shift(1) == 0) & (df["GED"].isin([2, 3]))).groupby(day).sum()
    return pd.DataFrame({"ged1_frac": g1, "ged3_frac": g3, "ged_churn": churn})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py -q`
Expected: PASS (4 passed). Deprecation warnings from pandas groupby are acceptable.

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_features.py V11_ALT_heuristics/tests/test_features.py
git commit -m "feat(v11-alt): pure features module + #4/#7/#8 (reg duty, crank effort, GED states)"
```

---

## Task 2: #1 dVSI/dRPM curve features

**Files:**
- Modify: `V11_ALT_heuristics/src/V11_ALT_heuristics_features.py` (append `vsi_rpm_curve`)
- Test: `V11_ALT_heuristics/tests/test_features.py` (append)

- [ ] **Step 1: Write the failing test** — append to `test_features.py`:

```python
def test_vsi_rpm_curve():
    rpm = np.linspace(600, 1500, 40)
    vsi = 20.0 + 0.005 * rpm            # slope 0.005 V/rpm; reaches 27 at rpm=1400
    plateau_rpm = np.full(20, 2000.0)
    plateau_vsi = np.full(20, 28.0)
    eo = pd.DataFrame({
        "RPM": np.r_[rpm, plateau_rpm],
        "vsi": np.r_[vsi, plateau_vsi],
        "day": np.r_[np.ones(40), np.ones(20)].astype(int),
    })
    out = F.vsi_rpm_curve(eo, cfg)
    assert abs(out.loc[1, "vsi_rpm_slope"] - 0.005) < 1e-4
    assert abs(out.loc[1, "vsi_ceiling"] - 28.0) < 1e-6
    assert abs(out.loc[1, "vsi_onset_rpm"] - 1400.0) < 25.0   # first RPM where vsi>=27
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py::test_vsi_rpm_curve -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'vsi_rpm_curve'`.

- [ ] **Step 3: Append the implementation** to `V11_ALT_heuristics_features.py`:

```python
def vsi_rpm_curve(eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#1 charging-ramp slope dVSI/dRPM (600-1500 rpm), regulation ceiling
    (mean VSI at RPM>1500), and charging-onset rpm (first RPM reaching VSI_TARGET)."""
    ramp = eo[(eo["RPM"] >= cfg.RAMP_RPM_LO) & (eo["RPM"] <= cfg.RAMP_RPM_HI)]
    slope = ramp.groupby("day").apply(
        lambda g: _ols_slope(g["RPM"].to_numpy(), g["vsi"].to_numpy()))
    ceiling = eo[eo["RPM"] > cfg.CEILING_RPM_MIN].groupby("day")["vsi"].mean()

    def _onset(g):
        hit = g.loc[g["vsi"] >= cfg.VSI_TARGET, "RPM"]
        return float(hit.min()) if len(hit) else np.nan

    onset = eo.groupby("day").apply(_onset)
    out = pd.DataFrame({"vsi_rpm_slope": slope, "vsi_ceiling": ceiling,
                        "vsi_onset_rpm": onset})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_features.py V11_ALT_heuristics/tests/test_features.py
git commit -m "feat(v11-alt): #1 dVSI/dRPM slope, ceiling, charging-onset rpm"
```

---

## Task 3: #2 load-normalised voltage residual (NF reference surface)

**Files:**
- Modify: `V11_ALT_heuristics/src/V11_ALT_heuristics_features.py` (append `build_load_reference`, `load_residual`)
- Test: `V11_ALT_heuristics/tests/test_features.py` (append)

- [ ] **Step 1: Write the failing test** — append to `test_features.py`:

```python
def test_load_residual():
    # NF reference: at (RPM~800, ANR~200, CSP~0) healthy median VSI = 28.0
    nf = pd.DataFrame({
        "RPM": np.full(100, 800.0), "anr": np.full(100, 200.0),
        "CSP": np.full(100, 0.0), "vsi": np.full(100, 28.0), "day": np.ones(100),
    })
    ref = F.build_load_reference(nf, cfg)
    # failing truck at same operating point but 1 V low
    fl = pd.DataFrame({
        "RPM": np.full(10, 800.0), "anr": np.full(10, 200.0),
        "CSP": np.full(10, 0.0), "vsi": np.full(10, 27.0), "day": np.ones(10),
    })
    out = F.load_residual(fl, ref, cfg)
    assert abs(out.loc[1.0, "vsi_resid_mean"] - (-1.0)) < 1e-6
    assert abs(out.loc[1.0, "vsi_resid_negfrac"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py::test_load_residual -q`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Append the implementation** to `V11_ALT_heuristics_features.py`:

```python
def _bin3(d: pd.DataFrame, cfg) -> pd.DataFrame:
    d = d.dropna(subset=["vsi", "anr"]).copy()
    d["rpm_bin"] = np.floor(d["RPM"] / cfg.REF_RPM_BIN).astype("int64")
    d["anr_bin"] = np.floor(d["anr"] / cfg.REF_ANR_BIN).astype("int64")
    d["csp_bin"] = np.floor(d["CSP"] / cfg.REF_CSP_BIN).astype("int64")
    return d


def build_load_reference(nf_eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#2 binned healthy-fleet surface E[VSI | RPM, ANR, CSP] (median per bin,
    only bins with >= REF_MIN_BIN_COUNT NF samples). Index = (rpm_bin,anr_bin,csp_bin)."""
    d = _bin3(nf_eo, cfg)
    g = d.groupby(["rpm_bin", "anr_bin", "csp_bin"])["vsi"]
    ref = pd.DataFrame({"vsi_med": g.median(), "n": g.size()})
    return ref[ref["n"] >= cfg.REF_MIN_BIN_COUNT]


def load_residual(eo: pd.DataFrame, ref: pd.DataFrame, cfg) -> pd.DataFrame:
    """#2 daily residual mean (obs - expected at actual operating point) and
    negative-residual fraction. Samples whose bin is absent from ref are dropped."""
    d = _bin3(eo, cfg)
    d = d.join(ref["vsi_med"], on=["rpm_bin", "anr_bin", "csp_bin"]).dropna(subset=["vsi_med"])
    if d.empty:
        return pd.DataFrame({"vsi_resid_mean": [], "vsi_resid_negfrac": []})
    d["resid"] = d["vsi"] - d["vsi_med"]
    rmean = d.groupby("day")["resid"].mean()
    rneg = (d["resid"] < 0).groupby(d["day"]).mean()
    return pd.DataFrame({"vsi_resid_mean": rmean, "vsi_resid_negfrac": rneg})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_features.py V11_ALT_heuristics/tests/test_features.py
git commit -m "feat(v11-alt): #2 load-normalised VSI residual + NF reference surface"
```

---

## Task 4: #3 post-crank voltage recovery dynamics

**Files:**
- Modify: `V11_ALT_heuristics/src/V11_ALT_heuristics_features.py` (append `crank_recovery`)
- Test: `V11_ALT_heuristics/tests/test_features.py` (append)

- [ ] **Step 1: Write the failing test** — append to `test_features.py`:

```python
def test_crank_recovery():
    # SMA falls 1->0 at idx 2 (t=10s). VSI climbs 22->27 reaching 27 at t=20s (idx4).
    df = pd.DataFrame({
        "SMA":   [1,   1,   0,    0,    0,    0],
        "vsi":   [22., 22., 22.0, 24.5, 27.0, 28.0],
        "t_s":   [0.,  5.,  10.,  15.,  20.,  25.],
        "day":   [1,   1,   1,    1,    1,    1],
    })
    out = F.crank_recovery(df, cfg)
    assert abs(out.loc[1, "crank_recovery_t"] - 10.0) < 1e-6   # 20s - 10s edge
    assert out.loc[1, "crank_recovery_slope"] > 0              # voltage rising
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py::test_crank_recovery -q`
Expected: FAIL — function not defined.

- [ ] **Step 3: Append the implementation** to `V11_ALT_heuristics_features.py`:

```python
def crank_recovery(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """#3 per SMA 1->0 edge: seconds for VSI to reach VSI_TARGET (censored at
    RECOVERY_WINDOW_S if never reached) and the recovery slope (V/s) over that
    window; returns per-day mean across start events."""
    s = df.sort_values("t_s")
    sma = s["SMA"].to_numpy()
    vsi = s["vsi"].to_numpy()
    t = s["t_s"].to_numpy()
    dayv = s["day"].to_numpy()
    prev = np.r_[0, sma[:-1]]
    fall = np.where((prev == 1) & (sma == 0))[0]
    rows = []
    W = cfg.RECOVERY_WINDOW_S
    for i in fall:
        t0 = t[i]
        j = i
        reached = np.nan
        xs, ys = [], []
        while j < len(t) and (t[j] - t0) <= W:
            v = vsi[j]
            if not np.isnan(v):
                xs.append(t[j] - t0)
                ys.append(v)
                if v >= cfg.VSI_TARGET and np.isnan(reached):
                    reached = t[j] - t0
            j += 1
        rt = reached if not np.isnan(reached) else (W if xs else np.nan)
        sl = _ols_slope(xs, ys, min_n=3, min_span=5.0)
        rows.append((dayv[i], rt, sl))
    if not rows:
        return pd.DataFrame({"crank_recovery_t": [], "crank_recovery_slope": []})
    r = pd.DataFrame(rows, columns=["day", "rt", "sl"])
    return pd.DataFrame({"crank_recovery_t": r.groupby("day")["rt"].mean(),
                         "crank_recovery_slope": r.groupby("day")["sl"].mean()})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_features.py V11_ALT_heuristics/tests/test_features.py
git commit -m "feat(v11-alt): #3 post-crank voltage recovery time + slope"
```

---

## Task 5: #9 idle hunting, #10 sag typing, #5 daily UV dose

**Files:**
- Modify: `V11_ALT_heuristics/src/V11_ALT_heuristics_features.py` (append `idle_hunting`, `sag_typing`, `uv_dose_daily`)
- Test: `V11_ALT_heuristics/tests/test_features.py` (append)

- [ ] **Step 1: Write the failing test** — append to `test_features.py`:

```python
def test_idle_hunting():
    # idle band rows (RPM 600-900, CSP<5); alternating VSI -> high zcr, variance>0
    n = 40
    eo = pd.DataFrame({
        "RPM": np.full(n, 700.0), "CSP": np.full(n, 0.0),
        "vsi": np.where(np.arange(n) % 2 == 0, 28.5, 27.5),
        "t_s": np.arange(n) * 5.0, "day": np.ones(n, dtype=int),
    })
    out = F.idle_hunting(eo, cfg)
    assert out.loc[1, "idle_vsi_var"] > 0
    assert out.loc[1, "idle_vsi_zcr"] > 0.8          # alternating -> near-1 crossing rate
    assert out.loc[1, "idle_vsi_acf1"] < 0           # alternation -> negative lag-1 autocorr


def test_sag_typing():
    eo = pd.DataFrame({
        "anr": [500.0, 500.0, 10.0, 10.0],          # 2 high-load, 2 low-load
        "vsi": [23.0, 28.0, 23.0, 23.0],            # high-load 1/2 sag; low-load 2/2 sag
        "day": [1, 1, 1, 1],
    })
    out = F.sag_typing(eo, cfg)
    assert abs(out.loc[1, "sag_highload_frac"] - 0.5) < 1e-9
    assert abs(out.loc[1, "sag_idle_frac"] - 1.0) < 1e-9


def test_uv_dose_daily():
    # two under-volt samples at 22V (deficit 2V) and 23V (deficit 1V), dt=5s each
    eo = pd.DataFrame({
        "vsi": [28.0, 22.0, 23.0], "t_s": [0.0, 5.0, 10.0], "day": [1, 1, 1],
    })
    out = F.uv_dose_daily(eo, cfg)
    assert abs(out.loc[1] - (2.0 * 5 + 1.0 * 5)) < 1e-6   # 15 volt-seconds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py -q -k "idle or sag or dose"`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Append the implementation** to `V11_ALT_heuristics_features.py`:

```python
def idle_hunting(eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#9 within-idle (RPM 550-950, CSP<5) VSI variance, lag-1 autocorr, and
    mean-crossing rate (regulator-hunting / ripple signature)."""
    idle = eo[(eo["RPM"].between(cfg.IDLE_RPM_LO, cfg.IDLE_RPM_HI))
              & (eo["CSP"] < cfg.IDLE_CSP_MAX)].sort_values("t_s")

    def _acf1(v):
        v = v.dropna().to_numpy()
        if len(v) < 10 or v.std(ddof=0) == 0:
            return np.nan
        return float(np.corrcoef(v[:-1], v[1:])[0, 1])

    def _zcr(v):
        v = v.dropna().to_numpy()
        if len(v) < 10:
            return np.nan
        c = np.sign(v - v.mean())
        c = c[c != 0]
        if len(c) < 2:
            return np.nan
        return float(np.mean(np.diff(c) != 0))

    var = idle.groupby("day")["vsi"].var()
    acf = idle.groupby("day")["vsi"].apply(_acf1)
    zcr = idle.groupby("day")["vsi"].apply(_zcr)
    return pd.DataFrame({"idle_vsi_var": var, "idle_vsi_acf1": acf, "idle_vsi_zcr": zcr})


def sag_typing(eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#10 under-voltage (VSI<SAG_V) fraction split by load: high-torque
    (ANR>HIGH_LOAD_NM, stator/diode) vs low-load (ANR<LOW_LOAD_NM, regulator)."""
    hi = eo[eo["anr"] > cfg.HIGH_LOAD_NM]
    lo = eo[eo["anr"] < cfg.LOW_LOAD_NM]
    hf = (hi["vsi"] < cfg.SAG_V).groupby(hi["day"]).mean()
    lf = (lo["vsi"] < cfg.SAG_V).groupby(lo["day"]).mean()
    return pd.DataFrame({"sag_highload_frac": hf, "sag_idle_frac": lf})


def uv_dose_daily(eo: pd.DataFrame, cfg) -> pd.Series:
    """#5 daily under-voltage dose increment = integral of (SAG_V - VSI)*dt while
    engine-on and VSI<SAG_V (volt-seconds), using real DATETIME deltas (gap-capped)."""
    d = eo.sort_values("t_s").copy()
    d["dt"] = d.groupby("day")["t_s"].diff().clip(upper=cfg.DOSE_DT_CAP_S).fillna(0.0)
    under = d["vsi"] < cfg.SAG_V
    d["dose"] = np.where(under & d["vsi"].notna(), (cfg.SAG_V - d["vsi"]) * d["dt"], 0.0)
    return d.groupby("day")["dose"].sum().rename("uv_dose_day")
```

Note: `test_uv_dose_daily` expects 15 volt-seconds because the first under-volt sample (idx1) has `dt` = 5 s (diff from idx0) and idx2 has `dt` = 5 s. The very first row of the day has `dt`=0 (fillna), which is correct (no preceding sample). This matches the assertion.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_features.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_features.py V11_ALT_heuristics/tests/test_features.py
git commit -m "feat(v11-alt): #9 idle hunting, #10 sag typing, #5 daily UV dose"
```

---

## Task 6: build_daily integration + 1-VIN smoke test

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_forensic.py` (build_daily only this task)
- Test: `V11_ALT_heuristics/tests/test_build_daily.py`

- [ ] **Step 1: Write the failing test**

Create `V11_ALT_heuristics/tests/test_build_daily.py`:

```python
import importlib.util
import pathlib
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FOR = _load("V11_ALT_heuristics_forensic")


def test_build_daily_has_all_feat_cols():
    vin = "VIN1_F_ALT"
    ref = FOR.build_reference()           # NF surface for #2
    d = FOR.build_daily(vin, ref)
    for col in cfg.FEAT_COLS:
        assert col in d.columns, f"missing {col}"
    assert "dtf" in d.columns and "day" in d.columns and "n_eo" in d.columns
    assert len(d) > 30                     # real VIN has many days
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_build_daily.py -q`
Expected: FAIL — forensic module / build_daily not defined.

- [ ] **Step 3: Create `V11_ALT_heuristics_forensic.py` (build_daily + build_reference)**

```python
"""V11_ALT_heuristics — Forensic Feature Engine (extended honest-gate harness).

Builds the expanded daily panel (V10.6.2's 16 features + 19 new heuristic
features) per VIN, then runs the UNCHANGED honest gate (within-truck z>=2 AND
outside NF p05-p95, MIN_EO_SAMPLES=200, horizons 90/60/45/30/14/7) over the new
feature set. Adds an NF self-test (each NF truck scored as if failing, LOO
envelope) so multiple-comparison false-alarm risk is reported, not hidden.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
F = _load("V11_ALT_heuristics_features")
FORENSICS = cfg.FORENSICS

RAW_COLS = ["RPM", "CSP", "ANR", "VSI", "GED", "SMA",
            "DATETIME", "DAYS_SINCE_SALE", "DAYS_TO_FAILURE"]


def _entropy(vals, lo=22.0, hi=31.0, step=0.25):
    if len(vals) < 5:
        return np.nan
    bins = np.arange(lo, hi + step, step)
    h, _ = np.histogram(vals, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def _raw(vin):
    return pathlib.Path(cfg.V52_PARQUET_DIR) / f"{cfg.V52_PARQUET_PREFIX}{vin}.parquet"


def _read_prepared(vin):
    df = pd.read_parquet(_raw(vin), columns=RAW_COLS)
    return F.prepare(df, cfg)


def build_reference() -> pd.DataFrame:
    """Pool NF engine-on valid rows and build the #2 reference surface."""
    nf_eo = []
    for vin in cfg.ALL_VINS:
        if vin in cfg.FAILED_VIN_SET:
            continue
        p = _read_prepared(vin)
        nf_eo.append(p[p["eo"] & p["vsi"].notna()][["RPM", "anr", "CSP", "vsi", "day"]])
    return F.build_load_reference(pd.concat(nf_eo, ignore_index=True), cfg)


def build_daily(vin: str, ref: pd.DataFrame) -> pd.DataFrame:
    df = _read_prepared(vin)
    day = "day"
    eo = df[df["eo"] & df["vsi"].notna()]
    g = eo.groupby(day)["vsi"]
    daily = pd.DataFrame({
        "n_eo": g.size(), "vsi_mean": g.mean(), "vsi_std": g.std(),
        "vsi_min": g.min(), "vsi_p05": g.quantile(0.05), "vsi_p95": g.quantile(0.95),
    })
    daily["vsi_cv"] = daily["vsi_std"] / daily["vsi_mean"]
    daily["vsi_range"] = daily["vsi_p95"] - daily["vsi_p05"]
    daily["vsi_entropy"] = g.apply(lambda s: _entropy(s.values))
    daily["vsi_sag_frac"] = eo.assign(sag=eo["vsi"] < cfg.SAG_V).groupby(day)["sag"].mean()
    idle = eo[(eo["RPM"].between(550, 950)) & (eo["CSP"] < 5)]
    daily["idle_vsi_mean"] = idle.groupby(day)["vsi"].mean()
    cruise = eo[eo["CSP"] > 40]
    daily["cruise_vsi_mean"] = cruise.groupby(day)["vsi"].mean()
    rest = df[df["off"] & df["vsi"].notna()]
    daily["resting_vsi_mean"] = rest.groupby(day)["vsi"].mean()
    crank = df[(df["SMA"] == 1) & df["vsi"].notna()]
    daily["crank_vsi_min"] = crank.groupby(day)["vsi"].min()
    daily["ged2_cnt"] = (df["GED"] == 2).groupby(df[day]).sum()
    daily["ged2_frac"] = (df["GED"] == 2).groupby(df[day]).mean()
    daily["sma_starts"] = ((df["SMA"] == 1) & (df["SMA"].shift(1) == 0)).groupby(df[day]).sum()
    daily["rpm_mean"] = eo.groupby(day)["RPM"].mean()

    # --- new heuristic features (joined on day) ---
    daily = daily.join(F.vsi_rpm_curve(eo, cfg))                         # #1
    daily = daily.join(F.load_residual(eo, ref, cfg))                    # #2
    daily = daily.join(F.crank_recovery(df, cfg))                        # #3
    daily = daily.join(F.reg_duty(eo, cfg))                             # #4
    daily = daily.join(F.crank_effort(df, eo, cfg))                     # #7
    daily = daily.join(F.ged_states(df, cfg))                           # #8
    daily = daily.join(F.idle_hunting(eo, cfg))                         # #9
    daily = daily.join(F.sag_typing(eo, cfg))                           # #10
    daily = daily.join(F.uv_dose_daily(eo, cfg))                        # #5 daily

    daily["dtf"] = df.groupby(day)["DAYS_TO_FAILURE"].median()
    daily = daily.reset_index().rename(columns={day: "day"})
    daily["vin_label"] = vin
    # guarantee all registered feature columns exist (NaN if a truck never hit a regime)
    for col in cfg.FEAT_COLS:
        if col not in daily.columns:
            daily[col] = np.nan
    return daily
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_build_daily.py -q`
Expected: PASS (1 passed). May take 30–90 s (reads 15 NF parquets for the reference + 1 failed VIN).

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_forensic.py V11_ALT_heuristics/tests/test_build_daily.py
git commit -m "feat(v11-alt): build_daily integrates all 19 new features + NF reference"
```

---

## Task 7: Forensic main() — honest gate + extended CSVs + NF self-test

**Files:**
- Modify: `V11_ALT_heuristics/src/V11_ALT_heuristics_forensic.py` (append `_gate_one_vin`, `main`)
- Test: `V11_ALT_heuristics/tests/test_gate.py`

- [ ] **Step 1: Write the failing test**

Create `V11_ALT_heuristics/tests/test_gate.py`:

```python
import importlib.util
import pathlib
import numpy as np
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FOR = _load("V11_ALT_heuristics_forensic")


def test_gate_one_vin_detects_planted_drop():
    # baseline mid-life days dtf 120-365 with reg_duty_frac ~0.9; final window ~0.2
    rng = np.arange(0, 400)
    dtf = 400 - rng
    duty = np.where(dtf <= 7, 0.2, 0.9)
    d = pd.DataFrame({"day": rng, "n_eo": 500, "dtf": dtf, "reg_duty_frac": duty})
    for c in cfg.FEAT_COLS:
        if c not in d.columns:
            d[c] = 0.9 if c not in cfg.BAD_HIGH else 0.0
    nfb = pd.DataFrame({"feature": cfg.FEAT_COLS}).set_index("feature")
    nfb["nf_p05"] = 0.5      # reg_duty p05 = 0.5; window 0.2 < 0.5 -> discriminative (BAD_LOW)
    nfb["nf_p95"] = 1.5
    best_h, best_feat, best_z, _ = FOR._gate_one_vin(d, nfb)
    assert best_feat == "reg_duty_frac"
    assert best_h == "7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_gate.py -q`
Expected: FAIL — `_gate_one_vin` not defined.

- [ ] **Step 3: Append `_gate_one_vin` and `main` to `V11_ALT_heuristics_forensic.py`:**

```python
def _gate_one_vin(d_all: pd.DataFrame, nfb: pd.DataFrame):
    """Run the V10.6.2 honest gate over FEAT_COLS for one VIN's daily panel.
    Returns (best_horizon_label, best_feature, best_z, dev_rows)."""
    d = d_all[d_all["n_eo"] >= cfg.MIN_EO_SAMPLES]
    base = d[(d["dtf"] >= 120) & (d["dtf"] <= 365)]
    if len(base) < 15:
        base = d.nsmallest(max(int(len(d) * 0.4), 15), "day")
    best_h, best_feat, best_z = None, None, 0.0
    dev_rows = []
    for lo, hi, lbl in cfg.HORIZON_BINS:
        win = d[(d["dtf"] > lo) & (d["dtf"] <= hi)]
        n_days = len(win)
        for f in cfg.FEAT_COLS:
            bmean, bstd = base[f].mean(), base[f].std()
            wmean = win[f].mean() if n_days else np.nan
            z = (wmean - bmean) / bstd if (bstd and bstd > 0 and not np.isnan(wmean)) else np.nan
            p05, p95 = nfb.loc[f, "nf_p05"], nfb.loc[f, "nf_p95"]
            if np.isnan(wmean):
                disc = False
            elif f in cfg.BAD_HIGH:
                disc = wmean > p95
            else:
                disc = wmean < p05
            dev_rows.append({
                "vin_label": d_all["vin_label"].iloc[0], "horizon_days": lbl,
                "n_days": n_days, "feature": f,
                "window_mean": round(wmean, 4) if not np.isnan(wmean) else "",
                "baseline_mean": round(bmean, 4) if not np.isnan(bmean) else "",
                "z_vs_baseline": round(z, 2) if not np.isnan(z) else "",
                "nf_p05": round(p05, 4), "nf_p95": round(p95, 4),
                "discriminative": bool(disc),
            })
            if (f in cfg.KEY_FEATURES and disc and not np.isnan(z) and abs(z) >= 2.0
                    and int(lbl) >= (int(best_h) if best_h else -1)):
                if best_h is None or int(lbl) > int(best_h) or abs(z) > abs(best_z):
                    best_h, best_feat, best_z = lbl, f, z
    return best_h, best_feat, best_z, dev_rows


def _nf_baseline(nf_dailies):
    nf = pd.concat(nf_dailies, ignore_index=True)
    nf = nf[nf["n_eo"] >= cfg.MIN_EO_SAMPLES]
    rows = []
    for f in cfg.FEAT_COLS:
        s = nf[f].dropna()
        rows.append({"feature": f, "nf_p05": s.quantile(0.05), "nf_p50": s.quantile(0.50),
                     "nf_p95": s.quantile(0.95), "nf_mean": s.mean(), "nf_std": s.std()})
    return pd.DataFrame(rows)


def main() -> None:
    FORENSICS.mkdir(parents=True, exist_ok=True)
    print("[v11 forensic] building NF reference surface ...")
    ref = build_reference()

    print("[v11 forensic] building daily panels for 25 VINs ...")
    dailies = {}
    for vin in cfg.ALL_VINS:
        d = build_daily(vin, ref)
        d.to_csv(FORENSICS / f"{vin}_daily.csv", index=False)
        dailies[vin] = d
        print(f"  {vin:<16} days={len(d):<4} uv_dose_total={np.nansum(d['uv_dose_day']):.0f}")

    nf_list = [dailies[v] for v in cfg.ALL_VINS if v not in cfg.FAILED_VIN_SET]
    nf_base = _nf_baseline(nf_list)
    nf_base.to_csv(FORENSICS / "nf_baseline.csv", index=False)
    nfb = nf_base.set_index("feature")

    # failed-VIN gate
    dev_rows, earliest_rows = [], []
    for vin in cfg.FAILED_VIN_SET:
        d_all = dailies[vin]
        best_h, best_feat, best_z, devs = _gate_one_vin(d_all, nfb)
        dev_rows.extend(devs)
        d = d_all[d_all["n_eo"] >= cfg.MIN_EO_SAMPLES]
        cov30 = len(d[(d["dtf"] >= 0) & (d["dtf"] <= 30)])
        earliest_rows.append({
            "vin_label": vin,
            "earliest_discriminative_horizon_days": (best_h if best_h else "none"),
            "feature": (best_feat if best_feat else ""),
            "z": (round(best_z, 2) if best_h else ""),
            "ged2_total": int(d["ged2_cnt"].sum()),
            "n_days_final_30d": cov30,
            "verdict": ("discriminative_precursor" if best_h else "no_discriminative_precursor"),
        })
    pd.DataFrame(dev_rows).to_csv(FORENSICS / "failed_window_deviations.csv", index=False)
    es = pd.DataFrame(earliest_rows)
    es.to_csv(FORENSICS / "earliest_signal_per_vin.csv", index=False)

    # NF self-test: each NF truck scored as if failing, with LOO envelope
    self_rows = []
    nf_vins = [v for v in cfg.ALL_VINS if v not in cfg.FAILED_VIN_SET]
    for vin in nf_vins:
        loo = [dailies[v] for v in nf_vins if v != vin]
        nfb_loo = _nf_baseline(loo).set_index("feature")
        best_h, best_feat, best_z, _ = _gate_one_vin(dailies[vin], nfb_loo)
        self_rows.append({
            "vin_label": vin,
            "false_precursor_horizon_days": (best_h if best_h else "none"),
            "feature": (best_feat if best_feat else ""),
            "z": (round(best_z, 2) if best_h else ""),
            "verdict": ("FALSE_ALARM" if best_h else "clean"),
        })
    pd.DataFrame(self_rows).to_csv(FORENSICS / "nf_self_test.csv", index=False)

    n_detect = int((es["verdict"] == "discriminative_precursor").sum())
    n_false = int(sum(r["verdict"] == "FALSE_ALARM" for r in self_rows))
    print(f"\n  V11: {n_detect}/10 failed VINs with a discriminative precursor at >=7d")
    print(f"  NF self-test false alarms: {n_false}/15")
    print(f"  Saved forensic artifacts to {FORENSICS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test, then the full forensic stage on real data**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_gate.py -q`
Expected: PASS (1 passed).

Run: `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_forensic.py"`
Expected: prints per-VIN lines for 25 VINs, then `V11: N/10 failed VINs ...` and `NF self-test false alarms: M/15`. Writes `cache/forensics/{<VIN>_daily.csv, nf_baseline.csv, failed_window_deviations.csv, earliest_signal_per_vin.csv, nf_self_test.csv}`. Runtime ~2–5 min.

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_forensic.py V11_ALT_heuristics/tests/test_gate.py V11_ALT_heuristics/cache/forensics
git commit -m "feat(v11-alt): forensic honest gate over 35 features + NF self-test; first real run"
```

---

## Task 8: #12 CUSUM change-point + #5 dose knee + #6 resting slope

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_changepoint.py`
- Test: `V11_ALT_heuristics/tests/test_changepoint.py`

- [ ] **Step 1: Write the failing test**

Create `V11_ALT_heuristics/tests/test_changepoint.py`:

```python
import importlib.util
import pathlib
import numpy as np
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
CP = _load("V11_ALT_heuristics_changepoint")


def test_cusum_detects_downward_shift():
    # 100 baseline points ~0, then 50 points shifted to -3 sigma
    x = np.r_[np.zeros(100), np.full(50, -3.0)]
    idx = CP.cusum_changepoint(x, direction="down", k=cfg.CUSUM_K, h=cfg.CUSUM_H)
    assert 100 <= idx <= 110          # detected shortly after the shift at index 100


def test_cusum_no_shift_returns_none():
    rng = np.sin(np.arange(200) * 0.1) * 0.1   # tiny stationary noise
    idx = CP.cusum_changepoint(rng, direction="down", k=cfg.CUSUM_K, h=cfg.CUSUM_H)
    assert idx is None


def test_knee_on_convex_dose():
    # flat then sharply rising cumulative curve -> knee near the elbow
    cum = np.r_[np.zeros(50), np.arange(1, 51) ** 2]
    k = CP.knee_index(cum)
    assert 45 <= k <= 70
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_changepoint.py -q`
Expected: FAIL — module not defined.

- [ ] **Step 3: Create `V11_ALT_heuristics_changepoint.py`:**

```python
"""V11_ALT_heuristics — per-truck change-point lead-time module.

#12 CUSUM change-point on each truck's own standardized feature series (the
change-point timestamp IS the lead time). #5 knee detection on the cumulative
under-voltage dose curve. #6 life-long resting-voltage decay slope vs the NF
slope envelope. Reads the daily panels written by the forensic stage.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FORENSICS = cfg.FORENSICS


def cusum_changepoint(x, direction="down", k=0.5, h=5.0):
    """Tabular CUSUM on z-standardized x. Returns the first index where the
    one-sided cumulative sum exceeds h (in sigma units), else None."""
    x = np.asarray(x, dtype=float)
    m = ~np.isnan(x)
    if m.sum() < 20:
        return None
    xs = x.copy()
    mu = np.nanmean(x[m][: max(10, m.sum() // 3)])   # baseline = early third
    sd = np.nanstd(x[m][: max(10, m.sum() // 3)])
    if sd <= 0:
        return None
    z = (xs - mu) / sd
    s = 0.0
    for i in range(len(z)):
        if np.isnan(z[i]):
            continue
        if direction == "down":
            s = min(0.0, s + z[i] + k)
            if s < -h:
                return i
        else:
            s = max(0.0, s + z[i] - k)
            if s > h:
                return i
    return None


def knee_index(cum):
    """Kneedle-style elbow: index of max distance from the chord joining the
    first and last points of the (monotone) cumulative curve."""
    y = np.asarray(cum, dtype=float)
    n = len(y)
    if n < 5 or y[-1] == y[0]:
        return None
    x = np.arange(n)
    # normalized chord distance
    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(np.argmax(d))


def _lead_at(d, idx):
    """dtf at panel row idx = lead time in days (None if idx invalid)."""
    if idx is None or idx < 0 or idx >= len(d):
        return None
    val = d["dtf"].iloc[idx]
    return None if pd.isna(val) else float(val)


def _resting_slope(d):
    x = d["day"].to_numpy(dtype=float)
    y = d["resting_vsi_mean"].to_numpy(dtype=float)
    m = ~np.isnan(y)
    if m.sum() < 15:
        return np.nan
    xm, ym = x[m].mean(), y[m].mean()
    sxx = ((x[m] - xm) ** 2).sum()
    return float(((x[m] - xm) * (y[m] - ym)).sum() / sxx) if sxx > 0 else np.nan


def main() -> None:
    # NF resting-slope envelope (p05) for the #6 discriminative test
    nf_slopes = []
    for vin in cfg.ALL_VINS:
        if vin in cfg.FAILED_VIN_SET:
            continue
        d = pd.read_csv(FORENSICS / f"{vin}_daily.csv").sort_values("day")
        d = d[d["n_eo"] >= cfg.MIN_EO_SAMPLES]
        nf_slopes.append(_resting_slope(d))
    nf_slopes = [s for s in nf_slopes if not np.isnan(s)]
    rest_p05 = float(np.quantile(nf_slopes, 0.05)) if nf_slopes else np.nan

    rows = []
    for vin in cfg.FAILED_VIN_SET:
        d = pd.read_csv(FORENSICS / f"{vin}_daily.csv").sort_values("day")
        d = d[d["n_eo"] >= cfg.MIN_EO_SAMPLES].reset_index(drop=True)
        cp_resid = cusum_changepoint(d["vsi_resid_mean"].to_numpy(), "down",
                                     cfg.CUSUM_K, cfg.CUSUM_H)
        cp_rest = cusum_changepoint(d["resting_vsi_mean"].to_numpy(), "down",
                                    cfg.CUSUM_K, cfg.CUSUM_H)
        cum = np.nancumsum(d["uv_dose_day"].to_numpy())
        knee = knee_index(cum) if np.nanmax(cum) > 0 else None
        slope = _resting_slope(d)
        rows.append({
            "vin_label": vin,
            "cp_resid_lead_days": _lead_at(d, cp_resid),
            "cp_resting_lead_days": _lead_at(d, cp_rest),
            "dose_knee_lead_days": _lead_at(d, knee),
            "resting_slope": round(slope, 5) if not np.isnan(slope) else "",
            "resting_slope_nf_p05": round(rest_p05, 5) if not np.isnan(rest_p05) else "",
            "resting_slope_disc": bool(not np.isnan(slope) and not np.isnan(rest_p05)
                                       and slope < rest_p05),
        })
    out = pd.DataFrame(rows)
    out.to_csv(FORENSICS / "changepoint_per_vin.csv", index=False)
    print("[v11 changepoint] per-VIN change-point / knee / resting-slope:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests, then the stage on real data**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_changepoint.py -q`
Expected: PASS (3 passed).

Run: `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_changepoint.py"`
Expected: prints a 10-row table; writes `cache/forensics/changepoint_per_vin.csv`.

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_changepoint.py V11_ALT_heuristics/tests/test_changepoint.py V11_ALT_heuristics/cache/forensics/changepoint_per_vin.csv
git commit -m "feat(v11-alt): #12 CUSUM change-point + #5 dose knee + #6 resting slope"
```

---

## Task 9: #11 compound voting alarm + LOVO

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_compound.py`
- Test: `V11_ALT_heuristics/tests/test_compound.py`

- [ ] **Step 1: Write the failing test**

Create `V11_ALT_heuristics/tests/test_compound.py`:

```python
import importlib.util
import pathlib
import numpy as np
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
C = _load("V11_ALT_heuristics_compound")


def test_votes_in_window_counts_bad_crossings():
    # 2 of the 5 vote channels cross their NF boundary in the bad direction
    win = pd.Series({
        "vsi_ceiling": 26.0,        # BAD_LOW; p05 27 -> votes
        "vsi_resid_mean": -0.9,     # BAD_LOW; p05 -0.2 -> votes
        "crank_recovery_t": 5.0,    # BAD_HIGH; p95 12 -> no vote
        "resting_vsi_mean": 26.0,   # BAD_LOW; p05 25 -> no vote (26 > 25)
        "ged_churn": 1.0,           # BAD_HIGH; p95 5 -> no vote
    })
    nfb = pd.DataFrame({
        "feature": cfg.VOTE_CHANNELS,
        "nf_p05": [27.0, -0.2, 0.0, 25.0, 0.0],
        "nf_p95": [29.0, 0.2, 12.0, 27.0, 5.0],
    }).set_index("feature")
    n = C.count_votes(win, nfb, cfg)
    assert n == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_compound.py -q`
Expected: FAIL — module not defined.

- [ ] **Step 3: Create `V11_ALT_heuristics_compound.py`:**

```python
"""V11_ALT_heuristics — #11 compound weak-vote early-watch alarm.

Fires "early-watch" at the earliest horizon where >= VOTE_MIN of the orthogonal
VOTE_CHANNELS cross their NF p05/p95 boundary in the bad direction (weak vote: no
within-truck z requirement). The GED==2 storm remains a separate high-precision
emergency and is NOT part of the vote. Evaluated over the 10 failed VINs and,
for false-alarm honesty, the 15 NF trucks (LOO envelope).
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FOR = _load("V11_ALT_heuristics_forensic")
FORENSICS = cfg.FORENSICS


def count_votes(win_means: pd.Series, nfb: pd.DataFrame, cfg) -> int:
    """Count vote channels whose window mean crosses the NF boundary the bad way."""
    n = 0
    for f in cfg.VOTE_CHANNELS:
        v = win_means.get(f, np.nan)
        if pd.isna(v):
            continue
        p05, p95 = nfb.loc[f, "nf_p05"], nfb.loc[f, "nf_p95"]
        if f in cfg.BAD_HIGH:
            n += int(v > p95)
        else:
            n += int(v < p05)
    return n


def first_trigger(d_all: pd.DataFrame, nfb: pd.DataFrame, cfg):
    """Earliest horizon label where >= VOTE_MIN channels vote; returns (label, n)."""
    d = d_all[d_all["n_eo"] >= cfg.MIN_EO_SAMPLES]
    best = None
    for lo, hi, lbl in cfg.HORIZON_BINS:        # 90 -> 7 (earliest first)
        win = d[(d["dtf"] > lo) & (d["dtf"] <= hi)]
        if win.empty:
            continue
        means = win[cfg.VOTE_CHANNELS].mean()
        nv = count_votes(means, nfb, cfg)
        if nv >= cfg.VOTE_MIN:
            return lbl, nv          # first (earliest) qualifying horizon
    return None, 0


def main() -> None:
    nfb = pd.read_csv(FORENSICS / "nf_baseline.csv").set_index("feature")
    nf_vins = [v for v in cfg.ALL_VINS if v not in cfg.FAILED_VIN_SET]

    rows = []
    for vin in cfg.FAILED_VIN_SET:
        d = pd.read_csv(FORENSICS / f"{vin}_daily.csv")
        lbl, nv = first_trigger(d, nfb, cfg)
        rows.append({"group": "FAILED", "vin_label": vin,
                     "early_watch_horizon_days": (lbl if lbl else "none"),
                     "n_votes": nv, "fired": bool(lbl)})

    # NF false-alarm test with LOO envelope rebuilt from daily csvs
    nf_dailies = {v: pd.read_csv(FORENSICS / f"{v}_daily.csv") for v in nf_vins}
    for vin in nf_vins:
        loo = [nf_dailies[v] for v in nf_vins if v != vin]
        nfb_loo = FOR._nf_baseline(loo).set_index("feature")
        lbl, nv = first_trigger(nf_dailies[vin], nfb_loo, cfg)
        rows.append({"group": "NF", "vin_label": vin,
                     "early_watch_horizon_days": (lbl if lbl else "none"),
                     "n_votes": nv, "fired": bool(lbl)})

    out = pd.DataFrame(rows)
    out.to_csv(FORENSICS / "compound_alarm_lovo.csv", index=False)
    fired_f = int(out[(out.group == "FAILED") & out.fired].shape[0])
    fired_nf = int(out[(out.group == "NF") & out.fired].shape[0])
    print(f"[v11 compound] early-watch recall: {fired_f}/10 failed; "
          f"false alarms: {fired_nf}/15 NF")
    leads = out[(out.group == "FAILED") & out.fired]["early_watch_horizon_days"].tolist()
    print(f"  first-trigger lead horizons (failed): {leads}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test, then the stage on real data**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_compound.py -q`
Expected: PASS (1 passed).

Run: `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_compound.py"`
Expected: prints recall `X/10` and false alarms `Y/15`; writes `cache/forensics/compound_alarm_lovo.csv`.

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_compound.py V11_ALT_heuristics/tests/test_compound.py V11_ALT_heuristics/cache/forensics/compound_alarm_lovo.csv
git commit -m "feat(v11-alt): #11 compound weak-vote early-watch alarm + LOVO false-alarm test"
```

---

## Task 10: Compare module — V11 vs V10.6.2 report

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_compare.py`
- Test: `V11_ALT_heuristics/tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

Create `V11_ALT_heuristics/tests/test_compare.py`:

```python
import importlib.util
import pathlib
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CMP = _load("V11_ALT_heuristics_compare")


def test_classify_feature():
    # generalizes: fires on >=2 failed, 0 NF false alarms
    assert CMP.classify_feature(failed_hits=3, nf_false=0) == "generalizes"
    assert CMP.classify_feature(failed_hits=1, nf_false=0) == "anecdotal"
    assert CMP.classify_feature(failed_hits=4, nf_false=2) == "false_alarm_prone"
    assert CMP.classify_feature(failed_hits=0, nf_false=0) == "no_signal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_compare.py -q`
Expected: FAIL — module not defined.

- [ ] **Step 3: Create `V11_ALT_heuristics_compare.py`:**

```python
"""V11_ALT_heuristics — V11-vs-V10.6.2 comparison + honest verdict report.

Reads V11 forensic/changepoint/compound outputs and the frozen V10.6.2
earliest_signal_per_vin.csv, then emits:
  results/V11_ALT_heuristics_comparison.csv  (per-failed-VIN head-to-head)
  reports/V11_ALT_heuristics_report.md       (summary + per-feature classification)
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FORENSICS = cfg.FORENSICS


def classify_feature(failed_hits: int, nf_false: int) -> str:
    if nf_false >= 2:
        return "false_alarm_prone"
    if failed_hits >= 2:
        return "generalizes"
    if failed_hits == 1:
        return "anecdotal"
    return "no_signal"


def _hzn(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def main() -> None:
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    v11 = pd.read_csv(FORENSICS / "earliest_signal_per_vin.csv")
    v1062 = pd.read_csv(cfg.V1062_FORENSICS / "earliest_signal_per_vin.csv")
    dev = pd.read_csv(FORENSICS / "failed_window_deviations.csv")
    self_test = pd.read_csv(FORENSICS / "nf_self_test.csv")
    cp = pd.read_csv(FORENSICS / "changepoint_per_vin.csv")
    comp = pd.read_csv(FORENSICS / "compound_alarm_lovo.csv")

    a = v1062.set_index("vin_label")
    b = v11.set_index("vin_label")
    rows = []
    for vin in cfg.FAILED_VIN_SET:
        ra = a.loc[vin] if vin in a.index else None
        rb = b.loc[vin] if vin in b.index else None
        rows.append({
            "vin_label": vin,
            "v1062_horizon": ra["earliest_discriminative_horizon_days"] if ra is not None else "none",
            "v1062_feature": ra["feature"] if ra is not None else "",
            "v11_horizon": rb["earliest_discriminative_horizon_days"] if rb is not None else "none",
            "v11_feature": rb["feature"] if rb is not None else "",
            "earlier": _hzn(rb["earliest_discriminative_horizon_days"]) > _hzn(ra["earliest_discriminative_horizon_days"])
                       if (ra is not None and rb is not None) else False,
            "new_in_v11": (ra is None or str(ra["earliest_discriminative_horizon_days"]) == "none")
                          and (rb is not None and str(rb["earliest_discriminative_horizon_days"]) != "none"),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv", index=False)

    # per-feature generalization classification (new features only)
    disc = dev[dev["discriminative"] == True]               # noqa: E712
    feat_rows = []
    nf_false_by_feat = self_test[self_test["verdict"] == "FALSE_ALARM"]["feature"].value_counts()
    for f in cfg.NEW_FEATS:
        failed_hits = disc[disc["feature"] == f]["vin_label"].nunique()
        nf_false = int(nf_false_by_feat.get(f, 0))
        feat_rows.append({"feature": f, "failed_vins_discriminative": failed_hits,
                          "nf_false_alarms": nf_false,
                          "class": classify_feature(failed_hits, nf_false)})
    feat_df = pd.DataFrame(feat_rows).sort_values(
        ["class", "failed_vins_discriminative"], ascending=[True, False])

    n11 = int((v11["verdict"] == "discriminative_precursor").sum())
    n1062 = int((v1062["verdict"] == "discriminative_precursor").sum())
    n_false = int((self_test["verdict"] == "FALSE_ALARM").sum())
    fired_f = int(comp[(comp.group == "FAILED") & (comp.fired == True)].shape[0])   # noqa: E712
    fired_nf = int(comp[(comp.group == "NF") & (comp.fired == True)].shape[0])      # noqa: E712

    # markdown report
    md = []
    md.append("# V11_ALT_heuristics — Lead-Time Heuristics vs V10.6.2\n")
    md.append(f"_Generated by V11_ALT_heuristics_compare. Honest gate: within-truck "
              f"z>=2 AND outside NF p05-p95, MIN_EO_SAMPLES={cfg.MIN_EO_SAMPLES}._\n")
    md.append("## Headline\n")
    md.append(f"- **Discriminative-precursor recall:** V11 **{n11}/10** vs V10.6.2 {n1062}/10\n")
    md.append(f"- **NF self-test false alarms (15 NF as-if-failing):** {n_false}/15\n")
    md.append(f"- **Compound early-watch (#11):** recall {fired_f}/10 failed, "
              f"{fired_nf}/15 NF false alarms\n")
    md.append("\n## Per-failed-VIN head-to-head\n")
    md.append(comparison.to_markdown(index=False))
    md.append("\n\n## New-feature generalization (which of the 12 helped)\n")
    md.append(feat_df.to_markdown(index=False))
    md.append("\n\n## Change-point / dose-knee / resting-slope lead times (#12/#5/#6)\n")
    md.append(cp.to_markdown(index=False))
    md.append("\n\n## Honest verdict\n")
    if n11 > n1062:
        md.append(f"- V11 raises discriminative recall by {n11 - n1062} VIN(s) over V10.6.2.\n")
    else:
        md.append("- V11 does NOT raise discriminative recall above V10.6.2; new features add "
                  "fault-mode coverage / lead but not new detections.\n")
    gen = feat_df[feat_df["class"] == "generalizes"]["feature"].tolist()
    anec = feat_df[feat_df["class"] == "anecdotal"]["feature"].tolist()
    fap = feat_df[feat_df["class"] == "false_alarm_prone"]["feature"].tolist()
    md.append(f"- Generalizing new features (>=2 failed, 0-1 NF false): {gen or 'none'}\n")
    md.append(f"- Anecdotal (single-VIN) features: {anec or 'none'}\n")
    md.append(f"- False-alarm-prone features (>=2 NF false): {fap or 'none'}\n")
    md.append("- Structural limit (V10.6.2 sec 5) unchanged: no per-truck daily RUL; "
              "deliverable stays WHICH (classifier) + WHEN-fleet (Weibull) + WHEN-emergency.\n")

    out = cfg.REPORTS_DIR / "V11_ALT_heuristics_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[v11 compare] wrote {out}")
    print(f"  recall V11 {n11}/10 vs V10.6.2 {n1062}/10; NF false {n_false}/15")


if __name__ == "__main__":
    main()
```

Note: `to_markdown` requires the `tabulate` package. If it is missing, the run will raise `ImportError`. Step 4 includes a fallback check.

- [ ] **Step 4: Run unit test; ensure tabulate; run the stage**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_compare.py -q`
Expected: PASS (1 passed).

Run: `py -3 -c "import tabulate; print(tabulate.__version__)"`
Expected: prints a version. If it fails with ImportError, run `py -3 -m pip install tabulate` and retry.

Run: `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_compare.py"`
Expected: prints recall line; writes `results/V11_ALT_heuristics_comparison.csv` and `reports/V11_ALT_heuristics_report.md`.

- [ ] **Step 5: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_compare.py V11_ALT_heuristics/tests/test_compare.py V11_ALT_heuristics/results V11_ALT_heuristics/reports
git commit -m "feat(v11-alt): V11-vs-V10.6.2 comparison report + per-feature generalization verdict"
```

---

## Task 11: Verify gate + orchestrator (full pipeline run)

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_verify.py`
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_orchestrator.py`
- Test: `V11_ALT_heuristics/tests/test_verify.py`

- [ ] **Step 1: Write the failing test**

Create `V11_ALT_heuristics/tests/test_verify.py`:

```python
import importlib.util
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load("V11_ALT_heuristics_verify")


def test_scan_forbidden_clean_on_features():
    src = (_SRC / "V11_ALT_heuristics_features.py").read_text(encoding="utf-8")
    assert V.scan_forbidden(src) == []     # no sklearn / Ridge( in feature code


def test_scan_forbidden_flags_sklearn():
    assert "sklearn" in str(V.scan_forbidden("import sklearn\nfrom x import Ridge("))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_verify.py -q`
Expected: FAIL — module not defined.

- [ ] **Step 3: Create `V11_ALT_heuristics_verify.py`:**

```python
"""V11_ALT_heuristics — honest verification gates.

G1 every registered feature has a direction (config self-test already asserts).
G2 forensic artifacts exist.
G3 recall does not regress below V10.6.2 (V11 superset must be >=).
G4 W6 forbidden-pattern scan: no re-fitting of the frozen classifier in src/.
Exits non-zero on any hard-gate failure so the orchestrator surfaces it.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FORENSICS = cfg.FORENSICS


def scan_forbidden(text: str):
    """Return forbidden classifier-refit patterns present in text."""
    return [p for p in cfg.FORBIDDEN_FIT_PATTERNS if p in text]


def main() -> None:
    failures = []

    # G2 artifacts
    required = ["earliest_signal_per_vin.csv", "nf_baseline.csv",
                "failed_window_deviations.csv", "nf_self_test.csv",
                "changepoint_per_vin.csv", "compound_alarm_lovo.csv"]
    for fn in required:
        if not (FORENSICS / fn).exists():
            failures.append(f"G2 missing artifact: {fn}")

    # G3 recall non-regression
    if (FORENSICS / "earliest_signal_per_vin.csv").exists():
        v11 = pd.read_csv(FORENSICS / "earliest_signal_per_vin.csv")
        v1062 = pd.read_csv(cfg.V1062_FORENSICS / "earliest_signal_per_vin.csv")
        n11 = int((v11["verdict"] == "discriminative_precursor").sum())
        n1062 = int((v1062["verdict"] == "discriminative_precursor").sum())
        print(f"  G3 recall: V11 {n11}/10  vs  V10.6.2 {n1062}/10")
        if n11 < n1062:
            failures.append(f"G3 recall regression: V11 {n11} < V10.6.2 {n1062}")

    # G4 forbidden-pattern scan over src/ (skip config.py which lists them literally)
    for py in sorted(_src.glob("V11_ALT_heuristics_*.py")):
        if py.name.endswith("config.py"):
            continue
        hits = scan_forbidden(py.read_text(encoding="utf-8"))
        if hits:
            failures.append(f"G4 forbidden pattern {hits} in {py.name}")

    if failures:
        print("[v11 verify] FAIL:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    print("[v11 verify] PASS — all honest gates green")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `V11_ALT_heuristics_orchestrator.py`:**

```python
"""V11_ALT_heuristics — pipeline orchestrator.

Runs the precursor channel end to end:
  forensic -> changepoint -> compound -> verify (hard gate) -> compare.
The frozen classifier / Weibull are reused by reference and not re-run.
"""
from __future__ import annotations

import importlib.util
import pathlib
import time

_src = pathlib.Path(__file__).resolve().parent
SCRIPTS = [
    "V11_ALT_heuristics_forensic",
    "V11_ALT_heuristics_changepoint",
    "V11_ALT_heuristics_compound",
    "V11_ALT_heuristics_verify",
    "V11_ALT_heuristics_compare",
]


def _run(name):
    path = _src / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    t0 = time.perf_counter()
    try:
        spec.loader.exec_module(mod)
        mod.main()
        return "OK", time.perf_counter() - t0, None
    except SystemExit as e:
        ok = e.code in (0, None)
        return ("OK" if ok else "GATE_FAIL"), time.perf_counter() - t0, f"exit={e.code}"


def main():
    print("=" * 70)
    print("V11_ALT_heuristics pipeline")
    print("=" * 70)
    results = []
    for name in SCRIPTS:
        print(f"\n>>> {name}")
        status, dt, note = _run(name)
        results.append((name, status, dt, note))
        print(f"<<< {name}: {status} ({dt:.1f}s){'' if not note else ' ' + note}")
        if status == "GATE_FAIL":
            print("\nPipeline halted: a hard gate failed.")
            break
    print("\n" + "=" * 70)
    for name, status, dt, note in results:
        print(f"  {status:<10} {name:<36} {dt:6.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run verify unit test, then the FULL pipeline**

Run: `py -3 -m pytest V11_ALT_heuristics/tests/test_verify.py -q`
Expected: PASS (2 passed).

Run: `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_orchestrator.py"`
Expected: each stage prints `OK`; final summary table all `OK`; verify prints `G3 recall: V11 N/10 vs V10.6.2 4/10` and `PASS`. Total runtime ~5–10 min.

- [ ] **Step 6: Commit**

```bash
git add V11_ALT_heuristics/src/V11_ALT_heuristics_verify.py V11_ALT_heuristics/src/V11_ALT_heuristics_orchestrator.py V11_ALT_heuristics/tests/test_verify.py V11_ALT_heuristics/cache V11_ALT_heuristics/results V11_ALT_heuristics/reports
git commit -m "feat(v11-alt): verify gates + orchestrator; full pipeline green"
```

---

## Task 12: Read results, finalize report, present comparison

**Files:**
- Read: `V11_ALT_heuristics/reports/V11_ALT_heuristics_report.md`
- Read: `V11_ALT_heuristics/results/V11_ALT_heuristics_comparison.csv`
- Modify (if needed): `V11_ALT_heuristics/reports/V11_ALT_heuristics_report.md` (prepend an executive summary paragraph with the real numbers)

- [ ] **Step 1: Run the whole test suite**

Run: `py -3 -m pytest V11_ALT_heuristics/tests -q`
Expected: all tests pass (≈22 tests across 6 files).

- [ ] **Step 2: Inspect the real outputs**

Run: `py -3 -c "import pandas as pd; print(pd.read_csv(r'V11_ALT_heuristics/results/V11_ALT_heuristics_comparison.csv').to_string(index=False))"`
Run: `py -3 -c "import pandas as pd; print(pd.read_csv(r'V11_ALT_heuristics/cache/forensics/nf_self_test.csv').to_string(index=False))"`
Read `V11_ALT_heuristics/reports/V11_ALT_heuristics_report.md` in full.

- [ ] **Step 3: Sanity-check the honest verdict against the numbers**

Confirm: (a) recall V11 ≥ 4/10 (cannot be lower — V11 is a feature superset); (b) any feature labeled `generalizes` truly fires on ≥2 failed VINs with ≤1 NF false alarm; (c) NF self-test false-alarm count is reported and reflected in the verdict; (d) if recall did not rise, the report says so plainly (no overclaim). If any check fails, the bug is in the corresponding module — fix and re-run the orchestrator before proceeding.

- [ ] **Step 4: Prepend an executive summary** to `reports/V11_ALT_heuristics_report.md` with the actual recall, the list of generalizing heuristics, the compound-alarm recall/false-alarm, and whether any new channel beat the GED==2 lead for any truck. (Use the real numbers from Step 2 — do not invent.)

- [ ] **Step 5: Final commit**

```bash
git add V11_ALT_heuristics
git commit -m "docs(v11-alt): executive summary + final V11-vs-V10.6.2 comparison"
```

---

## Self-Review (completed during plan authoring)

**Spec coverage:**
- §5 Group A #1→Task2, #2→Task3, #3→Task4, #4/#7/#8→Task1, #9/#10→Task5, #5 daily→Task5; #5 knee/#6/#12→Task8; #11→Task9. All 12 heuristics mapped. ✓
- §6 honest guards: NF self-test→Task7, anecdotal flag→Task10 (`classify_feature`), recall/lead/false-alarm metrics→Task9+Task10, recall non-regression gate→Task11. ✓
- §3 architecture (focused fork, frozen reuse-by-reference, no Weibull/Ridge re-run)→Tasks0/11 (W6 scan). ✓
- §6 output artifacts all produced (Tasks 7–10). ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete runnable code; every run step has an exact command + expected output. ✓

**Type/name consistency:** `prepare`, `_ols_slope`, `build_load_reference`/`load_residual`, `crank_recovery`, `reg_duty`, `crank_effort`, `ged_states`, `vsi_rpm_curve`, `idle_hunting`, `sag_typing`, `uv_dose_daily`, `build_daily(vin, ref)`, `build_reference`, `_gate_one_vin`, `_nf_baseline`, `cusum_changepoint`, `knee_index`, `count_votes`, `first_trigger`, `classify_feature`, `scan_forbidden` — names are consistent across the tasks that define and call them. `FEAT_COLS`/`NEW_FEATS`/`BAD_HIGH`/`BAD_LOW`/`KEY_FEATURES`/`VOTE_CHANNELS`/`VOTE_MIN` referenced consistently. ✓

**Known nuance (documented, not a gap):** #6 implements the life-long resting-voltage decay slope (the heuristic's primary claim); the secondary "overnight hold" decay is approximated by the resting-level trend and is not separately modeled this iteration — the report notes this.
