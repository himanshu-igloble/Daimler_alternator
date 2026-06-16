# V11.1_ALT Covariate-Informed RUL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redo every alternator pipeline phase except classification under `V11.1_ALT/`, extending the Bayesian Weibull to a covariate AFT model fed by the V11 heuristic channels, with hard honesty gates and a full artifact suite.

**Architecture:** Mirror of V10.6.2's stage structure. New math lives in two focused modules (`covariates`, `survival` extended with AFT grid posterior); stages communicate via cache CSVs/JSONs exactly like V10.6.2. Variant selection (M0 β=0 / M1 x1 / M2 x1+x2) happens in the backtest stage and is consumed downstream. Fallback to M0 is a coded path, gated by verify.

**Tech Stack:** Python 3.11 via `py -3`; numpy, scipy, pandas, lifelines (MLE reference only), matplotlib, python-pptx, openpyxl, pytest. NO sklearn (W6).

**Frozen / reused inputs (read-only, never recompute):**
- Classifier: `V10.6_ALT/cache/ridge/ridge_prob_rescaled.csv`, threshold 0.4456.
- Lifecycle: `V10.6_ALT/cache/lifecycle/vin_lifecycle.parquet` — cols verified:
  `vin_label, alt_t0, alt_t1, age_days_observed, est_km, est_engine_hrs, ttf_days, failed_flag, ...` (25 rows).
- V11 forensics: `V11_ALT_heuristics/cache/forensics/<VIN>_daily.csv` (cols incl `day`(=DAYS_SINCE_SALE), `n_eo`, `crank_recovery_t`, the 5 vote channels, `ged2_cnt`, `dtf`) and `nf_baseline.csv`.
- Reference source (V11.1 adapts these; engineer reads them directly):
  `V10.6.2_ALT/src/V10.6.2_ALT_{survival,weibull_fleet,predictive_rul,backtest,decisions,assemble_rul,narrative_rul,rul_graphs,markdown_report,excel_report,verify,orchestrator}.py`,
  `V10.6.2_ALT/visualizations/visual script/generate_all_rul_graphs_alternator.py` and `generate_fleet_overlay_graphs_alternator.py`,
  `V11_ALT_heuristics/presentation/build_*_presentation.py`.

**Leakage doctrine (drives Tasks 1, 5, 8):** covariates at time t use trusted rows (`n_eo>=200`) with `day <= t` only. dtf is NEVER used to compute a covariate (it is post-hoc knowledge). x2 uses trailing **age** windows, not dtf windows.

**Run dir:** repo root `D:\Daimler-starter_motor_alternator_battery`. Branch: create `v11.1-alt` from master after merging current work, or from current branch per controller's choice.

---

## Task 0: Scaffold + config

**Files:** Create `V11.1_ALT/src/V11_1_ALT_config.py` (note: module filename uses `V11_1` — Python-safe; artifact prefix string is "V11.1_ALT_").

- [ ] **Step 1: Create the config**

```python
"""V11.1_ALT — Config. Frozen surface: classification results only."""
from __future__ import annotations
import pathlib
from importlib.util import spec_from_file_location, module_from_spec

_v1062_path = (pathlib.Path(__file__).resolve().parents[2]
               / "V10.6.2_ALT" / "src" / "V10.6.2_ALT_config.py")
_s = spec_from_file_location("_v1062", str(_v1062_path)); _m = module_from_spec(_s); _s.loader.exec_module(_m)
for _a in dir(_m):
    if not _a.startswith("_"):
        globals()[_a] = getattr(_m, _a)

VERSION = "V11.1_ALT"
FILE_PREFIX = "V11.1_ALT_"
V11_1_ROOT = pathlib.Path(__file__).resolve().parents[1]
COV_CACHE      = V11_1_ROOT / "cache" / "covariates"
WEIBULL_CACHE  = V11_1_ROOT / "cache" / "weibull"      # overrides inherited
RUL_CACHE      = V11_1_ROOT / "cache" / "rul"
BACKTEST_CACHE = V11_1_ROOT / "cache" / "backtest"
EMERG_CACHE    = V11_1_ROOT / "cache" / "emergency"
RESULTS_DIR_V11_1 = V11_1_ROOT / "results"
REPORTS_DIR_V11_1 = V11_1_ROOT / "reports"
VIZ_DIR_V11_1     = V11_1_ROOT / "visualizations"

# reused-by-reference inputs
V11_FORENSICS = pathlib.Path(__file__).resolve().parents[2] / "V11_ALT_heuristics" / "cache" / "forensics"
# LIFECYCLE_PARQUET, RIDGE_PROB_CSV, RIDGE_DECISION_THR inherited from V10.6.2 config.

MIN_EO_SAMPLES = 200
# covariates
X2_TRAIL_DAYS = 90          # trailing age-window for compound-vote covariate
VOTE_CHANNELS = ["vsi_ceiling", "vsi_resid_mean", "crank_recovery_t",
                 "resting_vsi_mean", "ged_churn"]
VOTE_BAD_HIGH = {"crank_recovery_t", "ged_churn"}   # rest are BAD_LOW
VOTE_MIN = 2
# AFT grids (k, scale0 inherit V10.6.2 grid bounds/prior constants)
BETA1_LO, BETA1_HI, BETA1_N = -0.2, 1.0, 25
BETA2_LO, BETA2_HI, BETA2_N = -0.2, 1.5, 18
BETA_PRIOR_SD = 0.5
VARIANTS = ["M0", "M1", "M2"]
# emergency
EXCEED_TRAIL_DAYS = 30
EXCEED_K_START = 2          # calibration starts here; smallest k with 0/15 NF
# gates
PI_WIDTH_SHRINK_MIN = 0.15  # G-BETA alternate criterion
SIGNED_RANK_ALPHA = 0.05

if __name__ == "__main__":
    print(f"{VERSION} config OK; lifecycle exists: {LIFECYCLE_PARQUET.exists()}; "
          f"ridge csv exists: {RIDGE_PROB_CSV.exists()}; "
          f"V11 forensics exists: {V11_FORENSICS.exists()}")
    assert LIFECYCLE_PARQUET.exists() and RIDGE_PROB_CSV.exists() and V11_FORENSICS.exists()
```

- [ ] **Step 2: Run** `py -3 "V11.1_ALT\src\V11_1_ALT_config.py"` → prints OK, all three True, no assert.
- [ ] **Step 3: Commit** `git add V11.1_ALT/src/V11_1_ALT_config.py && git commit -m "feat(v11.1): scaffold config (frozen classifier refs, AFT grids, leakage constants)"`

---

## Task 1: Covariates module (leakage-safe x1, x2)

**Files:** Create `V11.1_ALT/src/V11_1_ALT_covariates.py`; Test `V11.1_ALT/tests/test_covariates.py`.

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util, pathlib
import numpy as np, pandas as pd
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cfg = _load("V11_1_ALT_config")
C = _load("V11_1_ALT_covariates")

def _panel():
    # trusted panel: days 1..100; crank_recovery_t exceeds p95(=0.05) on days 10, 50, 90
    d = pd.DataFrame({"day": range(1, 101), "n_eo": 500})
    d["crank_recovery_t"] = 0.0
    d.loc[d.day.isin([10, 50, 90]), "crank_recovery_t"] = 5.0
    for ch in cfg.VOTE_CHANNELS:
        if ch not in d.columns:
            d[ch] = 100.0    # safely inside any band by default in tests below
    return d

def test_x1_counts_only_up_to_t():
    d = _panel()
    assert C.x1_exceedance(d, t=60, p95=0.05) == np.log1p(2)   # days 10, 50
    assert C.x1_exceedance(d, t=100, p95=0.05) == np.log1p(3)
    assert C.x1_exceedance(d, t=5, p95=0.05) == np.log1p(0)

def test_x2_trailing_window_votes():
    d = _panel()
    nfb = pd.DataFrame({"feature": cfg.VOTE_CHANNELS,
                        "nf_p05": [0.0]*5, "nf_p95": [50.0]*5}).set_index("feature")
    # all channels ~100 -> BAD_LOW channels (3) are NOT below p05; BAD_HIGH (2) ARE above p95
    # crank_recovery_t mean in window & ged_churn=100 > 50 -> 2 votes -> fires
    assert C.x2_compound(d, t=100, nfb=nfb) == 1
    nfb2 = nfb.copy(); nfb2["nf_p95"] = 1e6      # nothing above p95 now
    assert C.x2_compound(d, t=100, nfb=nfb2) == 0

def test_x2_ignores_data_after_t():
    d = _panel()
    d.loc[d.day > 50, "ged_churn"] = 1e9
    nfb = pd.DataFrame({"feature": cfg.VOTE_CHANNELS,
                        "nf_p05": [-1e12]*5, "nf_p95": [1e6]*5}).set_index("feature")
    assert C.x2_compound(d, t=50, nfb=nfb) == 0   # the 1e9 rows are day>50, invisible at t=50
```

- [ ] **Step 2: Run to verify FAIL** — `py -3 -m pytest V11.1_ALT/tests/test_covariates.py -q` → import error.

- [ ] **Step 3: Implement**

```python
"""V11.1_ALT — leakage-safe covariates from V11 daily panels.
x1(t) = log1p(#trusted days <= t with crank_recovery_t > NF p95)
x2(t) = 1 if >= VOTE_MIN of VOTE_CHANNELS cross the NF boundary (bad direction)
        on the trailing X2_TRAIL_DAYS age-window mean ending at t.
dtf is NEVER read here (post-hoc knowledge)."""
from __future__ import annotations
import importlib.util, pathlib
import numpy as np, pandas as pd
_src = pathlib.Path(__file__).resolve().parent
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_src / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cfg = _load("V11_1_ALT_config")

def trusted(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["n_eo"] >= cfg.MIN_EO_SAMPLES]

def x1_exceedance(panel: pd.DataFrame, t: float, p95: float) -> float:
    d = trusted(panel)
    n = int(((d["day"] <= t) & (pd.to_numeric(d["crank_recovery_t"], errors="coerce") > p95)).sum())
    return float(np.log1p(n))

def x2_compound(panel: pd.DataFrame, t: float, nfb: pd.DataFrame) -> int:
    d = trusted(panel)
    win = d[(d["day"] > t - cfg.X2_TRAIL_DAYS) & (d["day"] <= t)]
    if win.empty:
        return 0
    votes = 0
    for ch in cfg.VOTE_CHANNELS:
        v = pd.to_numeric(win[ch], errors="coerce").mean()
        if pd.isna(v):
            continue
        if ch in cfg.VOTE_BAD_HIGH:
            votes += int(v > nfb.loc[ch, "nf_p95"])
        else:
            votes += int(v < nfb.loc[ch, "nf_p05"])
    return int(votes >= cfg.VOTE_MIN)

def load_panel(vin: str) -> pd.DataFrame:
    return pd.read_csv(cfg.V11_FORENSICS / f"{vin}_daily.csv")

def crank_p95_from_nf(nf_vins: list[str]) -> float:
    """NF p95 of crank_recovery_t pooled over the given NF trucks' trusted days."""
    vals = []
    for v in nf_vins:
        d = trusted(load_panel(v))
        vals.append(pd.to_numeric(d["crank_recovery_t"], errors="coerce").dropna())
    return float(pd.concat(vals).quantile(0.95))

def covariate_vector(vin: str, t: float, p95: float, nfb: pd.DataFrame) -> tuple[float, int]:
    panel = load_panel(vin)
    return x1_exceedance(panel, t, p95), x2_compound(panel, t, nfb)

def main() -> None:
    """Stage output: whole-fleet covariates at each truck's endpoint (fit-time x)."""
    cfg.COV_CACHE.mkdir(parents=True, exist_ok=True)
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    nf_vins = [v for v in lc[lc.failed_flag == False]["vin_label"]]          # noqa: E712
    p95 = crank_p95_from_nf(nf_vins)
    nfb = pd.read_csv(cfg.V11_FORENSICS / "nf_baseline.csv").set_index("feature")
    rows = []
    for _, r in lc.iterrows():
        t_end = float(r["age_days_observed"])
        x1, x2 = covariate_vector(r["vin_label"], t_end, p95, nfb)
        rows.append({"vin_label": r["vin_label"], "failed_flag": int(r["failed_flag"]),
                     "t_end": t_end, "x1": round(x1, 4), "x2": x2})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.COV_CACHE / "covariates_fit.csv", index=False)
    (cfg.COV_CACHE / "crank_p95.txt").write_text(str(p95), encoding="utf-8")
    print(f"[covariates] p95={p95:.4f}; fit-time covariates:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** → 3 passed. **Run stage on real data:** `py -3 "V11.1_ALT\src\V11_1_ALT_covariates.py"` → prints p95 (~0.05) and a 25-row table; failed trucks should generally show higher x1 than NF (sanity-read it).
- [ ] **Step 5: Commit** `git add V11.1_ALT/src/V11_1_ALT_covariates.py V11.1_ALT/tests/test_covariates.py V11.1_ALT/cache/covariates && git commit -m "feat(v11.1): leakage-safe covariates x1/x2 + fit-time stage output"`

---

## Task 2: AFT survival math (extends V10.6.2 helper)

**Files:** Create `V11.1_ALT/src/V11_1_ALT_survival.py`; Test `V11.1_ALT/tests/test_survival_aft.py`.

The module COPIES the five pure functions from `V10.6.2_ALT/src/V10.6.2_ALT_survival.py` verbatim (`weibull_log_sf`, `weibull_sf`, `weibull_log_pdf`, `weibull_median`, `conditional_predictive_rul`, `predictive_rul_summary` — they are the shared math) and ADDS the AFT grid posterior below.

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util, pathlib
import numpy as np
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
S = _load("V11_1_ALT_survival")

def test_aft_beta_zero_matches_plain_grid():
    rng = np.random.default_rng(0)
    t = rng.weibull(4.0, 40) * 700; e = np.ones(40, dtype=int)
    x = np.zeros((40, 1))
    post = S.fit_aft_grid_posterior(t, e, x, prior_shape=3.5, prior_scale=650,
        prior_shape_sd=1.5, prior_scale_sd=100, shape_lo=2, shape_hi=12, shape_n=40,
        scale_lo=500, scale_hi=1100, scale_n=40, beta_grids=[np.array([0.0])])
    assert 3.0 < post["map_shape"] < 6.0 and 600 < post["map_scale0"] < 800
    assert post["map_beta"] == [0.0]

def test_aft_recovers_positive_beta():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.integers(0, 2, size=n).astype(float).reshape(-1, 1)   # binary covariate
    true_scale = 700 * np.exp(-0.6 * x[:, 0])
    t = rng.weibull(4.0, n) * true_scale; e = np.ones(n, dtype=int)
    post = S.fit_aft_grid_posterior(t, e, x, prior_shape=3.5, prior_scale=650,
        prior_shape_sd=1.5, prior_scale_sd=150, shape_lo=2, shape_hi=12, shape_n=40,
        scale_lo=400, scale_hi=1100, scale_n=50,
        beta_grids=[np.linspace(-0.2, 1.2, 29)])
    assert 0.3 < post["map_beta"][0] < 0.9   # recovers ~0.6

def test_sample_aft_posterior_shapes():
    rng = np.random.default_rng(2)
    t = rng.weibull(4.0, 30) * 700; e = np.ones(30, dtype=int)
    x = np.zeros((30, 1))
    post = S.fit_aft_grid_posterior(t, e, x, prior_shape=3.5, prior_scale=650,
        prior_shape_sd=1.5, prior_scale_sd=100, shape_lo=2, shape_hi=12, shape_n=20,
        scale_lo=500, scale_hi=1100, scale_n=20, beta_grids=[np.linspace(-0.2, 1.0, 7)])
    ks, ls, bs = S.sample_aft_posterior(post, 500, rng)
    assert ks.shape == (500,) and ls.shape == (500,) and bs.shape == (500, 1)
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement** — module = verbatim copy of the six V10.6.2 survival functions, plus:

```python
def fit_aft_grid_posterior(durations, events, x, *, prior_shape, prior_scale,
                           prior_shape_sd, prior_scale_sd, shape_lo, shape_hi, shape_n,
                           scale_lo, scale_hi, scale_n, beta_grids,
                           beta_prior_sd=0.5):
    """Grid posterior over (shape, scale0, beta_1..beta_p).
    lambda_i = scale0 * exp(-(x_i @ beta)). beta_grids: list of 1-D arrays.
    Returns dict with axes, posterior (ndim = 2+p), MAP values, and meta."""
    from scipy import stats
    durations = np.asarray(durations, float); events = np.asarray(events, int)
    x = np.asarray(x, float)
    if x.ndim == 1: x = x.reshape(-1, 1)
    p = x.shape[1]
    assert len(beta_grids) == p
    shape_grid = np.linspace(shape_lo, shape_hi, shape_n)
    scale_grid = np.linspace(scale_lo, scale_hi, scale_n)
    axes = [shape_grid, scale_grid] + list(beta_grids)
    mesh = np.meshgrid(*axes, indexing="ij")
    SHAPE, SCALE0 = mesh[0], mesh[1]
    BETAS = mesh[2:]
    log_prior = (stats.norm.logpdf(SHAPE, prior_shape, prior_shape_sd)
                 + stats.norm.logpdf(SCALE0, prior_scale, prior_scale_sd))
    for B in BETAS:
        log_prior = log_prior + stats.norm.logpdf(B, 0.0, beta_prior_sd)
    log_lik = np.zeros_like(SHAPE)
    for t_i, e_i, x_i in zip(durations, events, x):
        eta = np.zeros_like(SHAPE)
        for j, B in enumerate(BETAS):
            eta = eta + B * x_i[j]
        lam_i = SCALE0 * np.exp(-eta)
        if e_i == 1:
            log_lik += weibull_log_pdf(t_i, SHAPE, lam_i)
        else:
            log_lik += weibull_log_sf(t_i, SHAPE, lam_i)
    log_post = log_prior + log_lik
    log_post -= log_post.max()
    post = np.exp(log_post); post /= post.sum()
    midx = np.unravel_index(np.argmax(post), post.shape)
    return {"axes": axes, "posterior": post,
            "map_shape": float(shape_grid[midx[0]]),
            "map_scale0": float(scale_grid[midx[1]]),
            "map_beta": [float(beta_grids[j][midx[2 + j]]) for j in range(p)],
            "p": p}

def sample_aft_posterior(post, n, rng):
    """Draw n tuples (shape, scale0, betas[n,p]) from the AFT grid posterior,
    cell-jittered like V10.6.2's sample_posterior."""
    axes, posterior, p = post["axes"], post["posterior"], post["p"]
    flat = posterior.ravel(); flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, replace=True, p=flat)
    multi = np.unravel_index(idx, posterior.shape)
    def jit(grid, ii):
        d = (grid[1] - grid[0]) if len(grid) > 1 else 0.0
        return grid[ii] + rng.uniform(-0.5, 0.5, size=n) * d
    ks = np.clip(jit(axes[0], multi[0]), 1e-3, None)
    ls = np.clip(jit(axes[1], multi[1]), 1e-3, None)
    bs = np.column_stack([jit(axes[2 + j], multi[2 + j]) for j in range(p)]) if p else np.zeros((n, 0))
    return ks, ls, bs

def scale_for(scale0_s, beta_s, x_vec):
    """Per-draw per-truck scale: scale0 * exp(-(x @ beta)). x_vec shape (p,)."""
    eta = beta_s @ np.asarray(x_vec, float) if beta_s.size else 0.0
    return scale0_s * np.exp(-eta)
```

- [ ] **Step 4: Run tests** → 3 passed (test 2 may take ~30 s — 40×50×29 grid × 60 obs).
- [ ] **Step 5: Commit** `git add V11.1_ALT/src/V11_1_ALT_survival.py V11.1_ALT/tests/test_survival_aft.py && git commit -m "feat(v11.1): AFT grid posterior + sampling (reduces to V10.6.2 at beta=0)"`

---

## Task 3: Weibull fleet stage (fit M0/M1/M2)

**Files:** Create `V11.1_ALT/src/V11_1_ALT_weibull_fleet.py`.

Adapt `V10.6.2_ALT/src/V10.6.2_ALT_weibull_fleet.py` (read it; same step structure, prints, MLE reference via lifelines) with these exact changes:
1. Load covariates from `cfg.COV_CACHE / "covariates_fit.csv"` joined to lifecycle on `vin_label`.
2. For each variant in `cfg.VARIANTS`: M0 → `x = zeros((25,1))`, `beta_grids=[np.array([0.0])]`; M1 → `x = covs[["x1"]]`, `beta_grids=[np.linspace(cfg.BETA1_LO, cfg.BETA1_HI, cfg.BETA1_N)]`; M2 → `x = covs[["x1","x2"]]`, beta_grids for β1 and β2 (`np.linspace(cfg.BETA2_LO, cfg.BETA2_HI, cfg.BETA2_N)`).
3. Fit with `S.fit_aft_grid_posterior` (priors/grid bounds from inherited cfg constants), sample `cfg.N_PREDICTIVE_DRAWS` with `S.sample_aft_posterior`, save per variant: `posterior_samples_<variant>.csv` (cols `shape,scale0,beta1[,beta2]`), `aft_params_<variant>.json` (MAP values + grid meta).
4. M0 additionally writes V10.6.2-compatible `fleet_weibull_params.json`, `fleet_survival_curve.csv`, `posterior_samples.csv` (cols shape,scale — scale=scale0) so all V10.6.2-style graph code works unchanged.
5. Sanity assert in-stage: M0's MAP shape within ±0.3 and scale within ±30 of V10.6.2's published (5.17, 771) — same data, same priors, must reproduce.

- [ ] Run: `py -3 "V11.1_ALT\src\V11_1_ALT_weibull_fleet.py"` → prints 3 variants' MAP (M0 ≈ shape 5.17/scale 771; M1/M2 report MAP β). M2 grid is 100×100×25×18 ≈ 4.5M cells × 25 trucks — expect 1–5 min.
- [ ] Commit: `git add V11.1_ALT/src/V11_1_ALT_weibull_fleet.py V11.1_ALT/cache/weibull && git commit -m "feat(v11.1): AFT fleet fits M0/M1/M2 (M0 reproduces V10.6.2)"`

---

## Task 4: Backtest with truncated covariates (decides the variant)

**Files:** Create `V11.1_ALT/src/V11_1_ALT_backtest.py`; Test `V11.1_ALT/tests/test_backtest_trunc.py`.

- [ ] **Step 1: Failing test for the truncation helper**

```python
import importlib.util, pathlib
import numpy as np, pandas as pd
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
B = _load("V11_1_ALT_backtest")

def test_rewind_time_and_truncation_contract():
    # truck failed at ttf=600; horizon 90 -> rewind t=510; covariate fn must be
    # called with exactly t=510 (asserted via a spy)
    calls = []
    def spy_cov(vin, t, p95, nfb):
        calls.append((vin, t)); return 0.5, 1
    x = B.rewind_covariates("VINX", ttf=600.0, horizon=90, cov_fn=spy_cov, p95=0.05, nfb=None)
    assert calls == [("VINX", 510.0)] and x == (0.5, 1)
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement the stage.** Adapt `V10.6.2_ALT/src/V10.6.2_ALT_backtest.py` (read it for the dummy-A fleet-clock definition, fold loop, metrics, and json schema — keep those identical) with these exact additions:

```python
def rewind_covariates(vin, ttf, horizon, cov_fn, p95, nfb):
    """Covariates at the rewind point t = ttf - horizon. The ONLY covariate
    entry point the backtest may use (G-LEAK audits this)."""
    return cov_fn(vin, float(ttf) - float(horizon), p95, nfb)
```

Per LOVO fold (held-out failed VIN v, horizons 270/180/90), per variant:
1. Training set = other 24 trucks; durations/events from lifecycle; training covariates from `covariates_fit.csv` (failed: x at failure; censored: x at censor) — for M0 zeros.
2. Refit `fit_aft_grid_posterior` on the 24 (per-fold refit, like V10.6.2). Use the SAME beta grids as Task 3.
3. Held-out covariate: `rewind_covariates(v, ttf_v, h, covariates.covariate_vector, p95, nfb)` — p95/nfb computed once from all 15 NF (they are never held out in LOVO; note this in the json as `nf_envelope: "all_15_nf"`).
4. Predict: sample posterior (2000 draws), per-draw scale via `S.scale_for`, conditional median RUL at age `a = ttf_v - h` via `S.conditional_predictive_rul`; error `|median - h|`; record PI(10,90) coverage of true h and PI width.
5. Aggregate per variant: `mae_model`, per-horizon MAEs, `pi_coverage`, `mean_pi_width`, signed-rank p vs dummy-A and vs M0's per-fold absolute errors (`scipy.stats.wilcoxon`).
6. **Variant selection** (coded, deterministic): chosen = best rewound overall MAE among variants that pass `pi_coverage >= 0.80`; ties → fewer parameters. Write `backtest_results.json` with all variants' metrics + `"chosen_variant"`, `"selection_rule"`, and per-fold residual CSV per variant.

- [ ] **Step 4: Run unit test** → 1 passed. **Run stage:** `py -3 "V11.1_ALT\src\V11_1_ALT_backtest.py"` — 10 folds × 3 variants refits; M2 fold fits dominate (~10–60 s each) → total possibly 20–45 min: RUN IN BACKGROUND and wait. Expected print: per-variant table (MAE vs dummy 49.7 vs M0 ~125–142) + `chosen_variant`.
- [ ] **Step 5: Commit** with cache: `git add V11.1_ALT/src/V11_1_ALT_backtest.py V11.1_ALT/tests/test_backtest_trunc.py V11.1_ALT/cache/backtest && git commit -m "feat(v11.1): rewound LOVO backtest with truncated covariates; variant selection"`

---

## Task 5: Predictive RUL (chosen variant)

**Files:** Create `V11.1_ALT/src/V11_1_ALT_predictive_rul.py`.

Adapt `V10.6.2_ALT/src/V10.6.2_ALT_predictive_rul.py` (read it; keep row schema + display-unit logic identical) with exact changes: read `chosen_variant` from `backtest_results.json`; load `posterior_samples_<chosen>.csv`; for each non-failed truck compute current covariates `covariate_vector(vin, age_days_observed, p95, nfb)`, per-draw scales via `S.scale_for`, then `S.predictive_rul_summary(a, shape_s, scale_i_s, rng, ...)` unchanged. Add columns `x1`, `x2`, `variant` to the output CSV `predictive_rul_per_vin.csv`.

- [ ] Run: `py -3 "V11.1_ALT\src\V11_1_ALT_predictive_rul.py"` → 25-row CSV; NF table printed; if chosen=M0 the numbers must match V10.6.2's predictive table to within sampling noise (~±5d).
- [ ] Commit.

---

## Task 6: Emergency layer (3 channels + k calibration)

**Files:** Create `V11.1_ALT/src/V11_1_ALT_emergency.py`; Test `V11.1_ALT/tests/test_emergency.py`.

- [ ] **Step 1: Failing tests**

```python
import importlib.util, pathlib
import numpy as np, pandas as pd
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
E = _load("V11_1_ALT_emergency")

def test_exceedance_trailing_window():
    d = pd.DataFrame({"day": range(1, 61), "n_eo": 500, "crank_recovery_t": 0.0})
    d.loc[d.day.isin([40, 45]), "crank_recovery_t"] = 9.0    # 2 exceedances within 30d
    first = E.exceedance_first_fire(d, p95=0.05, k=2, trail=30)
    assert first == 45                                        # fires on the day the 2nd lands
    assert E.exceedance_first_fire(d, p95=0.05, k=3, trail=30) is None

def test_exceedance_respects_window():
    d = pd.DataFrame({"day": range(1, 101), "n_eo": 500, "crank_recovery_t": 0.0})
    d.loc[d.day.isin([10, 80]), "crank_recovery_t"] = 9.0     # 70 days apart
    assert E.exceedance_first_fire(d, p95=0.05, k=2, trail=30) is None
```

- [ ] **Step 2: FAIL run.** **Step 3: Implement**

```python
def exceedance_first_fire(panel, p95, k, trail):
    d = panel[panel["n_eo"] >= cfg.MIN_EO_SAMPLES].sort_values("day")
    days = d.loc[pd.to_numeric(d["crank_recovery_t"], errors="coerce") > p95, "day"].to_numpy()
    for i in range(len(days)):
        if i + 1 >= k and days[i] - days[i + 1 - k] <= trail:
            return int(days[i])
    return None
```

`main()`: (a) GED=2 storm — per VIN daily `ged2_cnt` max from V11 panels, fired if any day ≥ `GED_EMERGENCY_DAILY_COUNT_MIN` (200), lead = dtf at first fire day (reporting only); (b) calibrate k = smallest integer ≥ `EXCEED_K_START` with **0/15 NF** first-fires, print the calibration table k→(NF fires, failed fires); (c) compound channel — `x2_compound(panel, t, nfb)` evaluated at every trusted day, first day it returns 1; NF false-fire check with the same envelope. Output `EMERG_CACHE/emergency_per_vin.csv`: vin, failed_flag, ged_fired/lead, exceed_fired/lead/k, compound_fired/lead, any_fired. Leads computed from dtf for failed trucks (post-hoc reporting is fine; the FIRING rule never uses dtf).

- [ ] **Step 4: Tests pass; run stage** (reads 25 CSVs; ~1–2 min). Expected: GED 2/10; exceedance and compound recalls printed with 0/15 NF (if NF fires at k=2, the calibration loop raises k and reports).
- [ ] **Step 5: Commit.**

---

## Task 7: Decisions + assemble + narratives

**Files:** Create `V11.1_ALT/src/V11_1_ALT_decisions.py`, `V11_1_ALT_assemble_rul.py`, `V11_1_ALT_narrative_rul.py`.

Adapt the three V10.6.2 counterparts (read each; keep their input/output schemas) with exact changes:
- decisions: risk band from frozen `RIDGE_PROB_CSV` + `RIDGE_DECISION_THR` (unchanged); time dimension from V11.1's `predictive_rul_per_vin.csv` (`short` if `rul_p10_days < SHORT_RUL_HORIZON_DAYS`); NEW third dimension `emergency_state` ∈ {none, early-watch, emergency} from `emergency_per_vin.csv` (early-watch = exceed_fired or compound_fired; emergency = ged_fired). Recommendation matrix: emergency → "service immediately"; early-watch + (red band or short) → "inspect within 2 weeks"; early-watch → "schedule inspection"; else V10.6.2 wording.
- assemble: merge predictive + decisions + emergency + covariates into `final_rul_per_vin.csv` (superset of V10.6.2 columns + `x1,x2,variant,emergency_state`).
- narratives: per-VIN sentences; must state variant used and, if M0 fallback, say "covariates did not improve timing; fleet-curve window shown".

- [ ] Run all three; spot-read 3 narratives; commit.

---

## Task 8: Verify gates + orchestrator

**Files:** Create `V11.1_ALT/src/V11_1_ALT_verify.py`, `V11_1_ALT_orchestrator.py`; Test `V11.1_ALT/tests/test_verify_gates.py`.

- [ ] **Step 1: Failing tests** for the two pure gate helpers:

```python
import importlib.util, pathlib
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V = _load("V11_1_ALT_verify")

def test_g_beta_ship_rules():
    m0 = {"mae": 140.0, "pi_coverage": 0.87, "mean_pi_width": 400.0}
    good = {"mae": 45.0, "pi_coverage": 0.85, "mean_pi_width": 380.0, "p_vs_dummy": 0.01, "p_vs_m0": 0.01}
    bad  = {"mae": 130.0, "pi_coverage": 0.85, "mean_pi_width": 395.0, "p_vs_dummy": 0.4, "p_vs_m0": 0.3}
    sharp = {"mae": 138.0, "pi_coverage": 0.83, "mean_pi_width": 320.0, "p_vs_dummy": 0.4, "p_vs_m0": 0.4}
    assert V.g_beta_ships(good, m0, dummy_mae=49.7) is True     # beats both, significant
    assert V.g_beta_ships(bad, m0, dummy_mae=49.7) is False
    assert V.g_beta_ships(sharp, m0, dummy_mae=49.7) is True    # >=15% narrower at >=80% coverage

def test_scan_forbidden():
    assert V.scan_forbidden("import sklearn") == ["sklearn"]
    assert V.scan_forbidden("import numpy") == []
```

- [ ] **Step 2: FAIL run.** **Step 3: Implement.** `g_beta_ships(variant, m0, dummy_mae)` returns True iff (`variant.mae < dummy_mae` AND `variant.mae < m0.mae` AND both p-values < `SIGNED_RANK_ALPHA`) OR (`variant.mean_pi_width <= (1 - PI_WIDTH_SHRINK_MIN) * m0.mean_pi_width` AND `variant.pi_coverage >= 0.80`). `scan_forbidden` as in V11. `main()` gates:
  - **G-LEAK**: recompute `rewind_covariates` for every (fold, horizon) from scratch and compare to the values logged by the backtest in `per_fold_residuals_<variant>.csv` (backtest must log x1,x2 per fold-horizon — ensure Task 4 wrote them); any |Δ| > 1e-9 → fail.
  - **G-BETA**: if `chosen_variant != "M0"`, assert `g_beta_ships(...)` is True; if M0, assert the json's `selection_rule` recorded why (NO_IMPROVEMENT path) — both honest outcomes pass; an unjustified covariate ship fails.
  - **G-W6**: forbidden scan over `V11.1_ALT/src` (config excluded).
  - **G-EMERG**: from `emergency_per_vin.csv`: NF any_fired == 0/15; failed ged_fired ≥ 2, compound_fired ≥ 4 (non-regression vs V11), exceed channel reported.
  - **G-COVER**: chosen variant `pi_coverage >= 0.80`.
  Exit 1 with the failure list, else print PASS. Write `results/V11.1_ALT_verification.json`.
- [ ] **Step 4: Orchestrator** — same `_run` pattern as `V11_ALT_heuristics_orchestrator.py`, SCRIPTS = [covariates, weibull_fleet, backtest, predictive_rul, emergency, decisions, assemble_rul, narrative_rul, verify] with GATE_FAIL halt. (Graphs/reports run in Tasks 9–11 after verify is green.)
- [ ] **Step 5: Unit tests pass; run verify standalone** (fast) → expect PASS with the real chosen variant. **Run the full orchestrator in BACKGROUND** as the end-to-end reproducibility check (~30–60 min, dominated by backtest). Commit after green.

---

## Task 9: Graphs (4 core + 25 per-VIN RUL curves + 3 fleet overlays)

**Files:** Create `V11.1_ALT/src/V11_1_ALT_rul_graphs.py`; Create `V11.1_ALT/visualizations/visual script/generate_all_rul_graphs_v11_1.py` and `generate_fleet_overlay_graphs_v11_1.py`.

Adapt the three existing generators (exact sources listed in the header) with ONLY these data-source changes — visual style, 4-layer architecture, zones, calendar axes, legends stay identical:
- All `V10.6.2_ALT/cache/...` paths → `V11.1_ALT/cache/...`; `final_rul_per_vin.csv` and `posterior_samples.csv` (M0-compatible file from Task 3, or chosen-variant samples with per-VIN scale — use chosen; the per-VIN curve code consumes shape/scale columns, so write a per-VIN effective-scale samples file `posterior_samples_pervin/<VIN>.csv` from Task 5's draws if chosen ≠ M0; if chosen == M0 reuse the fleet file).
- backtest_accuracy bars: add the per-variant comparison (M0/M1/M2 vs dummy) — grouped bars per horizon.
- per-VIN curve annotation: add emergency_state badge and x1/x2 values in the legend block.
- fleet overlays + `fleet_statistics_summary.csv/.xlsx` + `Fleet_graphs_generation_report.md` regenerate with V11.1 columns (add `variant`, `emergency_state`).

- [ ] Run all three generators (per-VIN script ~2–5 min); spot-open 2 PNGs (1 failed, 1 NF) to confirm rendering; commit.

---

## Task 10: Customer report (md + xlsx) + 3-way comparison

**Files:** Create `V11.1_ALT/src/V11_1_ALT_markdown_report.py`, `V11_1_ALT_excel_report.py`.

Adapt the two V10.6.2 counterparts with: V11.1 cache paths; an added section **"Covariate verdict"** stating the chosen variant, the G-BETA evidence (MAEs, p-values, PI widths), and the plain-language conclusion (improvement or NO_IMPROVEMENT); an added section **"Emergency channels"** (3 channels, recalls, 0/15 NF, calibration k); and a 3-way table V10.6.2 vs V11 vs V11.1 (classifier AUROC same; precursor recalls from V11; V11.1's backtest + emergency + decisions). Excel adds sheets `Covariates`, `Emergency`, `Variant_Comparison`.

- [ ] Run both; read the md fully; confirm no fabricated numbers (every figure from a V11.1 cache); commit.

---

## Task 11: Decks + DATA_SOURCES + AUDIT_REPORT

**Files:** Create `V11.1_ALT/presentation/build_technical_presentation.py`, `build_business_presentation.py`, `DATA_SOURCES.md`, `AUDIT_REPORT.md`.

Adapt the two V11 builders (same helper pattern, navy/gold, live `_nums()` from caches): technical ~15 slides = V11 deck structure + 3 new slides (Covariate model & verdict — embeds backtest bars; Emergency channels; Decision matrix). Business 5 slides with the covariate verdict stated honestly. DATA_SOURCES/AUDIT mirror V11's, every claim → file table.

- [ ] Run both builders; confirm pptx sizes > 0; write the 2 md docs; commit.

---

## Task 12: Final verification & wrap

- [ ] Full test suite: `py -3 -m pytest V11.1_ALT/tests -q` → all pass.
- [ ] Re-run `V11_1_ALT_verify.py` → PASS printed.
- [ ] Inventory check (graphs/reports/decks/caches all non-empty).
- [ ] Honesty cross-check: if chosen == M0, confirm the report/deck/narratives all state NO_IMPROVEMENT plainly and show M0 numbers ≡ V10.6.2; if a covariate variant shipped, confirm the G-BETA evidence appears verbatim in the report.
- [ ] Final commit.

---

## Self-Review

**Spec coverage:** §2 covariates→Task 1; §3 AFT variants→Tasks 2–3; §4 backtest+selection→Task 4; predictive→Task 5; §5 emergency 3-channel+k-calibration→Task 6; §6 decisions→Task 7; §7 gates G-LEAK/G-BETA/G-W6/G-EMERG/G-COVER→Task 8 (G-LEAK depends on Task 4 logging x1,x2 per fold — stated in both tasks); §8 artifact suite→Tasks 9–11; §10 TDD on new math→Tasks 1,2,4,6,8 have explicit tests. ✓
**Placeholders:** none — adaptation tasks name the exact committed source file to read and enumerate the exact changes; new-math tasks carry complete code. ✓
**Name consistency:** `fit_aft_grid_posterior` / `sample_aft_posterior` / `scale_for` (Task 2) used in Tasks 3–5; `covariate_vector` / `crank_p95_from_nf` / `x1_exceedance` / `x2_compound` (Task 1) used in Tasks 4–6; `rewind_covariates` (Task 4) audited in Task 8; `g_beta_ships` / `scan_forbidden` (Task 8) self-contained. Config names (`COV_CACHE`, `VARIANTS`, `BETA1_*`, `EXCEED_*`) defined in Task 0 before use. ✓
**Known risks (stated, not hidden):** M2 per-fold refits make the backtest the long pole (run in background); x2 may be all-zero for NF trucks at fit time (fine — β2 then identified only by failed trucks, the β prior regularizes); expectation remains that G-BETA may select M0 — that path is first-class.
