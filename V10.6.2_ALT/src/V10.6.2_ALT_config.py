"""
V10.6.2 Alternator Honest-Baseline RUL — Config & Constants
============================================================
Inherits the V10.6 intelligence-layer config (which itself inherits
V10.5.3 -> V5.2.1 -> V5.2), then adds the V10.6.2-alpha constants.

Design decisions baked in (see plan
C:/Users/suraj_pradhan/.claude/plans/lets-create-a-detailed-async-stonebraker.md):

  D3  Survival cohort = ALL 25 VINs (10 events + 15 censored);
      LOVO is over the 10 FAILED VINs.  The TRAIN/TEST/VAL split is
      *not* used for survival (it only ever mattered for the now-frozen
      classifier).
  D4  Use the POSTERIOR shape (~5.1), never the MLE shape (~7.6).
  D6  No historical ridge_prob exists (no V10.5.3 model artifact).
      Time-rewound backtest is calendar/usage-only; the ridge tier is
      static and explicitly labelled "whole-life input".
  D7  Predictive intervals condition on survival-to-now and are sampled
      from the Bayesian posterior grid (epistemic + aleatoric in one
      draw).  No params-only bootstrap CI, no divergent MLE bootstrap.
  D9  Success metric = beat the fleet-clock dummy on out-of-sample
      day-MAE + 80% predictive-interval coverage.
"""
from __future__ import annotations

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


# ---------------------------------------------------------------------------
# 1. Inherit V10.6 (-> V10.5.3 -> V5.2.1 -> V5.2)
#    Provides: ALL_VINS, TRAIN_VINS, TEST_VINS, VALIDATION_VINS,
#    FAILED_VIN_SET, SILENT_FAILURE_VINS, BAND_GREEN_MAX/BAND_AMBER_MAX,
#    GED_BURST_PER_HOUR, LIFECYCLE_STAGE_CUTS_KM, paths, etc.
# ---------------------------------------------------------------------------
_v10_6_cfg_path = (
    Path(__file__).resolve().parents[2]
    / "V10.6_ALT" / "src" / "V10.6_ALT_config.py"
)
_spec = spec_from_file_location("_v10_6_cfg", str(_v10_6_cfg_path))
_v10_6_cfg = module_from_spec(_spec)
_spec.loader.exec_module(_v10_6_cfg)

for _attr in dir(_v10_6_cfg):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_v10_6_cfg, _attr)

# ---------------------------------------------------------------------------
# 2. V10.6.2 IDENTITY
# ---------------------------------------------------------------------------
VERSION = "V10.6.2_ALT"
FILE_PREFIX = "V10.6.2_ALT_"

# ---------------------------------------------------------------------------
# 3. PATHS
# ---------------------------------------------------------------------------
V10_6_2_ROOT = Path(__file__).resolve().parents[1]

WEIBULL_CACHE       = V10_6_2_ROOT / "cache" / "weibull"
RUL_CACHE           = V10_6_2_ROOT / "cache" / "rul"
BACKTEST_CACHE      = V10_6_2_ROOT / "cache" / "backtest"
GED_EMERGENCY_CACHE = V10_6_2_ROOT / "cache" / "ged_emergency"

RESULTS_DIR_V2 = V10_6_2_ROOT / "results"
VIZ_DIR_V2     = V10_6_2_ROOT / "visualizations"
REPORTS_DIR_V2 = V10_6_2_ROOT / "reports"

# V10.6 caches (READ-ONLY — never write here)
V10_6_ROOT_DIR  = Path(__file__).resolve().parents[2] / "V10.6_ALT"
V10_6_CACHE     = V10_6_ROOT_DIR / "cache"
V10_6_WEEKLY    = V10_6_CACHE / "weekly"
V10_6_LIFECYCLE = V10_6_CACHE / "lifecycle"
V10_6_RIDGE     = V10_6_CACHE / "ridge"
V10_6_RULES     = V10_6_CACHE / "rules"
V10_6_RESULTS   = V10_6_ROOT_DIR / "results"

# Concrete read-only files
LIFECYCLE_PARQUET   = V10_6_LIFECYCLE / "vin_lifecycle.parquet"
RIDGE_PROB_CSV      = V10_6_RIDGE / "ridge_prob_rescaled.csv"
RULE_HITS_PARQUET   = V10_6_RULES / "rule_hits.parquet"
HYBRID_RISK_CSV     = V10_6_RESULTS / "V10.6_ALT_hybrid_risk.csv"

# ---------------------------------------------------------------------------
# 4. SURVIVAL COHORT (D3)
# ---------------------------------------------------------------------------
# Fit the fleet Weibull on ALL 25 VINs (10 events + 15 right-censored).
# LOVO validation iterates over the 10 FAILED VINs only.
SURVIVAL_USE_ALL_VINS = True          # ignore TRAIN/TEST/VAL for survival
# LOVO_FAILED resolved at runtime from FAILED_VIN_SET (inherited).

# ---------------------------------------------------------------------------
# 5. WEIBULL PRIORS + GRID POSTERIOR (copied from V10.6.1; wear-out mode)
# ---------------------------------------------------------------------------
WEIBULL_PRIOR_SHAPE    = 3.5
WEIBULL_PRIOR_SCALE    = 650.0
WEIBULL_PRIOR_SHAPE_SD = 1.5
WEIBULL_PRIOR_SCALE_SD = 100.0

GRID_SHAPE_LO, GRID_SHAPE_HI, GRID_SHAPE_N = 2.0, 12.0, 200
GRID_SCALE_LO, GRID_SCALE_HI, GRID_SCALE_N = 500.0, 1100.0, 200

# ---------------------------------------------------------------------------
# 6. PREDICTIVE INTERVALS (D4, D7)
# ---------------------------------------------------------------------------
# Point estimate = posterior-grid MAP/mean shape & scale (NOT the MLE).
USE_POSTERIOR_SHAPE = True
N_PREDICTIVE_DRAWS  = 10_000          # per-VIN samples for the conditional PI
PI_LOWER_PCT = 10.0                   # 80% predictive interval
PI_UPPER_PCT = 90.0
RNG_SEED = 42

# ---------------------------------------------------------------------------
# 7. BACKTEST (D8, D9)
# ---------------------------------------------------------------------------
BACKTEST_HORIZONS_DAYS = [270, 180, 90]   # time-rewound evaluation points
PI_COVERAGE_TARGET = 0.80                 # >= 8/10 LOVO folds
# Decision gate: model day-MAE must beat Dummy-A (fleet-clock).

# ---------------------------------------------------------------------------
# 8. FROZEN CLASSIFIER DECISION THRESHOLD
# ---------------------------------------------------------------------------
# Youden-optimal threshold from the FROZEN V10.5.3 ridge spec
# (V5.2_ALT/models/classification/V10.5.3_20_5_ALT_ridge_spec.json).
RIDGE_DECISION_THR = 0.4456
# Tier bands (inherited as BAND_GREEN_MAX=35, BAND_AMBER_MAX=55 on 0-100).

# ---------------------------------------------------------------------------
# 9. DECISION ENGINE (W4 — 2x2 risk x time)
# ---------------------------------------------------------------------------
SHORT_RUL_HORIZON_DAYS = 90       # "short" if PI lower (p10) < this

# ---------------------------------------------------------------------------
# 10. GED=2 EMERGENCY LAYER (W3)
# ---------------------------------------------------------------------------
# Primary fire signal is the validated weekly burst flag (any hour in the
# week with > GED_BURST_PER_HOUR GED=2 events).  A weekly-count floor guards
# against single-hour noise.
GED_EMERGENCY_USE_WEEKLY_FLAG = True
GED_EMERGENCY_WEEKLY_COUNT_MIN = 50      # >= this many GED=2 events in a week
# Primary signal: re-derived DAILY GED=2 count from the raw per-VIN parquets.
# Daily-max GED=2 counts are strongly BIMODAL across the fleet:
#   healthy-truck blips top out at 142/day (VIN11_NF), 98 (VIN4_NF), <=27 rest;
#   firing failures reach 2,897/day (VIN10_F) and 12,367/day (VIN1_F).
# A 20x gap (142 -> 2,897) lets the threshold sit safely in the gap; 200 is an
# order of magnitude below the firing failures and well above the healthy
# blips, giving 0 false alarms without tuning to a target.  (The cited >=100 in
# lead_time_analysis.md narrowly catches VIN11_NF's 142 blip.)
# DAYS_TO_FAILURE in the raw parquet gives exact lead times.
# V52_PARQUET_DIR / V52_PARQUET_PREFIX inherited from V10.6 cfg.
GED_EMERGENCY_DAILY_COUNT_MIN = 200

# ---------------------------------------------------------------------------
# 11. FLEET AVERAGE CONVERSION FACTORS (fallbacks for km / engine-hours)
# ---------------------------------------------------------------------------
FLEET_AVG_KM_PER_DAY   = 200.4
FLEET_AVG_EHRS_PER_DAY = 7.55

# ---------------------------------------------------------------------------
# 12. FROZEN RIDGE INTEGRITY (W6 gate)
# ---------------------------------------------------------------------------
# SHA-256 of ridge_prob_rescaled.csv is recorded at verify time for audit.
# Forbidden patterns are CLASSIFIER-specific: survival fitting (lifelines
# WeibullFitter.fit) is legitimate and allowed; only re-training / re-deriving
# the frozen V10.5.3 classifier (or the V10.6.1 isotonic Tier-B leakage) is
# banned.  Scanned across src/ EXCEPT config.py (which lists them literally).
FORBIDDEN_FIT_PATTERNS = ["RidgeClassifier", "IsotonicRegression", "sklearn", "Ridge("]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"V10.6.2 config loaded: {VERSION}")
    print(f"  ALL_VINS:        {len(ALL_VINS)}")
    print(f"  FAILED_VIN_SET:  {len(FAILED_VIN_SET)}  -> LOVO folds")
    print(f"  RIDGE_DECISION_THR: {RIDGE_DECISION_THR}")
    print(f"  BAND cuts:       green<{BAND_GREEN_MAX}  amber<{BAND_AMBER_MAX}  red>=")
    print(f"  LIFECYCLE_PARQUET exists: {LIFECYCLE_PARQUET.exists()}")
    print(f"  RIDGE_PROB_CSV   exists: {RIDGE_PROB_CSV.exists()}")
    print(f"  RULE_HITS_PARQUET exists: {RULE_HITS_PARQUET.exists()}")
    print(f"  HYBRID_RISK_CSV  exists: {HYBRID_RISK_CSV.exists()}")
    print(f"  GED_BURST_PER_HOUR (inherited): {GED_BURST_PER_HOUR}")
    print("Config OK.")
