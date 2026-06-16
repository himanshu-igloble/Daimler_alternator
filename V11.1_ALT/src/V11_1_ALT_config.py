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
