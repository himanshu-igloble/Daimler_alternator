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
            "ged2_frac", "ged2_cnt", "sma_starts",
            "vsi_onset_rpm", "vsi_resid_negfrac", "crank_recovery_t",
            "cranks_per_ehr", "crank_dur_mean", "ged1_frac", "ged3_frac",
            "ged_churn", "idle_vsi_var", "idle_vsi_zcr", "sag_highload_frac",
            "sag_idle_frac", "uv_dose_day"}
BAD_LOW = {"vsi_mean", "vsi_min", "vsi_p05", "idle_vsi_mean", "cruise_vsi_mean",
           "resting_vsi_mean", "crank_vsi_min", "rpm_mean",
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
