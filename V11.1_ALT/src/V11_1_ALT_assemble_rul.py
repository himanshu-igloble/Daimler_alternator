"""
V11.1_ALT — Assemble final deliverable table  (Task 7)
=======================================================
Merges all validated per-VIN artefacts into one superset table.

Sources (read-only unless explicitly our output):
  predictive_rul_per_vin.csv   — RUL bands + x1/x2 + variant (from our RUL module)
  decisions_per_vin.csv        — risk_band/above_thr/time_dim/emergency_state/recommendation
  emergency_per_vin.csv        — ged_fired/compound_current/early_watch_current
  covariates_fit.csv           — x1/x2 (assert must match predictive's to within 1e-6)
  V10.6.2_ALT/cache/rul/fleet_window.json  — empirical median_ttf_days (READ-ONLY reference)

Outputs:
  cache/rul/final_rul_per_vin.csv   — full superset (all intermediate columns)

Columns kept:
  all predictive cols + risk_band/above_thr/time_dim/emergency_state/recommendation
  + ged_fired/compound_current/early_watch_current
  + x1/x2  (from predictive; asserted consistent with covariates_fit for non-failed)
  + variant
  + fleet_window_median_days  (empirical 601.0 from V10.6.2 fleet_window.json)

Honesty notes:
  * The chosen_variant is M0 (NO_IMPROVEMENT from backtest): covariates did not
    improve per-truck timing; x1/x2 are informational only.
  * fleet_window_median_days = empirical failed-truck median; the survival-model
    median is not used here (biased by right-censoring).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config", "V11_1_ALT_config.py")

# READ-ONLY reference to V10.6.2 fleet window (not recomputed here)
_FLEET_WINDOW_JSON = (
    pathlib.Path(__file__).resolve().parents[2]
    / "V10.6.2_ALT" / "cache" / "rul" / "fleet_window.json"
)


def main() -> None:
    rul_cache = pathlib.Path(cfg.RUL_CACHE)
    rul_cache.mkdir(parents=True, exist_ok=True)

    # ---- load all inputs ------------------------------------------------
    rul = pd.read_csv(rul_cache / "predictive_rul_per_vin.csv")
    dec = pd.read_csv(rul_cache / "decisions_per_vin.csv")
    emerg = pd.read_csv(pathlib.Path(cfg.EMERG_CACHE) / "emergency_per_vin.csv")
    cov = pd.read_csv(pathlib.Path(cfg.COV_CACHE) / "covariates_fit.csv")

    # ---- fleet window scalar (read-only reference) ----------------------
    with open(_FLEET_WINDOW_JSON) as fh:
        fw = json.load(fh)
    fleet_window_median = float(fw["median_ttf_days"])   # 601.0

    # ---- assert x1/x2 consistency between predictive and covariates ----
    # Only check non-failed VINs (failed VINs have RUL rows from the RUL module
    # which may carry forward the cov x1/x2; also covariates_fit covers all 25).
    rul_nf = rul[rul["failed_flag"] == 0][["vin_label", "x1", "x2"]].set_index("vin_label")
    cov_nf = cov[cov["failed_flag"] == 0][["vin_label", "x1", "x2"]].set_index("vin_label")
    common = rul_nf.index.intersection(cov_nf.index)
    if len(common) > 0:
        diff_x1 = (rul_nf.loc[common, "x1"] - cov_nf.loc[common, "x1"]).abs().max()
        diff_x2 = (rul_nf.loc[common, "x2"] - cov_nf.loc[common, "x2"]).abs().max()
        assert diff_x1 < 1e-6, (
            f"x1 mismatch between predictive_rul and covariates_fit: max_abs_diff={diff_x1}"
        )
        assert diff_x2 < 1e-6, (
            f"x2 mismatch between predictive_rul and covariates_fit: max_abs_diff={diff_x2}"
        )
        print(f"[assemble] x1/x2 consistency OK  "
              f"(max_abs_diff x1={diff_x1:.2e}, x2={diff_x2:.2e}) "
              f"over {len(common)} non-failed VINs")

    # ---- merge ----------------------------------------------------------
    # Start from predictive (full column set including x1, x2, variant)
    df = rul.copy()

    # Decisions: drop duplicate failed_flag; keep ridge_prob + band columns
    dec_cols = ["vin_label", "ridge_prob", "risk_band", "above_thr", "time_dim",
                "emergency_state", "recommendation"]
    df = df.merge(dec[dec_cols], on="vin_label", how="left")

    # Emergency: keep ged_fired / compound_current / early_watch_current
    emerg_cols = ["vin_label", "ged_fired", "compound_current", "early_watch_current"]
    df = df.merge(emerg[emerg_cols], on="vin_label", how="left")

    # Fleet window constant column
    df["fleet_window_median_days"] = fleet_window_median

    # ---- sort: non-failed first, then by risk desc, then RUL asc --------
    _band_order = {"red": 0, "amber": 1, "green": 2}
    df["_band_ord"] = df["risk_band"].map(_band_order).fillna(9)
    df = df.sort_values(
        ["failed_flag", "_band_ord", "median_rul_days"],
        ascending=[True, True, True],
    ).drop(columns="_band_ord").reset_index(drop=True)

    # ---- save -----------------------------------------------------------
    out_path = rul_cache / "final_rul_per_vin.csv"
    df.to_csv(out_path, index=False)

    print(f"[assemble] Saved final_rul_per_vin.csv  shape={df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"  fleet_window_median_days = {fleet_window_median}")
    nf_count = (df["failed_flag"] == 0).sum()
    f_count = (df["failed_flag"] == 1).sum()
    print(f"  Non-failed: {nf_count}  |  Failed: {f_count}  |  Total: {len(df)}")


if __name__ == "__main__":
    main()
