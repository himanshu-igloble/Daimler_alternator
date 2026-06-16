"""
V10.6.2 Alternator — Assemble final deliverable table  (plan W5a)
=================================================================
Joins the validated pieces into one per-VIN table and computes the
EMPIRICALLY-ANCHORED fleet replacement window.

Key honesty choices:
  * Failed VINs: median_rul_days = 0, plus `would_have_flagged_lead_days`
    sourced from the GED=2 emergency layer (the only real precursor) — not a
    bare "already failed".
  * The headline fleet window is anchored on the EMPIRICAL failed-truck TTF
    distribution (backtest-validated best point estimate, ~602d), NOT the
    censoring-biased survival median (~718d, kept for context only).
  * Per-truck RUL band (survival-conditioned) is carried but flagged: its
    point estimate does NOT beat the fleet clock (backtest); its intervals
    are well-calibrated.

Outputs:
  results/V10.6.2_ALT_rul_predictions.csv   (customer-facing)
  cache/rul/final_rul_per_vin.csv           (full intermediates)
  cache/rul/fleet_window.json               (headline replacement window)
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")


def main() -> None:
    rul_cache = pathlib.Path(cfg.RUL_CACHE)
    results = pathlib.Path(cfg.RESULTS_DIR_V2)
    results.mkdir(parents=True, exist_ok=True)

    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    rul = pd.read_csv(rul_cache / "predictive_rul_per_vin.csv")
    dec = pd.read_csv(rul_cache / "decisions_per_vin.csv")
    ged = pd.read_csv(pathlib.Path(cfg.GED_EMERGENCY_CACHE) / "ged_emergency.csv")

    # ---- empirically-anchored fleet replacement window --------------------
    failed = lc[lc["failed_flag"] == True]
    ttf = failed["ttf_days"].astype(float).values
    with open(pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_weibull_params.json") as f:
        wb = json.load(f)
    fleet_window = {
        "anchor": "empirical_failed_ttf",
        "n_events": int(len(ttf)),
        "median_ttf_days": round(float(np.median(ttf)), 1),
        "p25_ttf_days": round(float(np.percentile(ttf, 25)), 1),
        "p75_ttf_days": round(float(np.percentile(ttf, 75)), 1),
        "min_ttf_days": round(float(ttf.min()), 1),
        "max_ttf_days": round(float(ttf.max()), 1),
        "median_ttf_km_est": round(float(np.median(ttf)) * cfg.FLEET_AVG_KM_PER_DAY, 0),
        "median_ttf_ehrs_est": round(float(np.median(ttf)) * cfg.FLEET_AVG_EHRS_PER_DAY, 0),
        "survival_median_context_days": wb.get("median_ttf_days"),
        "note": ("Headline window uses the empirical failed-truck median (backtest-"
                 "validated best point estimate). The survival-model median "
                 f"({wb.get('median_ttf_days')}d) over-predicts by ~"
                 f"{round(wb.get('median_ttf_days', 0) - float(np.median(ttf)))}d due to "
                 "right-censoring and is shown for context only."),
    }
    with open(rul_cache / "fleet_window.json", "w") as f:
        json.dump(fleet_window, f, indent=2)

    # ---- per-VIN merge ----------------------------------------------------
    ged_lead = dict(zip(ged["vin_label"], ged["first_fire_lead_days"]))
    df = (lc[["vin_label", "failed_flag", "lifecycle_stage", "age_days_observed",
              "est_km", "est_engine_hrs", "km_is_estimated", "ttf_days"]]
          .merge(rul.drop(columns=["failed_flag", "lifecycle_stage"], errors="ignore"),
                 on="vin_label", how="left")
          .merge(dec.drop(columns=["failed_flag", "rul_p10_days"], errors="ignore"),
                 on="vin_label", how="left"))   # p10 sourced from predictive_rul (avoid merge collision)

    def _would_have(vin):
        v = ged_lead.get(vin, "")
        return v if v not in ("", None) and not (isinstance(v, float) and np.isnan(v)) else "no_precursor"

    df["would_have_flagged_lead_days"] = [
        (_would_have(v) if f else "") for v, f in zip(df["vin_label"], df["failed_flag"])
    ]

    # customer-facing column subset (RUL band kept but clearly secondary)
    cust_cols = [
        "vin_label", "failed_flag", "lifecycle_stage", "current_age_days",
        "ridge_prob", "ridge_band", "risk_tier", "risk_confidence",
        "median_rul_days", "rul_p10_days", "rul_p90_days",
        "timing_flag", "timing_confidence",
        "ged_emergency", "would_have_flagged_lead_days",
        "median_rul_km", "median_rul_ehrs", "km_is_estimated",
        "recommendation",
    ]
    for c in cust_cols:
        if c not in df.columns:
            df[c] = ""
    cust = df[cust_cols].copy()

    # sort: failed last; among non-failed, by risk then ascending RUL
    cust["_risk_ord"] = (cust["risk_tier"] == "HIGH_RISK").astype(int)
    cust = cust.sort_values(
        ["failed_flag", "_risk_ord", "median_rul_days"], ascending=[True, False, True]
    ).drop(columns="_risk_ord")

    cust.to_csv(results / "V10.6.2_ALT_rul_predictions.csv", index=False)
    df.to_csv(rul_cache / "final_rul_per_vin.csv", index=False)

    print(f"[assemble] Saved V10.6.2_ALT_rul_predictions.csv ({len(cust)} rows)")
    print(f"\n  FLEET REPLACEMENT WINDOW (empirical, n={fleet_window['n_events']} events):")
    print(f"    median TTF = {fleet_window['median_ttf_days']:.0f}d "
          f"(p25-p75: {fleet_window['p25_ttf_days']:.0f}-{fleet_window['p75_ttf_days']:.0f}d, "
          f"range {fleet_window['min_ttf_days']:.0f}-{fleet_window['max_ttf_days']:.0f})")
    print(f"    ~ {fleet_window['median_ttf_km_est']:.0f} km / "
          f"{fleet_window['median_ttf_ehrs_est']:.0f} engine-hours (estimated)")
    print(f"    survival-model median (context, long-biased): "
          f"{fleet_window['survival_median_context_days']}d")
    crit = cust[cust["recommendation"].str.startswith("CRITICAL", na=False)]
    emer = cust[cust["ged_emergency"] == True]
    print(f"\n  Action summary: {len(crit)} CRITICAL, {len(emer)} GED-emergency, "
          f"{(cust['risk_tier']=='HIGH_RISK').sum()} high-risk non-failed")


if __name__ == "__main__":
    main()
