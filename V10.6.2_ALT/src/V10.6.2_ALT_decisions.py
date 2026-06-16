"""
V10.6.2 Alternator — 2x2 Decision Engine  (plan W4)
====================================================
Replaces the incoherent V10.6.1 logic (which keyed CRITICAL/MEDIUM_HIGH off
lifecycle_stage, so a healthy truck got "CRITICAL within 90 days" and the
lowest-risk truck got "MEDIUM_HIGH").

Decisions are anchored on the TWO signals that actually validate:
  RISK  axis = frozen V10.5.3 classifier tier  (ridge_prob vs Youden 0.4456)
               -> the only signal with out-of-sample discrimination (0.927).
  TIME  axis = survival-conditioned RUL band    (p10 < SHORT_RUL_HORIZON_DAYS)
               -> "near fleet-replacement age".  Honest caveat: timing does
               NOT beat the fleet clock (backtest), so it only ever NUDGES.
  GED   override = excitation-fault storm  -> immediate, regardless of axes.

Guard (kills the V10.6.1 defect): CRITICAL requires HIGH risk.  A low-risk
truck is never CRITICAL no matter how old it is.

Honesty: we emit risk_confidence (HIGH — validated classifier) and
timing_confidence (LOW — backtest showed no day-MAE edge over fleet clock)
SEPARATELY, rather than one misleading blended "confidence".

Output: cfg.RUL_CACHE / decisions_per_vin.csv
"""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")


def _band(ridge_100: float) -> str:
    if ridge_100 < cfg.BAND_GREEN_MAX:
        return "GREEN"
    if ridge_100 < cfg.BAND_AMBER_MAX:
        return "AMBER"
    return "RED"


def _decide(failed, ridge_prob, p10_days, ged_fired):
    """Return (recommendation, risk_tier, timing_flag, risk_conf, timing_conf)."""
    if failed:
        return ("Post-hoc: alternator already failed (in dataset).",
                "N/A", "N/A", "N/A", "N/A")

    hi_risk = ridge_prob >= cfg.RIDGE_DECISION_THR
    near_term = p10_days < cfg.SHORT_RUL_HORIZON_DAYS
    risk_tier = "HIGH_RISK" if hi_risk else "LOW_RISK"
    timing_flag = "NEAR_TERM" if near_term else "NOMINAL"

    # risk confidence from distance to the frozen Youden threshold
    risk_conf = "HIGH" if abs(ridge_prob - cfg.RIDGE_DECISION_THR) >= 0.15 else "MEDIUM"
    timing_conf = "LOW"   # backtest: RUL timing does not beat the fleet clock

    # GED storm overrides everything
    if ged_fired:
        return ("EMERGENCY: GED=2 excitation-fault storm — inspect alternator within days.",
                risk_tier, timing_flag, risk_conf, "HIGH")

    # 2x2 matrix (CRITICAL requires HIGH risk — the guard)
    if hi_risk and near_term:
        rec = "CRITICAL: high failure risk and near fleet-replacement age — schedule inspection now."
    elif hi_risk and not near_term:
        rec = "Elevated risk — add alternator inspection to next scheduled service."
    elif (not hi_risk) and near_term:
        rec = "Watch: near fleet-replacement age but low classifier risk — review at next depot visit."
    else:
        rec = "Routine — re-assess at next data refresh."
    return (rec, risk_tier, timing_flag, risk_conf, timing_conf)


def main() -> None:
    out = pathlib.Path(cfg.RUL_CACHE)
    out.mkdir(parents=True, exist_ok=True)

    ridge = pd.read_csv(cfg.RIDGE_PROB_CSV)
    rul = pd.read_csv(out / "predictive_rul_per_vin.csv")
    ged = pd.read_csv(pathlib.Path(cfg.GED_EMERGENCY_CACHE) / "ged_emergency.csv")

    df = (ridge[["vin_label", "ridge_prob", "ridge_100", "failed_flag"]]
          .merge(rul[["vin_label", "median_rul_days", "rul_p10_days", "rul_p90_days"]],
                 on="vin_label", how="left")
          .merge(ged[["vin_label", "ever_fired", "first_fire_lead_days"]],
                 on="vin_label", how="left"))
    df["ever_fired"] = df["ever_fired"].fillna(False).astype(bool)

    recs = []
    for _, r in df.iterrows():
        failed = bool(r["failed_flag"])
        p10 = float(r["rul_p10_days"]) if pd.notna(r["rul_p10_days"]) else 0.0
        rec, risk_tier, timing_flag, risk_conf, timing_conf = _decide(
            failed, float(r["ridge_prob"]), p10, bool(r["ever_fired"])
        )
        recs.append({
            "vin_label": r["vin_label"],
            "failed_flag": int(failed),
            "ridge_prob": round(float(r["ridge_prob"]), 4),
            "ridge_band": _band(float(r["ridge_100"])),
            "risk_tier": risk_tier,
            "rul_p10_days": (round(p10, 1) if not failed else 0.0),
            "timing_flag": timing_flag,
            "ged_emergency": bool(r["ever_fired"]),
            "risk_confidence": risk_conf,
            "timing_confidence": timing_conf,
            "recommendation": rec,
        })

    res = pd.DataFrame(recs)
    res.to_csv(out / "decisions_per_vin.csv", index=False)

    # sanity guard: no CRITICAL for a low-risk truck
    bad = res[(res["recommendation"].str.startswith("CRITICAL"))
              & (res["ridge_prob"] < cfg.RIDGE_DECISION_THR)]
    assert len(bad) == 0, f"GUARD FAILED: CRITICAL on low-risk VINs: {list(bad['vin_label'])}"

    print(f"[decisions] Saved decisions_per_vin.csv ({len(res)} rows)")
    print(f"  Guard OK: 0 CRITICAL recommendations for ridge_prob < {cfg.RIDGE_DECISION_THR}")
    nf = res[res["failed_flag"] == 0]
    print(f"\n  {'VIN':<16}{'ridge':>7}{'tier':>11}{'p10':>7}{'GED':>5}  recommendation")
    for _, r in nf.sort_values("ridge_prob", ascending=False).iterrows():
        print(f"  {r['vin_label']:<16}{r['ridge_prob']:>7.3f}{r['risk_tier']:>11}"
              f"{r['rul_p10_days']:>7.0f}{str(r['ged_emergency']):>5}  {r['recommendation'][:54]}")


if __name__ == "__main__":
    main()
