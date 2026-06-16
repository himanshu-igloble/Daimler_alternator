"""
V11.1_ALT — 3-D Decision Engine  (Task 7)
==========================================
Extends the V10.6.2 2x2 risk × time matrix with an EMERGENCY third dimension
sourced from the compound / GED-storm emergency layer (V11.1 emergency module).

Three decision axes per non-failed VIN:
  RISK axis      — frozen V10.5.3 classifier tier (ridge_100 band + above_thr
                   flag).  Exactly as V10.6.2: GREEN / AMBER / RED on the
                   0-100 rescaled score with BAND_GREEN_MAX / BAND_AMBER_MAX;
                   above_thr = ridge_prob >= RIDGE_DECISION_THR.
  TIME axis      — "SHORT" if rul_p10_days < SHORT_RUL_HORIZON_DAYS, else
                   "MID".
  EMERGENCY axis — "emergency"   if ged_fired (ever GED=2 storm)
                   "early-watch" if early_watch_current == 1 and not ged_fired
                   "none"        otherwise

Recommendation matrix (V11.1 strings — exact):
  priority 1 — emergency:
    "SERVICE IMMEDIATELY — GED=2 excitation storm active"
  priority 2 — early-watch AND (red OR SHORT):
    "Inspect within 2 weeks — compound electrical distress + elevated risk/short window"
  priority 3 — early-watch:
    "Schedule inspection — compound electrical distress signature"
  priority 4 — red AND SHORT:
    "Plan replacement within fleet window — high risk, short RUL band"
  default:
    "Routine monitoring — within fleet wear-out window"

Failed VINs:  included with their emergency_state row and
  recommendation = "FAILED — historical record"
  (mirroring V10.6.2 which includes failed VINs in decisions_per_vin.csv
  with a post-hoc recommendation string.)

Output: cfg.RUL_CACHE / "decisions_per_vin.csv"
Columns: vin_label, ridge_prob, risk_band, above_thr, time_dim,
         emergency_state, recommendation
"""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config", "V11_1_ALT_config.py")


def _band(ridge_100: float) -> str:
    """Map 0-100 rescaled ridge score to risk band (exactly as V10.6.2)."""
    if ridge_100 < cfg.BAND_GREEN_MAX:
        return "green"
    if ridge_100 < cfg.BAND_AMBER_MAX:
        return "amber"
    return "red"


def _emergency_state(ged_fired: bool, early_watch_current: int) -> str:
    if ged_fired:
        return "emergency"
    if early_watch_current == 1:
        return "early-watch"
    return "none"


def _recommend(failed: bool,
               risk_band: str,
               time_dim: str,
               emergency_state: str) -> str:
    """Return the recommendation string for one VIN."""
    if failed:
        return "FAILED — historical record"
    if emergency_state == "emergency":
        return "SERVICE IMMEDIATELY — GED=2 excitation storm active"
    if emergency_state == "early-watch" and (risk_band == "red" or time_dim == "SHORT"):
        return ("Inspect within 2 weeks — compound electrical distress + "
                "elevated risk/short window")
    if emergency_state == "early-watch":
        return "Schedule inspection — compound electrical distress signature"
    if risk_band == "red" and time_dim == "SHORT":
        return "Plan replacement within fleet window — high risk, short RUL band"
    return "Routine monitoring — within fleet wear-out window"


def main() -> None:
    out = pathlib.Path(cfg.RUL_CACHE)
    out.mkdir(parents=True, exist_ok=True)

    ridge = pd.read_csv(cfg.RIDGE_PROB_CSV)
    rul = pd.read_csv(out / "predictive_rul_per_vin.csv")
    emerg = pd.read_csv(pathlib.Path(cfg.EMERG_CACHE) / "emergency_per_vin.csv")

    df = (
        ridge[["vin_label", "ridge_prob", "ridge_100", "failed_flag"]]
        .merge(
            rul[["vin_label", "rul_p10_days"]],
            on="vin_label",
            how="left",
        )
        .merge(
            emerg[["vin_label", "ged_fired", "early_watch_current"]],
            on="vin_label",
            how="left",
        )
    )

    df["ged_fired"] = df["ged_fired"].fillna(False).astype(bool)
    df["early_watch_current"] = df["early_watch_current"].fillna(0).astype(int)

    recs = []
    for _, r in df.iterrows():
        failed = bool(r["failed_flag"])
        ridge_prob = float(r["ridge_prob"])
        ridge_100 = float(r["ridge_100"])
        p10 = float(r["rul_p10_days"]) if pd.notna(r.get("rul_p10_days")) else 0.0
        ged_fired = bool(r["ged_fired"])
        early_watch = int(r["early_watch_current"])

        rb = _band(ridge_100)
        above_thr = ridge_prob >= cfg.RIDGE_DECISION_THR
        time_dim = "SHORT" if p10 < cfg.SHORT_RUL_HORIZON_DAYS else "MID"
        # Failed VINs: time_dim is not meaningful but still computed from p10=0
        es = _emergency_state(ged_fired, early_watch)
        rec = _recommend(failed, rb, time_dim, es)

        recs.append({
            "vin_label": r["vin_label"],
            "failed_flag": int(failed),
            "ridge_prob": round(ridge_prob, 4),
            "risk_band": rb,
            "above_thr": bool(above_thr),
            "time_dim": time_dim,
            "emergency_state": es,
            "recommendation": rec,
        })

    res = pd.DataFrame(recs)
    res.to_csv(out / "decisions_per_vin.csv", index=False)

    print(f"[decisions] Saved decisions_per_vin.csv ({len(res)} rows)")

    nf = res[res["failed_flag"] == 0].sort_values("ridge_prob", ascending=False)
    print(f"\n  Non-failed VIN decisions ({len(nf)} trucks):")
    print(f"  {'VIN':<16} {'ridge':>6} {'band':>6} {'thr':>5} {'time':>5} {'emergency':<12}  recommendation")
    for _, r in nf.iterrows():
        print(f"  {r['vin_label']:<16} {r['ridge_prob']:>6.3f} {r['risk_band']:>6} "
              f"{'Y' if r['above_thr'] else 'N':>5} {r['time_dim']:>5} "
              f"{r['emergency_state']:<12}  {r['recommendation'][:60]}")

    f_rows = res[res["failed_flag"] == 1]
    print(f"\n  Failed VINs ({len(f_rows)} rows, recommendation='FAILED — historical record'):")
    for _, r in f_rows.iterrows():
        print(f"    {r['vin_label']:<16}  emergency_state={r['emergency_state']}")


if __name__ == "__main__":
    main()
