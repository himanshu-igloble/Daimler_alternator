"""
V10.6.2 Alternator — TRACK A: Deployable Maintenance Service
=============================================================
The honest, shippable product. Integrates ONLY validated signals into a single
per-truck operational service + fleet replacement schedule + alerts feed, with
two new operational monitors (data-gap, same-day collapse).

It deliberately emits NO per-truck failure date. Per-truck timing is "RUL =
fleet clock" (601d wear-out window) unless a verified precursor fires.

Components (all grounded in V10.6.2 verified artifacts):
  1. Risk ranking      — frozen V10.5.3 ridge_prob (AUROC 0.927), HIGH/LOW @0.4456
  2. Fleet-clock RUL   — empirical 601d window (NOT the biased 718d survival median)
  3. 3-state monitor   — NORMAL (abstain) / DEGRADING (GED2 storm) / LATE (collapse)
  4. GED2 alarm        — daily >=200 GED2/day sustained; n=1-supported (caveated)
  5. Data-gap monitor  — <50% trusted-day coverage in trailing 30/60d -> fleet-clock fallback
  6. Collapse detector — same-day reactive voltage-collapse alert (NOT a precursor)

In-service fleet = the 15 non-failed trucks (live). The 10 failed trucks are
included as HISTORICAL, showing what the service would have done.

Inputs (read-only):
  cache/forensics/<VIN>_daily.csv, cache/forensics/nf_baseline.csv
  cache/rul/fleet_window.json, cache/rul/predictive_rul_per_vin.csv
  cache/ged_emergency/ged_emergency.csv
  <V10.6 ridge> ridge_prob_rescaled.csv ; lifecycle parquet
Outputs (service/):
  V10.6.2_ALT_service_per_vin.csv, V10.6.2_ALT_fleet_replacement_schedule.csv,
  V10.6.2_ALT_alerts.csv, V10.6.2_ALT_track_a_service_report.md
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("V10_6_2_ALT_config", str(_src / "V10.6.2_ALT_config.py"))
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

FOR = pathlib.Path(cfg.V10_6_2_ROOT) / "cache" / "forensics"
SERVICE = pathlib.Path(cfg.V10_6_2_ROOT) / "service"

# Verified operational rules (from V10.6.2 forensics + adversarial verification)
GED_DAILY = cfg.GED_EMERGENCY_DAILY_COUNT_MIN          # 200 GED2/day
GED_SUSTAIN_DAYS = 3                                    # >=3 days in trailing window
GED_TRAIL = 60                                          # trailing window for "active storm"
COLLAPSE = dict(vsi_mean=27.8, vsi_std=0.485, vsi_sag_frac=0.0017)  # same-day reactive
MIN_EO = 200                                            # trusted day
COV_TRAIL = [30, 60]
COV_FLOOR = 0.50
STALE_DAYS = 30                                          # in-service truck silent > this vs fleet cutoff = unmonitorable


def _daily(vin):
    return pd.read_csv(FOR / f"{vin}_daily.csv")


def _state_and_monitors(vin, age):
    """Compute 3-state, GED active-storm, data-gap, collapse from the daily panel.
    'Current' = the truck's most recent day (max DAYS_SINCE_SALE)."""
    d = _daily(vin)
    max_day = int(d["day"].max())
    trusted = d[d["n_eo"] >= MIN_EO]

    # --- within-window coverage (trailing w calendar days, exclusive lower
    # bound so exactly w day-slots -> fraction never exceeds 1.0) ---
    cov = {}
    for w in COV_TRAIL:
        win_trusted = trusted[trusted["day"] > max_day - w]
        cov[w] = (len(win_trusted) / w) if w else np.nan
    coverage_gap = any(cov[w] < COV_FLOOR for w in COV_TRAIL)

    # --- GED active storm (DEGRADING) ---
    ged_win = trusted[trusted["day"] >= max_day - GED_TRAIL]
    ged_storm_days = int((ged_win["ged2_cnt"] >= GED_DAILY).sum())
    degrading = ged_storm_days >= GED_SUSTAIN_DAYS

    # --- same-day collapse (reactive) over the last 5 trusted days ---
    # (single-latest-day misses a collapse that "recovers" post jump-start, e.g.
    #  VIN2_F collapsed at dtf 2-3 then read ~28V at dtf 0.)
    recent = trusted.sort_values("day").tail(5)
    collapse_rows = recent[
        (recent["vsi_mean"] < COLLAPSE["vsi_mean"]) &
        (recent["vsi_std"] > COLLAPSE["vsi_std"]) &
        (recent["vsi_sag_frac"] > COLLAPSE["vsi_sag_frac"])
    ]
    collapse = len(collapse_rows) > 0

    # --- 3-state ---
    if collapse:
        state = "LATE"
    elif degrading:
        state = "DEGRADING"
    else:
        state = "NORMAL"

    return {
        "state": state,
        "ged_storm_active": degrading,
        "ged_storm_days_60d": ged_storm_days,
        "collapse_alert": collapse,
        "cov_30d": round(cov[30], 2),
        "cov_60d": round(cov[60], 2),
        "coverage_gap": bool(coverage_gap),
        "last_obs_age_days": max_day,
    }


def _action(rec):
    """Operational instruction. Order = severity."""
    if rec["collapse_alert"]:
        return "IMMEDIATE: voltage-collapse signature today — dispatch to depot now (reactive, not preventive)."
    if rec["state"] == "DEGRADING":
        return "URGENT: GED2 excitation-fault storm active — inspect alternator within ~3 weeks."
    if rec["data_gap_flag"]:
        return "BLIND: telemetry stale/sparse — cannot monitor; fall back to fleet-clock schedule & fix telematics (or confirm decommissioned)."
    if rec["risk_tier"] == "HIGH_RISK" and rec["replacement_status"].startswith(("OVERDUE", "DUE")):
        return "PRIORITY: high failure risk + past/near fleet replacement age — add alternator inspection at next depot visit."
    if rec["replacement_status"].startswith("OVERDUE"):
        return "Schedule alternator inspection: past fleet median wear-out age."
    if rec["replacement_status"].startswith("DUE"):
        return "Plan alternator replacement within ~2 months (approaching fleet wear-out window)."
    if rec["risk_tier"] == "HIGH_RISK":
        return "Watch: elevated classifier risk — review at next depot visit."
    return "Routine — re-assess at next data refresh."


def _replacement_status(age, rul, median):
    if age >= median:
        return f"OVERDUE (+{age - median:.0f}d past fleet median {median:.0f}d)"
    if rul < 60:
        return f"DUE SOON (~{rul:.0f}d to fleet median)"
    if rul < 180:
        return f"Plan <6mo (~{rul:.0f}d to fleet median)"
    return f"Monitor (~{rul:.0f}d to fleet median)"


def main() -> None:
    SERVICE.mkdir(parents=True, exist_ok=True)

    fw = json.loads((pathlib.Path(cfg.RUL_CACHE) / "fleet_window.json").read_text())
    median = float(fw["median_ttf_days"])               # 601 (empirical, NOT 718 survival)

    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    lc["alt_t1"] = pd.to_datetime(lc["alt_t1"])
    # live "current" date = latest last-observation across the in-service fleet
    fleet_cutoff = lc.loc[lc.failed_flag == False, "alt_t1"].max()
    ridge = pd.read_csv(cfg.RIDGE_PROB_CSV)
    ged = pd.read_csv(pathlib.Path(cfg.GED_EMERGENCY_CACHE) / "ged_emergency.csv")
    ged_map = ged.set_index("vin_label").to_dict("index")

    rows = []
    for _, r in lc.iterrows():
        vin = r["vin_label"]
        failed = bool(r["failed_flag"])
        age = float(r["age_days_observed"])
        rp = float(ridge.loc[ridge.vin_label == vin, "ridge_prob"].iloc[0])
        rul = max(median - age, 0.0)

        rec = {"vin_label": vin, "in_service": (not failed), "historical_failure": failed,
               "current_age_days": round(age, 0), "ridge_prob": round(rp, 4),
               "risk_tier": "HIGH_RISK" if rp >= cfg.RIDGE_DECISION_THR else "LOW_RISK",
               "fleet_clock_rul_days": round(rul, 0)}
        rec["replacement_status"] = _replacement_status(age, rul, median)
        rec.update(_state_and_monitors(vin, age))

        # B1 fix: staleness vs the live fleet cutoff catches STOPPED telemetry
        # streams (a within-window coverage check alone cannot — it anchors to
        # the truck's own last point). Only meaningful for in-service trucks.
        staleness = int((fleet_cutoff - r["alt_t1"]).days)
        rec["staleness_days"] = staleness if not failed else ""
        rec["data_gap_flag"] = bool(rec["coverage_gap"] or (not failed and staleness > STALE_DAYS))

        # GED: live storm (trailing-window, set by _state_and_monitors, drives
        # DEGRADING) vs lifetime fire (history/context from the emergency layer)
        g = ged_map.get(vin, {})
        rec["ged_ever_fired"] = bool(g.get("ever_fired", False))
        rec["ged_lead_days"] = g.get("first_fire_lead_days", "")

        rec["recommended_action"] = _action(rec)
        rows.append(rec)

    df = pd.DataFrame(rows)
    cols = ["vin_label", "in_service", "historical_failure", "current_age_days",
            "ridge_prob", "risk_tier", "state", "fleet_clock_rul_days", "replacement_status",
            "ged_storm_active", "ged_ever_fired", "ged_lead_days", "ged_storm_days_60d",
            "collapse_alert", "cov_30d", "cov_60d", "last_obs_age_days", "staleness_days",
            "data_gap_flag", "recommended_action"]
    df = df[cols]

    # risk rank among in-service trucks (1 = highest risk)
    df["risk_rank_in_service"] = ""
    insvc = df[df.in_service].sort_values("ridge_prob", ascending=False)
    df.loc[insvc.index, "risk_rank_in_service"] = range(1, len(insvc) + 1)

    df.to_csv(SERVICE / "V10.6.2_ALT_service_per_vin.csv", index=False)

    # fleet replacement schedule (in-service) — PRIORITY order: HIGH_RISK first,
    # then soonest fleet-clock, then most-overdue (so the high-risk overdue truck
    # surfaces at the top instead of being buried among rul=0 ties).
    sched_src = df[df.in_service].copy()
    sched_src["_hi"] = (sched_src.risk_tier == "HIGH_RISK").astype(int)
    sched_src["_overdue"] = median - sched_src.current_age_days  # negative = overdue
    sched = sched_src.sort_values(["_hi", "fleet_clock_rul_days", "_overdue"],
                                  ascending=[False, True, True])[
        ["vin_label", "current_age_days", "fleet_clock_rul_days", "replacement_status",
         "risk_tier", "state", "recommended_action"]]
    sched.to_csv(SERVICE / "V10.6.2_ALT_fleet_replacement_schedule.csv", index=False)

    # active alerts (in-service trucks needing action now)
    alert_mask = df.in_service & (
        df.collapse_alert | (df.state != "NORMAL") | df.data_gap_flag | df.ged_storm_active |
        ((df.risk_tier == "HIGH_RISK") & df.replacement_status.str.startswith(("OVERDUE", "DUE"))))
    alerts = df[alert_mask][["vin_label", "risk_tier", "state", "ged_storm_active",
                             "collapse_alert", "data_gap_flag", "recommended_action"]]
    alerts.to_csv(SERVICE / "V10.6.2_ALT_alerts.csv", index=False)

    _write_report(df, sched, alerts, fw, ged)

    # console summary
    insv = df[df.in_service]
    print("[track_a] TRACK A SERVICE BUILT")
    print(f"  In-service trucks: {len(insv)}  |  historical failures shown: {(~df.in_service).sum()}")
    print(f"  States (in-service): {insv['state'].value_counts().to_dict()}")
    print(f"  HIGH_RISK in-service: {(insv.risk_tier=='HIGH_RISK').sum()}  "
          f"| GED storms (in-service): {insv.ged_storm_active.sum()}  "
          f"| collapse alerts: {insv.collapse_alert.sum()}  | data-gap: {insv.data_gap_flag.sum()}")
    print(f"  Active alerts (in-service): {len(alerts)}")
    print(f"  Replacement schedule top-3 (soonest):")
    for _, s in sched.head(3).iterrows():
        print(f"    {s.vin_label:<14} age={s.current_age_days:.0f} {s.replacement_status}")
    print(f"  Saved service/ (4 files)")


def _write_report(df, sched, alerts, fw, ged):
    insv = df[df.in_service]
    f_fire = ged[(ged.failed_flag == 1) & (ged.ever_fired)]
    L = []
    L.append("# V10.6.2 Alternator — Track A Maintenance Service")
    L.append("")
    L.append("**The honest, shippable product.** Risk ranking + fleet-clock replacement window + "
             "3-state monitor + GED2 alarm + data-gap & collapse monitors. **No per-truck failure "
             "date is emitted** — per-truck timing is the fleet clock unless a verified precursor fires.")
    L.append("")
    L.append("## What this service does (and its honest limits)")
    L.append(f"- **Which trucks** (validated, AUROC 0.927): ranks failure risk; "
             f"{(insv.risk_tier=='HIGH_RISK').sum()}/{len(insv)} in-service trucks are HIGH_RISK.")
    L.append(f"- **When (fleet-level)**: replacement window = empirical **{fw['median_ttf_days']:.0f} d** "
             f"(IQR {fw['p25_ttf_days']:.0f}–{fw['p75_ttf_days']:.0f}; ≈{fw['median_ttf_km_est']:.0f} km / "
             f"{fw['median_ttf_ehrs_est']:.0f} engine-hr, estimated). Survival median {fw['survival_median_context_days']} d "
             f"is shown elsewhere as long-biased context only.")
    L.append(f"- **Early warning**: GED2 storm alarm — verified ~21-day lead on **1** historical truck "
             f"(VIN1); **rests on n=1**, re-verified on every new failure. Most failures give no precursor.")
    L.append("- **Per-truck RUL date**: deliberately NOT emitted — it loses to the fleet clock (142 d vs 50 d backtest).")
    L.append("")
    L.append("## Active alerts (in-service)")
    if len(alerts):
        L.append(_md(alerts))
    else:
        L.append("_No live trucks currently fire GED/collapse/data-gap alerts. High-risk + overdue trucks listed in the schedule._")
    L.append("")
    L.append("## Fleet replacement schedule (in-service, soonest first)")
    L.append(_md(sched))
    L.append("")
    L.append("## Risk ranking (in-service, highest first)")
    rr = insv.sort_values("ridge_prob", ascending=False)[
        ["vin_label", "ridge_prob", "risk_tier", "state", "replacement_status"]]
    L.append(_md(rr))
    L.append("")
    L.append("## Historical failures — what the service would have done")
    hist = df[df.historical_failure][["vin_label", "ridge_prob", "risk_tier", "state",
                                      "ged_ever_fired", "ged_lead_days", "collapse_alert", "data_gap_flag"]]
    L.append(_md(hist))
    L.append("")
    L.append("## Honest operating metrics (do not oversell)")
    L.append("- Actionable early-warning recall: **1/10** (~21 d, n=1). GED fires 2/10 but VIN10 only 1 d.")
    L.append("- Per-truck RUL accuracy: **= fleet clock** (no per-truck timing skill exists at n=10).")
    L.append("- Classifier: frozen V10.5.3, LOVO AUROC 0.927 (which-truck only; temporal AUROC ~0.53).")
    L.append("- Collapse detector is **reactive** (same-day), not preventive; not envelope-clean.")
    L.append("- Failure-day labels are inferred (no maintenance records) — all leads provisional.")
    L.append("")
    L.append("_Generated by V10.6.2 Track A service._")
    (SERVICE / "V10.6.2_ALT_track_a_service_report.md").write_text("\n".join(L), encoding="utf-8")


def _md(d):
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


if __name__ == "__main__":
    main()
