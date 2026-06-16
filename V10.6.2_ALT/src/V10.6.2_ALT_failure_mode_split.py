#!/usr/bin/env python3
"""
V10.6.2 Alternator — Failure-Mode Split of the 10 Failed Trucks
================================================================
Classifies each of the 10 alternator failures by the TELEMETRY SIGNATURE in its
pre-failure window, then maps that signature to a (hedged) likely failure-mode
hypothesis and a precursor lead time.  Purely a re-interpretation of the existing
deterministic forensic artifacts — it re-fits NOTHING and does not touch the
survival math.

Inputs (read-only, already on disk):
  cache/forensics/failed_window_deviations.csv  per-VIN x horizon x feature z vs NF envelope
  cache/forensics/earliest_signal_per_vin.csv   earliest discriminative precursor + 30d coverage
  cache/forensics/nf_baseline.csv               healthy-fleet p05/p95 envelope
  cache/rul/final_rul_per_vin.csv               ged_emergency flag, would_have_flagged_lead_days
  V10.6_ALT/cache/lifecycle/vin_lifecycle.parquet  ttf_days

Output:
  cache/forensics/failure_mode_split.csv        one row per failed VIN

CLASSIFICATION RUBRIC (transparent, signature-based — NOT a physical diagnosis).
A 'robust' charging-bus signature requires the pipeline's `discriminative` flag
(window mean outside the healthy-fleet p05-p95) AND a non-sparse window (n_days
>= MIN_DAYS) AND |z| >= Z_MIN, to reject low-sample artifacts.

  EXCITATION_STORM : ged2_cnt window-mean >= GED2_VERIFIED/day  (validated field-
                     disturbance threshold)  -> regulator / field-excitation fault
  RIPPLE           : any of {vsi_std, vsi_cv, vsi_range, vsi_sag_frac} robust
                     -> rectifier/diode-bridge AC ripple
  DROOP            : any charging-regime voltage {vsi_mean, idle_vsi_mean,
                     cruise_vsi_mean, vsi_p05} below NF p05, robust
                     -> undercharge (regulator / belt-slip / brush)
  BATTERY_ONLY     : only crank_vsi_min or resting_vsi_mean flagged (engine-off /
                     cranking) -> battery/starter-side, NOT alternator charging
  NONE + cov>0     : flat to failure with good coverage -> ABRUPT (bearing /
                     winding open / shorted regulator / load-dump) - no precursor
  NONE + cov==0    : no telemetry near failure -> UNOBSERVABLE (data gap)

  Lead time = earliest (largest) horizon at which a robust charging signature
  (STORM/RIPPLE/DROOP) appears.  Actionability: >=14 d actionable; 1-13 d
  terminal-only; else none.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("V10_6_2_ALT_config", str(_src / "V10.6.2_ALT_config.py"))
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

FOR = pathlib.Path(cfg.V10_6_2_ROOT) / "cache" / "forensics"

# Robustness thresholds
MIN_DAYS = 5                 # reject windows with < 5 trusted days
Z_MIN = 2.5                  # |z| vs the truck's own baseline
GED2_VERIFIED = 200          # cfg.GED_EMERGENCY_DAILY_COUNT_MIN — 0 NF false alarms

RIPPLE_FEATS = ["vsi_std", "vsi_cv", "vsi_range", "vsi_sag_frac"]
DROOP_FEATS = ["vsi_mean", "idle_vsi_mean", "cruise_vsi_mean", "vsi_p05"]
BATTERY_FEATS = ["crank_vsi_min", "resting_vsi_mean"]
# Absolute materiality floors: some features (notably vsi_sag_frac) sit near zero
# in a healthy truck, so a tiny absolute change yields a huge z and crosses the
# tight NF p95 — a statistical artifact, not a real fault.  A flag must clear BOTH
# the discriminative/z test AND this absolute floor.  (vsi_sag_frac 0.005 = 0.5%
# of engine-on samples sagging < 24 V; VIN10 0.035 / VIN2 0.054 pass, the VIN6
# 0.0019 artifact does not.)
MATERIALITY = {"vsi_sag_frac": 0.005}


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def _daily_onset(vin):
    """Precise per-day onset of a material charging disturbance (overrides the
    coarse horizon-bin lead).  A 'disturbed' day = verified GED2 storm
    (ged2_cnt >= 200) OR material undervoltage sag (vsi_sag_frac >= floor).
    Returns (onset_dtf, n_disturbed_days, has_ged2_storm)."""
    d = pd.read_csv(FOR / f"{vin}_daily.csv")
    d = d[(d["n_eo"] >= 200) & (d["dtf"] >= 0) & (d["dtf"] <= 90)].copy()
    if d.empty:
        return 0, 0, False
    d["dist"] = (d["ged2_cnt"] >= GED2_VERIFIED) | (d["vsi_sag_frac"] >= MATERIALITY["vsi_sag_frac"])
    dd = d[d["dist"]]
    if dd.empty:
        return 0, 0, False
    return int(dd["dtf"].max()), int(len(dd)), bool((dd["ged2_cnt"] >= GED2_VERIFIED).any())


def main():
    dev = pd.read_csv(FOR / "failed_window_deviations.csv")
    es = pd.read_csv(FOR / "earliest_signal_per_vin.csv").set_index("vin_label")
    fr = pd.read_csv(pathlib.Path(cfg.RUL_CACHE) / "final_rul_per_vin.csv").set_index("vin_label")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET).set_index("vin_label")

    dev["z"] = dev["z_vs_baseline"].map(_num)
    dev["wmean"] = dev["window_mean"].map(_num)
    dev["horizon_days"] = dev["horizon_days"].astype(int)

    rows = []
    for vin in cfg.FAILED_VIN_SET:
        d = dev[dev.vin_label == vin]
        cov30 = int(es.loc[vin, "n_days_final_30d"])
        ged2_total = int(es.loc[vin, "ged2_total"])
        ttf = float(lc.loc[vin, "ttf_days"])
        ged_emerg = bool(fr.loc[vin, "ged_emergency"])

        # robust per-horizon signature detection
        storm_h, ripple_h, droop_h, batt_h = [], [], [], []
        evidence = {}
        for h, g in d.groupby("horizon_days"):
            n_days = int(g["n_days"].iloc[0])
            if n_days < MIN_DAYS:
                continue
            gi = g.set_index("feature")

            # EXCITATION STORM (validated absolute count threshold)
            ged2 = gi.loc["ged2_cnt", "wmean"] if "ged2_cnt" in gi.index else np.nan
            if not np.isnan(ged2) and ged2 >= GED2_VERIFIED:
                storm_h.append(h)
                evidence.setdefault("storm", f"ged2_cnt~{ged2:.0f}/d @T-{h}d")

            def _robust(feat):
                if feat not in gi.index:
                    return False
                r = gi.loc[feat]
                if not (bool(r["discriminative"]) and abs(_num(r["z"])) >= Z_MIN):
                    return False
                floor = MATERIALITY.get(feat)
                return floor is None or _num(r["wmean"]) >= floor

            if any(_robust(f) for f in RIPPLE_FEATS):
                ripple_h.append(h)
                hot = max((f for f in RIPPLE_FEATS if _robust(f)),
                          key=lambda f: abs(_num(gi.loc[f, "z"])))
                evidence.setdefault("ripple", f"{hot} z={_num(gi.loc[hot,'z']):.0f} @T-{h}d")
            if any(_robust(f) for f in DROOP_FEATS):
                droop_h.append(h)
                hot = min((f for f in DROOP_FEATS if _robust(f)),
                          key=lambda f: _num(gi.loc[f, "z"]))
                evidence.setdefault("droop", f"{hot} z={_num(gi.loc[hot,'z']):.1f} @T-{h}d")
            if any(_robust(f) for f in BATTERY_FEATS):
                batt_h.append(h)
                evidence.setdefault("battery", "engine-off/crank voltage low")

        # earliest (largest-horizon) robust charging signature
        charging_h = storm_h + ripple_h + droop_h
        lead = max(charging_h) if charging_h else 0

        # signature class (priority: storm > ripple > droop > battery > none)
        if storm_h:
            sig = "EXCITATION_STORM"
            mode = "Voltage-regulator / field-excitation fault"
        elif ripple_h:
            sig = "RIPPLE"
            mode = "Rectifier / diode-bridge (AC ripple)"
        elif droop_h:
            sig = "DROOP"
            mode = "Undercharge — regulator / belt-slip / brush"
        elif batt_h:
            sig = "BATTERY_ONLY"
            mode = "Battery/starter-side — inconclusive (not charging-bus)"
            lead = 0
        elif cov30 == 0:
            sig = "UNOBSERVABLE"
            mode = "Telemetry gap near failure — unclassifiable"
        elif cov30 < MIN_DAYS:
            sig = "UNDER_OBSERVED"
            mode = f"Too few trusted days near failure ({cov30}d) — inconclusive"
        else:
            sig = "NONE_ABRUPT"
            mode = "Abrupt — bearing / winding open / shorted regulator / load-dump"

        # Refine charging-signature classes with the precise daily onset, and
        # separate a sustained electrical precursor from a terminal/transient
        # blip.  GED2 storm = alternator-specific excitation signal (keep
        # gradual-electrical); a bus-voltage-only transient that does not
        # persist is ambiguous (could be a connection/battery dropout) -> the
        # adversarial review downgraded VIN2 to inconclusive on exactly this.
        if sig in ("EXCITATION_STORM", "RIPPLE"):
            onset, n_dist, has_storm = _daily_onset(vin)
            lead = onset
            sustained = n_dist >= 3
            if has_storm:
                sig = "EXCITATION_STORM"
                mode = "Excitation/field-control disturbance (regulator IC / brush-slip-ring / field circuit — not separable)"
            elif sustained and onset >= 14:
                sig = "RIPPLE"
                mode = "Sustained voltage instability — rectifier/diode-bridge signature (unconfirmed)"
            else:
                sig = "VOLTAGE_TRANSIENT"
                mode = "Transient voltage instability, recovers before failure — inconclusive"
                lead = 0

        gradual = sig in ("EXCITATION_STORM", "RIPPLE", "DROOP")
        if lead >= 14 and gradual:
            action = "actionable"
        elif lead > 0 and gradual:
            action = "terminal-only"
        else:
            action = "none"

        rows.append({
            "vin": vin,
            "ttf_days": round(ttf),
            "cov_30d": cov30,
            "ged2_total": ged2_total,
            "signature_class": sig,
            "failure_class": "gradual-electrical" if gradual else
                             ("inconclusive" if sig in ("BATTERY_ONLY", "UNOBSERVABLE",
                              "UNDER_OBSERVED", "VOLTAGE_TRANSIENT")
                              else "abrupt/no-precursor"),
            "likely_mode_hypothesis": mode,
            "lead_days": lead,
            "actionability": action,
            "ged_emergency_verified": ged_emerg,
            "key_evidence": "; ".join(f"{k}:{v}" for k, v in evidence.items()) or "flat — no robust deviation",
        })

    out = pd.DataFrame(rows).sort_values(
        ["failure_class", "lead_days"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(FOR / "failure_mode_split.csv", index=False)

    # ---- console summary ----
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("=" * 78)
    print("ALTERNATOR FAILURE-MODE SPLIT — 10 failed trucks (signature-based)")
    print("=" * 78)
    show = out[["vin", "ttf_days", "cov_30d", "ged2_total", "signature_class",
                "likely_mode_hypothesis", "lead_days", "actionability"]]
    print(show.to_string(index=False))
    print("\n--- Fleet split ---")
    print(out["failure_class"].value_counts().to_string())
    print("\n--- Precursor actionability ---")
    print(out["actionability"].value_counts().to_string())
    n_act = int((out.actionability == "actionable").sum())
    n_term = int((out.actionability == "terminal-only").sum())
    n_none = int((out.actionability == "none").sum())
    print(f"\nAchievable lead: {n_act}/10 actionable (>=14d), {n_term}/10 terminal-only (<14d), "
          f"{n_none}/10 no precursor.")
    print(f"Saved -> {FOR / 'failure_mode_split.csv'}")


if __name__ == "__main__":
    main()
