"""
V10.6.2 Alternator — GED=2 Emergency Layer  (plan W3)
=====================================================
The ONLY genuine short-horizon ("days") signal in the 6 CAN channels, promoted
out of the rules engine into its own independent, event-driven alert — kept
strictly SEPARATE from the RUL band (it must not inflate headline RUL accuracy).

Signal: re-derived DAILY GED=2 event count from the raw per-VIN telemetry
parquets (V5.2 features).  Fire when a single calendar day has
>= GED_EMERGENCY_DAILY_COUNT_MIN GED=2 (alternator excitation disturbance)
events.  This matches the validated threshold in V5.2.1 lead_time_analysis.md
(~0 false positives; VIN1 ~T-21d; VIN10 ~T-30d).

The raw parquet carries DAYS_TO_FAILURE, so the lead time of the earliest
fire is read directly (no date arithmetic).  Each day's count uses only that
day's data, so first-fire is inherently a real-time-detectable event.

Expected: high precision / low recall (~2/10 failed fire; ~0 healthy fire).

Output: cfg.GED_EMERGENCY_CACHE / ged_emergency.csv
"""
from __future__ import annotations

import importlib.util
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


def _raw_parquet(vin: str) -> pathlib.Path:
    return pathlib.Path(cfg.V52_PARQUET_DIR) / f"{cfg.V52_PARQUET_PREFIX}{vin}.parquet"


def main() -> None:
    out = pathlib.Path(cfg.GED_EMERGENCY_CACHE)
    out.mkdir(parents=True, exist_ok=True)
    thr = cfg.GED_EMERGENCY_DAILY_COUNT_MIN

    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    failed_set = set(cfg.FAILED_VIN_SET)

    print(f"[ged_emergency] Daily GED=2 scan (threshold >= {thr}/day) ...")
    rows = []
    for vin in cfg.ALL_VINS:
        rp = _raw_parquet(vin)
        if not rp.exists():
            print(f"  WARN: no raw parquet for {vin} ({rp.name})")
            continue
        df = pd.read_parquet(rp, columns=["GED", "DAYS_SINCE_SALE", "DAYS_TO_FAILURE"])
        df["is_ged2"] = (df["GED"] == 2)

        daily = df.groupby("DAYS_SINCE_SALE").agg(
            ged2=("is_ged2", "sum"),
            dtf=("DAYS_TO_FAILURE", "first"),
        )
        fire = daily[daily["ged2"] >= thr]
        ever = bool(len(fire) > 0)
        n_fire_days = int(len(fire))
        total_ged2 = int(df["is_ged2"].sum())
        failed = vin in failed_set

        lead = ""
        if ever and failed:
            # earliest fire = largest days-to-failure among fire days
            lead_val = float(fire["dtf"].max())
            if lead_val >= 0:
                lead = round(lead_val, 0)

        rows.append({
            "vin_label": vin,
            "failed_flag": int(failed),
            "ever_fired": ever,
            "n_fire_days": n_fire_days,
            "first_fire_lead_days": lead,
            "total_ged2_lifetime": total_ged2,
            "daily_threshold": thr,
        })
        tag = "F" if failed else "NF"
        print(f"  {vin:<16}[{tag:>2}] fired={str(ever):<5} days={n_fire_days:<4} "
              f"lead={str(lead):<6} ged2_total={total_ged2}")

    df_out = pd.DataFrame(rows).sort_values(
        ["ever_fired", "failed_flag"], ascending=[False, False]
    )
    df_out.to_csv(out / "ged_emergency.csv", index=False)

    failed_df = df_out[df_out["failed_flag"] == 1]
    nf_df = df_out[df_out["failed_flag"] == 0]
    n_failed_fire = int(failed_df["ever_fired"].sum())
    n_nf_fire = int(nf_df["ever_fired"].sum())

    print("\n  " + "=" * 56)
    print("  GED=2 EMERGENCY LAYER (high precision / low recall)")
    print("  " + "=" * 56)
    print(f"  Sensitivity (failed firing): {n_failed_fire}/{len(failed_df)}")
    print(f"  False alarms (healthy firing): {n_nf_fire}/{len(nf_df)}")
    leads = [r["first_fire_lead_days"] for _, r in failed_df.iterrows()
             if r["ever_fired"] and r["first_fire_lead_days"] != ""]
    if leads:
        print(f"  Lead times (failed, days): {sorted(leads, reverse=True)}")
    print("\n  NOTE: independent alert, NOT part of the RUL accuracy claim.")
    print(f"  Saved ged_emergency.csv ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
