"""Task 4 — Sub-daily structure of the two GED=2 storms (R3 independent recompute).

Loads raw rows for VIN1_F_ALT and VIN10_F_ALT via ged_common.load_vin.
Writes: V12_ALT_GED/results/r3_storm_subdaily.json
"""
import sys
import json
import pathlib
import importlib.util
from datetime import timedelta

import polars as pl
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

_SRC = pathlib.Path(__file__).resolve().parent
RESULTS = _SRC.parent / "results"


def _load_mod(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_mod("ged_common")

print("=" * 70)
print("TASK 4 — SUB-DAILY STORM STRUCTURE")
print("=" * 70)

output = {}

for vin, storm_days_dtf in [("VIN1_F_ALT", [21, 20, 19]), ("VIN10_F_ALT", [1])]:
    print(f"\nLoading {vin}...")
    raw = C.load_vin(vin)

    # Filter GED==2
    ged2_rows = raw.filter(pl.col("GED") == 2)

    # Get days_to_failure col
    dtf_col = C.DTF_COL  # "DAYS_TO_FAILURE"

    vin_output = {}

    for target_dtf in storm_days_dtf:
        # find the calendar day corresponding to this dtf
        day_rows = ged2_rows.filter(pl.col(dtf_col) == target_dtf)

        if len(day_rows) == 0:
            print(f"  dtf={target_dtf}: NO GED=2 rows found!")
            vin_output[f"dtf_{target_dtf}"] = {"n_ged2": 0, "note": "no GED2 rows"}
            continue

        # Sort by DATETIME
        day_rows = day_rows.sort(C.DT_COL)

        n_ged2 = len(day_rows)
        ts_first = day_rows[C.DT_COL][0]
        ts_last  = day_rows[C.DT_COL][-1]

        hour_first = ts_first.hour if hasattr(ts_first, 'hour') else ts_first.to_pydatetime().hour
        hour_last  = ts_last.hour if hasattr(ts_last, 'hour') else ts_last.to_pydatetime().hour

        # Episode detection: gap > 10 minutes splits episodes
        dt_col_series = day_rows[C.DT_COL].to_list()
        n_eps = 1
        total_gap_mins = 0.0
        for i in range(1, len(dt_col_series)):
            gap_s = (dt_col_series[i] - dt_col_series[i-1]).total_seconds()
            if gap_s > 600:
                n_eps += 1
            total_gap_mins += gap_s / 60.0

        # Total GED2 minutes: sum of (consecutive gap when gap<=10min)
        ged2_minutes = 0.0
        for i in range(1, len(dt_col_series)):
            gap_s = (dt_col_series[i] - dt_col_series[i-1]).total_seconds()
            if gap_s <= 600:
                ged2_minutes += gap_s / 60.0

        # Fraction in first 2 hours after onset
        first_ts = dt_col_series[0]
        cutoff_2h = first_ts + timedelta(hours=2)
        n_in_2h = sum(1 for t in dt_col_series if t <= cutoff_2h)
        frac_2h = n_in_2h / n_ged2

        day_result = {
            "n_ged2":       n_ged2,
            "hour_first":   int(hour_first),
            "hour_last":    int(hour_last),
            "n_episodes":   n_eps,
            "ged2_minutes": round(float(ged2_minutes), 1),
            "frac_in_first_2h": round(float(frac_2h), 4),
        }

        # Special: VIN10 at dtf=1 — time to cumulative 50 and 200 events
        if vin == "VIN10_F_ALT" and target_dtf == 1:
            ts50  = dt_col_series[49]  if n_ged2 >= 50  else None
            ts200 = dt_col_series[199] if n_ged2 >= 200 else None
            hours_to_50  = (ts50  - first_ts).total_seconds() / 3600.0 if ts50  else None
            hours_to_200 = (ts200 - first_ts).total_seconds() / 3600.0 if ts200 else None
            day_result["hours_to_50_events"]  = round(float(hours_to_50),  3) if hours_to_50  is not None else None
            day_result["hours_to_200_events"] = round(float(hours_to_200), 3) if hours_to_200 is not None else None
            if hours_to_50 is not None:
                print(f"  VIN10 dtf=1: first_hour={hour_first}  hours_to_50={hours_to_50:.3f}h  hours_to_200={hours_to_200:.3f}h")
            gap_hrs = (hours_to_200 - hours_to_50) if (hours_to_50 and hours_to_200) else None
            day_result["hours_50_to_200"] = round(float(gap_hrs), 3) if gap_hrs else None
            print(f"  => could Tier-2 (>=200) have fired {gap_hrs:.2f}h after Tier-1 (>=50) on same day? {'YES if alerting intra-day' if gap_hrs else 'N/A'}")

        print(f"  {vin} dtf={target_dtf}: n_ged2={n_ged2}  first_hour={hour_first}  "
              f"last_hour={hour_last}  episodes={n_eps}  ged2_min={ged2_minutes:.1f}  frac_2h={frac_2h:.3f}")

        vin_output[f"dtf_{target_dtf}"] = day_result

    output[vin] = vin_output

# ── write json ────────────────────────────────────────────────────────────────
out_path = RESULTS / "r3_storm_subdaily.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)
print()
print(f"JSON written: {out_path}")
print("=" * 70)
