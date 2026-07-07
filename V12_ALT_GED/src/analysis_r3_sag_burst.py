"""Task 3 — Pre-storm VSI-sag burst, self-normalized (R3 independent recompute).

Sag-day: vsi_p10 <= 24.0 on engine-on days
         (engine-on = regime_idle_rows + regime_cruise_rows + regime_heavy_rows > 0)

Self-normalize per VIN:
  trailing30 = rolling 30-day count of sag-days
  burst = (trailing30 - VIN_median) / max(VIN_IQR, 1.0)

NOTE on denominator floor: spec says (VIN_IQR + 1e-9). With 1e-9 the formula
degenerates for zero-IQR VINs that have any occasional sag day (gives ~1e9
values). We use max(IQR, 1.0) — one sag-day is the minimum meaningful unit —
and report both the practical burst and the degenerate raw result for transparency.

Writes: V12_ALT_GED/results/r3_sag_burst.json
"""
import sys
import json
import pathlib

import numpy as np
import pandas as pd
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
CACHE   = RESULTS / "ged_daily_cache.parquet"

df = pl.read_parquet(CACHE)

# ── engine-on filter ──────────────────────────────────────────────────────────
engine_on = df.filter(
    (pl.col("regime_idle_rows") + pl.col("regime_cruise_rows") + pl.col("regime_heavy_rows")) > 0
)

# ── sag indicator ─────────────────────────────────────────────────────────────
engine_on = engine_on.with_columns(
    (pl.col("vsi_p10") <= 24.0).cast(pl.Int32).alias("is_sag")
)

FAILED_VINS    = [f"VIN{i}_F_ALT" for i in range(1, 11)]
NONFAILED_VINS = [f"VIN{i}_NF_ALT" for i in range(1, 16)]
ALL_VINS       = FAILED_VINS + NONFAILED_VINS

results = {}

print("=" * 70)
print("TASK 3 — VSI-SAG BURST, SELF-NORMALIZED")
print("  Denominator: max(IQR, 1.0)  [spec: IQR+1e-9, see docstring]")
print("=" * 70)
print()

nf_max_bursts = {}

for vin in ALL_VINS:
    sub = (
        engine_on.filter(pl.col("vin") == vin)
        .select(["day", "dtf", "vsi_p10", "is_sag", "failed"])
        .sort("day")
    )
    if len(sub) == 0:
        results[vin] = {"note": "no engine-on days"}
        continue

    days_pd = sub.to_pandas()
    days_pd["day"] = pd.to_datetime(days_pd["day"])
    days_pd = days_pd.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    if len(days_pd) == 0:
        results[vin] = {"note": "all NaT dates dropped"}
        continue

    # rolling 30-day sag count
    days_pd_idx = days_pd.set_index("day")
    t30_series = days_pd_idx["is_sag"].rolling("30D", min_periods=1).sum().reset_index()
    t30_series.columns = ["day", "trailing30_sag"]
    days_pd = days_pd.merge(t30_series, on="day", how="left")

    t30       = days_pd["trailing30_sag"].values.astype(float)
    vin_med   = float(np.median(t30))
    vin_q25   = float(np.percentile(t30, 25))
    vin_q75   = float(np.percentile(t30, 75))
    vin_iqr   = vin_q75 - vin_q25

    # practical denominator (floor at 1.0 sag-day unit)
    denom     = max(vin_iqr, 1.0)
    burst     = (t30 - vin_med) / denom
    # degenerate version for reporting
    burst_raw = (t30 - vin_med) / (vin_iqr + 1e-9)

    days_pd["burst"]     = burst
    days_pd["burst_raw"] = burst_raw

    failed_flag = bool(days_pd["failed"].iloc[0])

    if vin == "VIN1_F_ALT":
        dtf_vals = days_pd["dtf"].values.astype(float)
        # pre-storm window dtf 76 → 22 (inclusive)
        mask_prestorm = np.isfinite(dtf_vals) & (dtf_vals >= 22) & (dtf_vals <= 76)
        if mask_prestorm.sum() > 0:
            prestorm_burst  = burst[mask_prestorm]
            max_burst_val   = float(prestorm_burst.max())
            max_raw_val     = float(burst_raw[mask_prestorm].max())
            # which dtf?
            best_idx_local  = np.argmax(prestorm_burst)
            matching_dtf    = dtf_vals[mask_prestorm][best_idx_local]
            max_burst_dtf   = int(matching_dtf)
        else:
            max_burst_val = max_raw_val = None
            max_burst_dtf = None

        results["VIN1_F_ALT"] = {
            "vin":                  "VIN1_F_ALT",
            "failed":               True,
            "vin_median_t30":       round(vin_med, 4),
            "vin_iqr_t30":          round(vin_iqr, 4),
            "denom_used":           round(denom, 4),
            "prestorm_max_burst":   round(max_burst_val, 4) if max_burst_val is not None else None,
            "prestorm_max_raw":     round(max_raw_val,   4) if max_raw_val   is not None else None,
            "prestorm_max_burst_dtf": max_burst_dtf,
            "engine_on_days":       int(len(days_pd)),
            "n_sag_days":           int(days_pd["is_sag"].sum()),
        }
        print(f"VIN1_F_ALT:")
        print(f"  engine-on days={len(days_pd)}, sag-days={int(days_pd['is_sag'].sum())}")
        print(f"  VIN median t30={vin_med:.2f}, IQR={vin_iqr:.2f}, denom_used={denom:.2f}")
        print(f"  Pre-storm (dtf 76-22): max_burst={max_burst_val:.3f} at dtf={max_burst_dtf}")
        print(f"  Pre-storm raw (IQR+1e-9): max={max_raw_val:.3f}")
    else:
        max_b     = float(burst.max()) if len(burst) > 0 else None
        max_b_raw = float(burst_raw.max()) if len(burst_raw) > 0 else None
        if not failed_flag:
            nf_max_bursts[vin] = max_b
        results[vin] = {
            "vin":              vin,
            "failed":           failed_flag,
            "vin_median_t30":   round(vin_med, 4),
            "vin_iqr_t30":      round(vin_iqr, 4),
            "denom_used":       round(denom, 4),
            "max_burst_ever":   round(max_b, 4) if max_b is not None else None,
            "max_burst_raw":    round(max_b_raw, 4) if max_b_raw is not None else None,
            "engine_on_days":   int(len(days_pd)),
            "n_sag_days":       int(days_pd["is_sag"].sum()),
        }

print()
print("NF MAX BURST (over full history, denom=max(IQR,1.0)):")
for vin_nf, mb in sorted(nf_max_bursts.items()):
    r = results[vin_nf]
    print(f"  {vin_nf:<16s}  max_burst={mb:8.3f}  iqr={r['vin_iqr_t30']:.2f}  n_sag={r['n_sag_days']}")

nf_max_ceil  = max(nf_max_bursts.values()) if nf_max_bursts else None
nf_max_vin   = max(nf_max_bursts, key=nf_max_bursts.get) if nf_max_bursts else None

print()
vin1_res = results.get("VIN1_F_ALT", {})
vin1_pb  = vin1_res.get("prestorm_max_burst")

print("VERDICT:")
print(f"  VIN1 pre-storm max burst (denom=max(IQR,1))  = {vin1_pb:.3f}" if vin1_pb is not None else "  VIN1 pre-storm burst UNAVAILABLE")
print(f"  NF max-burst ceiling = {nf_max_ceil:.3f} ({nf_max_vin})" if nf_max_ceil is not None else "  NF ceiling UNAVAILABLE")

if vin1_pb is not None and nf_max_ceil is not None:
    if vin1_pb > nf_max_ceil:
        margin = vin1_pb - nf_max_ceil
        print(f"  => VIN1 burst ABOVE NF ceiling (margin=+{margin:.3f}) — USABLE early warning")
    else:
        margin = vin1_pb - nf_max_ceil
        print(f"  => VIN1 burst INSIDE NF range (margin={margin:.3f}) — KILL / not separable")

print()
print("NOTE: with literal (IQR+1e-9), many NF VINs with IQR=0 but any sag day")
print("      get infinite-scale burst (e.g. 3/1e-9 = 3e9). That is degenerate.")
print("      The practical max(IQR,1.0) floor is the meaningful comparison.")

# ── write json ────────────────────────────────────────────────────────────────
out = {
    "task":                      "r3_sag_burst",
    "sag_threshold_vsi_p10":     24.0,
    "engine_on_definition":      "regime_idle_rows + regime_cruise_rows + regime_heavy_rows > 0",
    "denominator_note":          "max(IQR, 1.0) used; spec says (IQR+1e-9) which degenerates when IQR=0",
    "per_vin":                   results,
    "nf_max_burst_ceiling":      float(nf_max_ceil) if nf_max_ceil is not None else None,
    "nf_max_burst_vin":          nf_max_vin,
    "vin1_prestorm_max_burst":   vin1_pb,
    "vin1_prestorm_max_burst_dtf": vin1_res.get("prestorm_max_burst_dtf"),
    "verdict":                   (
        "ABOVE_NF_CEILING" if (vin1_pb is not None and nf_max_ceil is not None and vin1_pb > nf_max_ceil)
        else "INSIDE_NF_RANGE" if (vin1_pb is not None and nf_max_ceil is not None)
        else "UNAVAILABLE"
    ),
}
out_path = RESULTS / "r3_sag_burst.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print()
print(f"JSON written: {out_path}")
print("=" * 70)
