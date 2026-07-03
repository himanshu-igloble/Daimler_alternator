"""Task 1 — Depth-qualified two-tier emergency rule (R3 independent recompute).

Writes:
  V12_ALT_GED/results/r3_emergency_2tier.csv
Prints:
  per-day table, operating-point sweep, combined 2-tier verdict.
"""
import sys
import pathlib
import itertools

import polars as pl
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
CACHE = RESULTS / "ged_daily_cache.parquet"

df = pl.read_parquet(CACHE)

# ── extract every VIN-day that had at least one GED=2 event ──────────────────
ged2_days = (
    df.filter(pl.col("ged_cnt_2") > 0)
    .select([
        "vin", "failed", "day", "dtf",
        "ged_cnt_2",
        "vsi_when_ged2_mean",
        "vsi_mean",
    ])
    .with_columns(
        (pl.col("vsi_when_ged2_mean") - pl.col("vsi_mean")).alias("depth_delta")
    )
    .sort(["vin", "day"])
)

# Convert to pandas for rich printing
g2pd = ged2_days.to_pandas()

print("=" * 90)
print("TASK 1 — TWO-TIER GED EMERGENCY RULE — INDEPENDENT RECOMPUTE")
print("=" * 90)
print()
print("(a) FULL PER-DAY TABLE (all VINs, all days with ged_cnt_2 > 0)")
print("-" * 90)
fmt = "{:<14s} {:>6s} {:>6s} {:>10s} {:>10s} {:>10s} {:>10s} {:>12s}"
print(fmt.format("vin", "failed", "dtf", "ged_cnt_2", "vsi_ged2", "vsi_mean", "depth_d", "group"))
print("-" * 90)

for _, row in g2pd.sort_values(["vin", "day"]).iterrows():
    dtf_s = str(int(row["dtf"])) if pd.notna(row["dtf"]) else "NF"
    ged_cnt = int(row["ged_cnt_2"])
    vsi_g = row["vsi_when_ged2_mean"]
    vsi_m = row["vsi_mean"]
    dd = row["depth_delta"]
    grp = "VIN1-storm" if (row["vin"] == "VIN1_F_ALT" and ged_cnt >= 50) else \
          "VIN1-other" if row["vin"] == "VIN1_F_ALT" else \
          "VIN10-F"    if row["vin"] == "VIN10_F_ALT" else "NF"
    print(fmt.format(
        row["vin"], str(row["failed"]), dtf_s,
        f"{ged_cnt}", f"{vsi_g:.3f}", f"{vsi_m:.3f}", f"{dd:+.3f}", grp
    ))

print()
print("(a) SUMMARY — VIN1 storm-days (ged_cnt_2>=50) vs NF ged2 days:")
vin1_storm = g2pd[(g2pd["vin"] == "VIN1_F_ALT") & (g2pd["ged_cnt_2"] >= 50)]
nf_ged2    = g2pd[g2pd["failed"] == False]
print(f"  VIN1 storm days     : {len(vin1_storm)}")
print(f"    vsi_when_ged2 range: {vin1_storm['vsi_when_ged2_mean'].min():.3f} – {vin1_storm['vsi_when_ged2_mean'].max():.3f} V")
print(f"    depth_delta range  : {vin1_storm['depth_delta'].min():+.3f} – {vin1_storm['depth_delta'].max():+.3f} V  (claim: ABOVE vsi_mean ~+0.3V)")
print(f"  NF ged2 days        : {len(nf_ged2)}")
print(f"    vsi_when_ged2 range: {nf_ged2['vsi_when_ged2_mean'].min():.3f} – {nf_ged2['vsi_when_ged2_mean'].max():.3f} V  (claim: BELOW, 20.8-27.2)")
nf_max_vsi_ged2 = nf_ged2["vsi_when_ged2_mean"].max()
vin1_storm_min_vsi_ged2 = vin1_storm["vsi_when_ged2_mean"].min()
claim_separation = vin1_storm_min_vsi_ged2 > nf_max_vsi_ged2
print(f"  Claim (VIN1-storm > ALL-NF vsi_when_ged2): {'CONFIRMED' if claim_separation else 'REFUTED'}")
print(f"    VIN1-storm min = {vin1_storm_min_vsi_ged2:.3f}  |  NF max = {nf_max_vsi_ged2:.3f}")
print()

# ── (b) Tier-1 sweep ─────────────────────────────────────────────────────────
print("(b) TIER-1 SWEEP — C1 x V1 operating points")
C1_vals = [10, 25, 50, 100]
V1_vals = [26.5, 27.0, 27.5]
print()
hdr = "{:>5s} {:>5s} | {:>10s} {:>14s} {:>14s} {:>14s}"
print(hdr.format("C1", "V1", "NF-fires", "F-fires", "F-first-dtfs", "Notes"))
print("-" * 75)

tier1_rows = []
for C1, V1 in itertools.product(C1_vals, V1_vals):
    cand = g2pd[(g2pd["ged_cnt_2"] >= C1) & (g2pd["vsi_when_ged2_mean"] >= V1)]
    # per-VIN first fire
    f_fires  = cand[cand["failed"] == True]
    nf_fires = cand[cand["failed"] == False]

    f_vins = sorted(f_fires["vin"].unique())
    nf_count = nf_fires["vin"].nunique()

    # first-fire dtf per failed VIN
    first_dtf = {}
    for v in f_vins:
        rows_v = f_fires[f_fires["vin"] == v]
        first_dtf[v] = rows_v["dtf"].max()  # max dtf = earliest chronologically

    dtf_str = " ".join([f"{v.replace('_F_ALT','')}@{int(d)}" for v, d in sorted(first_dtf.items())])
    print(f"  C1={C1:>3d} V1={V1:.1f} | NF-fires={nf_count:>3d}  F-fires={len(f_vins):>3d}/{10:<2d}  first-dtf={dtf_str}")
    tier1_rows.append({"C1": C1, "V1": V1, "nf_fires": nf_count, "f_vins": len(f_vins), "dtfs": dtf_str})

print()

# ── (c) Tier-2 confirmation ───────────────────────────────────────────────────
print("(c) TIER-2 — ged_cnt_2 >= 200 (unchanged)")
tier2 = g2pd[g2pd["ged_cnt_2"] >= 200]
print(f"  VINs firing Tier-2 ({len(tier2)} VIN-days total):")
for _, row in tier2.sort_values(["vin","day"]).iterrows():
    dtf_s = str(int(row["dtf"])) if pd.notna(row["dtf"]) else "NF"
    print(f"    {row['vin']}  dtf={dtf_s}  ged_cnt_2={int(row['ged_cnt_2'])}  vsi_when_ged2={row['vsi_when_ged2_mean']:.3f}")

vin10_t2 = tier2[tier2["vin"] == "VIN10_F_ALT"]
nf_t2    = tier2[tier2["failed"] == False]
vin10_dtf = int(vin10_t2["dtf"].iloc[0]) if len(vin10_t2) else "ABSENT"
print(f"  VIN10_F_ALT Tier-2 dtf = {vin10_dtf}  (claim: 1)  vsi_when_ged2 = {vin10_t2['vsi_when_ged2_mean'].iloc[0]:.3f}" if len(vin10_t2) else "  VIN10_F_ALT NOT in Tier-2!")
print(f"  NF false-fires Tier-2  = {nf_t2['vin'].nunique()} VINs  (claim: 0)")
print()

# ── (d) Combined 2-tier summary ───────────────────────────────────────────────
print("(d) COMBINED 2-TIER (C1=50, V1=27.0 + Tier-2>=200)")
t1 = g2pd[(g2pd["ged_cnt_2"] >= 50) & (g2pd["vsi_when_ged2_mean"] >= 27.0)]
t2 = g2pd[g2pd["ged_cnt_2"] >= 200]

combined_vin_days = pd.concat([t1, t2]).drop_duplicates()
f_comb  = combined_vin_days[combined_vin_days["failed"] == True]
nf_comb = combined_vin_days[combined_vin_days["failed"] == False]

f_vins_comb = sorted(f_comb["vin"].unique())
nf_vins_comb = sorted(nf_comb["vin"].unique())

print(f"  Failed VINs firing: {len(f_vins_comb)}/10  {f_vins_comb}")
for v in f_vins_comb:
    rows_v = f_comb[f_comb["vin"] == v]
    best_dtf = rows_v["dtf"].max()
    print(f"    {v}: first-fire dtf={int(best_dtf) if pd.notna(best_dtf) else 'NF'}")

print(f"  NF VINs firing: {len(nf_vins_comb)}/15  {nf_vins_comb}")
print()

# Margins
# Count margin: VIN1 min storm-day count vs best NF day count
vin1_storm_min_cnt = vin1_storm["ged_cnt_2"].min()
nf_max_cnt = nf_ged2["ged_cnt_2"].max()
cnt_margin = vin1_storm_min_cnt - nf_max_cnt
print(f"  COUNT MARGIN  : VIN1-storm min={vin1_storm_min_cnt} vs NF max={nf_max_cnt}  => margin={cnt_margin}")

# Voltage margin: VIN1 storm min vsi_ged2 vs NF max vsi_ged2
v_margin = vin1_storm_min_vsi_ged2 - nf_max_vsi_ged2
print(f"  VOLTAGE MARGIN: VIN1-storm min vsi_ged2={vin1_storm_min_vsi_ged2:.3f} vs NF max vsi_ged2={nf_max_vsi_ged2:.3f}  => margin={v_margin:+.3f} V")

print()
print("VERDICT:")
f_count = len(f_vins_comb)
nf_count_comb = len(nf_vins_comb)
print(f"  2-tier preserves {f_count}/10 F + {nf_count_comb}/15 NF false-fires")
print(f"  Voltage dimension adds second qualifying axis: {'YES' if v_margin > 0 else 'NO (overlapping range)'}")
print(f"  Claim (2/10 F, 0/15 NF): {'CONFIRMED' if f_count == 2 and nf_count_comb == 0 else 'DEVIATION — see above'}")

# ── write CSV ─────────────────────────────────────────────────────────────────
out = g2pd[["vin","failed","day","dtf","ged_cnt_2","vsi_when_ged2_mean","vsi_mean","depth_delta"]].copy()
out.to_csv(RESULTS / "r3_emergency_2tier.csv", index=False)
print()
print(f"CSV written: {RESULTS / 'r3_emergency_2tier.csv'}")
print("=" * 90)
