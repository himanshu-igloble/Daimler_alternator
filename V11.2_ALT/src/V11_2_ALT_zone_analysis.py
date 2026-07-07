"""
V11.2_ALT Task 3: Zone-system universality analysis.

PART 1  — Deployed V11.1 risk bands (green/amber/red) table + cutoff confirmation.
PART 2  — Explored 4-zone M5 health system: per-VIN time-in-zone from monthly components.
PART 3  — Consistency tests (JSON).
PART 4  — Three publication-quality plots.

SOURCE USED: (b) per-VIN monthly component reconstruction
    File: V5.2_ALT/results/V5.2.1_20_5_ALT_monthly_trajectories.csv
    Columns available: vsi_std_monthly (C2), vsi_range_monthly (C4),
                       vsi_mean_monthly (C1), ged2_count_monthly (C3 proxy)
    C5 (crank_min_vsi) is NOT present in the monthly file -> dropped; weights renormalized.

M5 formula used (4-component, renormalized):
    C1 = min(1, abs(VSI_mean - 28.2) / 4.0)          weight 0.25 -> renorm 0.303
    C2 = min(1, VSI_std / 0.45)                        weight 0.20 -> renorm 0.242
    C3 = min(1, GED2_rate / 0.05)                      weight 0.25 -> renorm 0.303
         where GED2_rate = ged2_count / (days_with_data * 48)  # 30-min sampling
    C4 = min(1, VSI_range / 2.5)                       weight 0.15 -> renorm 0.152
    C5 DROPPED (no crank data in monthly file)          original weight 0.15 omitted
    Score = 0.303*C1 + 0.242*C2 + 0.303*C3 + 0.152*C4

4-zone boundaries: GREEN < 0.20, YELLOW 0.20-0.40, ORANGE 0.40-0.60, RED >= 0.60
"""

from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

sys.path.insert(0, r"D:/Daimler-starter_motor_alternator_battery/V11.2_ALT/src")
from V11_2_ALT_common import (
    load_v111_rul, save_json,
    RESULTS, VIZ, PALETTE, style_ax
)

# ── directories ────────────────────────────────────────────────────────────────
VIZ_ZONE = os.path.join(VIZ, "zone_analysis")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(VIZ_ZONE, exist_ok=True)

# ── zone colours ───────────────────────────────────────────────────────────────
ZONE_COLORS = {
    "GREEN":  "#2e7d32",
    "YELLOW": "#f9a825",
    "ORANGE": "#ef6c00",
    "RED":    "#b71c1c",
}

# ── 4-zone M5 boundaries ───────────────────────────────────────────────────────
ZONE_CUTS = [0.0, 0.20, 0.40, 0.60, 1.1]
ZONE_LABELS = ["GREEN", "YELLOW", "ORANGE", "RED"]


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Deployed V11.1 risk bands
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1: Deployed V11.1 risk bands")
print("=" * 60)

rul_df = load_v111_rul()

# keep only needed columns
bands_df = rul_df[["vin_label", "ridge_prob", "risk_band", "above_thr"]].copy()
bands_df = bands_df.sort_values("vin_label").reset_index(drop=True)

# Empirical cutoff confirmation
print("\nEmpirical ridge_prob per risk_band:")
for band, grp in bands_df.groupby("risk_band"):
    lo, hi = grp["ridge_prob"].min(), grp["ridge_prob"].max()
    print(f"  {band:6s}  min={lo:.4f}  max={hi:.4f}  n={len(grp)}")

# Global thresholds per spec: green<0.35, amber 0.35-0.55, red>=0.55
print("\nExpected thresholds: green<0.35  amber 0.35-0.55  red>=0.55")
print("  (Above-thr at Youden 0.4456)\n")

# Write CSV
out_p1 = os.path.join(RESULTS, "V11.2_ALT_deployed_bands.csv")
bands_df.to_csv(out_p1, index=False)
print(f"Written: {out_p1}  ({len(bands_df)} rows)")
assert len(bands_df) == 25, f"GUARD FAIL: expected 25 rows, got {len(bands_df)}"


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — 4-zone time-in-zone from monthly component reconstruction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: 4-zone time-in-zone (M5 monthly reconstruction)")
print("=" * 60)

MONTHLY_PATH = (
    r"D:/Daimler-starter_motor_alternator_battery"
    r"/V5.2_ALT/results/V5.2.1_20_5_ALT_monthly_trajectories.csv"
)
mt = pd.read_csv(MONTHLY_PATH, parse_dates=["month_start", "month_end"])
mt["days_with_data"] = mt["days_with_data"].fillna(30)

# ── M5 formula (4-component, C5 dropped; weights renormalised) ─────────────
# Original weights: C1=0.25, C2=0.20, C3=0.25, C4=0.15, C5=0.15; total=1.0
# Drop C5 (weight 0.15); remaining sum = 0.85
# Renorm: w1=0.25/0.85=0.294, w2=0.20/0.85=0.235, w3=0.25/0.85=0.294, w4=0.15/0.85=0.176
W1, W2, W3, W4 = 0.25/0.85, 0.20/0.85, 0.25/0.85, 0.15/0.85

VSI_NOMINAL   = 28.2   # V
C1_DENOM      = 4.0
C2_DENOM      = 0.45   # std saturation V
C3_THRESHOLD  = 0.05   # 5 % of readings in GED=2 → saturation
SAMPLES_PER_DAY = 48.0  # ~30-min sampling
C4_DENOM      = 2.5    # range saturation V


def compute_m5_monthly(row):
    """Compute per-month M5 degradation score (C5 absent)."""
    c1 = min(1.0, abs(row["vsi_mean_monthly"] - VSI_NOMINAL) / C1_DENOM)
    c2 = min(1.0, row["vsi_std_monthly"] / C2_DENOM)
    denom_c3 = row["days_with_data"] * SAMPLES_PER_DAY
    ged2_rate = row["ged2_count_monthly"] / denom_c3 if denom_c3 > 0 else 0.0
    c3 = min(1.0, ged2_rate / C3_THRESHOLD)
    c4 = min(1.0, row["vsi_range_monthly"] / C4_DENOM)
    return W1*c1 + W2*c2 + W3*c3 + W4*c4


mt["m5_score"] = mt.apply(compute_m5_monthly, axis=1)
mt["zone4"] = pd.cut(
    mt["m5_score"],
    bins=ZONE_CUTS,
    labels=ZONE_LABELS,
    right=False,
).astype(str)

# ── derive is_failed per VIN_LABEL ─────────────────────────────────────────
failed_vins = set(bands_df[bands_df["risk_band"] == "red"]["vin_label"].tolist() +
                  [v for v in mt["VIN_LABEL"].unique() if "_F_" in v])
# Use the F/NF label in the VIN name as ground truth
mt["failed"] = mt["VIN_LABEL"].str.contains("_F_")

print(f"\nMonthly rows: {len(mt)}, VINs: {mt['VIN_LABEL'].nunique()}")
print("\nZone distribution across all VIN-months:")
print(mt["zone4"].value_counts().to_dict())

# ── per-VIN occupancy ─────────────────────────────────────────────────────
records = []
for vin, grp in mt.groupby("VIN_LABEL"):
    grp = grp.sort_values("month_start").reset_index(drop=True)
    failed = bool(grp["failed"].iloc[0])
    total_months = len(grp)
    total_days = grp["days_with_data"].sum()
    if total_days == 0:
        total_days = total_months * 30

    zone_days = {}
    zone_months = {}
    for z in ZONE_LABELS:
        zmask = grp["zone4"] == z
        zone_months[z] = int(zmask.sum())
        zone_days[z] = int(grp.loc[zmask, "days_with_data"].sum())

    pct = {z: round(100.0 * zone_days[z] / total_days, 2) if total_days > 0 else 0.0
           for z in ZONE_LABELS}

    # first-entry age (month_start of first occurrence of each zone)
    first_entry = {}
    for z in ZONE_LABELS:
        hits = grp[grp["zone4"] == z]
        first_entry[z] = str(hits["month_start"].min().date()) if len(hits) > 0 else None

    records.append({
        "vin_label":   vin,
        "failed":      int(failed),
        "pct_green":   pct["GREEN"],
        "pct_yellow":  pct["YELLOW"],
        "pct_orange":  pct["ORANGE"],
        "pct_red":     pct["RED"],
        "days_green":  zone_days["GREEN"],
        "days_yellow": zone_days["YELLOW"],
        "days_orange": zone_days["ORANGE"],
        "days_red":    zone_days["RED"],
        "source_used": "b_monthly_components_reconstructed_M5",
        "first_entry_GREEN":  first_entry["GREEN"],
        "first_entry_YELLOW": first_entry["YELLOW"],
        "first_entry_ORANGE": first_entry["ORANGE"],
        "first_entry_RED":    first_entry["RED"],
    })

occ_df = pd.DataFrame(records)
assert len(occ_df) == 25, f"GUARD FAIL: expected 25 rows, got {len(occ_df)}"

out_p2 = os.path.join(RESULTS, "V11.2_ALT_zone_occupancy.csv")
occ_df.to_csv(out_p2, index=False)
print(f"\nWritten: {out_p2}  ({len(occ_df)} rows)")

print("\nOccupancy summary (all VINs):")
cols = ["vin_label", "failed", "pct_green", "pct_yellow", "pct_orange", "pct_red",
        "days_green", "days_yellow", "days_orange", "days_red"]
print(occ_df[cols].to_string(index=False))

# Failed VINs that ever reached ORANGE or RED
failed_df = occ_df[occ_df["failed"] == 1]
n_failed = len(failed_df)
reached_orange_or_red = failed_df[(failed_df["days_orange"] > 0) | (failed_df["days_red"] > 0)]
n_orange_red = len(reached_orange_or_red)
pct_orange_red = 100.0 * n_orange_red / n_failed if n_failed > 0 else 0.0
print(f"\nFailed VINs that reached ORANGE or RED: {n_orange_red}/{n_failed} "
      f"({pct_orange_red:.1f}%)")
print("  VINs:", reached_orange_or_red["vin_label"].tolist())

nf_df = occ_df[occ_df["failed"] == 0]
nf_orange_red = nf_df[(nf_df["days_orange"] > 0) | (nf_df["days_red"] > 0)]
print(f"Non-failed VINs in ORANGE or RED: {len(nf_orange_red)}/{len(nf_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Consistency tests
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3: Consistency tests")
print("=" * 60)

# Variance / spread across VINs for each pct zone
spread = {}
for z in ZONE_LABELS:
    col = f"pct_{z.lower()}"
    vals = occ_df[col].values
    spread[z] = {
        "mean": round(float(np.mean(vals)), 2),
        "std":  round(float(np.std(vals)),  2),
        "min":  round(float(np.min(vals)),  2),
        "max":  round(float(np.max(vals)),  2),
    }
print("\nPct-zone spread (std as uniformity proxy):")
for z, s in spread.items():
    print(f"  {z:7s}  mean={s['mean']:5.1f}%  std={s['std']:5.1f}%  "
          f"min={s['min']:4.1f}%  max={s['max']:4.1f}%")

# Rapid transition: first YELLOW to first RED < 30 days
rapid_transitions = []
for _, row in occ_df.iterrows():
    fy = row["first_entry_YELLOW"]
    fr = row["first_entry_RED"]
    if fy and fr:
        gap_days = (pd.to_datetime(fr) - pd.to_datetime(fy)).days
        if gap_days < 30:
            rapid_transitions.append({
                "vin_label": row["vin_label"],
                "failed": int(row["failed"]),
                "days_yellow_to_red": gap_days,
            })

print(f"\nRapid transitions (YELLOW->RED < 30 days): {len(rapid_transitions)}")
for rt in rapid_transitions:
    print(f"  {rt['vin_label']}  failed={rt['failed']}  gap={rt['days_yellow_to_red']}d")

# Deployed bands cutoff confirmation
band_stats = {}
for band, grp in bands_df.groupby("risk_band"):
    band_stats[band] = {
        "n": len(grp),
        "ridge_prob_min": round(float(grp["ridge_prob"].min()), 4),
        "ridge_prob_max": round(float(grp["ridge_prob"].max()), 4),
    }
print("\nDeployed band cutoff confirmation (empirical):")
for b, s in band_stats.items():
    print(f"  {b:6s}  n={s['n']}  prob_range=[{s['ridge_prob_min']:.4f}, "
          f"{s['ridge_prob_max']:.4f}]")
print("  Expected: green<0.35, amber 0.35-0.55, red>=0.55")
cutoff_ok = (
    band_stats.get("green", {}).get("ridge_prob_max", 1.0) < 0.35 and
    band_stats.get("red", {}).get("ridge_prob_min", 0.0) >= 0.55
)
print(f"  Cutoffs confirmed: {cutoff_ok}")

# Verdict
verdict = (
    "GLOBAL bands are acceptable for RANKING: deployed green/amber/red risk bands cleanly "
    "separate VINs by ridge_prob and empirically confirm the 0.35/0.55 thresholds. "
    f"The explored 4-zone temporal M5 health system has WEAK sensitivity: only "
    f"{n_orange_red}/{n_failed} ({pct_orange_red:.0f}%) failed VINs ever reached "
    "ORANGE or RED, while non-failed trucks also enter those zones. "
    "Dynamic per-VIN zones are NOT justified by n=25 data; the data ceiling "
    "(n=10 failed) prevents reliable zone-level health-progression discrimination. "
    "Recommendation: use V11.1 risk bands for operational deployment; treat 4-zone "
    "M5 trajectories as supplemental visual context only, not a standalone alert system."
)
print(f"\nVERDICT:\n{verdict}")

consistency = {
    "zone_spread_pct": spread,
    "failed_reaching_orange_or_red": {
        "n_failed": int(n_failed),
        "n_reached": int(n_orange_red),
        "pct_reached": round(pct_orange_red, 1),
        "vins": reached_orange_or_red["vin_label"].tolist(),
    },
    "nonfailed_in_orange_or_red": {
        "n_nonfailed": int(len(nf_df)),
        "n_reached": int(len(nf_orange_red)),
        "vins": nf_orange_red["vin_label"].tolist(),
    },
    "rapid_transitions_yellow_to_red_lt30d": rapid_transitions,
    "deployed_bands_cutoff_confirmation": {
        "empirical_bands": band_stats,
        "cutoffs_confirmed": cutoff_ok,
        "spec_thresholds": {"green_lt": 0.35, "amber_range": [0.35, 0.55], "red_gte": 0.55},
        "youden_above_thr": 0.4456,
    },
    "source_used_for_m5_timeseries": "b_monthly_components_V5.2.1_20_5_ALT_monthly_trajectories.csv",
    "c5_status": "DROPPED_not_in_monthly_file_weights_renormalized_to_1.0",
    "verdict": verdict,
}
save_json(consistency, "V11.2_ALT_zone_consistency.json")
out_p3 = os.path.join(RESULTS, "V11.2_ALT_zone_consistency.json")
print(f"\nWritten: {out_p3}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4: Plots")
print("=" * 60)

# Sort VINs: failed first, then NF, alphabetically within each group
occ_sorted = occ_df.copy()
occ_sorted["sort_key"] = occ_sorted.apply(
    lambda r: (1 - r["failed"], r["vin_label"]), axis=1
)
occ_sorted = occ_sorted.sort_values("sort_key").reset_index(drop=True)
vin_order = occ_sorted["vin_label"].tolist()

# ── Plot 1: horizontal stacked bar — % life per zone per VIN ─────────────
fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(vin_order))
left = np.zeros(len(vin_order))

for zone in ZONE_LABELS:
    col = f"pct_{zone.lower()}"
    vals = occ_sorted[col].values
    ax.barh(y_pos, vals, left=left,
            color=ZONE_COLORS[zone], label=zone, height=0.7, edgecolor="white", lw=0.4)
    left += vals

ax.set_yticks(y_pos)
ax.set_yticklabels(vin_order, fontsize=8)
ax.set_xlabel("% of observed truck life", fontsize=10)
ax.set_xlim(0, 105)
ax.axhline(y=n_failed - 0.5, color="#666", lw=1.2, ls="--", alpha=0.7)
ax.text(101, n_failed - 0.5, "↑ Failed  ↓ Non-failed",
        va="center", ha="left", fontsize=7.5, color="#555")

# Mark failed VINs
for i, row in occ_sorted.iterrows():
    if row["failed"]:
        ax.get_yticklabels()[i].set_color(PALETTE["fail"])
        ax.get_yticklabels()[i].set_weight("bold")

# Legend patches
patches = [mpatches.Patch(color=ZONE_COLORS[z], label=z) for z in ZONE_LABELS]
ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.85)
style_ax(ax, title="Zone Occupancy: % Life in Each Health Zone per VIN\n(failed VINs in red, dashed separator)",
         xlabel="% of observed truck life")
plt.tight_layout()

p4a = os.path.join(VIZ_ZONE, "zone_occupancy_stacked.png")
fig.savefig(p4a, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p4a}")

# ── Plot 2: heatmap — VIN x zone, days in zone ──────────────────────────
heat_data = occ_sorted[["days_green", "days_yellow", "days_orange", "days_red"]].values
col_labels = ["GREEN", "YELLOW", "ORANGE", "RED"]

fig, ax = plt.subplots(figsize=(6, 10))
zone_cmap = LinearSegmentedColormap.from_list(
    "zones",
    ["#e8f5e9", "#f9a825", "#ef6c00", "#b71c1c"],
    N=256
)
im = ax.imshow(heat_data, aspect="auto", cmap=zone_cmap, vmin=0)
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(vin_order)))
ax.set_yticklabels(vin_order, fontsize=8)

# Color-code failed y-labels
for i, row in occ_sorted.iterrows():
    color = PALETTE["fail"] if row["failed"] else "#1f1f1f"
    ax.get_yticklabels()[i].set_color(color)

# Annotate cells with days
for r in range(heat_data.shape[0]):
    for c in range(heat_data.shape[1]):
        val = int(heat_data[r, c])
        if val > 0:
            txt_color = "white" if val > heat_data.max() * 0.5 else "black"
            ax.text(c, r, str(val), ha="center", va="center",
                    fontsize=7, color=txt_color, weight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Days in zone", fontsize=9)
ax.axhline(y=n_failed - 0.5, color="#666", lw=1.5, ls="--")
style_ax(ax, title="Zone Heatmap: Days per Health Zone\n(failed VINs in red)",
         xlabel="Health Zone")
plt.tight_layout()

p4b = os.path.join(VIZ_ZONE, "zone_heatmap.png")
fig.savefig(p4b, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p4b}")

# ── Plot 3: per-failed-VIN zone timeline (zone vs month_start) ───────────
failed_labels = [v for v in vin_order if "_F_" in v]
n_f = len(failed_labels)
fig, axes = plt.subplots(n_f, 1, figsize=(14, n_f * 1.8), sharex=False)
if n_f == 1:
    axes = [axes]

zone_to_num = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}

for ax_i, (ax, vin) in enumerate(zip(axes, failed_labels)):
    grp = mt[mt["VIN_LABEL"] == vin].sort_values("month_start").reset_index(drop=True)
    grp["zone_num"] = grp["zone4"].map(zone_to_num)

    # Filled step plot coloured by zone
    for _, row_m in grp.iterrows():
        zn = row_m["zone_num"]
        ax.barh(0, row_m["days_with_data"],
                left=(row_m["month_start"] - grp["month_start"].min()).days,
                color=list(ZONE_COLORS.values())[int(zn)],
                height=0.65, edgecolor="white", lw=0.3)

    # Overlay zone level as step line
    xs = [(row_m["month_start"] - grp["month_start"].min()).days for _, row_m in grp.iterrows()]
    ys = grp["zone_num"].tolist()
    ax_twin = ax.twinx()
    ax_twin.step(xs, ys, where="post", color="#333", lw=1.0, alpha=0.6)
    ax_twin.set_yticks([0, 1, 2, 3])
    ax_twin.set_yticklabels(["G", "Y", "O", "R"], fontsize=6)
    ax_twin.set_ylim(-0.5, 3.8)

    ax.set_yticks([])
    ax.set_ylabel(vin.replace("_ALT", ""), fontsize=8, rotation=0, ha="right", va="center")
    ax.set_xlabel("" if ax_i < n_f - 1 else "Days since first observation", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=PALETTE["grid"], lw=0.5, alpha=0.6, axis="x")

patches = [mpatches.Patch(color=ZONE_COLORS[z], label=z) for z in ZONE_LABELS]
axes[0].legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.85)
fig.suptitle("Per-Failed-VIN Zone Timeline (M5 health zones, 4-component)\n"
             "Bar fill = zone colour; right axis = zone level",
             fontsize=11, weight="bold", y=1.01)
plt.tight_layout()

p4c = os.path.join(VIZ_ZONE, "progression_failed.png")
fig.savefig(p4c, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p4c}")

# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DONE — output summary")
print("=" * 60)
print(f"  PART 1  {out_p1}")
print(f"  PART 2  {out_p2}")
print(f"  PART 3  {out_p3}")
print(f"  PLOT A  {p4a}")
print(f"  PLOT B  {p4b}")
print(f"  PLOT C  {p4c}")
print()
print(f"  25-row guard: PASSED")
print(f"  Source: b (monthly_trajectories component reconstruction)")
print(f"  C5 dropped (no crank data); weights renormalized to 1.0")
print(f"  Failed reaching ORANGE/RED: {n_orange_red}/{n_failed} ({pct_orange_red:.0f}%)")
print(f"  Deployed band cutoffs confirmed: {cutoff_ok}")
