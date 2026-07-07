"""V11.2_ALT Task 1 — Statistical validation of prediction heuristics.

Computes failed-vs-healthy distribution stats and separation metrics for:
  Family A: 6 Ridge classification features (production_features.csv)
  Family B: 5 emergency/lead-time heuristics (per-VIN daily panels, late-life window)

Outputs:
  V11.2_ALT/results/V11.2_ALT_heuristic_stats.csv  (11 rows)
  V11.2_ALT/visualizations/heuristic_distributions/<heuristic>.png  (11 PNGs)
  Markdown table printed to stdout
"""

import matplotlib
matplotlib.use("Agg")

import sys, os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde

sys.path.insert(0, r"D:/Daimler-starter_motor_alternator_battery/V11.2_ALT/src")
from V11_2_ALT_common import (
    FAMILY_A, FAMILY_B, PERM_IMPORTANCE,
    auroc_from_scores, cliffs_delta,
    RESULTS, VIZ, PALETTE, style_ax, save_json,
    ROOT,
)

# ── paths ──────────────────────────────────────────────────────────────────────
PROD_FEAT = os.path.join(ROOT, "V5.2_ALT/features/V5.2.1_20_5_ALT_production_features.csv")
FORENSICS  = os.path.join(ROOT, "V11_ALT_heuristics/cache/forensics")
OUT_CSV    = os.path.join(RESULTS, "V11.2_ALT_heuristic_stats.csv")
OUT_VIZ    = os.path.join(VIZ, "heuristic_distributions")
os.makedirs(OUT_VIZ, exist_ok=True)

# ── engineering meaning (one line, electrical interpretation) ──────────────────
ENG_MEANING = {
    "vsi_std_ratio_30d":            "late-life voltage scatter vs early-life baseline",
    "vsi_dominant_freq":            "dominant FFT frequency of daily voltage (rhythm vs drift)",
    "vsi_spectral_entropy":         "voltage spectrum disorder (clean rhythm vs broadband noise)",
    "bat_charge_delta_trend_right": "trend of cruise-minus-resting voltage (charging headroom loss)",
    "vsi_range_trend_last30d":      "slope of daily voltage min-max range late in life",
    "progressive_drift":            "cumulative drift of daily voltage from baseline",
    "crank_recovery_t":             "seconds for voltage to recover to 27V after a crank",
    "vsi_ceiling":                  "regulation plateau voltage at high RPM",
    "vsi_resid_mean":               "voltage residual vs healthy-fleet expected surface",
    "resting_vsi_mean":             "engine-off resting voltage level",
    "ged_churn":                    "rate of GED state-transition churn (excitation instability)",
}

# Family B importance: discriminative-VIN recall from V11 heuristics report
# Source: V11_ALT heuristics validation report (2025-11-xx) per-heuristic recall over 10 failed VINs
FAMILY_B_IMPORTANCE = {
    "crank_recovery_t":  6 / 10,
    "vsi_ceiling":       2 / 10,
    "vsi_resid_mean":    2 / 10,
    "resting_vsi_mean":  3 / 10,
    "ged_churn":         2 / 10,
}


# ── helpers ────────────────────────────────────────────────────────────────────
def percentiles(arr, qs=(10, 25, 50, 75, 90)):
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def dist_stats(arr):
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return dict(mean=float("nan"), median=float("nan"), std=float("nan"),
                    **{f"p{q}": float("nan") for q in (10, 25, 50, 75, 90)})
    return dict(
        mean=float(np.nanmean(a)),
        median=float(np.nanmedian(a)),
        std=float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"),
        **percentiles(a),
    )


def separation_label(cd_abs):
    if cd_abs >= 0.474:
        return "STRONG"
    elif cd_abs >= 0.330:
        return "MEDIUM"
    elif cd_abs >= 0.147:
        return "SMALL"
    else:
        return "NEGLIGIBLE"


def mwu_p(failed_vals, healthy_vals):
    f = np.asarray(failed_vals, float); h = np.asarray(healthy_vals, float)
    f = f[~np.isnan(f)]; h = h[~np.isnan(h)]
    try:
        _, p = scipy_stats.mannwhitneyu(f, h, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


# ── Family A: load from production features CSV ────────────────────────────────
print("=" * 65)
print("Loading Family A features …")
df_prod = pd.read_csv(PROD_FEAT)
assert set(FAMILY_A).issubset(df_prod.columns), \
    f"Missing Family A columns: {set(FAMILY_A) - set(df_prod.columns)}"
df_failed  = df_prod[df_prod["failed"] == 1]
df_healthy = df_prod[df_prod["failed"] == 0]
print(f"  Failed: {len(df_failed)} rows | Healthy: {len(df_healthy)} rows")

family_a_data = {}  # feat -> {"failed": list, "healthy": list}
for feat in FAMILY_A:
    family_a_data[feat] = {
        "failed":  df_failed[feat].dropna().tolist(),
        "healthy": df_healthy[feat].dropna().tolist(),
    }
    print(f"  {feat}: F={len(family_a_data[feat]['failed'])} H={len(family_a_data[feat]['healthy'])}")


# ── Family B: per-VIN scalar from late-life daily panels ──────────────────────
print("\nLoading Family B features from daily panels …")

daily_files = sorted(glob.glob(os.path.join(FORENSICS, "*_daily.csv")))
print(f"  Found {len(daily_files)} daily CSVs")

# Inspect first daily for column availability
_sample = pd.read_csv(daily_files[0])
for feat in FAMILY_B:
    avail = feat in _sample.columns
    print(f"  {feat}: {'FOUND' if avail else 'MISSING'} in daily panels")

LATE_LIFE_N = 30  # last N rows with n_eo > 0 (engine-on days)

family_b_data = {feat: {"failed": [], "healthy": []} for feat in FAMILY_B}

for fpath in daily_files:
    fname   = os.path.basename(fpath)
    is_fail = "_F_" in fname
    label   = "failed" if is_fail else "healthy"

    df_vin = pd.read_csv(fpath)

    # engine-on rows (n_eo > 0)
    eo_mask = df_vin["n_eo"] > 0
    df_eo = df_vin[eo_mask].copy()

    # late-life window = last LATE_LIFE_N engine-on rows
    df_late = df_eo.tail(LATE_LIFE_N) if len(df_eo) >= LATE_LIFE_N else df_eo

    for feat in FAMILY_B:
        if feat in df_late.columns:
            vals = df_late[feat].dropna()
            scalar = float(vals.mean()) if len(vals) > 0 else float("nan")
        else:
            scalar = float("nan")
            warnings.warn(f"{feat} not found in {fname}; recording NaN")
        family_b_data[feat][label].append(scalar)

for feat in FAMILY_B:
    d = family_b_data[feat]
    print(f"  {feat}: F_n={len(d['failed'])} H_n={len(d['healthy'])} "
          f"| F_nans={sum(np.isnan(d['failed']))} H_nans={sum(np.isnan(d['healthy']))}")


# ── Compute stats for all 11 heuristics ───────────────────────────────────────
print("\nComputing separation metrics …")

rows = []

all_heuristics = (
    [(feat, "A") for feat in FAMILY_A] +
    [(feat, "B") for feat in FAMILY_B]
)

for feat, family in all_heuristics:
    if family == "A":
        f_vals = np.asarray(family_a_data[feat]["failed"],  float)
        h_vals = np.asarray(family_a_data[feat]["healthy"], float)
        importance = PERM_IMPORTANCE.get(feat, float("nan"))
    else:
        f_vals = np.asarray(family_b_data[feat]["failed"],  float)
        h_vals = np.asarray(family_b_data[feat]["healthy"], float)
        importance = FAMILY_B_IMPORTANCE.get(feat, float("nan"))

    # drop NaNs before metrics
    f_clean = f_vals[~np.isnan(f_vals)]
    h_clean = h_vals[~np.isnan(h_vals)]

    auroc_val  = auroc_from_scores(f_clean, h_clean) if (len(f_clean) and len(h_clean)) else float("nan")
    cd_val     = cliffs_delta(f_clean, h_clean)      if (len(f_clean) and len(h_clean)) else float("nan")
    p_val      = mwu_p(f_vals, h_vals)
    sep        = separation_label(abs(cd_val) if not np.isnan(cd_val) else 0.0)

    f_stats = dist_stats(f_vals)
    h_stats = dist_stats(h_vals)

    row = dict(
        heuristic          = feat,
        family             = family,
        engineering_meaning= ENG_MEANING[feat],
        failed_mean        = f_stats["mean"],
        healthy_mean       = h_stats["mean"],
        failed_median      = f_stats["median"],
        healthy_median     = h_stats["median"],
        failed_std         = f_stats["std"],
        healthy_std        = h_stats["std"],
        failed_p10         = f_stats["p10"],
        failed_p25         = f_stats["p25"],
        failed_p50         = f_stats["p50"],
        failed_p75         = f_stats["p75"],
        failed_p90         = f_stats["p90"],
        healthy_p10        = h_stats["p10"],
        healthy_p25        = h_stats["p25"],
        healthy_p50        = h_stats["p50"],
        healthy_p75        = h_stats["p75"],
        healthy_p90        = h_stats["p90"],
        auroc              = auroc_val,
        cliffs_delta       = cd_val,
        mwu_p              = p_val,
        separation         = sep,
        importance         = importance,
    )
    rows.append(row)
    print(f"  {feat:35s} | family={family} | AUROC={auroc_val:.3f} | CD={cd_val:.3f} | sep={sep}")

df_stats = pd.DataFrame(rows)

# ── Guard: vsi_std_ratio_30d AUROC >= 0.80 ────────────────────────────────────
vsi_auroc = float(df_stats.loc[df_stats["heuristic"] == "vsi_std_ratio_30d", "auroc"].iloc[0])
if vsi_auroc < 0.80:
    print(f"\nWARNING: vsi_std_ratio_30d AUROC={vsi_auroc:.3f} < 0.80 (expected >=0.80)")
else:
    print(f"\nOK: vsi_std_ratio_30d AUROC={vsi_auroc:.3f} >= 0.80")

# ── Guard: CSV must have exactly 11 rows ──────────────────────────────────────
assert len(df_stats) == 11, f"Expected 11 rows, got {len(df_stats)}"

# ── Save CSV ──────────────────────────────────────────────────────────────────
df_stats.to_csv(OUT_CSV, index=False, float_format="%.6f")
print(f"\nCSV written: {OUT_CSV}  ({len(df_stats)} rows)")


# ── Per-heuristic 2×2 plots ───────────────────────────────────────────────────
print("\nGenerating heuristic distribution plots …")

def plot_heuristic(feat, f_vals, h_vals, title_suffix=""):
    f_clean = f_vals[~np.isnan(f_vals)]
    h_clean = h_vals[~np.isnan(h_vals)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{feat}{title_suffix}", fontsize=14, weight="bold", y=1.01)
    fig.patch.set_facecolor("white")

    fc = PALETTE["fail"]
    hc = PALETTE["healthy"]

    # 1. Box plot
    ax = axes[0, 0]
    data_box = [f_clean, h_clean]
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, color in zip(bp["boxes"], [fc, hc]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Failed", "Healthy"])
    style_ax(ax, title="Box Plot", ylabel="Value")

    # 2. Violin plot
    ax = axes[0, 1]
    for i, (arr, color, label) in enumerate([(f_clean, fc, "Failed"), (h_clean, hc, "Healthy")]):
        if len(arr) >= 2:
            vp = ax.violinplot(arr, positions=[i + 1], showmedians=True, showextrema=True)
            for part in vp["bodies"]:
                part.set_facecolor(color); part.set_alpha(0.6)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                if key in vp:
                    vp[key].set_edgecolor(color)
        else:
            ax.scatter([i + 1], arr, color=color, s=60, zorder=5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Failed", "Healthy"])
    style_ax(ax, title="Violin Plot", ylabel="Value")

    # 3. KDE / distribution
    ax = axes[1, 0]
    for arr, color, label in [(f_clean, fc, "Failed"), (h_clean, hc, "Healthy")]:
        if len(arr) >= 3:
            try:
                kde = gaussian_kde(arr)
                xs  = np.linspace(arr.min() - arr.std(), arr.max() + arr.std(), 200)
                ax.fill_between(xs, kde(xs), alpha=0.35, color=color, label=label)
                ax.plot(xs, kde(xs), color=color, lw=1.8)
            except Exception:
                ax.hist(arr, bins=5, density=True, alpha=0.5, color=color, label=label)
        elif len(arr) > 0:
            ax.hist(arr, bins=max(2, len(arr)), density=True, alpha=0.5, color=color, label=label)
    ax.legend(fontsize=9)
    style_ax(ax, title="KDE Distribution", xlabel="Value", ylabel="Density")

    # 4. Overlaid histogram
    ax = axes[1, 1]
    all_vals = np.concatenate([f_clean, h_clean]) if (len(f_clean) and len(h_clean)) else (f_clean if len(f_clean) else h_clean)
    n_bins = min(15, max(5, int(len(all_vals) ** 0.5 * 2)))
    if len(f_clean):
        ax.hist(f_clean, bins=n_bins, alpha=0.55, color=fc, label="Failed", density=True)
    if len(h_clean):
        ax.hist(h_clean, bins=n_bins, alpha=0.55, color=hc, label="Healthy", density=True)
    ax.legend(fontsize=9)
    style_ax(ax, title="Overlaid Histogram", xlabel="Value", ylabel="Density")

    plt.tight_layout()
    out_path = os.path.join(OUT_VIZ, f"{feat}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


png_paths = []
for feat, family in all_heuristics:
    if family == "A":
        f_vals = np.asarray(family_a_data[feat]["failed"],  float)
        h_vals = np.asarray(family_a_data[feat]["healthy"], float)
    else:
        f_vals = np.asarray(family_b_data[feat]["failed"],  float)
        h_vals = np.asarray(family_b_data[feat]["healthy"], float)

    out_path = plot_heuristic(feat, f_vals, h_vals)
    png_paths.append(out_path)
    print(f"  Saved: {os.path.basename(out_path)}")

assert len(png_paths) == 11, f"Expected 11 PNGs, produced {len(png_paths)}"
print(f"\nAll {len(png_paths)} PNGs produced.")


# ── Markdown table ─────────────────────────────────────────────────────────────
print("\n")
header = ("| Heuristic | Engineering Meaning | Failed Mean | Healthy Mean "
          "| Separation Strength | Importance |")
sep    = ("|-----------|---------------------|-------------|--------------|"
          "---------------------|------------|")
print(header)
print(sep)
for _, r in df_stats.iterrows():
    fm   = f"{r['failed_mean']:.4f}"  if not np.isnan(r["failed_mean"])  else "NaN"
    hm   = f"{r['healthy_mean']:.4f}" if not np.isnan(r["healthy_mean"]) else "NaN"
    imp  = f"{r['importance']:.3f}"   if not np.isnan(r["importance"])   else "NaN"
    line = (f"| {r['heuristic']} | {r['engineering_meaning']} "
            f"| {fm} | {hm} | {r['separation']} | {imp} |")
    print(line)

print("\nDONE: V11.2_ALT Task 1 complete.")
print(f"  CSV:  {OUT_CSV}")
print(f"  PNGs: {OUT_VIZ}/")
