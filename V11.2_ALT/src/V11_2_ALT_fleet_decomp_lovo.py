"""
V11.2_ALT — Fleet contribution decomposition, sorted by LOVO out-of-fold score.

A COPY of fleet_decomposition.png re-ordered by the honest Leave-One-VIN-Out
(LOVO) ranking instead of the in-sample full-fit risk_linear. The per-feature
contributions are unchanged (coef x z from the frozen-spec inspection fit);
only the x-axis ordering changes to match the deployable AUROC-0.927 ranking.
Method name 'LOVO' appears in both the chart title and the output filename.

Input : V11.2_ALT/results/V11.2_ALT_contribution.csv  (contributions, full-fit)
        V5.2_ALT/results/V10.5.3_20_5_ALT_ridge_predictions.csv  (LOVO ridge_prob)
Output: V11.2_ALT/visualizations/contribution/fleet_decomposition_LOVO.png
"""
from __future__ import annotations
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:/Daimler-starter_motor_alternator_battery/V11.2_ALT/src")
from V11_2_ALT_common import FAMILY_A, VIZ, RESULTS, ROOT, PALETTE, style_ax

CONTRIB_DIR = os.path.join(VIZ, "contribution")

# identical feature colour map to fleet_decomposition.png
FEAT_COLORS = {
    "vsi_std_ratio_30d":            "#0B5394",
    "vsi_dominant_freq":            "#e69800",
    "vsi_spectral_entropy":         "#6aa84f",
    "bat_charge_delta_trend_right": "#cc0000",
    "vsi_range_trend_last30d":      "#674ea7",
    "progressive_drift":            "#999999",
}


def _key(label, failed):
    """Stable join key: failedflag_number (e.g. VIN8_F_ALT,1 -> '1_8')."""
    n = re.search(r"(\d+)", str(label)).group(1)
    return f"{int(failed)}_{n}"


# 1. per-VIN per-feature contributions (full-fit decomposition, already computed)
contrib = pd.read_csv(os.path.join(RESULTS, "V11.2_ALT_contribution.csv"))
contrib["key"] = [_key(v, f) for v, f in zip(contrib["vin"], contrib["failed"])]

# 2. LOVO out-of-fold scores (honest deployment ranking behind AUROC 0.927)
pred = pd.read_csv(os.path.join(ROOT, "V5.2_ALT/results/V10.5.3_20_5_ALT_ridge_predictions.csv"))
pred["key"] = [_key(l, f) for l, f in zip(pred["VIN_LABEL"], pred["failed"])]
lovo = dict(zip(pred["key"], pred["ridge_prob"]))

# 3. pivot contributions, attach LOVO score, sort ascending by LOVO (lowest risk left)
pivot = contrib.pivot_table(index=["vin", "failed", "key"], columns="feature",
                            values="contribution", aggfunc="first").reset_index()
pivot["lovo_prob"] = pivot["key"].map(lovo)
assert pivot["lovo_prob"].notna().all(), "missing LOVO score for some VIN"
pivot = pivot.sort_values("lovo_prob", ascending=True).reset_index(drop=True)
failed_flags = pivot["failed"].tolist()

# 4. plot (identical style to fleet_decomposition.png, only ordering + title differ)
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(pivot))
bottom_pos = np.zeros(len(pivot))
bottom_neg = np.zeros(len(pivot))
for feat in FAMILY_A:
    vals = pivot[feat].values.astype(float)
    pos = np.where(vals > 0, vals, 0)
    neg = np.where(vals < 0, vals, 0)
    col = FEAT_COLORS.get(feat, "#aaaaaa")
    ax.bar(x, pos, bottom=bottom_pos, color=col, width=0.7,
           label=feat.replace("_", " "), edgecolor="white", linewidth=0.3)
    ax.bar(x, neg, bottom=bottom_neg, color=col, width=0.7,
           edgecolor="white", linewidth=0.3)
    bottom_pos += pos
    bottom_neg += neg

short = [v.replace("_ALT", "") for v in pivot["vin"]]
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
for tick, flag in zip(ax.get_xticklabels(), failed_flags):
    tick.set_color(PALETTE["fail"] if flag else PALETTE["healthy"])

ax.axhline(0, color="#333333", lw=0.8)
ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.8)
style_ax(ax,
         title="Fleet Feature Contribution Decomposition (sorted by LOVO out-of-fold score)",
         xlabel="VIN (red=Failed, green=Non-Failed)  —  ordered by honest LOVO ranking",
         ylabel="Contribution to Composite Risk Score")
plt.tight_layout()
out = os.path.join(CONTRIB_DIR, "fleet_decomposition_LOVO.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved:", out)
print("\nLeft-to-right order (lowest -> highest LOVO risk):")
for _, r in pivot.iterrows():
    print(f"  {r['vin']:14} lovo_prob={r['lovo_prob']:.4f}  {'FAILED' if r['failed'] else 'healthy'}")
