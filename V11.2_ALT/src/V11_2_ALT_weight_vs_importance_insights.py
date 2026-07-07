"""
V11.2_ALT — Feature importance, three views (insight-rich redo of weight_vs_importance.png).

Beyond the original 2-bar chart this adds:
  • a 3rd metric — UNIVARIATE AUROC (standalone failed-vs-healthy separation) on a 2nd axis,
    with a 0.5 "no separation" reference (progressive_drift falls BELOW it -> inverted, flagged red).
  • signed Ridge coefficients in the x labels (+/-, flags the one negative feature).
  • value labels on every bar and marker.
  • a tidy two-line caption carrying the metric definitions + the key divergence insights
    (kept out of the plot area to avoid clutter).

Inputs : results/V11.2_ALT_weightage_summary.json  (coef, |mean contribution|, perm importance)
         results/V11.2_ALT_heuristic_stats.csv      (univariate AUROC per feature)
Output : visualizations/contribution/weight_vs_importance_insights.png
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:/Daimler-starter_motor_alternator_battery/V11.2_ALT/src")
from V11_2_ALT_common import FAMILY_A, RESULTS, VIZ, PALETTE, style_ax

CONTRIB_DIR = os.path.join(VIZ, "contribution")
TEAL = "#1b6b50"
REDF = "#9a3b3b"

# ── load the three metrics ────────────────────────────────────────────────────
S = json.load(open(os.path.join(RESULTS, "V11.2_ALT_weightage_summary.json")))
coef         = S["ridge_coefficients"]
mean_contrib = S["per_feature_mean_abs_contribution"]
perm         = S["perm_importance"]

hs = pd.read_csv(os.path.join(RESULTS, "V11.2_ALT_heuristic_stats.csv")).set_index("heuristic")
uni_auroc = {f: float(hs.loc[f, "auroc"]) for f in FAMILY_A}

feats = sorted(FAMILY_A, key=lambda f: -mean_contrib[f])   # order by |mean contribution| desc
mc = [mean_contrib[f] for f in feats]
pi = [perm[f] for f in feats]
au = [uni_auroc[f] for f in feats]

def pretty(f):
    return f.replace("vsi_", "vsi ").replace("_", " ")
xlabels = [f"{pretty(f)}\n(coef {coef[f]:+.2f})" for f in feats]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13.5, 7.4))
x = np.arange(len(feats)); w = 0.38

b1 = ax.bar(x - w/2, mc, w, label="|Mean contribution|  (in-sample swing)",
            color=PALETTE["accent"], edgecolor="white", zorder=3)
b2 = ax.bar(x + w/2, pi, w, label="Permutation importance  (generalisation)",
            color=PALETTE["fail"], alpha=0.85, edgecolor="white", zorder=3)
for xi, v in zip(x - w/2, mc):
    ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=PALETTE["accent"])
for xi, v in zip(x + w/2, pi):
    ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=PALETTE["fail"])

ax.set_ylim(0, max(mc) * 1.35)
style_ax(ax,
         title="Feature Importance — Three Views: in-sample weight, generalisation, and standalone separation",
         xlabel="Feature  (coefficient shown; only progressive drift is negative)",
         ylabel="Importance score  (bars)")
ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8.5)

# secondary axis: univariate AUROC — scaled so markers sit clear of the legend
ax2 = ax.twinx()
ax2.set_ylim(0.20, 1.15)
ax2.set_ylabel("Univariate AUROC  (standalone separation)", color=TEAL)
ax2.tick_params(axis="y", labelcolor=TEAL)
ax2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax2.axhline(0.5, ls="--", lw=1.1, color="#888888", zorder=1)
ax2.text(len(feats) - 0.5, 0.51, "0.5 = no standalone separation", ha="right", va="bottom",
         fontsize=8, color="#888888", style="italic")
ax2.plot(x, au, "-", color=TEAL, lw=1.6, zorder=4)
# markers: red for the inverted feature (AUROC < 0.5), teal otherwise
mcolors = [REDF if v < 0.5 else TEAL for v in au]
ax2.scatter(x, au, s=70, marker="D", c=mcolors, edgecolors="white", linewidths=0.8, zorder=5)
for xi, v in zip(x, au):
    ax2.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=(REDF if v < 0.5 else TEAL))

# combined legend (top-right, above the short right-hand bars/markers)
from matplotlib.lines import Line2D
legend_handles = [b1, b2,
                  Line2D([0], [0], color=TEAL, marker="D", lw=1.6, mfc=TEAL, mec="white",
                         label="Univariate AUROC  (standalone)")]
ax.legend(legend_handles, [h.get_label() for h in legend_handles],
          loc="upper right", fontsize=8.5, framealpha=0.92)

# ── narrative caption (kept below the plot to keep the figure clean) ───────────
defn = ("Bars: |mean contribution| = how much the signal MOVES this fleet's scores (outlier-sensitive)   |   "
        "permutation importance = how much it IMPROVES held-out ranking (generalisation).   "
        "Line: univariate AUROC = standalone failed-vs-healthy separation (0.5 = none).")
ins = ("Insights:  vsi_std_ratio_30d is #1 by weight & importance yet only 0.69 alone -> powerful in combination.   "
       "vsi_spectral_entropy is the best standalone signal (0.81) but carries modest weight.   "
       "bat_charge_delta swings scores (#3) but has the LOWEST generalisation (0.03) and weakest standalone (0.56) "
       "-> driven by a few outlier trucks.   progressive_drift is INVERTED (AUROC 0.26 < 0.5) with a negative coef -> exposure artifact.")
fig.text(0.5, 0.075, defn, ha="center", va="bottom", fontsize=8, color="#444444", wrap=True)
fig.text(0.5, 0.015, ins, ha="center", va="bottom", fontsize=8, color="#222222", wrap=True)

plt.tight_layout(rect=(0, 0.135, 1, 1))
out = os.path.join(CONTRIB_DIR, "weight_vs_importance_insights.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)
print("\nfeature                       coef    |contrib|  perm    uni_AUROC  rank_contrib/perm/uni")
rc = {f: i+1 for i, f in enumerate(sorted(feats, key=lambda f: -mean_contrib[f]))}
rp = {f: i+1 for i, f in enumerate(sorted(feats, key=lambda f: -perm[f]))}
ru = {f: i+1 for i, f in enumerate(sorted(feats, key=lambda f: -uni_auroc[f]))}
for f in feats:
    print(f"  {f:<28} {coef[f]:+.3f}   {mean_contrib[f]:.3f}     {perm[f]:.3f}   {uni_auroc[f]:.3f}      {rc[f]}/{rp[f]}/{ru[f]}")
