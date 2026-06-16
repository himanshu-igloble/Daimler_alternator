"""
V11.1_ALT — Visualizations (Task 9)
=====================================
Four honest core plots for the covariate-verdict run:
  1. fleet_survival_curve.png   — posterior survival band + actual failed TTF
                                   + empirical vs survival median.
  2. backtest_accuracy.png      — NEW grouped-bar layout: 4 horizons x 4 series
                                   (M0, M1, M2, fleet-clock dummy).
                                   Title: "Covariate verdict: NO_IMPROVEMENT — M0 ships"
  3. rul_band_waterfall.png     — per-truck 80% RUL band, coloured by risk_band;
                                   marker/annotation if early_watch_current==1.
  4. ged_emergency.png          — GED leads for 2 firing trucks + second panel
                                   showing current-state early-watch counts.

Output: V11.1_ALT/visualizations/rul_core/
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config", "V11_1_ALT_config.py")
C_HI, C_LO = "#c0392b", "#2980b9"

# Variant colours (M0 = purple, M1 = teal, M2 = olive)
_VAR_COLORS = {"M0": "#8e44ad", "M1": "#16a085", "M2": "#d4ac0d"}


def main() -> None:
    viz = pathlib.Path(cfg.V11_1_ROOT) / "visualizations" / "rul_core"
    viz.mkdir(parents=True, exist_ok=True)

    wb = json.loads((pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_weibull_params.json").read_text())
    fw = json.loads((pathlib.Path(cfg.V10_6_2_ROOT) / "cache" / "rul" / "fleet_window.json").read_text())
    bt = json.loads((pathlib.Path(cfg.BACKTEST_CACHE) / "backtest_results.json").read_text())
    surv = pd.read_csv(pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_survival_curve.csv")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    final_rul = pd.read_csv(pathlib.Path(cfg.RUL_CACHE) / "final_rul_per_vin.csv")
    emerg = pd.read_csv(pathlib.Path(cfg.EMERG_CACHE) / "emergency_per_vin.csv")

    # 1. fleet survival curve --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(surv["t_days"], surv["S_lower"], surv["S_upper"],
                    color="#95a5a6", alpha=0.3, label="80% posterior band")
    ax.plot(surv["t_days"], surv["S_t"], color="#2c3e50", lw=2,
            label="Survival S(t) (posterior MAP)")
    ttf = lc[lc["failed_flag"] == True]["ttf_days"].astype(float)
    for i, t in enumerate(ttf):
        ax.axvline(t, color=C_HI, alpha=0.35, lw=1,
                   label="actual failures" if i == 0 else None)
    ax.axvline(fw["median_ttf_days"], color="#27ae60", lw=2.2, ls="--",
               label=f"empirical median {fw['median_ttf_days']:.0f}d")
    ax.axvline(wb["median_ttf_days"], color="#8e44ad", lw=2.0, ls=":",
               label=f"survival median {wb['median_ttf_days']:.0f}d (long-biased)")
    ax.set_xlabel("days since sale")
    ax.set_ylabel("survival probability")
    ax.set_title("Fleet alternator survival — posterior band vs actual failures")
    ax.set_xlim(0, 1000)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(viz / "fleet_survival_curve.png", dpi=130)
    plt.close(fig)

    # 2. backtest accuracy — NEW grouped-bar layout ----------------------------
    #  Groups: T-270d | T-180d | T-90d | total-TTF  (4 groups)
    #  Series per group: M0, M1, M2, fleet-clock dummy  (4 bars)
    variants = ["M0", "M1", "M2"]
    horizons = ["270", "180", "90"]
    hor_labels = [f"T-{h}d" for h in horizons] + ["total-TTF"]
    n_groups = len(hor_labels)

    # Build bar arrays: rows = series, cols = groups
    var_data = {}
    for v in variants:
        vd = bt["variants"][v]
        row = [vd["per_horizon_mae"][h] for h in horizons] + [vd["mae_model"]]
        var_data[v] = row
    # dummy is the same for all horizons (fleet-clock, no per-horizon breakdown)
    dummy_mae = bt["variants"]["M0"]["mae_dummy"]  # same value across all variants
    dummy_row = [dummy_mae] * n_groups

    x = np.arange(n_groups)
    n_series = 4  # M0, M1, M2, dummy
    total_w = 0.72
    w = total_w / n_series
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_series)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for si, v in enumerate(variants):
        bars = ax.bar(x + offsets[si], var_data[v], w,
                      label=f"{v} (MAE {var_data[v][-1]:.1f}d)",
                      color=_VAR_COLORS[v], alpha=0.88)
        for bar_xi, val in zip(x + offsets[si], var_data[v]):
            ax.text(bar_xi, val + 1.2, f"{val:.0f}", ha="center", fontsize=7.5,
                    color=_VAR_COLORS[v])
    # fleet-clock dummy
    si_dum = 3
    bars_d = ax.bar(x + offsets[si_dum], dummy_row, w,
                    label=f"fleet-clock dummy ({dummy_mae:.1f}d)",
                    color="#27ae60", alpha=0.88)
    for bar_xi, val in zip(x + offsets[si_dum], dummy_row):
        ax.text(bar_xi, val + 1.2, f"{val:.0f}", ha="center", fontsize=7.5,
                color="#27ae60")

    ax.set_xticks(x)
    ax.set_xticklabels(hor_labels)
    ax.set_ylabel("out-of-sample MAE (days)")
    ax.set_title("Covariate verdict: NO_IMPROVEMENT — M0 ships\n"
                 "(lower is better; variant vs fleet-clock dummy)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # Footnote inside figure
    fig.text(0.01, 0.01,
             "no variant beats the fleet-clock dummy (49.7d); "
             "covariates do not individualize RUL",
             fontsize=8, color="#7f8c8d", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(viz / "backtest_accuracy.png", dpi=130)
    plt.close(fig)

    # 3. RUL band waterfall ---------------------------------------------------
    #  Source: final_rul_per_vin.csv — non-failed rows, colour by risk_band
    #  Marker/annotation if early_watch_current == 1
    nf = (final_rul[final_rul["failed_flag"] == 0]
          .copy()
          .sort_values("median_rul_days"))

    band_color_map = {"red": C_HI, "amber": "#e67e22", "green": "#27ae60"}

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(nf))
    for i, (_, r) in enumerate(nf.iterrows()):
        col = band_color_map.get(str(r["risk_band"]).lower(), C_LO)
        ax.plot([r["rul_p10_days"], r["rul_p90_days"]], [i, i],
                color=col, lw=3, alpha=0.6)
        ax.plot(r["median_rul_days"], i, "o", color=col, ms=6)
        # early-watch annotation
        if int(r.get("early_watch_current", 0)) == 1:
            ax.annotate("EW",
                        xy=(r["median_rul_days"], i),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color="#c0392b", fontweight="bold",
                        va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(nf["vin_label"], fontsize=8)
    ax.axvline(cfg.SHORT_RUL_HORIZON_DAYS, color="#e67e22", ls="--",
               label=f"near-term horizon {cfg.SHORT_RUL_HORIZON_DAYS}d")
    ax.set_xlabel("predicted RUL (days), 80% band")
    ax.set_title("Per-truck RUL band (non-failed) — colour by risk_band\n"
                 "point estimate ~ fleet clock; the BAND is the trustworthy part")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=C_HI, lw=3, label="red risk band"),
        Line2D([0], [0], color="#e67e22", lw=3, label="amber risk band"),
        Line2D([0], [0], color="#27ae60", lw=3, label="green risk band"),
        Line2D([0], [0], color="#e67e22", ls="--",
               label=f"{cfg.SHORT_RUL_HORIZON_DAYS}d horizon"),
    ], fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(viz / "rul_band_waterfall.png", dpi=130)
    plt.close(fig)

    # 4. GED emergency — two-panel figure --------------------------------------
    #  Panel A (top): GED=2 lead-time bars for the 2 firing trucks (ged_fired=True,
    #                 failed_flag=1)
    #  Panel B (bottom): current-state early-watch counts (3/10 failed,  0/15 NF)
    fire = emerg[(emerg["ged_fired"] == True) & (emerg["failed_flag"] == 1)].copy()
    fire["lead"] = pd.to_numeric(fire["ged_lead_days"], errors="coerce").fillna(0)

    ew_failed = int(emerg[emerg["failed_flag"] == 1]["early_watch_current"].sum())
    ew_nf = int(emerg[emerg["failed_flag"] == 0]["early_watch_current"].sum())
    n_failed_total = int((emerg["failed_flag"] == 1).sum())
    n_nf_total = int((emerg["failed_flag"] == 0).sum())

    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(2, 1, hspace=0.52, height_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # -- Panel A: GED lead bars
    if len(fire):
        ax1.barh(fire["vin_label"], fire["lead"], color=C_HI)
        for i, v in enumerate(fire["lead"]):
            ax1.text(v + 0.3, i, f"{v:.0f}d", va="center", fontsize=9)
    ax1.axvline(21, color="#27ae60", ls="--", label="3-week target")
    ax1.set_xlabel("GED=2 lead time (days before failure)")
    ax1.set_title(
        f"GED=2 emergency: {len(fire)}/10 failures fire (0/15 false alarms)",
        fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", alpha=0.25)

    # -- Panel B: current early-watch counts
    labels_b = ["Failed\n(n=10)", "Non-failed\n(n=15)"]
    counts_b = [ew_failed, ew_nf]
    colors_b = [C_HI, "#2980b9"]
    bars_b = ax2.bar(labels_b, counts_b, color=colors_b, width=0.35, alpha=0.85)
    for bar, val in zip(bars_b, counts_b):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("trucks in early-watch state")
    ax2.set_title(
        f"Current-state early-watch: {ew_failed}/{n_failed_total} failed, "
        f"{ew_nf}/{n_nf_total} NF",
        fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, max(counts_b) + 1.5)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("GED=2 Emergency Channel — V11.1_ALT", fontsize=12,
                 fontweight="bold", y=1.01)
    fig.savefig(viz / "ged_emergency.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[V11_1_rul_graphs] Saved 4 figures to {viz}")


if __name__ == "__main__":
    main()
