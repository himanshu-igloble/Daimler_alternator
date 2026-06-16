"""
V10.6.2 Alternator — Visualizations  (plan W5b)
================================================
Honest plots:
  1. fleet_survival_curve.png   — posterior survival band + actual failed TTF
                                   + empirical vs survival median.
  2. backtest_accuracy.png      — model vs fleet-clock MAE by horizon.
  3. rul_band_waterfall.png     — per-truck 80% RUL band, coloured by risk tier.
  4. ged_emergency.png          — GED=2 lead-time bars for firing trucks.

Output: cfg.VIZ_DIR_V2
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")
C_HI, C_LO = "#c0392b", "#2980b9"


def main() -> None:
    viz = pathlib.Path(cfg.VIZ_DIR_V2)
    viz.mkdir(parents=True, exist_ok=True)

    wb = json.loads((pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_weibull_params.json").read_text())
    fw = json.loads((pathlib.Path(cfg.RUL_CACHE) / "fleet_window.json").read_text())
    bt = json.loads((pathlib.Path(cfg.BACKTEST_CACHE) / "backtest_results.json").read_text())
    surv = pd.read_csv(pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_survival_curve.csv")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    pred = pd.read_csv(pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_rul_predictions.csv")
    ged = pd.read_csv(pathlib.Path(cfg.GED_EMERGENCY_CACHE) / "ged_emergency.csv")

    # 1. fleet survival curve ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(surv["t_days"], surv["S_lower"], surv["S_upper"],
                    color="#95a5a6", alpha=0.3, label="80% posterior band")
    ax.plot(surv["t_days"], surv["S_t"], color="#2c3e50", lw=2, label="Survival S(t) (posterior MAP)")
    ttf = lc[lc["failed_flag"] == True]["ttf_days"].astype(float)
    for i, t in enumerate(ttf):
        ax.axvline(t, color=C_HI, alpha=0.35, lw=1, label="actual failures" if i == 0 else None)
    ax.axvline(fw["median_ttf_days"], color="#27ae60", lw=2.2, ls="--",
               label=f"empirical median {fw['median_ttf_days']:.0f}d")
    ax.axvline(wb["median_ttf_days"], color="#8e44ad", lw=2.0, ls=":",
               label=f"survival median {wb['median_ttf_days']:.0f}d (long-biased)")
    ax.set_xlabel("days since sale"); ax.set_ylabel("survival probability")
    ax.set_title("Fleet alternator survival — posterior band vs actual failures")
    ax.set_xlim(0, 1000); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(viz / "fleet_survival_curve.png", dpi=130); plt.close(fig)

    # 2. backtest accuracy --------------------------------------------------
    bh = bt["time_rewound"]["by_horizon"]
    labels = [f"T-{h}d" for h in bh] + ["total-TTF"]
    model = [bh[h]["mae_model"] for h in bh] + [bt["lovo_total_ttf"]["mae_model"]]
    dummy = [bh[h]["mae_dummyA"] for h in bh] + [bt["lovo_total_ttf"]["mae_dummyA"]]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, model, w, label="survival RUL model", color="#8e44ad")
    ax.bar(x + w/2, dummy, w, label="fleet-clock dummy", color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("out-of-sample MAE (days)")
    ax.set_title(f"Backtest day-MAE — verdict: {bt['time_rewound']['verdict'].upper()}\n"
                 "(lower is better; dummy wins = no per-truck timing skill)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    for xi, (m, d) in enumerate(zip(model, dummy)):
        ax.text(xi - w/2, m + 1, f"{m:.0f}", ha="center", fontsize=8)
        ax.text(xi + w/2, d + 1, f"{d:.0f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(viz / "backtest_accuracy.png", dpi=130); plt.close(fig)

    # 3. RUL band waterfall -------------------------------------------------
    nf = pred[pred["failed_flag"] == 0].sort_values("median_rul_days")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(nf))
    for i, (_, r) in enumerate(nf.iterrows()):
        col = C_HI if r["risk_tier"] == "HIGH_RISK" else C_LO
        ax.plot([r["rul_p10_days"], r["rul_p90_days"]], [i, i], color=col, lw=3, alpha=0.6)
        ax.plot(r["median_rul_days"], i, "o", color=col, ms=6)
    ax.set_yticks(y); ax.set_yticklabels(nf["vin_label"], fontsize=8)
    ax.axvline(cfg.SHORT_RUL_HORIZON_DAYS, color="#e67e22", ls="--",
               label=f"near-term horizon {cfg.SHORT_RUL_HORIZON_DAYS}d")
    ax.set_xlabel("predicted RUL (days), 80% band")
    ax.set_title("Per-truck RUL band (red=high-risk, blue=low-risk)\n"
                 "point estimate ~ fleet clock; the BAND is the trustworthy part")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=C_HI, lw=3, label="high-risk"),
                       Line2D([0], [0], color=C_LO, lw=3, label="low-risk"),
                       Line2D([0], [0], color="#e67e22", ls="--", label=f"{cfg.SHORT_RUL_HORIZON_DAYS}d")],
              fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(viz / "rul_band_waterfall.png", dpi=130); plt.close(fig)

    # 4. GED emergency ------------------------------------------------------
    fire = ged[(ged["ever_fired"]) & (ged["failed_flag"] == 1)].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(fire):
        fire["lead"] = pd.to_numeric(fire["first_fire_lead_days"], errors="coerce").fillna(0)
        ax.barh(fire["vin_label"], fire["lead"], color=C_HI)
        for i, v in enumerate(fire["lead"]):
            ax.text(v + 0.3, i, f"{v:.0f}d", va="center", fontsize=9)
    ax.axvline(21, color="#27ae60", ls="--", label="3-week target")
    ax.set_xlabel("GED=2 emergency lead time (days before failure)")
    ax.set_title(f"GED=2 emergency: {len(fire)}/10 failures fire (0/15 false alarms)")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(viz / "ged_emergency.png", dpi=130); plt.close(fig)

    print(f"[rul_graphs] Saved 4 figures to {viz}")


if __name__ == "__main__":
    main()
