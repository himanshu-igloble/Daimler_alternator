"""
V10.6.2 Alternator — Reliability/PHM Charts (n=25)  [STYLE v3]
=============================================================
Implements the v3 redesign (docs/2026-06-05-12-58-30-v10.6.2-chart-redesign-
technical-plan.md). NO dual axis, NO fleet-clock diagonal, NO fabricated
per-truck degradation trend.

Figure 1 (per-VIN, engineer view) — 3 stacked panels + a gated precursor inset:
  A  Fleet reliability S(t) (Weibull) + posterior band + "you are here" + n=10
     failure-age rug + empirical 601d median.
  B  Conditional RUL DISTRIBUTION at the truck's current age (right-skewed
     violin from posterior samples) + p10/median/p90 + fleet-clock point.
     Failed trucks: "EVENT OBSERVED (RUL=0)" against the fleet conditional density.
  C  alpha-lambda validation: predicted-vs-actual remaining life (rewound LOVO),
     fleet-clock dummy vs survival model, +/-20% cone, MAE textbox.
  Inset (gated): GED2 precursor over the last 30d, ONLY for adversarially-verified
     precursor trucks (VIN1_F ~21d; VIN10_F single-day/no-lead). Omitted otherwise.

Figure 2 (fleet, ops view) — ranked maintenance "runway": fleet_maintenance_board.png

Honesty: RUL is fleet-posterior-derived; only per-truck marks are age (fact) and
risk rank (frozen classifier, STATIC). km/hrs are estimated.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10,
                     "axes.labelsize": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

_src = pathlib.Path(__file__).resolve().parent


def _load(mod, fn):
    spec = importlib.util.spec_from_file_location(mod, str(_src / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")
surv = _load("V10_6_2_ALT_survival", "V10.6.2_ALT_survival.py")

OUT = pathlib.Path(cfg.V10_6_2_ROOT) / "visualizations" / "rul_curves"
VIZ = pathlib.Path(cfg.V10_6_2_ROOT) / "visualizations"

# v3 palette (muted, one accent per panel)
C_SURV, C_SURV_BAND = "#1F6F78", "#1F6F78"
C_RULDIST = "#117A65"
C_MODEL, C_DUMMY = "#C0392B", "#2E7D32"
C_FLEETCLOCK = "#7f5a00"
C_GED = "#B5532A"
C_GRID = "#E6E6E6"
C_HI, C_LO = "#C0392B", "#2980B9"
FS_ANN = 8.5

# Adversarially-VERIFIED precursors only (NOT the raw deterministic flags).
# VIN6/VIN2/VIN8 were REFUTED (fire on healthy trucks / window artifacts).
VERIFIED_PRECURSOR = {
    "VIN1_F_ALT": ("ged2_frac", 21, "GED2 excitation storm — verified ~21d lead"),
    "VIN10_F_ALT": ("ged2_frac", 1, "GED2 single-day storm — no usable lead"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wb = json.loads((pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_weibull_params.json").read_text())
    fw = json.loads((pathlib.Path(cfg.RUL_CACHE) / "fleet_window.json").read_text())
    fleet_med = float(fw["median_ttf_days"])
    surv_curve = pd.read_csv(pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_survival_curve.csv")
    ps = pd.read_csv(pathlib.Path(cfg.WEIBULL_CACHE) / "posterior_samples.csv")
    shape_s, scale_s = ps["shape"].values, ps["scale"].values
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    fail_ttf = lc.loc[lc.failed_flag == True, "ttf_days"].astype(float).values
    ridge = pd.read_csv(cfg.RIDGE_PROB_CSV).set_index("vin_label")
    svc = pd.read_csv(VIZ.parent / "service" / "V10.6.2_ALT_service_per_vin.csv")
    svc_i = svc.set_index("vin_label")
    pr = pd.read_csv(pathlib.Path(cfg.RUL_CACHE) / "predictive_rul_per_vin.csv").set_index("vin_label")
    rew = pd.read_csv(pathlib.Path(cfg.BACKTEST_CACHE) / "per_fold_residuals.csv")
    rew = rew[rew.test == "rewound"].copy()
    rng = np.random.default_rng(cfg.RNG_SEED)

    for vin in cfg.ALL_VINS:
        _plot_vin(vin, lc=lc, ridge=ridge, svc=svc_i, pr=pr, surv_curve=surv_curve,
                  shape_s=shape_s, scale_s=scale_s, fail_ttf=fail_ttf, rew=rew,
                  fleet_med=fleet_med, rng=rng)
    _fleet_board(svc, fw)
    print(f"[rul_curves] Saved 25 engineer charts + fleet board to {VIZ}")


def _plot_vin(vin, *, lc, ridge, svc, pr, surv_curve, shape_s, scale_s, fail_ttf, rew, fleet_med, rng):
    r = lc[lc.vin_label == vin].iloc[0]
    failed = bool(r["failed_flag"]); age = float(r["age_days_observed"])
    est_km = float(r["est_km"]) if pd.notna(r["est_km"]) else float("nan")
    prob = float(ridge.loc[vin, "ridge_prob"]); s = svc.loc[vin]
    tier = s["risk_tier"]
    S_age = float(np.interp(age, surv_curve["t_days"], surv_curve["S_t"]))

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 0.95], hspace=0.45,
                          left=0.075, right=0.965, top=0.90, bottom=0.07)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    for a in (axA, axB, axC):
        a.grid(True, axis="both", ls="--", color=C_GRID, alpha=0.8)

    # ---------- Panel A: fleet reliability S(t) ----------
    axA.plot(surv_curve["t_days"], surv_curve["S_t"], color=C_SURV, lw=2.2, label="Fleet survival S(t) (Weibull)")
    axA.fill_between(surv_curve["t_days"], surv_curve["S_lower"], surv_curve["S_upper"],
                     color=C_SURV_BAND, alpha=0.15, label="80% posterior band")
    axA.plot(fail_ttf, np.full(len(fail_ttf), 0.015), marker="|", ls="none", color=C_MODEL,
             ms=12, mew=1.4, alpha=0.8, label=f"observed failures (n={len(fail_ttf)})")
    axA.axvline(fleet_med, color="#666", ls=":", lw=1.1, alpha=0.7)
    axA.text(fleet_med, 1.02, f"fleet median {fleet_med:.0f}d", ha="center", va="bottom",
             fontsize=FS_ANN, color="#666", transform=axA.get_xaxis_transform())
    axA.axvline(age, color="#222", ls="--", lw=1.3, alpha=0.8)
    axA.scatter([age], [S_age], s=90, color="#222", zorder=6)
    axA.annotate(f"Age {age:.0f}d · S={S_age:.2f}", xy=(age, S_age), xytext=(age + 18, min(S_age + 0.18, 0.95)),
                 fontsize=FS_ANN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#222", lw=1))
    axA.set_xlim(0, max(age, fleet_med, fail_ttf.max()) + 120)
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("Survival probability S(t)")
    axA.set_xlabel("Age (days since first observation)")
    axA.set_title("A · Fleet reliability — this truck's position on the population survival curve", loc="left")
    axA.legend(loc="upper right", fontsize=FS_ANN, framealpha=0.9)

    # ---------- Panel B: conditional RUL distribution ----------
    draws = surv.conditional_predictive_rul(age, shape_s, scale_s, rng)
    fleet_clock = max(fleet_med - age, 0.0)
    if not failed:
        vp = axB.violinplot([draws], positions=[0], vert=False, widths=0.85, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(C_RULDIST); b.set_alpha(0.35); b.set_edgecolor(C_RULDIST)
        p10, med, p90 = float(pr.loc[vin, "rul_p10_days"]), float(pr.loc[vin, "median_rul_days"]), float(pr.loc[vin, "rul_p90_days"])
        for x, lab, c in [(p10, "p10", "#555"), (med, "median", C_RULDIST), (p90, "p90", "#555")]:
            axB.plot([x, x], [-0.42, 0.42], color=c, lw=1.6 if lab == "median" else 1.1)
            axB.text(x, 0.5, f"{lab} {x:.0f}d", ha="center", va="bottom", fontsize=FS_ANN, color=c)
        axB.axvline(fleet_clock, color=C_FLEETCLOCK, ls="-.", lw=1.6)
        axB.text(fleet_clock, -0.62, f"fleet-clock {fleet_clock:.0f}d\n(deployed; backtest 50d beats model 125d)",
                 ha="center", va="top", fontsize=FS_ANN, color=C_FLEETCLOCK)
        km = fleet_clock * float(pr.loc[vin, "km_per_day_est"])
        axB.set_title(f"B · Conditional RUL distribution at current age — 80% PI ≈ {p90 - p10:.0f}d "
                      f"(fleet-clock ≈ {fleet_clock:.0f}d / {km:,.0f} km)", loc="left")
        axB.set_xlim(0, max(p90 * 1.25, 350))
    else:
        vp = axB.violinplot([draws], positions=[0], vert=False, widths=0.85, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("#999"); b.set_alpha(0.3); b.set_edgecolor("#777")
        axB.axvline(0, color=C_MODEL, lw=2.2)
        axB.text(5, 0.45, "EVENT OBSERVED\nactual RUL = 0", color=C_MODEL, fontsize=FS_ANN, fontweight="bold", va="top")
        axB.text(np.median(draws), -0.62, "grey = fleet conditional density the truck was drawn from\n"
                 "(it validates the curve; we did not 'predict' this date)", ha="center", va="top",
                 fontsize=FS_ANN, color="#555")
        axB.set_title("B · Conditional RUL distribution at failure age (failed truck)", loc="left")
        axB.set_xlim(-10, max(np.percentile(draws, 95) * 1.2, 350))
    axB.set_yticks([]); axB.set_ylim(-1.0, 1.0)
    axB.set_xlabel("Remaining useful life (days)")

    # ---------- Panel C: alpha-lambda validation ----------
    xmax = max(rew["actual_days"].max(), rew["model_pred_days"].max()) * 1.1
    xs = np.linspace(0, xmax, 50)
    axC.fill_between(xs, 0.8 * xs, 1.2 * xs, color="#bbb", alpha=0.25, label="±20% accuracy cone")
    axC.plot(xs, xs, color="#333", lw=1.2, ls="--", label="ideal (pred = actual)")
    hz_color = {270: "#f0b27a", 180: "#e67e22", 90: "#922b21"}
    axC.scatter(rew["actual_days"], rew["dummyA_pred_days"], s=34, color=C_DUMMY, marker="s",
                alpha=0.8, label="fleet-clock dummy", zorder=4)
    for hz, g in rew.groupby("horizon_days"):
        axC.scatter(g["actual_days"], g["model_pred_days"], s=40, color=hz_color.get(int(hz), "#900"),
                    edgecolor="k", lw=0.4, alpha=0.9, label=f"survival model T-{int(hz)}d", zorder=5)
    if failed and (rew.vin_label == vin).any():
        cur = rew[rew.vin_label == vin]
        axC.scatter(cur["actual_days"], cur["model_pred_days"], s=150, facecolor="none",
                    edgecolor="#000", lw=1.6, zorder=6, label=f"{vin}")
    mae_m = rew["model_abs_err"].mean(); mae_d = rew["dummyA_abs_err"].mean()
    by_h = rew.groupby("horizon_days")["model_abs_err"].mean()
    axC.text(0.015, 0.97, f"LOVO rewound MAE: model {mae_m:.0f}d  vs  fleet-clock {mae_d:.0f}d\n"
             f"accuracy worsens nearer failure: " + " ".join(f"T-{int(h)}:{v:.0f}d" for h, v in by_h.items()) +
             "  →  fleet clock deployed", transform=axC.transAxes, va="top", fontsize=FS_ANN,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#999"))
    axC.set_xlim(0, xmax); axC.set_ylim(0, xmax)
    axC.set_xlabel("Actual remaining life (days)"); axC.set_ylabel("Predicted RUL (days)")
    axC.set_title("C · Validation (α–λ): per-truck model loses to the fleet-clock baseline", loc="left")
    axC.legend(loc="lower right", fontsize=FS_ANN - 0.5, ncol=2, framealpha=0.9)

    # ---------- gated precursor inset ----------
    if vin in VERIFIED_PRECURSOR:
        feat, lead, lab = VERIFIED_PRECURSOR[vin]
        d = pd.read_csv(pathlib.Path(cfg.V10_6_2_ROOT) / "cache" / "forensics" / f"{vin}_daily.csv")
        win = d[(d["dtf"] >= 0) & (d["dtf"] <= 40)].sort_values("dtf", ascending=False)
        ins = axA.inset_axes([0.06, 0.12, 0.32, 0.5])
        ins.plot(win["dtf"], win[feat], color=C_GED, lw=1.3)
        ins.axvline(lead, color="#222", ls=":", lw=1)
        ins.invert_xaxis()
        ins.set_title(lab, fontsize=7.2)
        ins.set_xlabel("days to failure", fontsize=7); ins.set_ylabel(feat, fontsize=7)
        ins.tick_params(labelsize=6.5)

    # header + footer
    status = "FAILED" if failed else "in-service"
    fig.suptitle(f"V10.6.2 Alternator — Reliability & RUL  ·  {vin}", fontsize=14, fontweight="bold", y=0.965)
    fig.text(0.075, 0.925, f"Risk: ridge_prob={prob:.2f} ({tier}, STATIC whole-life rank)  ·  Age {age:.0f}d  ·  "
             f"State: {s['state']}  ·  S(age)={S_age:.2f}  ·  est. {est_km:,.0f} km  ·  {status}",
             fontsize=9.5, color="#444")
    fig.text(0.008, 0.012, "Fleet-clock RUL (Weibull n=10, shape 5.17/scale 771) · per-truck timing not predictable "
             "(backtest 142d > 50d floor) · classifier ranks which, not when · km/hrs est. | Confidential",
             fontsize=7.3, color="#888")
    fig.savefig(OUT / f"V10.6.2_ALT_{vin}_rul.png", dpi=130)
    plt.close(fig)


def _fleet_board(svc, fw):
    med = float(fw["median_ttf_days"]); p25 = float(fw["p25_ttf_days"]); p75 = float(fw["p75_ttf_days"])
    d = svc[svc.in_service == True].copy().sort_values("risk_rank_in_service")
    n = len(d); y = np.arange(n)[::-1]   # rank 1 at top

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axvspan(p25, p75, color="#d9c08a", alpha=0.35, zorder=0)
    ax.axvline(med, color="#7a5a00", lw=1.6, zorder=1)
    ax.text(med, n - 0.3, f"fleet wear-out median {med:.0f}d", ha="center", va="bottom", fontsize=FS_ANN, color="#7a5a00")

    for yi, (_, r) in zip(y, d.iterrows()):
        stale = bool(r["data_gap_flag"]); hi = (r["risk_tier"] == "HIGH_RISK")
        col = C_HI if hi else C_LO
        ax.barh(yi, r["current_age_days"], height=0.6, color=("#cfcfcf" if stale else col),
                alpha=(0.45 if stale else 0.55), hatch=("//" if stale else None), edgecolor="white", zorder=2)
        ax.scatter([r["current_age_days"]], [yi], s=70, color=("#888" if stale else col),
                   marker=("o" if hi else "o"), edgecolor="k", lw=0.6, zorder=4)
        tag = (f"STALE {r['staleness_days']:.0f}d / BLIND" if stale else r["replacement_status"])
        ax.text(max(d["current_age_days"].max(), med) * 1.02, yi, tag, va="center", fontsize=FS_ANN,
                color=("#888" if stale else "#333"))
        ax.text(-30, yi, ("●" if hi else "○"), va="center", ha="right", fontsize=12, color=col)

    ax.set_yticks(y); ax.set_yticklabels(d["vin_label"], fontsize=9)
    ax.set_xlabel("Age (days since sale)  →  RUL = distance from dot to the fleet wear-out band", fontweight="bold")
    ax.set_xlim(0, max(d["current_age_days"].max(), med) * 1.28)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title("V10.6.2 Alternator — Fleet Maintenance Runway (in-service, ranked by failure risk)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", ls="--", color=C_GRID, alpha=0.8)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HI, markeredgecolor="k", label="HIGH risk (classifier)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=C_LO, markeredgecolor="k", label="LOW risk"),
           Patch(fc="#cfcfcf", hatch="//", label="STALE / BLIND (no telemetry)"),
           Patch(fc="#d9c08a", alpha=0.5, label=f"fleet wear-out window {p25:.0f}-{p75:.0f}d")]
    ax.legend(handles=leg, loc="lower right", fontsize=FS_ANN, framealpha=0.95)
    ax.text(0.5, -0.12, "Risk = frozen classifier (which trucks fail, AUROC 0.927).  Timing = fleet wear-out window "
            f"({med:.0f}d); per-truck failure dates are not predictable.  Most survivors are already past the median — "
            "the band is population wear-out, NOT an expiry date.", transform=ax.transAxes, ha="center",
            fontsize=FS_ANN, color="#555")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.16)
    fig.savefig(VIZ / "fleet_maintenance_board.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
