"""V11_ALT_heuristics — graph suite (V11-native lead-time deliverables).

Reads committed V11 caches and writes professional PNG (+ HD) graphs mirroring
the V10.6.2 house style. No pipeline re-runs. matplotlib Agg backend.
"""
from __future__ import annotations

import importlib.util
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FOR = cfg.FORENSICS
OUT = cfg.V11_ROOT / "visualizations" / "V11_graphs"

NAVY, GOLD, GREEN, RED, BLUE, GREY = "#0D1B2A", "#C58B1F", "#27AE60", "#C0392B", "#2980B9", "#95A5A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": NAVY,
                     "axes.labelcolor": NAVY, "text.color": NAVY,
                     "xtick.color": NAVY, "ytick.color": NAVY, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 110})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}_hd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


def _hzn(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def g1_recall_comparison():
    comp = pd.read_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv")
    n11 = int((comp["v11_horizon"].astype(str) != "none").sum())
    n1062 = int((comp["v1062_horizon"].astype(str) != "none").sum())
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    comp_fired = int(cl[(cl.group == "FAILED") & (cl.fired == True)].shape[0])  # noqa: E712
    st = pd.read_csv(FOR / "nf_self_test.csv")
    nf_false = int((st["verdict"] == "FALSE_ALARM").sum())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ["V10.6.2\nstrict gate", "V11\nstrict gate", "V11\ncompound (#11)"]
    vals = [n1062, n11, comp_fired]
    colors = [GREY, GREEN, BLUE]
    b = ax.bar(bars, vals, color=colors, width=0.6)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, f"{v}/10",
                ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Failed trucks with a precursor (of 10)")
    ax.set_title("V11 lead-time recall vs V10.6.2  -  NF false alarms: "
                 f"{nf_false}/15", fontweight="bold", color=NAVY)
    ax.text(0.5, -0.18, "Honest gate: within-truck z>=2 AND outside healthy-fleet p05-p95",
            transform=ax.transAxes, ha="center", fontsize=9, color=GREY)
    return _save(fig, "G1_recall_comparison")


def g3_leadtime_dumbbell():
    comp = pd.read_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv").copy()
    comp["a"] = comp["v1062_horizon"].map(_hzn)
    comp["b"] = comp["v11_horizon"].map(_hzn)
    comp = comp.sort_values("b")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(comp))
    for i, (_, r) in enumerate(comp.iterrows()):
        ax.plot([r["a"], r["b"]], [i, i], color=GREY, zorder=1, lw=2)
        ax.scatter(r["a"], i, color=GREY, s=70, zorder=2, label="V10.6.2" if i == 0 else "")
        col = GOLD if r["new_in_v11"] else (GREEN if r["earlier"] else BLUE)
        ax.scatter(r["b"], i, color=col, s=90, zorder=3, label="V11" if i == 0 else "")
    ax.set_yticks(y)
    ax.set_yticklabels(comp["vin_label"])
    ax.set_xlabel("Earliest discriminative horizon (days before failure)")
    ax.set_title("Per-truck earliest precursor: V10.6.2 to V11\n"
                 "(gold = newly detected, green = earlier lead)", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, "G3_leadtime_dumbbell")


def _feat_class(failed_hits, nf_false):
    if nf_false >= 2:
        return "false_alarm_prone", RED
    if failed_hits >= 2:
        return "generalizes", GREEN
    if failed_hits == 1:
        return "anecdotal", GOLD
    return "no_signal", GREY


def g2_feature_generalization():
    dev = pd.read_csv(FOR / "failed_window_deviations.csv")
    st = pd.read_csv(FOR / "nf_self_test.csv")
    disc = dev[dev["discriminative"] == True]                      # noqa: E712
    nf_false_by = st[st["verdict"] == "FALSE_ALARM"]["feature"].value_counts()
    rows = []
    for f in cfg.NEW_FEATS:
        hits = disc[disc["feature"] == f]["vin_label"].nunique()
        nff = int(nf_false_by.get(f, 0))
        cls, col = _feat_class(hits, nff)
        rows.append((f, hits, nff, cls, col))
    d = pd.DataFrame(rows, columns=["feature", "hits", "nf_false", "cls", "col"]).sort_values("hits")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(d["feature"], d["hits"], color=d["col"])
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["hits"] + 0.05, i, str(int(r["hits"])), va="center", fontsize=9)
        if r["feature"] == "crank_recovery_t":
            ax.text(r["hits"] + 0.5, i, "  <- MVP (#3 post-crank recovery)",
                    va="center", color=RED, fontweight="bold", fontsize=10)
    ax.set_xlabel("Failed trucks discriminative (of 10)")
    ax.set_title("New-heuristic generalization (19 V11 features)\n"
                 "green=generalizes(>=2)  gold=anecdotal(1)  grey=no signal", fontweight="bold")
    ax.set_xlim(0, max(7, d["hits"].max() + 2))
    return _save(fig, "G2_feature_generalization")


def g4_compound_alarm_leads():
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    f = cl[(cl.group == "FAILED") & (cl.fired == True)].copy()      # noqa: E712
    f["h"] = f["early_watch_horizon_days"].map(_hzn)
    f = f.sort_values("h")
    nf_false = int(cl[(cl.group == "NF") & (cl.fired == True)].shape[0])  # noqa: E712
    fig, ax = plt.subplots(figsize=(9, 5))
    b = ax.barh(f["vin_label"], f["h"], color=BLUE)
    for rect, (_, r) in zip(b, f.iterrows()):
        ax.text(r["h"] + 0.7, rect.get_y() + rect.get_height() / 2,
                f"{int(r['h'])}d  ({int(r['n_votes'])} votes)", va="center", fontsize=10)
    ax.set_xlabel("First-trigger lead (days before failure)")
    ax.set_title(f"Compound 2-of-5 early-watch alarm (#11)  -  NF false alarms: {nf_false}/15",
                 fontweight="bold")
    ax.set_xlim(0, max(100, (f["h"].max() if len(f) else 0) + 25))
    return _save(fig, "G4_compound_alarm_leads")


def g5_crank_recovery_trajectories():
    nfb = pd.read_csv(FOR / "nf_baseline.csv").set_index("feature")
    p95 = float(nfb.loc["crank_recovery_t", "nf_p95"])
    vins = ["VIN1_F_ALT", "VIN8_F_ALT", "VIN9_F_ALT"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vin in zip(axes, vins):
        d = pd.read_csv(FOR / f"{vin}_daily.csv")
        d = d[(d["n_eo"] >= cfg.MIN_EO_SAMPLES) & d["crank_recovery_t"].notna()].sort_values("dtf", ascending=False)
        ax.plot(d["dtf"], d["crank_recovery_t"], color=BLUE, lw=1.4, marker="o", ms=3)
        ax.axhline(p95, color=RED, ls="--", lw=1.5, label=f"healthy-fleet p95 ({p95:.1f}s)")
        ax.invert_xaxis()
        ax.set_title(vin, fontweight="bold")
        ax.set_xlabel("days before failure")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_ylabel("post-crank voltage recovery time (s)")
    fig.suptitle("Post-crank recovery (#3): episodic slow-recovery events exceed the near-zero "
                 "healthy baseline\n(isolated spikes, not a smooth trend; VIN9's events span its "
                 "whole life — read with the z-magnitude caveat)",
                 fontweight="bold", fontsize=11, y=1.06)
    return _save(fig, "G5_crank_recovery_trajectories")


def g6_changepoint_resting():
    cp = pd.read_csv(FOR / "changepoint_per_vin.csv")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    kinds = ["cp_resid_lead_days", "cp_resting_lead_days", "dose_knee_lead_days"]
    x = np.arange(len(cp)); w = 0.25
    for i, k in enumerate(kinds):
        vals = pd.to_numeric(cp[k], errors="coerce").fillna(0)
        axL.bar(x + (i - 1) * w, vals, width=w, label=k.replace("_lead_days", ""))
    axL.axhline(90, color=RED, ls="--", lw=1.5, label="actionable < 90d")
    axL.set_xticks(x); axL.set_xticklabels(cp["vin_label"], rotation=60, ha="right", fontsize=8)
    axL.set_ylabel("change-point lead (days)")
    axL.set_title("Change-point leads (#12/#5) - mostly fire too early (exploratory)", fontweight="bold")
    axL.legend(frameon=False, fontsize=8)
    rs = pd.to_numeric(cp["resting_slope"], errors="coerce")
    p05 = pd.to_numeric(cp["resting_slope_nf_p05"], errors="coerce").iloc[0]
    cols = [RED if b else GREY for b in cp["resting_slope_disc"]]
    axR.barh(cp["vin_label"], rs, color=cols)
    axR.axvline(p05, color=NAVY, ls="--", lw=1.5, label=f"NF p05 ({p05:.4f})")
    axR.set_xlabel("resting-voltage decay slope (V/day)")
    axR.set_title("Resting-voltage decay (#6) - red = discriminative", fontweight="bold")
    axR.legend(frameon=False, fontsize=8)
    return _save(fig, "G6_changepoint_resting")


def _write_report(written):
    md = ["# V11 Graphs - generation report\n",
          "_All graphs sourced from committed V11 caches; no pipeline re-runs._\n",
          "| graph | file | source |", "|---|---|---|",
          "| G1 recall | G1_recall_comparison.png | comparison.csv, nf_self_test, compound_alarm_lovo |",
          "| G2 feature generalization | G2_feature_generalization.png | failed_window_deviations, nf_self_test |",
          "| G3 lead-time dumbbell | G3_leadtime_dumbbell.png | comparison.csv |",
          "| G4 compound leads | G4_compound_alarm_leads.png | compound_alarm_lovo |",
          "| G5 crank-recovery (MVP) | G5_crank_recovery_trajectories.png | <VIN>_daily.csv, nf_baseline |",
          "| G6 changepoint/resting | G6_changepoint_resting.png | changepoint_per_vin |",
          "\n## Honesty notes\n",
          "- G5: crank_recovery_t within-truck z-scores are inflated by near-zero baseline "
          "variance + the 30s censor; the trustworthy guard is 0/15 NF false alarms.\n",
          "- G6: change-point fires far too early (220-599d) to be actionable - exploratory only.\n",
          "- No RUL/Weibull content: V11 has no RUL data (referenced to V10.6.2).\n"]
    (OUT / "Graphs_generation_report.md").write_text("\n".join(md), encoding="utf-8")


def main():
    written = [g1_recall_comparison(), g2_feature_generalization(), g3_leadtime_dumbbell(),
               g4_compound_alarm_leads(), g5_crank_recovery_trajectories(), g6_changepoint_resting()]
    _write_report(written)
    print("[v11 graphs] wrote:", written)
    print("[v11 graphs] report:", OUT / "Graphs_generation_report.md")


if __name__ == "__main__":
    main()
