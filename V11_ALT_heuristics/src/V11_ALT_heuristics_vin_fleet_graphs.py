"""V11_ALT_heuristics — per-VIN precursor dashboards + fleet overlays (V11-native).

Mirrors the V10.6.2 per-VIN + failed/non-failed/comparison fleet-overlay structure,
but plots V11 precursor signals (not RUL — V11 has no RUL data). No pipeline re-runs.
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
OUTV = cfg.V11_ROOT / "visualizations" / "V11_per_vin_graphs"
OUTF = cfg.V11_ROOT / "visualizations" / "V11_Fleet_graphs"
NAVY, GOLD, GREEN, RED, BLUE, GREY = "#0D1B2A", "#C58B1F", "#27AE60", "#C0392B", "#2980B9", "#95A5A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": NAVY, "axes.labelcolor": NAVY,
                     "text.color": NAVY, "xtick.color": NAVY, "ytick.color": NAVY, "figure.dpi": 110})


def _save(fig, out, name, pdf=False):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}_hd.png", dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


def _trusted(vin):
    d = pd.read_csv(FOR / f"{vin}_daily.csv")
    return d[d["n_eo"] >= cfg.MIN_EO_SAMPLES].copy()


def per_vin():
    nfb = pd.read_csv(FOR / "nf_baseline.csv").set_index("feature")
    es = pd.read_csv(FOR / "earliest_signal_per_vin.csv").set_index("vin_label")
    st = pd.read_csv(FOR / "nf_self_test.csv").set_index("vin_label")
    chans = cfg.VOTE_CHANNELS
    written = 0
    for vin in cfg.ALL_VINS:
        d = _trusted(vin)
        failed = vin in cfg.FAILED_VIN_SET
        if failed:
            d = d.sort_values("dtf", ascending=False); x = d["dtf"]; xlab = "days before failure"
        else:
            d = d.sort_values("day"); x = d["day"]; xlab = "vehicle age (days)"
        fig, axes = plt.subplots(len(chans), 1, figsize=(11, 2.0 * len(chans)), sharex=True)
        for ax, f in zip(axes, chans):
            p05, p95 = float(nfb.loc[f, "nf_p05"]), float(nfb.loc[f, "nf_p95"])
            ax.axhspan(p05, p95, color=GREEN, alpha=0.10)
            bad = p95 if f in cfg.BAD_HIGH else p05
            ax.axhline(bad, color=RED, ls="--", lw=1.0, alpha=0.7)
            y = pd.to_numeric(d[f], errors="coerce")
            ax.plot(x, y, color=BLUE, lw=1.1, marker="o", ms=2.2)
            ax.set_ylabel(f, fontsize=8)
            ax.grid(alpha=0.2)
        if failed:
            for ax in axes:
                ax.invert_xaxis()
        axes[-1].set_xlabel(xlab)
        if failed and vin in es.index:
            r = es.loc[vin]
            verdict = (f"{r['verdict']}  (earliest: {r['feature']} @ "
                       f"{r['earliest_discriminative_horizon_days']}d, z={r['z']})")
        elif (not failed) and vin in st.index:
            verdict = f"NF self-test: {st.loc[vin]['verdict']}"
        else:
            verdict = ""
        status = "FAILED" if failed else "NON-FAILED"
        fig.suptitle(f"{vin}  [{status}]  -  V11 precursor channels (green=healthy p05-p95)\n{verdict}",
                     fontweight="bold", fontsize=11, y=0.997)
        _save(fig, OUTV, f"{vin}_precursors")
        written += 1
    return written


def _overlay(ax, vins, title, p95):
    cmap = plt.cm.tab20
    for i, v in enumerate(vins):
        d = _trusted(v).sort_values("day")
        ax.plot(d["day"], pd.to_numeric(d["crank_recovery_t"], errors="coerce"),
                lw=1.0, alpha=0.85, label=v, color=cmap(i % 20))
    ax.axhline(p95, color=RED, ls="--", lw=1.6, label=f"healthy p95 ({p95:.2f}s)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("vehicle age (days)"); ax.set_ylabel("post-crank recovery time (s)")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)


def fleet():
    nfb = pd.read_csv(FOR / "nf_baseline.csv").set_index("feature")
    p95 = float(nfb.loc["crank_recovery_t", "nf_p95"])
    failed = [v for v in cfg.ALL_VINS if v in cfg.FAILED_VIN_SET]
    nf = [v for v in cfg.ALL_VINS if v not in cfg.FAILED_VIN_SET]

    fig, ax = plt.subplots(figsize=(13, 6))
    _overlay(ax, failed, "Failed fleet - post-crank recovery (MVP signal #3)", p95)
    _save(fig, OUTF, "failed_vehicle_fleet_overlay", pdf=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    _overlay(ax, nf, "Non-failed fleet - post-crank recovery (stays in band: 0/15 fire)", p95)
    _save(fig, OUTF, "non_failed_vehicle_fleet_overlay", pdf=True)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 11))
    _overlay(a1, failed, "Failed fleet (top)", p95)
    _overlay(a2, nf, "Non-failed fleet (bottom)", p95)
    _save(fig, OUTF, "fleet_comparison_overlay", pdf=True)

    # stats summary
    es = pd.read_csv(FOR / "earliest_signal_per_vin.csv").set_index("vin_label")
    st = pd.read_csv(FOR / "nf_self_test.csv").set_index("vin_label")
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    dev = pd.read_csv(FOR / "failed_window_deviations.csv")
    rows = []
    for vin in cfg.ALL_VINS:
        d = _trusted(vin)
        is_failed = vin in cfg.FAILED_VIN_SET
        span = int(d["day"].max() - d["day"].min()) if len(d) else 0
        if is_failed and vin in es.index:
            r = es.loc[vin]
            horizon, feat = r["earliest_discriminative_horizon_days"], r["feature"]
            ndisc = int(dev[(dev.vin_label == vin) & (dev.discriminative == True)]["feature"].nunique())  # noqa: E712
            ged2 = int(r["ged2_total"])
        elif (not is_failed) and vin in st.index:
            rr = st.loc[vin]
            horizon, feat, ndisc, ged2 = rr["false_precursor_horizon_days"], rr["feature"], 0, ""
        else:
            horizon, feat, ndisc, ged2 = "none", "", 0, ""
        fired = bool(cl[(cl.vin_label == vin) & (cl.fired == True)].shape[0])  # noqa: E712
        rows.append({"vin_label": vin, "failure_status": "FAILED" if is_failed else "NON_FAILED",
                     "trusted_days": len(d), "data_span_days": span,
                     "earliest_detection_horizon": horizon, "detecting_feature": feat,
                     "n_discriminative_features": ndisc, "compound_fired": fired, "ged2_total": ged2})
    stats = pd.DataFrame(rows)
    stats.to_csv(OUTF / "fleet_statistics_summary.csv", index=False)
    with pd.ExcelWriter(OUTF / "fleet_statistics_summary.xlsx", engine="openpyxl") as xl:
        stats.to_excel(xl, sheet_name="Fleet_Statistics", index=False)

    rep = ["# V11 Fleet & Per-VIN Graphs - generation report\n",
           "_V11-native: plots precursor signals, NOT RUL (V11 has no RUL data; see V10.6.2 for RUL)._\n",
           f"- Per-VIN precursor dashboards: {len(cfg.ALL_VINS)} VINs in V11_per_vin_graphs/ (5 vote-channel panels each, healthy p05-p95 band).\n",
           "- Fleet overlays (failed / non-failed / comparison) of crank_recovery_t vs vehicle age, healthy p95 line; PNG+HD+PDF.\n",
           "- fleet_statistics_summary.csv/.xlsx: per-VIN status, span, earliest detection, detecting feature, #discriminative features, compound fired, GED2 total.\n",
           "\n## Honesty notes\n",
           "- crank_recovery_t is episodic (isolated spikes) against a near-zero healthy baseline (~0.05s); guard is 0/15 NF false alarms, not z-magnitude.\n",
           "- Failed plotted vs days-before-failure (per-VIN) / vehicle age (overlays); non-failed vs vehicle age.\n",
           "- No RUL/Weibull curves: not available for V11.\n"]
    (OUTF / "Fleet_graphs_generation_report.md").write_text("\n".join(rep), encoding="utf-8")
    return stats


def main():
    n = per_vin()
    stats = fleet()
    print(f"[v11 vin+fleet graphs] per-VIN dashboards: {n}")
    print(f"[v11 vin+fleet graphs] fleet overlays + stats ({len(stats)} VINs) -> {OUTF}")


if __name__ == "__main__":
    main()
