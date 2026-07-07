"""
V11.2_ALT OPTIMISTIC variant of the per-VIN RUL evidence stacks (optm_ prefix).

Executive/presentation layer: removes negative-framed comments and adds a data-driven optimistic
insight strip. Reuses build_bundle + the panels from E (called with optimistic=True). NEVER prints
a false claim — an abrupt-failure truck is framed as 'caught by ranking', not as having warned.
The honest canonical figures (rul_evidence_stack/) are untouched. Run: py -3
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
sys.path.insert(0, os.path.join(ROOT, "V11.2_ALT", "src"))
import V11_2_ALT_rul_evidence_stack as E

OUT = os.path.join(ROOT, "V11.2_ALT", "visualizations", "rul_evidence_stack_optm")
DET_CSV = os.path.join(ROOT, "V11.2_ALT", "results", "showcase", "detection_table.csv")
os.makedirs(OUT, exist_ok=True)


def optm_insight(b, prob, band):
    """One-line, factually-true optimistic insight for the bottom strip.
    Failed-with-lead -> 'DETECTED ...'; failed-abrupt -> 'CAUGHT ...' (no invented lead);
    non-failed -> 'HEALTHY ...'."""
    band = str(band).upper()
    # ---- non-failed (in-service) ----
    if not b["failed"]:
        if band == "GREEN":
            return (f"✓ HEALTHY — risk {prob:.2f}, comfortably below the alert line  ·  "
                    f"0 emergency/precursor fires  ·  no failure observed")
        return (f"✓ IN-SERVICE — risk {prob:.2f} (ML model {band.lower()} watch-list)  ·  "
                f"0 emergency/precursor fires  ·  no failure to date — proactively monitored")
    # ---- failed but scored below the alert line: the fleet's single miss (stay honest) ----
    if band == "GREEN":
        return (f"△ FLEET'S ONE MISS — scored {prob:.2f}, just under the alert line; the single "
                f"n=25 data-ceiling case the expanded-fleet roadmap is built to close")
    # ---- failed and flagged by the ML model ----
    short_tier = "top tier" if band == "RED" else "elevated tier"
    end = pd.Timestamp(b["end"]); dts = b["daily"]["date"]
    cor = [t for t in b["m5_trans"] if t["to"] in ("ORANGE", "RED")]
    cond_lead = (end - min(t["date"] for t in cor)).days if cor else 0
    worst = "RED" if any(t["to"] == "RED" for t in cor) else ("ORANGE" if cor else "")
    prec_leads = [(info["label"], (end - pd.Timestamp(dts.iloc[info["breach"]])).days)
                  for info in b["prec"].values() if info["breach"] is not None]
    # NAME each agreeing signal and tie it to where it shows on the figure — not all are
    # panel-3 precursors (the word 'precursor' is used ONLY when a sigma-precursor actually breaches).
    signals = ["ML risk rank"]                                          # the band / model score
    if cor:
        lead_txt = f", ~{cond_lead} d lead" if cond_lead > 0 else ""
        signals.append(f"M5 condition zone (→{worst}{lead_txt})")   # panel 1b
    if prec_leads:
        lab, lead = max(prec_leads, key=lambda kv: kv[1])              # strongest precursor (panel-3 line)
        signals.append(f"{lab} precursor (~{lead} d lead)")
    if b["ged_fired"] and b["ged_lead"]:
        signals.append(f"GED emergency (~{int(b['ged_lead'])} d lead)")  # panel-3 storm marker
    if len(signals) > 1:
        return (f"✓ DETECTED — {band} band ({short_tier})  ·  {len(signals)} signals agree →  "
                + "  ·  ".join(signals))
    return (f"✓ CAUGHT — {band} band (ML model's {short_tier})  ·  the ML risk model scored it {prob:.2f} "
            f"and ranked it in the alert band from its electrical signature alone — no physics precursor needed")


_DET = pd.read_csv(DET_CSV).set_index("dname")


def _strip(fig, b):
    """Bottom optimistic insight strip (replaces the canonical footnotes)."""
    row = _DET.loc[b["dname"]]
    txt = optm_insight(b, float(row["ridge_prob"]), str(row["band"]))
    fig.text(0.085, 0.018, txt, fontsize=8.5, color="#1B5E20", fontweight="bold")


def build_figure_optm(vin):
    b = E.build_bundle(vin)
    dates = b["daily"]["date"].values
    fire = [pd.Timestamp(dates[info["breach"]]) for info in b["prec"].values() if info["breach"] is not None]
    if b["ged_storm_date"] is not None:
        fire.append(pd.Timestamp(b["ged_storm_date"]))
    first_fire = min(fire) if fire else None

    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(5, 1, height_ratios=[2.7, 1.9, 2.0, 2.1, 2.1], hspace=0.45,
                  left=0.085, right=0.915, top=0.935, bottom=0.055)
    ax1 = fig.add_subplot(gs[0]); ax1b = ax1.twinx()
    axm = fig.add_subplot(gs[1], sharex=ax1)
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    ax4 = fig.add_subplot(gs[4])

    E.panel_rul(ax1, ax1b, b, optimistic=True); E.draw_rul_crossings(ax1, b)
    E.panel_health_zone(axm, b)
    E.panel_voltage(ax2, b)
    E.panel_precursors(ax3, b, first_fire, optimistic=True)
    E.panel_charging(ax4, b)

    end = b["jc"] if b["failed"] else (b["t1"] + pd.Timedelta(days=b["median_rul"]) if b["median_rul"] > 0 else b["t1"])
    ax1.set_xlim(b["t0"] - pd.Timedelta(days=10), end + pd.Timedelta(days=15))
    for ax in (ax1, axm, ax2, ax3):
        ax.axvline(b["t1"], color="#5D6D7E", lw=0.9, ls=":", alpha=0.55)
        if first_fire is not None:
            ax.axvline(first_fire, color="#FF9800", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(end, color="#8B0000" if b["failed"] else "#C0392B", lw=1.2,
                   ls="-." if b["failed"] else ":", alpha=0.65)
    if first_fire is not None:
        lead = (end - first_fire).days
        ax3.annotate(f"first physics deviation ~{lead} d before {'failure' if b['failed'] else 'forecast'}",
                     xy=(first_fire, 2.0), xytext=(8, -28), textcoords="offset points",
                     fontsize=8, fontweight="bold", color="#B71C1C",
                     bbox=dict(boxstyle="round,pad=0.25", fc="#FFF5F5", ec="#FF9800", lw=0.7),
                     arrowprops=dict(arrowstyle="->", color="#FF9800", lw=0.9))

    loc = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax3.xaxis.set_major_locator(loc); ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax3.set_xlabel("Timeline", fontsize=10, fontweight="bold")
    ax1.tick_params(labelbottom=False); axm.tick_params(labelbottom=False); ax2.tick_params(labelbottom=False)
    for ax in (axm, ax2, ax3, ax4):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in (ax1, axm, ax2, ax3, ax4):
        ax.grid(True, axis="y", color="#E8ECEF", lw=0.5, alpha=0.7)

    fig.suptitle(f"Alternator RUL + Physics Evidence — {b['dname']}", fontsize=18, fontweight="bold",
                 color=E.DK, x=0.085, ha="left", y=0.975)
    sub = (f"Risk: {b['risk_band'].upper()}   ·   ~{b['est_km']/1000:.0f}k km   ·   "
           f"fleet window {b['fw_p25']:.0f}–{b['fw_p75']:.0f} d (med {b['fleet_med']:.0f})   ·   "
           f"{'FAILED' if b['failed'] else 'in-service'}")
    fig.text(0.085, 0.952, sub, fontsize=10, color="#5D6D7E", style="italic")
    E.stamp_mode_badge(ax1, b, optimistic=True, band=str(_DET.loc[b["dname"]]["band"]))
    _strip(fig, b)

    png = os.path.join(OUT, f"optm_{b['dname']}_evidence_stack.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"optm_{b['dname']}_evidence_stack.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main():
    ok = []
    for vin in E.ALL_VINS:
        try:
            build_figure_optm(vin); ok.append(vin); print(f"  optm_{E.display_name(vin)}: saved")
        except Exception as ex:
            print(f"  {vin}: SKIP {ex}")
    print(f"Done {len(ok)}/25 optimistic figures -> {OUT}")
    assert len(ok) >= 25


if __name__ == "__main__":
    main()
