"""
V11.2_ALT snapshot-table variant: a component x key-moment table on the evidence stack.

Read DOWN a single column to see every component's exact state at that moment. New variant —
does NOT touch the rul_evidence_stack/ or rul_evidence_stack_effective/ figures. Reuses build_bundle
+ panels from E and compute_effective/panel_effective from X. Run: py -3
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
sys.path.insert(0, os.path.join(ROOT, "V11.2_ALT", "src"))
import V11_2_ALT_rul_evidence_stack as E
import V11_2_ALT_rul_evidence_stack_effective as X

OUT = os.path.join(ROOT, "V11.2_ALT", "visualizations", "rul_evidence_stack_snapshot")
CSVDIR = os.path.join(ROOT, "V11.2_ALT", "results", "snapshot")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CSVDIR, exist_ok=True)

ZCOL = {"GREEN": "#27AE60", "YELLOW": "#F5A623", "ORANGE": "#E67E22", "RED": "#C0392B", "BLACK": "#2C3E50"}
GREY = "#ECEFF1"


def _tint(hex_color, a=0.30):
    h = hex_color.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (1 - (1 - r / 255) * a, 1 - (1 - g / 255) * a, 1 - (1 - b / 255) * a)


def _vsi_col(v):
    return ZCOL["RED"] if v < 26 else ZCOL["YELLOW"] if v < 27 else ZCOL["GREEN"]


def _sig_col(z):
    return ZCOL["RED"] if z >= 2 else ZCOL["YELLOW"] if z >= 1 else ZCOL["GREEN"]


def compute_snapshot(b):
    dates = pd.to_datetime(b["daily"]["date"].values)
    eff_dates, eff, drv, sched, cond = X.compute_effective(b)
    m5 = b["m5"]
    day = np.asarray((dates - b["t0"]).days, float)
    wday = np.asarray((pd.to_datetime(b["weeks"].values) - b["t0"]).days, float)
    rul = np.interp(day, wday, np.asarray(b["rul_med"], float))
    fire = [pd.Timestamp(dates[i["breach"]]) for i in b["prec"].values() if i["breach"] is not None]
    if b["ged_storm_date"] is not None:
        fire.append(pd.Timestamp(b["ged_storm_date"]))
    first_fire = min(fire) if fire else None

    def first_cond(level):
        idx = np.where(np.array([E._ZORDER[E.zone_of_m5(float(x))] for x in m5]) >= level)[0]
        return pd.Timestamp(dates[idx[0]]) if len(idx) else None

    cand = [("Early", b["t0"] + pd.Timedelta(days=60)),
            ("First deviation", first_fire),
            ("Cond→ORANGE", first_cond(2)),
            ("Cond→RED", first_cond(3)),
            ("Failure" if b["failed"] else "Now", b["end"])]
    # keep only moments that occurred, in chronological (left→right) order
    cols = sorted([(lab, dt) for lab, dt in cand if dt is not None], key=lambda c: c[1])
    col_index = {c[0]: k for k, c in enumerate(cols)}

    def nidx(dt):
        return int(np.abs((dates - dt).days).argmin())

    ZN = {0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED"}
    dl = b["daily"]

    def crank(i):
        p = b["prec"].get("crank_recovery_t")
        return (f"{p['z'][i]:+.1f}σ", _sig_col(p["z"][i])) if p else ("n/a", GREY)

    def ceil_(i):
        p = b["prec"].get("vsi_ceiling")
        return (f"{p['z'][i]:+.1f}σ", _sig_col(p["z"][i])) if p else ("n/a", GREY)

    def ged(i, dt):
        on = b["ged_storm_date"] is not None and abs((pd.Timestamp(b["ged_storm_date"]) - dt).days) <= 7
        return ("STORM", ZCOL["RED"]) if on else ("—", GREY)

    def sagcol(s):
        return ZCOL["RED"] if s > 0.05 else (ZCOL["YELLOW"] if s > 0 else ZCOL["GREEN"])

    rowdefs = [
        ("Effective zone", lambda i, dt: (ZN[int(eff[i])], ZCOL[ZN[int(eff[i])]])),
        ("RUL (schedule)", lambda i, dt: (f"{rul[i]:.0f} d ({ZN[int(sched[i])]})", ZCOL[ZN[int(sched[i])]])),
        ("Condition (M5)", lambda i, dt: (f"{float(m5[i]):.2f} ({E.zone_of_m5(float(m5[i]))})", ZCOL[E.zone_of_m5(float(m5[i]))])),
        ("VSI mean", lambda i, dt: (f"{float(dl['vsi_mean'].iloc[i]):.1f} V", _vsi_col(float(dl['vsi_mean'].iloc[i])))),
        ("Resting V", lambda i, dt: (f"{float(dl['resting_vsi_mean'].iloc[i]):.1f} V", _vsi_col(float(dl['resting_vsi_mean'].iloc[i])))),
        ("Sag rate", lambda i, dt: (f"{float(dl['vsi_sag_frac'].iloc[i]):.2f}", sagcol(float(dl['vsi_sag_frac'].iloc[i])))),
        ("Crank-recovery", lambda i, dt: crank(i)),
        ("Reg. ceiling", lambda i, dt: ceil_(i)),
        ("GED2", lambda i, dt: ged(i, dt)),
    ]
    rows = {}
    for name, fn in rowdefs:
        cells = []
        for lab, dt in cols:
            cells.append(("—", GREY) if dt is None else fn(nidx(dt), dt))
        rows[name] = cells
    return dict(cols=cols, col_index=col_index, rows=rows, charge=(b["chg"],),
               risk=b["risk_band"], dname=b["dname"])


def compute_snapshot_for(vin):
    return compute_snapshot(E.build_bundle(vin))


def panel_snapshot(ax, b, snap):
    ax.axis("off")
    col_lab = [f"{lab}\n{(dt.strftime('%Y-%m-%d') if dt is not None else '—')}" for lab, dt in snap["cols"]]
    rownames = list(snap["rows"].keys())
    text = [[snap["rows"][r][c][0] for c in range(len(snap["cols"]))] for r in rownames]
    colour = [[_tint(snap["rows"][r][c][1]) for c in range(len(snap["cols"]))] for r in rownames]
    tbl = ax.table(cellText=text, cellColours=colour, rowLabels=rownames, colLabels=col_lab,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    ax.set_title("5 · SNAPSHOT — every component at each key moment (read DOWN a column)", **E.TITLE_KW)
    e, l = snap["charge"][0]["early"], snap["charge"][0]["late"]

    def ceil(bins):
        c, m = bins; sel = c >= 1500
        return float(np.median(m[sel])) if (len(m) and sel.any()) else (float(np.median(m)) if len(m) else float("nan"))

    cd = ceil(l) - ceil(e)
    ax.text(0.5, -0.04, f"Risk band: {snap['risk'].upper()}    ·    Charging ceiling early→late Δ = {cd:+.2f} V"
            "    ·    columns = the key moments that occurred for this truck (chronological)",
            transform=ax.transAxes, ha="center", fontsize=8, color="#5D6D7E", style="italic")


def build_figure_snapshot(vin):
    b = E.build_bundle(vin); snap = compute_snapshot(b)
    eff_data = X.compute_effective(b)
    dts = b["daily"]["date"].values
    fire = [pd.Timestamp(dts[i["breach"]]) for i in b["prec"].values() if i["breach"] is not None]
    if b["ged_storm_date"] is not None:
        fire.append(pd.Timestamp(b["ged_storm_date"]))
    first_fire = min(fire) if fire else None

    fig = plt.figure(figsize=(15, 24))
    gs = GridSpec(7, 1, height_ratios=[1.0, 2.6, 1.9, 2.0, 2.1, 2.1, 3.0], hspace=0.5,
                  left=0.085, right=0.915, top=0.945, bottom=0.04)
    axe = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1], sharex=axe); ax1b = ax1.twinx()
    axm = fig.add_subplot(gs[2], sharex=axe); ax2 = fig.add_subplot(gs[3], sharex=axe)
    ax3 = fig.add_subplot(gs[4], sharex=axe); ax4 = fig.add_subplot(gs[5]); axt = fig.add_subplot(gs[6])

    X.panel_effective(axe, b, eff_data)
    E.panel_rul(ax1, ax1b, b); E.draw_rul_crossings(ax1, b)
    E.panel_health_zone(axm, b); E.panel_voltage(ax2, b)
    E.panel_precursors(ax3, b, first_fire); E.panel_charging(ax4, b)
    panel_snapshot(axt, b, snap)

    end = b["jc"] if b["failed"] else (b["t1"] + pd.Timedelta(days=b["median_rul"]) if b["median_rul"] > 0 else b["t1"])
    axe.set_xlim(b["t0"] - pd.Timedelta(days=10), end + pd.Timedelta(days=15))
    for ax in (axe, ax1, axm, ax2, ax3):
        ax.axvline(b["t1"], color="#5D6D7E", lw=0.9, ls=":", alpha=0.55)
        if first_fire is not None:
            ax.axvline(first_fire, color="#FF9800", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(end, color="#8B0000" if b["failed"] else "#C0392B", lw=1.2,
                   ls="-." if b["failed"] else ":", alpha=0.65)
    loc = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax3.xaxis.set_major_locator(loc); ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax3.set_xlabel("Timeline", fontsize=10, fontweight="bold")
    for ax in (axe, ax1, axm, ax2):
        ax.tick_params(labelbottom=False)
    for ax in (ax1, axm, ax2, ax3, ax4):
        for s in ("top", "right"):
            if ax is not ax1:
                ax.spines[s].set_visible(False)
        ax.grid(True, axis="y", color="#E8ECEF", lw=0.5, alpha=0.7)

    fig.suptitle(f"Alternator Evidence Stack + Snapshot — {b['dname']}", fontsize=18, fontweight="bold",
                 color=E.DK, x=0.085, ha="left", y=0.965)
    E.stamp_mode_badge(ax1, b)
    fig.text(0.085, 0.018,
             "Read the SNAPSHOT (bottom) down a column to see every component at that moment. RUL = fleet "
             "schedule (age); Condition/Effective = this truck's physics. n=25 data ceiling. Confidential.",
             fontsize=7.5, color="#95A5A6", style="italic")

    png = os.path.join(OUT, f"{b['dname']}_snapshot.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"{b['dname']}_snapshot.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with open(os.path.join(CSVDIR, f"{b['dname']}_snapshot.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["component"] + [f"{lab} ({dt.date() if dt is not None else 'n/a'})" for lab, dt in snap["cols"]])
        for r, cells in snap["rows"].items():
            w.writerow([r] + [c[0] for c in cells])
    return png


def main():
    ok = []
    for vin in E.ALL_VINS:
        try:
            build_figure_snapshot(vin); ok.append(vin); print(f"  {E.display_name(vin)}: saved")
        except Exception as ex:
            print(f"  {vin}: SKIP {ex}")
    print(f"Done {len(ok)}/25")
    assert len(ok) >= 23


if __name__ == "__main__":
    main()
