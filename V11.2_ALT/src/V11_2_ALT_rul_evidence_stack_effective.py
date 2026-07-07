"""
V11.2_ALT — per-VIN evidence stack WITH a combined 'Effective Alert' strip (NEW variant).

Adds one top strip = the worst of {schedule zone (Panel 1, age) , condition zone (Panel 1b,
physics)} at each moment, so DICV gets ONE colour to act on while the two component panels stay
visible underneath. Honest by construction: it never invents a transition — it just surfaces the
worse of the two real signals. Does NOT modify the original figures; writes to a separate folder.

Reuses build_bundle + all panels from V11_2_ALT_rul_evidence_stack. Run: py -3
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
sys.path.insert(0, os.path.join(ROOT, "V11.2_ALT", "src"))
import V11_2_ALT_rul_evidence_stack as E

OUT = os.path.join(ROOT, "V11.2_ALT", "visualizations", "rul_evidence_stack_effective")
os.makedirs(OUT, exist_ok=True)

EFF_COLORS = {0: "#27AE60", 1: "#F5A623", 2: "#E67E22", 3: "#C0392B"}
EFF_NAME = {0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED"}


def _sched_level(r):
    return 0 if r > 180 else 1 if r > 90 else 2 if r > 30 else 3


MIN_DWELL_DAYS = 14   # the condition must HOLD a zone this many consecutive days before the
                      # effective alert switches — debounces M5 boundary-chatter (cf. VIN3, whose
                      # M5 hugs the 0.15 line and would otherwise flap GREEN<->YELLOW ~11 times).
COND_ALERT_DEADBAND = 0.02   # the act-on strip escalates a condition zone only once M5 CLEARS the
                             # boundary by this margin (Panel 1b keeps the raw 0.15/0.35/0.55 zones
                             # for diagnostics). Removes the multi-week shallow wobbles that sit
                             # 0.000-0.012 above 0.15 — within M5 noise — without hiding real swings.


def _cond_alert_level(m):
    """M5 score -> condition ALERT level (0=GREEN..3=RED) using the act-on entry deadband."""
    m = float(m)
    if m >= 0.55 + COND_ALERT_DEADBAND:
        return 3
    if m >= 0.35 + COND_ALERT_DEADBAND:
        return 2
    if m >= 0.15 + COND_ALERT_DEADBAND:
        return 1
    return 0


def _debounce_levels(dates, levels, min_dwell_days=MIN_DWELL_DAYS):
    """Dwell-time debounce on a daily zone-level series (0=GREEN..3=RED).

    Walks contiguous runs of equal raw level and ADOPTS a run's level only if it lasts
    >= min_dwell_days; otherwise the previously adopted level carries through. Starts GREEN,
    so the alert escalates AND recovers only on a sustained run — short excursions across a
    threshold (in either direction) are smoothed. Symmetric by construction. The schedule is
    NOT passed through here: it is monotone and never chatters, so it stays raw.
    """
    levels = np.asarray(levels, int); n = len(levels)
    out = np.zeros(n, int); adopted = 0; i = 0
    while i < n:
        j = i
        while j + 1 < n and levels[j + 1] == levels[i]:
            j += 1
        span = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days + 1
        if span >= min_dwell_days:
            adopted = int(levels[i])
        out[i:j + 1] = adopted
        i = j + 1
    return out


def compute_effective(b):
    """Daily effective alert = max(schedule level, DEBOUNCED condition level) + driver.

    The condition (M5) is debounced (>= MIN_DWELL_DAYS dwell) to kill boundary-chatter; the
    schedule is monotone so it passes through raw and real age-escalations are never hidden.
    """
    dates = pd.to_datetime(b["daily"]["date"].values)
    # interpolate on integer DAYS-since-t0 — robust to datetime64 unit differences
    # (daily series is ns, weekly series can be us; mixing raw .asi8 silently clamps interp).
    day = np.asarray((dates - b["t0"]).days, float)
    wday = np.asarray((pd.to_datetime(b["weeks"].values) - b["t0"]).days, float)
    rul_daily = np.interp(day, wday, np.asarray(b["rul_med"], float))
    sched = np.array([_sched_level(r) for r in rul_daily])
    cond_raw = np.array([_cond_alert_level(m) for m in b["m5"]])   # entry-deadband levels
    cond = _debounce_levels(dates, cond_raw)                        # then >=14-day dwell
    eff = np.maximum(sched, cond)
    driver = np.where(sched > cond, "schedule", np.where(cond > sched, "condition", "both"))
    return dates, eff, driver, sched, cond


def panel_effective(ax, b, eff_data):
    dates, eff, driver, sched, cond = eff_data
    n = len(eff)
    # contiguous colour runs
    i = 0
    while i < n:
        j = i
        while j + 1 < n and eff[j + 1] == eff[i]:
            j += 1
        d0, d1 = dates[i], dates[j]
        ax.axvspan(d0, d1, color=EFF_COLORS[int(eff[i])], alpha=0.62, zorder=1)
        if (d1 - d0).days >= 40:  # name the run if wide enough
            mid = d0 + (d1 - d0) / 2
            ax.text(mid, 0.5, EFF_NAME[int(eff[i])], ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=3)
        i = j + 1
    # first entry into each worse level -> labelled transition with driver
    seen = set()
    for k in range(1, n):
        if eff[k] > eff[k - 1] and int(eff[k]) not in seen:
            ax.axvline(dates[k], color=E.DK, lw=0.8, alpha=0.55, zorder=4)
            ax.annotate(f"→{EFF_NAME[int(eff[k])]} ({driver[k]})", xy=(dates[k], 1.0),
                        xytext=(2, 7), textcoords="offset points", ha="left", va="bottom",
                        fontsize=7.5, fontweight="bold", color=EFF_COLORS[int(eff[k])], clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=EFF_COLORS[int(eff[k])], lw=0.6))
            seen.add(int(eff[k]))
    # current state at right edge
    ax.text(1.005, 0.5, f"now:\n{EFF_NAME[int(eff[-1])]}", transform=ax.transAxes, va="center",
            ha="left", fontsize=8, fontweight="bold", color=EFF_COLORS[int(eff[-1])])
    ax.set_yticks([]); ax.set_ylim(0, 1)
    ax.set_ylabel("Act on\nthis", fontsize=9, fontweight="bold", rotation=0, ha="right", va="center", labelpad=22)
    # honesty caption: how many raw condition excursions the dwell-debounce removed
    cond_raw = np.array([E._ZORDER[E.zone_of_m5(float(m))] for m in b["m5"]])
    n_smoothed = int((np.diff(np.maximum(sched, cond_raw)) != 0).sum() - (np.diff(eff) != 0).sum())
    ax.set_title("0 · EFFECTIVE ALERT — worst of schedule & debounced condition (the single colour to act on)", **E.TITLE_KW)
    if n_smoothed > 0:
        ax.text(1.0, 1.05, f"act-on: clear +{COND_ALERT_DEADBAND:.2f} & hold ≥{MIN_DWELL_DAYS} d · {n_smoothed} excursion(s) smoothed",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#7F8C8D", style="italic")


def build_figure_effective(vin):
    b = E.build_bundle(vin)
    eff_data = compute_effective(b)
    dts = b["daily"]["date"].values
    fire = [pd.Timestamp(dts[info["breach"]]) for info in b["prec"].values() if info["breach"] is not None]
    if b["ged_storm_date"] is not None:
        fire.append(pd.Timestamp(b["ged_storm_date"]))
    first_fire = min(fire) if fire else None

    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(6, 1, height_ratios=[1.0, 2.6, 1.9, 2.0, 2.1, 2.1], hspace=0.42,
                  left=0.085, right=0.915, top=0.94, bottom=0.05)
    axe = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=axe); ax1b = ax1.twinx()
    axm = fig.add_subplot(gs[2], sharex=axe)
    ax2 = fig.add_subplot(gs[3], sharex=axe)
    ax3 = fig.add_subplot(gs[4], sharex=axe)
    ax4 = fig.add_subplot(gs[5])

    panel_effective(axe, b, eff_data)
    E.panel_rul(ax1, ax1b, b); E.draw_rul_crossings(ax1, b)
    E.panel_health_zone(axm, b)
    E.panel_voltage(ax2, b)
    E.panel_precursors(ax3, b, first_fire)
    E.panel_charging(ax4, b)

    end = b["jc"] if b["failed"] else (b["t1"] + pd.Timedelta(days=b["median_rul"]) if b["median_rul"] > 0 else b["t1"])
    axe.set_xlim(b["t0"] - pd.Timedelta(days=10), end + pd.Timedelta(days=15))
    for ax in (axe, ax1, axm, ax2, ax3):
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
    for ax in (axe, ax1, axm, ax2):
        ax.tick_params(labelbottom=False)
    for ax in (ax1, axm, ax2, ax3, ax4):
        for s in ("top", "right"):
            if ax is not ax1:
                ax.spines[s].set_visible(False)
    for ax in (ax1, axm, ax2, ax3, ax4):
        ax.grid(True, axis="y", color="#E8ECEF", lw=0.5, alpha=0.7)
    for s in ("top", "right", "left"):
        axe.spines[s].set_visible(False)

    status = "FAILED" if b["failed"] else "in-service"
    fig.suptitle(f"Alternator RUL + Physics Evidence (with Effective Alert) — {b['dname']}",
                 fontsize=18, fontweight="bold", color=E.DK, x=0.085, ha="left", y=0.975)
    sub = (f"Risk: {b['risk_band'].upper()}   ·   ~{b['est_km']/1000:.0f}k km   ·   "
           f"fleet window {b['fw_p25']:.0f}–{b['fw_p75']:.0f} d (med {b['fleet_med']:.0f})   ·   {status}")
    fig.text(0.085, 0.952, sub, fontsize=10, color="#5D6D7E", style="italic")
    E.stamp_mode_badge(ax1, b)
    fig.text(0.085, 0.022,
             "Top strip = EFFECTIVE ALERT (worse of the age-schedule and the physics-condition). The condition only "
             f"escalates once M5 CLEARS a zone by {COND_ALERT_DEADBAND:.2f} and HOLDS it ≥{MIN_DWELL_DAYS} days, so the "
             "act-on colour never flaps on boundary noise; the monotone schedule passes through raw, so real "
             "age-escalations are never hidden. Panel 1b keeps the raw 0.15/0.35/0.55 zones for diagnostics.",
             fontsize=7.5, color="#95A5A6", style="italic")

    png = os.path.join(OUT, f"{b['dname']}_effective.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"{b['dname']}_effective.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main():
    ok, skip = [], []
    for vin in E.ALL_VINS:
        try:
            build_figure_effective(vin); ok.append(vin); print(f"  {vin} -> {E.display_name(vin)}: saved")
        except Exception as ex:
            skip.append((vin, str(ex))); print(f"  {vin}: SKIP {ex}")
    print(f"\nDone: {len(ok)}/25 effective-alert figures. Skipped: {skip}")
    assert len(ok) >= 23, f"too many skips: {skip}"


if __name__ == "__main__":
    main()
