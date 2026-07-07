"""
V11.2_ALT per-VIN DETECTION SHOWCASE — optimistic-yet-honest graphs.

Reframes the per-VIN story from RUL *days* (fleet, proven un-beatable per-truck) to per-truck
DETECTION / ranking / lead-time (genuinely strong). Four figures + a shared data layer:
  1 scorecard       — ridge_prob of all 25 vs green/amber/red bands (RED = 7 fail / 0 healthy)
  2 leadtime_wins   — VIN1/2/6/10 hero panels, each truck's own signal firing N d before failure
  3 coverage_matrix — 10 failed x {ranking, condition, precursor sigma, GED emergency}
  4 leadtime_bar    — hard lead vs condition lead per failed VIN (6 are honestly 0)

Honest guardrails: hard lead = GED storm or genuine_breach precursor (sigma>=2 AT failure);
condition lead = first SUSTAINED M5 ORANGE/RED — shown separately, never summed. Run: py -3
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

OUT = os.path.join(ROOT, "V11.2_ALT", "visualizations", "showcase")
RESD = os.path.join(ROOT, "V11.2_ALT", "results", "showcase")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RESD, exist_ok=True)

GREENc, AMBERc, REDc, DK = "#27AE60", "#F5A623", "#C0392B", E.DK
MODE_COL = {"abrupt": "#7F8C8D", "short-lead": "#E67E22", "gradual": "#7B1FA2"}
AMBER_LO, RED_LO, YOUDEN = 0.39, 0.55, 0.4456   # display band edges + operating alert threshold
HEROES = ["VIN1_F_ALT", "VIN2_F_ALT", "VIN6_F_ALT", "VIN10_F_ALT"]
FOOT = ("Per-truck intelligence = DETECTION + lead-time (this), not a per-truck RUL number "
        "(that stays the fleet window). n=25 data ceiling. Confidential.")


def _firing_prec(b):
    """The precursor that genuinely fired, with the largest lead (days before failure)."""
    end = pd.Timestamp(b["end"]); dts = b["daily"]["date"]
    cands = [(col, info, (end - pd.Timestamp(dts.iloc[info["breach"]])).days)
             for col, info in b["prec"].items() if info["breach"] is not None]
    return max(cands, key=lambda c: c[2]) if cands else None


def build_detection_table():
    fr = pd.read_csv(os.path.join(E.cfg.RUL_CACHE, "final_rul_per_vin.csv")).set_index("vin_label")
    rows, bundles = [], {}
    for vin in E.ALL_VINS:
        failed = bool(fr.loc[vin, "failed_flag"])
        prob = float(fr.loc[vin, "ridge_prob"]); band = str(fr.loc[vin, "risk_band"])
        rec = dict(vin=vin, dname=E.display_name(vin), failed=failed, ridge_prob=prob, band=band,
                   alert=band != "green", red=band == "red", ged_lead=0, hard_lead=0,
                   hard_channel="", cond_lead=0, cond_zone="", prec_fired=False,
                   earliest_signal=0, mode="")
        if failed:
            b = E.build_bundle(vin); bundles[vin] = b
            end = pd.Timestamp(b["end"])
            gl = int(b["ged_lead"]) if b["ged_fired"] and b["ged_lead"] else 0
            fp = _firing_prec(b)
            cands = {}
            if gl:
                cands["GED storm"] = gl
            if fp:
                cands[fp[1]["label"]] = fp[2]
            hard = max(cands.items(), key=lambda kv: kv[1]) if cands else ("", 0)
            cor = [t for t in b["m5_trans"] if t["to"] in ("ORANGE", "RED")]
            cl = (end - min(t["date"] for t in cor)).days if cor else 0
            cz = min(cor, key=lambda t: t["date"])["to"] if cor else ""
            hl = int(hard[1]); cli = int(cl); earliest = max(hl, cli)
            rec.update(ged_lead=gl, hard_lead=hl, hard_channel=hard[0], cond_lead=cli, cond_zone=cz,
                       prec_fired=fp is not None, earliest_signal=earliest,
                       mode="abrupt" if earliest == 0 else ("short-lead" if earliest <= 30 else "gradual"))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESD, "detection_table.csv"), index=False)
    return df, bundles


# ---------------------------------------------------------------- 1 · scorecard
def fig_scorecard(df):
    fig, ax = plt.subplots(figsize=(13, 8))
    for lo, hi, c in [(0, AMBER_LO, GREENc), (AMBER_LO, RED_LO, AMBERc), (RED_LO, 1.0, REDc)]:
        ax.axhspan(lo, hi, color=c, alpha=0.08)
    ax.axhline(YOUDEN, color=DK, lw=1.1, ls="--", alpha=0.7)
    # FAILED: one x-slot each (lollipop), sorted to an ascending staircase -> never overlaps
    fl = df[df.failed].sort_values("ridge_prob").reset_index(drop=True)
    fx = np.arange(len(fl))
    for xi, r in zip(fx, fl.itertuples()):
        c = REDc if r.band == "red" else AMBERc if r.band == "amber" else GREENc
        ax.plot([xi, xi], [0, r.ridge_prob], color="#9AA4AC", lw=1.0, alpha=0.4, zorder=2)
        ax.scatter([xi], [r.ridge_prob], s=210, color=c, edgecolors=DK, lw=1.3, zorder=5)
    # HEALTHY: jittered cluster to the right (no labels needed)
    hx0 = len(fl) + 1.6
    hl = df[~df.failed]
    rng = np.random.default_rng(0)
    hx = hx0 + rng.uniform(-0.55, 0.55, len(hl))
    for xi, (_, r) in zip(hx, hl.iterrows()):
        c = AMBERc if r.band == "amber" else GREENc
        ax.scatter([xi], [r.ridge_prob], s=85, color=c, edgecolors=DK, lw=1.0, alpha=0.6, zorder=4)
    ax.set_xticks(list(fx))
    ax.set_xticklabels([r.dname.replace("_F_ALT", "") for r in fl.itertuples()], fontsize=9, fontweight="bold")
    ax.text((len(fl) - 1) / 2, -0.085, "FAILED (10)", ha="center", fontsize=11, fontweight="bold", color=DK)
    ax.text(hx0, -0.085, "HEALTHY (15)", ha="center", fontsize=11, fontweight="bold", color=DK)
    n_alert = int((df.failed & df.alert).sum()); n_red = int((df.failed & df.red).sum())
    n_red_h = int((~df.failed & df.red).sum())
    ax.text(0.015, 0.975, f"RED band: {n_red} failures · {n_red_h} false alarms\n"
            f"{n_alert}/10 failures in the alert zone\nranking AUROC 0.927", transform=ax.transAxes,
            va="top", ha="left", fontsize=10.5, fontweight="bold", color=DK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F4FBF6", ec=GREENc, lw=1.2))
    miss = fl[~fl.alert]
    if len(miss):
        r = miss.iloc[0]
        ax.annotate(f"{r.dname.replace('_F_ALT','')} — the one miss\n(scored {r.ridge_prob:.2f}, below threshold)",
                    (0, r.ridge_prob), xytext=(34, 6), textcoords="offset points", fontsize=8.5,
                    color=REDc, fontweight="bold", arrowprops=dict(arrowstyle="->", color=REDc, lw=1))
    xr = hx0 + 1.15
    ax.text(xr, YOUDEN, f" alert threshold {YOUDEN:.2f}", va="center", fontsize=8, color=DK)
    for y, t, c in [(0.19, "GREEN", GREENc), (0.47, "AMBER", AMBERc), (0.78, "RED", REDc)]:
        ax.text(xr + 0.25, y, t, color=c, fontsize=10, fontweight="bold", va="center", rotation=90)
    ax.set_xlim(-0.8, xr + 0.6); ax.set_ylim(-0.13, 1.0)
    ax.set_ylabel("Risk score (ridge_prob, LOVO)", fontsize=11, fontweight="bold")
    ax.set_title("Per-VIN Detection Scorecard — each truck's own risk score", fontsize=15,
                 fontweight="bold", color=DK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.012, FOOT, ha="center", fontsize=7.5, color="#95A5A6", style="italic")
    _save(fig, "scorecard")


# ---------------------------------------------------------- 2 · lead-time wins
def fig_leadtime_wins(bundles):
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, hspace=0.34, wspace=0.18, left=0.07, right=0.965, top=0.9, bottom=0.09)
    for k, vin in enumerate(HEROES):
        b = bundles[vin]; ax = fig.add_subplot(gs[k // 2, k % 2])
        end = pd.Timestamp(b["end"]); dts = pd.to_datetime(b["daily"]["date"])
        if vin == "VIN1_F_ALT":
            ax.plot(dts, b["m5"], color="#7B1FA2", lw=2.0, zorder=5)
            for lo, hi, c in [(0, .15, GREENc), (.15, .35, AMBERc), (.35, .55, "#E67E22"), (.55, 1, REDc)]:
                ax.axhspan(lo, hi, color=c, alpha=0.07)
            ax.set_ylim(0, max(0.75, float(np.max(b["m5"])) * 1.1)); ax.set_ylabel("M5 condition", fontsize=9)
            cor = [t for t in b["m5_trans"] if t["to"] in ("ORANGE", "RED")]
            if cor:
                d0 = min(t["date"] for t in cor)
                ax.annotate(f"condition decline\n≈{(end - d0).days} d before failure",
                            (d0, 0.4), xytext=(8, 8), textcoords="offset points", fontsize=8,
                            color="#7B1FA2", fontweight="bold")
            if b["ged_storm_date"] is not None:
                gd = pd.Timestamp(b["ged_storm_date"])
                ax.axvline(gd, color=REDc, lw=1.4, ls="--")
                ax.annotate(f"GED storm\n≈{b['ged_lead']} d", (gd, 0.62), xytext=(-58, -2),
                            textcoords="offset points", fontsize=8, color=REDc, fontweight="bold")
            sig = "M5 condition + GED emergency"
        else:
            fp = _firing_prec(b); col, info, lead = fp
            z = np.asarray(info["z"], float); ax.plot(dts, z, color="#1565C0", lw=2.0, zorder=5)
            x_left = end - pd.Timedelta(days=140)
            ax.set_xlim(x_left, end + pd.Timedelta(days=6))
            peak = float(np.nanmax(z)); top = min(max(peak * 1.1, 3.5), 8.0)
            ax.set_ylim(min(0.0, float(np.nanmin(z))), top)
            ax.axhline(2.0, color=REDc, lw=1.0, ls="--", alpha=0.8)
            ax.text(x_left, 2.12, " 2σ alert", fontsize=7.5, color=REDc, va="bottom")
            bi = info["breach"]; ax.scatter([dts.iloc[bi]], [min(z[bi], top)], s=120, color=REDc,
                                            edgecolors=DK, lw=1.2, zorder=7)
            ax.annotate(f"{info['label']}\nfired ≈{lead} d before failure", (dts.iloc[bi], min(z[bi], top)),
                        xytext=(-8, -24), textcoords="offset points", ha="right", fontsize=8.5,
                        color="#1565C0", fontweight="bold")
            if peak > top:
                ax.annotate(f"peak ≈{peak:.0f}σ\nat failure", (end, top), xytext=(-6, -4),
                            textcoords="offset points", ha="right", va="top", fontsize=7.5, color="#8B0000")
            ax.set_ylabel("signal σ vs healthy", fontsize=9); sig = info["label"]
        ax.axvline(end, color="#8B0000", lw=1.6, ls="-.", alpha=0.7)
        ax.annotate("failure", (end, ax.get_ylim()[1]), xytext=(-30, -14), textcoords="offset points",
                    fontsize=8, color="#8B0000", fontweight="bold")
        ax.set_title(f"{b['dname'].replace('_F_ALT','')} — {sig}", fontsize=11, fontweight="bold", color=DK)
        loc = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="y", color="#E8ECEF", lw=0.5)
    fig.suptitle("Lead-Time Wins — the trucks whose own signals warned before failure",
                 fontsize=15, fontweight="bold", color=DK, x=0.07, ha="left")
    fig.text(0.07, 0.945, "4 of 10 failures gave an actionable hard lead (11–21 d); VIN1 also declined "
             "in condition ~6 months out. The other 6 were lead-time-silent (caught by ranking).",
             fontsize=9.5, color="#5D6D7E", style="italic")
    fig.text(0.5, 0.014, FOOT, ha="center", fontsize=7.5, color="#95A5A6", style="italic")
    _save(fig, "leadtime_wins")


# -------------------------------------------------------- 3 · coverage matrix
def fig_coverage_matrix(df):
    f = df[df.failed].sort_values("ridge_prob", ascending=False).reset_index(drop=True)
    cols = ["Risk ranking", "Condition zone", "Precursor σ", "GED emergency"]
    fig, ax = plt.subplots(figsize=(12, 7.5)); ax.axis("off")
    cell, colour = [], []
    for _, r in f.iterrows():
        rank = f"✓ {r.band.upper()}" if r.alert else "✗ missed"
        cond = f"✓ {r.cond_zone} (≈{r.cond_lead}d)" if r.cond_zone else "—"
        prec = f"✓ {r.hard_channel} (≈{r.hard_lead}d)" if (r.prec_fired and r.hard_channel != "GED storm") else "—"
        ged = f"✓ ≈{r.ged_lead}d" if r.ged_lead else "—"
        cell.append([rank, cond, prec, ged])
        colour.append([
            _tint(REDc if r.band == "red" else AMBERc if r.band == "amber" else "#BDBDBD") if r.alert else _tint(REDc, .5),
            _tint("#E67E22") if r.cond_zone else "#F4F6F7",
            _tint("#1565C0") if (r.prec_fired and r.hard_channel != "GED storm") else "#F4F6F7",
            _tint(REDc) if r.ged_lead else "#F4F6F7"])
    t = ax.table(cellText=cell, cellColours=colour, rowLabels=[r.dname.replace("_F_ALT", "") for _, r in f.iterrows()],
                 colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 2.0)
    n_any = int(((f.alert) | (f.cond_zone != "") | (f.prec_fired) | (f.ged_lead > 0)).sum())
    ax.set_title("Defense-in-Depth Coverage — every failed truck × every detection channel",
                 fontsize=14, fontweight="bold", color=DK, pad=24)
    ax.text(0.5, 1.02, f"{int(f.alert.sum())}/10 caught by ranking · {n_any}/10 caught by ≥1 channel · "
            "4 with a time-based lead", transform=ax.transAxes, ha="center", fontsize=10,
            color=DK, fontweight="bold")
    fig.text(0.5, 0.02, FOOT, ha="center", fontsize=7.5, color="#95A5A6", style="italic")
    _save(fig, "coverage_matrix")


# --------------------------------------------------------- 4 · lead-time bar
def fig_leadtime_bar(df):
    f = df[df.failed].sort_values("hard_lead", ascending=False).reset_index(drop=True)
    x = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x, f.hard_lead, 0.62, color="#1565C0", zorder=3)
    for xi, r in zip(x, f.itertuples()):
        if r.hard_lead:
            ax.text(xi, r.hard_lead + 0.5, f"{r.hard_lead} d", ha="center", fontsize=9.5,
                    fontweight="bold", color="#1565C0")
        else:
            ax.text(xi, 0.4, "silent", ha="center", va="bottom", fontsize=8, color="#B0B0B0", rotation=90)
    ax.set_ylim(0, 27)
    ax.set_xticks(x); ax.set_xticklabels([r.dname.replace("_F_ALT", "") for _, r in f.iterrows()])
    ax.set_ylabel("days before failure (hard lead)", fontsize=11, fontweight="bold")
    ax.set_title("Per-VIN Lead Time — earliest actionable (hard) warning", fontsize=15,
                 fontweight="bold", color=DK, loc="left")
    ax.text(0.98, 0.96, "4/10 give a hard, schedulable lead (11–21 d).\n6/10 are lead-time-silent (shown as 0).\n"
            "VIN1 also declined in condition ~199 d out\n(softer signal, off this scale).",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#5D6D7E", style="italic")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#E8ECEF", lw=0.5)
    fig.text(0.5, 0.012, FOOT, ha="center", fontsize=7.5, color="#95A5A6", style="italic")
    _save(fig, "leadtime_bar")


# --------------------------------------------------- 5 · how alternators fail
def fig_failure_modes(df, bundles):
    f = df[df.failed].copy()
    counts = {m: int((f["mode"] == m).sum()) for m in ("abrupt", "short-lead", "gradual")}
    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 1, height_ratios=[1, 3.2], hspace=0.42, left=0.08, right=0.95, top=0.9, bottom=0.09)

    axt = fig.add_subplot(gs[0]); axt.axis("off"); x = 0
    for m, lab in [("abrupt", "ABRUPT\nno warning"), ("short-lead", "SHORT-LEAD\n11–16 d"),
                   ("gradual", "GRADUAL\n199 d")]:
        w = counts[m]
        if w:
            axt.barh(0, w, left=x, height=0.6, color=MODE_COL[m], edgecolor="white", lw=2)
            axt.text(x + w / 2, 0, f"{w}\n{lab}", ha="center", va="center", color="white",
                     fontsize=10.5, fontweight="bold")
        x += w
    axt.set_xlim(0, 10); axt.set_ylim(-0.5, 0.5)
    axt.set_title("How alternators actually fail — 6/10 give NO telemetry warning",
                  fontsize=15, fontweight="bold", color=DK, loc="left")
    axt.text(0, -0.62, "Alternators are electrical parts that mostly stop ABRUPTLY (regulator / diode / "
             "brush). The bar is the 10 failures by warning available in the data.", fontsize=9,
             color="#5D6D7E", style="italic")

    ax = fig.add_subplot(gs[1])
    for lo, hi, c in [(0, .15, GREENc), (.15, .35, AMBERc), (.35, .55, "#E67E22"), (.55, 1, REDc)]:
        ax.axhspan(lo, hi, color=c, alpha=0.06)
    for _, r in f.iterrows():
        b = bundles[r.vin]; dates = pd.to_datetime(b["daily"]["date"]); end = pd.Timestamp(b["end"])
        m5 = np.asarray(b["m5"], float); mask = (dates >= end - pd.Timedelta(days=60)).values
        xx = (dates[mask] - end).dt.days.values; yy = m5[mask]
        ab = r["mode"] == "abrupt"
        ax.plot(xx, yy, color=MODE_COL[r["mode"]], lw=1.2 if ab else 2.2,
                alpha=0.45 if ab else 0.95, zorder=3 if ab else 6)
        if not ab and len(xx):
            ax.annotate(r.dname.replace("_F_ALT", ""), (xx[-1], yy[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8, fontweight="bold",
                        color=MODE_COL[r["mode"]])
    ax.axvline(0, color="#8B0000", lw=1.8, ls="-.", zorder=7)
    ax.text(0.5, 0.72, "sudden\nstop", color="#8B0000", fontsize=9, fontweight="bold", va="top")
    ax.text(-58, 0.10, "6 ABRUPT failures — flat & healthy (M5≈0.1) to the very end, then stop",
            fontsize=9.5, color="#5D6D7E", fontweight="bold", va="center")
    ax.set_xlim(-61, 6); ax.set_ylim(0, 0.78)
    ax.set_xlabel("days before failure", fontsize=11, fontweight="bold")
    ax.set_ylabel("M5 condition (this truck's physics)", fontsize=11, fontweight="bold")
    ax.set_title("Condition in the final 60 days, aligned to failure", fontsize=12, color=DK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#E8ECEF", lw=0.5)
    fig.text(0.08, 0.022, "Honest implication: for ABRUPT failures the moment cannot be predicted from "
             "telemetry — mitigate by POPULATION (fleet window / age replacement). Condition monitoring "
             "buys lead time only for the gradual minority. " + FOOT, fontsize=7.5, color="#95A5A6",
             style="italic", wrap=True)
    _save(fig, "failure_modes")


def _tint(hex_color, a=0.32):
    h = hex_color.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (1 - (1 - r / 255) * a, 1 - (1 - g / 255) * a, 1 - (1 - b / 255) * a)


def _save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


def main():
    df, bundles = build_detection_table()
    print(f"detection table: {len(df)} VINs ({int(df.failed.sum())} failed) -> results/showcase/")
    fig_scorecard(df)
    fig_leadtime_wins(bundles)
    fig_coverage_matrix(df)
    fig_leadtime_bar(df)
    fig_failure_modes(df, bundles)
    print("done: 5 figures -> visualizations/showcase/")


if __name__ == "__main__":
    main()
