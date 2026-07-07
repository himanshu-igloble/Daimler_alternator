"""
V11.2_ALT per-VIN "RUL + Physics Evidence Stack".

Five time-aligned panels per VIN:
  1)  RUL clock (fleet schedule, reuses V11.1 conditional-Weibull math) + zone-crossing callouts
  1b) Health zone (parameter-driven M5 score) with decomposed transition drivers
  2)  Voltage health (VSI band, regulation band, sag line, resting voltage)
  3)  Precursor deterioration (6 signals as sigma-vs-healthy; breaching bold, rest muted; GED2 storm)
  4)  Charging signature (V-RPM early vs late life)

Display naming (unique 1-25): failed = VIN1..VIN10 (F), non-failed = VIN11..VIN25 (NF).
Honest framing: the RUL line is a FLEET clock (not per-truck timing). Read-only inputs. py -3
"""
from __future__ import annotations
import os, sys, json, importlib.util, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
sys.path.insert(0, os.path.join(ROOT, "V11.2_ALT", "src"))
from V11_2_ALT_common import JCOPENDATE  # frozen failure dates

OUT = os.path.join(ROOT, "V11.2_ALT", "visualizations", "rul_evidence_stack")
os.makedirs(OUT, exist_ok=True)

# palette
PRD = "#E8490F"; VOLT = "#0B5394"; REST = "#674ea7"; SAGC = "#B71C1C"
ZG = "#27AE60"; ZY = "#F5A623"; ZO = "#E67E22"; ZB = "#2C3E50"; DK = "#1B2838"
M5C = "#7B1FA2"
PREC_COLORS = ["#0B5394", "#E67E22", "#674ea7", "#1b6b50", "#B71C1C"]
TITLE_KW = dict(loc="left", fontsize=11.5, fontweight="bold", color=DK, pad=6)


# =============================================================================
# DISPLAY NAMING (unique 1-25: failed 1-10, non-failed 11-25)
# =============================================================================
_DISPLAY = {f"VIN{i}_F_ALT": f"VIN{i}_F_ALT" for i in range(1, 11)}
for _idx, _nf in enumerate([f"VIN{i}_NF_ALT" for i in range(1, 16)], start=11):
    _DISPLAY[_nf] = f"VIN{_idx}_NF_ALT"


def display_name(vin):
    return _DISPLAY.get(vin, vin)


# =============================================================================
# PURE HELPERS (unit-tested)
# =============================================================================
def zscore_vs_baseline(values, baseline_mask, invert=False):
    """z of each value vs the healthy-baseline mean/std. invert=True => a DROP scores higher (worse)."""
    v = np.asarray(values, float)
    base = v[np.asarray(baseline_mask, bool) & ~np.isnan(v)]
    mu = float(np.nanmean(base)) if base.size else float(np.nanmean(v))
    sd = float(np.nanstd(base)) if base.size else float(np.nanstd(v))
    if not np.isfinite(sd) or sd == 0:
        sd = (float(np.nanstd(v)) or 1.0)
    z = (v - mu) / sd
    return -z if invert else z


def first_breach_index(z, threshold=2.0):
    z = np.asarray(z, float)
    idx = np.where(z >= threshold)[0]
    return int(idx[0]) if idx.size else None


def genuine_breach(z, fire_thr=2.0, shoulder=1.5):
    """Onset of the FINAL sustained elevation that is still breaching at the end.
    Returns None unless the signal is clearly elevated (z>=fire_thr) at the last sample —
    so early-life baseline crossings that recover before failure are NOT counted as precursors.
    The lead is then (end - date[onset]), i.e. the duration of the genuine run-up to failure."""
    z = np.asarray(z, float)
    if z.size == 0 or not np.isfinite(z[-1]) or z[-1] < fire_thr:
        return None
    i = z.size - 1
    while i > 0 and z[i - 1] >= shoulder:
        i -= 1
    return i


def charging_bins(rpm, vsi, rpm_lo=600.0, rpm_hi=2500.0, bin_w=100.0):
    rpm = np.asarray(rpm, float); vsi = np.asarray(vsi, float)
    m = (rpm >= rpm_lo) & (rpm <= rpm_hi) & np.isfinite(vsi)
    rpm, vsi = rpm[m], vsi[m]
    edges = np.arange(rpm_lo, rpm_hi + bin_w, bin_w)
    centers, med = [], []
    for i in range(len(edges) - 1):
        sel = (rpm >= edges[i]) & (rpm < edges[i + 1])
        if sel.sum() >= 20:
            centers.append((edges[i] + edges[i + 1]) / 2.0)
            med.append(float(np.median(vsi[sel])))
    return np.array(centers), np.array(med)


# =============================================================================
# M5 PARAMETER-DRIVEN HEALTH-ZONE (decomposable) + RUL-crossing physics
# =============================================================================
M5_NOMINAL = 28.2; M5_DIV = 4.0; NF_STD_P90 = 0.45; GED2_NORM = 0.05; NF_RANGE_P90 = 2.5
M5_W = {"C1": 0.294, "C2": 0.235, "C3": 0.294, "C4": 0.176}
M5_THR = {"green": 0.15, "yellow": 0.35, "orange": 0.55}
M5_LABEL = {"C1": "VSI deviation", "C2": "VSI volatility", "C3": "GED2 rate", "C4": "VSI range"}
_ZORDER = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}


def _roll(x):
    return pd.Series(x, dtype=float).rolling(21, min_periods=7, center=True).mean().bfill().ffill().values


def compute_m5(daily):
    n = len(daily)
    vsi_m = _roll(daily["vsi_mean"].values) if "vsi_mean" in daily else np.full(n, M5_NOMINAL)
    C1 = np.clip(np.abs(vsi_m - M5_NOMINAL) / M5_DIV, 0, 1)
    C2 = np.clip(_roll(daily["vsi_std"].values) / NF_STD_P90, 0, 1) if "vsi_std" in daily else np.zeros(n)
    ged = daily["ged2_frac"].values if "ged2_frac" in daily else np.zeros(n)
    C3 = np.clip(_roll(ged) / GED2_NORM, 0, 1)
    C4 = np.clip(_roll(daily["vsi_range"].values) / NF_RANGE_P90, 0, 1) if "vsi_range" in daily else np.zeros(n)
    contrib = {"C1": M5_W["C1"] * C1, "C2": M5_W["C2"] * C2, "C3": M5_W["C3"] * C3, "C4": M5_W["C4"] * C4}
    m5 = contrib["C1"] + contrib["C2"] + contrib["C3"] + contrib["C4"]
    return m5, contrib


def zone_of_m5(score):
    if score < M5_THR["green"]:
        return "GREEN"
    if score < M5_THR["yellow"]:
        return "YELLOW"
    if score < M5_THR["orange"]:
        return "ORANGE"
    return "RED"


def m5_transitions(dates, m5, contrib):
    zones = [zone_of_m5(v) for v in m5]; out = []; seen = set()
    for i in range(1, len(zones)):
        if _ZORDER[zones[i]] > _ZORDER[zones[i - 1]] and zones[i] not in seen:
            dom = max(contrib, key=lambda k: contrib[k][i])
            out.append(dict(date=pd.Timestamp(dates[i]), to=zones[i], frm=zones[i - 1],
                            driver=M5_LABEL[dom], score=float(m5[i])))
            seen.add(zones[i])
    return out


def rul_crossings(weeks, med, daily, prec, ged_storm):
    out = []
    dd = daily.reset_index(drop=True)
    for thr, name in [(180, "GREEN→YELLOW"), (90, "YELLOW→ORANGE"), (30, "ORANGE→BLACK")]:
        below = np.where(np.asarray(med) <= thr)[0]
        if len(below) == 0 or med[0] <= thr:
            continue
        ci = int(below[0]); xdate = pd.Timestamp(weeks.iloc[ci]); ry = float(med[ci])
        di = int((dd["date"] - xdate).abs().values.argmin())
        breaching = []
        for col, info in prec.items():
            if info["breach"] is not None and di >= info["breach"]:
                breaching.append((info["label"], float(info["z"][di])))
        breaching.sort(key=lambda t: -t[1])
        ged_active = (ged_storm is not None and abs((pd.Timestamp(ged_storm) - xdate).days) < 30)
        out.append(dict(thr=thr, name=name, date=xdate, rul=ry,
                        breaching=breaching, ged=ged_active, corrob=bool(breaching) or ged_active))
    return out


# =============================================================================
# V11.1 RUL REUSE
# =============================================================================
def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_V11 = os.path.join(ROOT, "V11.1_ALT", "src")
cfg = _load_mod("V11_1_ALT_config", os.path.join(_V11, "V11_1_ALT_config.py"))
surv = _load_mod("V11_1_ALT_survival", os.path.join(_V11, "V11_1_ALT_survival.py"))


def conditional_rul_curve(ages, shape_s, scale_s, rng):
    med, lo, hi = [], [], []
    for a in ages:
        d = surv.conditional_predictive_rul(float(a), shape_s, scale_s, rng)
        med.append(float(np.median(d))); lo.append(float(np.percentile(d, 10))); hi.append(float(np.percentile(d, 90)))
    return (np.minimum.accumulate(np.array(med)),
            np.minimum.accumulate(np.array(lo)),
            np.minimum.accumulate(np.array(hi)))


PRECURSORS = [
    ("crank_recovery_t", "crank-recovery t", False),
    ("vsi_ceiling",      "regulation ceiling", True),
    ("resting_vsi_mean", "resting voltage",   True),
    ("vsi_resid_mean",   "voltage residual",  True),
    ("vsi_sag_frac",     "under-voltage sag", False),
]


# =============================================================================
# DATA BUNDLE
# =============================================================================
def build_bundle(vin):
    import polars as pl
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET).set_index("vin_label").loc[vin]
    failed = bool(lc["failed_flag"])
    t0 = pd.Timestamp(lc["alt_t0"]); t1 = pd.Timestamp(lc["alt_t1"])
    jc = pd.Timestamp(JCOPENDATE[vin]) if (failed and vin in JCOPENDATE) else None
    fr = pd.read_csv(os.path.join(cfg.RUL_CACHE, "final_rul_per_vin.csv")).set_index("vin_label").loc[vin]
    fw = json.loads(pathlib.Path(os.path.join(ROOT, "V10.6.2_ALT/cache/rul/fleet_window.json")).read_text())

    ps = pd.read_csv(os.path.join(cfg.WEIBULL_CACHE, "posterior_samples.csv"))
    rng = np.random.default_rng(cfg.RNG_SEED)
    wk = pd.read_parquet(os.path.join(cfg.V10_6_WEEKLY, f"{vin}.parquet"))
    wk["week"] = pd.to_datetime(wk["week"])
    wk = wk[(wk["week"] >= t0) & (wk["week"] <= t1)].sort_values("week")
    weeks = wk["week"].reset_index(drop=True)
    ages = (weeks - t0).dt.days.clip(lower=0).values.astype(float)
    rul_med, rul_p10, rul_p90 = conditional_rul_curve(ages, ps["shape"].values, ps["scale"].values, rng)

    d = pd.read_csv(os.path.join(ROOT, "V11_ALT_heuristics/cache/forensics", f"{vin}_daily.csv"))
    if np.issubdtype(d["day"].dtype, np.number):
        d["date"] = t0 + pd.to_timedelta(d["day"].astype(int), unit="D")
    else:
        d["date"] = pd.to_datetime(d["day"])
    end = jc if (failed and jc is not None) else t1
    if failed:
        d = d[d["date"] <= end]
    d = d.reset_index(drop=True)
    base_mask = ((d["date"] >= t0 + pd.Timedelta(days=40)) &
                 (d["date"] <= end - pd.Timedelta(days=120))).values
    if base_mask.sum() < 10:
        base_mask = (np.arange(len(d)) < max(int(0.4 * len(d)), 5))

    prec = {}
    for col, lab, inv in PRECURSORS:
        if col in d.columns:
            z_raw = zscore_vs_baseline(d[col].values, base_mask, invert=inv)
            z = pd.Series(z_raw).rolling(21, min_periods=10, center=True).mean().bfill().ffill().values
            prec[col] = dict(label=lab, z=z, breach=genuine_breach(z), z_end=float(z[-1]))

    # Authoritative GED2 storm from the V11.1 emergency channel (validated 2/10 F, 0/15 NF)
    emg = pd.read_csv(os.path.join(cfg.EMERG_CACHE, "emergency_per_vin.csv")).set_index("vin_label")
    ged_fired = bool(emg.loc[vin, "ged_fired"]) if (vin in emg.index and "ged_fired" in emg.columns) else False
    _gl = emg.loc[vin, "ged_lead_days"] if (vin in emg.index and "ged_lead_days" in emg.columns) else None
    ged_lead = int(_gl) if (ged_fired and pd.notna(_gl)) else None
    ged_storm_date = (end - pd.Timedelta(days=ged_lead)) if ged_lead is not None else None

    raw = pl.read_parquet(os.path.join(ROOT, f"V5.2_ALT/features/parquets/V5.2_20_5_ALT_{vin}.parquet"))
    raw = raw.select(["DATETIME", "RPM", "VSI"]).to_pandas()
    raw["DATETIME"] = pd.to_datetime(raw["DATETIME"])
    raw = raw[(raw["DATETIME"] >= t0) & (raw["DATETIME"] <= end)]
    span = max((end - t0).days, 1)
    early = raw[raw["DATETIME"] <= t0 + pd.Timedelta(days=int(0.25 * span))]
    late = raw[raw["DATETIME"] >= end - pd.Timedelta(days=int(0.25 * span))]
    chg = dict(early=charging_bins(early["RPM"].values, early["VSI"].values),
               late=charging_bins(late["RPM"].values, late["VSI"].values))

    m5, m5c = compute_m5(d)
    _all_trans = m5_transitions(d["date"].values, m5, m5c)
    final_zone = zone_of_m5(float(m5[-1]))
    _fz = _ZORDER[final_zone]
    m5_trans = [t for t in _all_trans if _ZORDER[t["to"]] <= _fz]   # sustained-to-end only
    _peak = max((_ZORDER[t["to"]] for t in _all_trans), default=0)
    _inv = {v: k for k, v in _ZORDER.items()}
    m5_peak_transient = _inv[_peak] if _peak > _fz else None        # touched then recovered
    rcross = rul_crossings(weeks, rul_med, d, prec, ged_storm_date)

    return dict(vin=vin, dname=display_name(vin), failed=failed, t0=t0, t1=t1, jc=jc, end=end,
                age_now=float(lc["age_days_observed"]),
                weeks=weeks, rul_med=rul_med, rul_p10=rul_p10, rul_p90=rul_p90,
                risk_band=str(fr["risk_band"]), km_per_day=float(fr["km_per_day_est"]),
                median_rul=float(fr["median_rul_days"]), est_km=float(lc["est_km"]),
                fleet_med=float(fw["median_ttf_days"]), fw_p25=float(fw["p25_ttf_days"]),
                fw_p75=float(fw["p75_ttf_days"]),
                daily=d, prec=prec, ged_fired=ged_fired, ged_lead=ged_lead,
                ged_storm_date=ged_storm_date, chg=chg,
                m5=m5, m5_contrib=m5c, m5_trans=m5_trans, m5_final=final_zone,
                m5_peak_transient=m5_peak_transient, rul_cross=rcross)


MODE_COLOR = {"ABRUPT": "#7F8C8D", "SHORT-LEAD": "#E67E22", "GRADUAL": "#7B1FA2"}


def failure_mode(b):
    """(mode, earliest_signal_days) via the earliest GENUINE signal for a failed truck;
    ('', 0) in-service. ABRUPT = no signal, SHORT-LEAD = <=30 d, GRADUAL = > 30 d.
    Mirrors the showcase taxonomy; alternators mostly fail ABRUPTLY."""
    if not b["failed"]:
        return ("", 0)
    end = pd.Timestamp(b["end"]); dts = b["daily"]["date"]
    hard = int(b["ged_lead"]) if (b["ged_fired"] and b["ged_lead"]) else 0
    for info in b["prec"].values():
        if info["breach"] is not None:
            hard = max(hard, (end - pd.Timestamp(dts.iloc[info["breach"]])).days)
    cor = [t for t in b["m5_trans"] if t["to"] in ("ORANGE", "RED")]
    cond = (end - min(t["date"] for t in cor)).days if cor else 0
    earliest = max(hard, cond)
    return ("ABRUPT" if earliest == 0 else "SHORT-LEAD" if earliest <= 30 else "GRADUAL", earliest)


def stamp_mode_badge(ax, b, optimistic=False, band=None):
    """Failure-mode pill in the RUL panel's top-right (failed trucks only).
    optimistic+band lets the optimistic variant render an honest badge for the one
    below-the-line miss (GREEN band) instead of a false 'caught'."""
    if not b["failed"]:
        return
    mode, e = failure_mode(b)
    if optimistic:
        if band is not None and str(band).upper() == "GREEN":
            col, txt = "#E67E22", "△ FLEET'S 1-OF-10 MISS"   # honest: this one was not flagged
        else:
            col = "#2E7D32"
            txt = {"ABRUPT": "✓ CAUGHT BY ML MODEL",
                   "SHORT-LEAD": f"✓ WARNED ~{e} d AHEAD",
                   "GRADUAL": f"✓ ~{e} d CONDITION RUNWAY"}[mode]
    else:
        col = MODE_COLOR[mode]
        txt = {"ABRUPT": "ABRUPT FAILURE · no telemetry warning",
               "SHORT-LEAD": f"SHORT-LEAD · warned ~{e} d before failure",
               "GRADUAL": f"GRADUAL · condition runway ~{e} d"}[mode]
    ax.text(0.986, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            fontweight="bold", color="white", zorder=30,
            bbox=dict(boxstyle="round,pad=0.45", fc=col, ec="white", lw=1.0))


# =============================================================================
# PANELS  (titles set above each axes via set_title -> never overlap data)
# =============================================================================
def panel_rul(ax, ax2, b, optimistic=False):
    weeks, med, p10, p90 = b["weeks"], b["rul_med"], b["rul_p10"], b["rul_p90"]
    ymax = int(np.ceil(max(med.max(), 200) / 50) * 50) + 60
    ylo = -15
    ax.axhspan(180, ymax, color=ZG, alpha=0.06); ax.axhspan(90, 180, color=ZY, alpha=0.07)
    ax.axhspan(30, 90, color=ZO, alpha=0.08); ax.axhspan(ylo, 30, color=ZB, alpha=0.06)
    for y in (180, 90, 30):
        ax.axhline(y, color="#bbb", lw=0.6, ls="--", alpha=0.5)
    ax.fill_between(weeks, p10, p90, color=PRD, alpha=0.10)
    ax.plot(weeks, med, color=PRD, lw=2.4, zorder=6)
    rl = float(med[-1])
    if b["failed"]:
        gap = (b["jc"] - b["t1"]).days
        over = int(round(rl - gap))
        # schedule's last view (open marker): the clock still predicted this many days
        ax.scatter([weeks.iloc[-1]], [rl], marker="o", s=55, facecolor="white",
                   edgecolors="#8B0000", lw=1.6, zorder=8)
        # ACTUAL outcome (dashed = not a predicted decline) to RUL=0 at JCOPENDATE
        ax.plot([weeks.iloc[-1], b["jc"]], [rl, 0], color="#8B0000", lw=1.4,
                ls=(0, (4, 2)), alpha=0.85, zorder=6)
        ax.scatter([b["jc"]], [0], marker="X", s=210, color="#8B0000", edgecolors=DK, lw=1.8, zorder=9)
        if not optimistic:
            ax.annotate(f"Fleet schedule OVER-PREDICTED\n~{rl:.0f} d expected  vs  {gap} d actual  →  {over} d early\n"
                        f"(per-truck RUL is indicative — act on condition + fleet window)",
                        xy=(b["jc"], 0), xytext=(-12, 64), textcoords="offset points", ha="right",
                        fontsize=8, fontweight="bold", color="#8B0000",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#FDECEA", ec="#8B0000", lw=0.9),
                        arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.0))
    elif b["median_rul"] > 0:
        fc = b["t1"] + pd.Timedelta(days=b["median_rul"])
        ax.plot([weeks.iloc[-1], fc], [rl, 0], color="#C0392B", lw=1.4, ls=":", alpha=0.8, zorder=6)
        ax.scatter([fc], [0], marker="X", s=140, color="#C0392B", edgecolors=DK, lw=1.2, zorder=8)
        if not optimistic:
            ax.annotate("fleet-schedule forecast\n(indicative, not per-truck)", xy=(fc, 0), xytext=(-8, 40),
                        textcoords="offset points", ha="right", fontsize=7.5, color="#C0392B",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF5F5", ec="#C0392B", lw=0.6))
    ax.set_ylim(ylo, ymax)
    ax.set_ylabel("Remaining life (days)\nFLEET SCHEDULE (age-based)", fontsize=10, fontweight="bold")
    ax.set_title("1 · Remaining life — FLEET SCHEDULE (age-based average, not per-truck)", **TITLE_KW)
    ax2.set_ylim(ylo, ymax)
    ticks = [t for t in (180, 90, 30) if t <= ymax]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"{t * b['km_per_day'] / 1000:.0f}k km" for t in ticks], fontsize=8, color="#7F8C8D")
    ax2.set_ylabel("est. distance", fontsize=8, color="#7F8C8D", style="italic")
    ax2.spines["top"].set_visible(False)


def draw_rul_crossings(ax, b):
    crosses = b["rul_cross"]
    for k, c in enumerate(crosses):
        ax.scatter([c["date"]], [c["rul"]], marker="v", s=95, color="#9C27B0", edgecolors=DK, lw=1, zorder=9)
        if c["corrob"]:
            top = c["breaching"][0][0] if c["breaching"] else "GED2 storm"
            tag = f"{c['name']} · physics✓ {top}"; fc = "#FDECEA"; ec = "#C0392B"
        else:
            tag = f"{c['name']} · schedule-only"; fc = "#EEF2F5"; ec = "#7F8C8D"
        ax.annotate(tag, xy=(c["date"], c["rul"]), xytext=(-6, 22 + k * 20), textcoords="offset points",
                    fontsize=7, ha="right", color="#333",
                    bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec=ec, lw=0.6),
                    arrowprops=dict(arrowstyle="->", color=ec, lw=0.7))
    # condition-zone transition reference lines (link to Panel 1b) — reference markers ONLY,
    # NOT schedule zone changes; the schedule zones still switch on RUL days (180/90/30).
    yhi = ax.get_ylim()[1]
    for t in b["m5_trans"]:
        ax.axvline(t["date"], color=M5C, lw=0.9, ls=(0, (3, 2)), alpha=0.45, zorder=3)
        ax.annotate(f"cond→{t['to']}", xy=(t["date"], yhi), xytext=(-3, -4), textcoords="offset points",
                    ha="right", va="top", rotation=90, fontsize=6.5, color=M5C, fontweight="bold")


def panel_health_zone(ax, b):
    d = b["daily"]; m5 = b["m5"]
    ax.axhspan(0.55, 1.0, color=ZB, alpha=0.08); ax.axhspan(0.35, 0.55, color=ZO, alpha=0.08)
    ax.axhspan(0.15, 0.35, color=ZY, alpha=0.08); ax.axhspan(0.0, 0.15, color=ZG, alpha=0.08)
    for y in (0.15, 0.35, 0.55):
        ax.axhline(y, color="#bbb", lw=0.6, ls="--", alpha=0.5)
    ax.plot(d["date"], m5, color=M5C, lw=2.0, zorder=5)
    rows = []
    for i, t in enumerate(b["m5_trans"]):
        ax.axvline(t["date"], color=M5C, lw=0.8, ls=":", alpha=0.55)
        ax.scatter([t["date"]], [t["score"]], s=130, color=M5C, edgecolors="white", lw=1.2, zorder=7)
        ax.text(t["date"], t["score"], str(i + 1), color="white", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=8)
        rows.append(f"{i + 1}  {t['date'].strftime('%b %Y')} → {t['to']} · {t['driver']}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Health-zone score\n(parametric M5)", fontsize=10, fontweight="bold")
    ax.set_title("1b · CONDITION — this truck's own signals (parameter-driven health zone)", **TITLE_KW)
    if rows:  # sustained transition list in the empty upper-left (M5 line is low early)
        ax.text(0.012, 0.95, "Sustained zone changes:\n" + "\n".join(rows),
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left", color="#5E35B1",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec=M5C, lw=0.6))
    else:
        ax.text(0.012, 0.90, f"ends {b['m5_final']} — no sustained parametric zone change",
                transform=ax.transAxes, ha="left", fontsize=9.5, color="#888", style="italic")
    if b.get("m5_peak_transient"):
        ax.text(0.99, 0.07, f"(transient peak: {b['m5_peak_transient']}, recovered)",
                transform=ax.transAxes, ha="right", fontsize=7.5, color="#999", style="italic")


def panel_voltage(ax, b):
    d = b["daily"]
    ax.axhspan(27, 29, color=ZG, alpha=0.10)
    ax.axhline(28.2, ls=":", color="#888", lw=0.8)
    ax.axhline(24, ls="--", color=SAGC, lw=1.0)
    if "vsi_p05" in d and "vsi_p95" in d:
        ax.fill_between(d["date"], d["vsi_p05"], d["vsi_p95"], color=VOLT, alpha=0.10)
    ax.plot(d["date"], d["vsi_mean"], color=VOLT, lw=1.7, zorder=5, label="VSI mean")
    if "resting_vsi_mean" in d:
        ax.plot(d["date"], d["resting_vsi_mean"], color=REST, lw=1.1, ls="--", alpha=0.85, label="resting V")
    if "vsi_sag_frac" in d:
        sag = d.loc[d["vsi_sag_frac"] > 0, "date"]
        ax.scatter(sag, np.full(len(sag), 20.6), marker="|", s=22, color=SAGC, alpha=0.7)
    ax.set_ylim(20, 31)
    ax.set_ylabel("Voltage (V)", fontsize=10, fontweight="bold")
    ax.set_title("2 · Voltage health — condition  (band 27–29 V · sag 24 V · ticks = sag days)", **TITLE_KW)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, ncol=2)


def panel_precursors(ax, b, first_fire, optimistic=False):
    d = b["daily"]; dates = d["date"].values
    ax.axhline(2.0, ls="--", color="#999", lw=1.0)
    ax.axhline(0, color="#ddd", lw=0.6)
    handles = []
    for i, (col, info) in enumerate(b["prec"].items()):
        z = info["z"]; col_c = PREC_COLORS[i % len(PREC_COLORS)]
        if info["breach"] is not None:
            ax.plot(dates, z, lw=2.0, color=col_c, zorder=5)
            handles.append(Line2D([], [], color=col_c, lw=2.0, label=info["label"]))
        else:
            ax.plot(dates, z, lw=1.0, color="#c2c2c2", alpha=0.7, zorder=3)
    ytop = 3.2
    if b["ged_storm_date"] is not None:
        ax.axvline(b["ged_storm_date"], color=PRD, lw=1.3, ls="-", alpha=0.55, zorder=6)
        ax.scatter([b["ged_storm_date"]], [ytop - 0.15], marker="v", s=80, color=PRD,
                   edgecolors=DK, lw=0.8, zorder=7)
        handles.append(Line2D([], [], color=PRD, marker="v", lw=0, ms=8,
                              label=f"GED2 storm (~{b['ged_lead']} d)"))
    ax.set_ylim(-1.5, ytop + 0.3)
    ax.set_ylabel("Deterioration\n(σ vs healthy, up=worse)", fontsize=10, fontweight="bold")
    ax.set_title("3 · Precursor signals — physics early-warning  (bold = breaches σ=2; grey = quiet)", **TITLE_KW)
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)
    elif not optimistic:
        ax.text(0.5, 0.5, "No precursor detected (abrupt-failure archetype)",
                transform=ax.transAxes, ha="center", fontsize=11, color="#888", style="italic")


def panel_charging(ax, b):
    ec, em = b["chg"]["early"]; lc, lm = b["chg"]["late"]
    drew = False
    if len(ec) >= 3:
        ax.plot(ec, em, "-o", color=ZG, ms=3.5, lw=1.8, label="early life"); drew = True
    if len(lc) >= 3:
        ax.plot(lc, lm, "-o", color="#B71C1C", ms=3.5, lw=1.8, label="late life"); drew = True
    ax.axhline(27, ls=":", color="#888", lw=0.8)
    ax.set_xlabel("Engine speed (RPM)", fontsize=10, fontweight="bold")
    ax.set_ylabel("VSI (V)", fontsize=10, fontweight="bold")
    ax.set_title("4 · Charging signature (V–RPM) — early vs late life", **TITLE_KW)
    if drew:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9); ax.set_ylim(20, 31)
    else:
        ax.text(0.5, 0.5, "insufficient engine-on data for charging curve",
                transform=ax.transAxes, ha="center", fontsize=10, color="#888", style="italic")


# =============================================================================
# ASSEMBLE
# =============================================================================
def build_figure(vin):
    b = build_bundle(vin)
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

    panel_rul(ax1, ax1b, b); draw_rul_crossings(ax1, b)
    panel_health_zone(axm, b)
    panel_voltage(ax2, b)
    panel_precursors(ax3, b, first_fire)
    panel_charging(ax4, b)

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

    status = "FAILED" if b["failed"] else "in-service"
    fig.suptitle(f"Alternator RUL + Physics Evidence — {b['dname']}", fontsize=18, fontweight="bold",
                 color=DK, x=0.085, ha="left", y=0.975)
    sub = (f"Risk: {b['risk_band'].upper()}   ·   ~{b['est_km']/1000:.0f}k km   ·   "
           f"fleet window {b['fw_p25']:.0f}–{b['fw_p75']:.0f} d (med {b['fleet_med']:.0f})   ·   {status}")
    fig.text(0.085, 0.952, sub, fontsize=10, color="#5D6D7E", style="italic")
    stamp_mode_badge(ax1, b)
    fig.text(0.085, 0.022,
             "Panel 1 (SCHEDULE, age-based fleet average) and Panel 1b (CONDITION, this truck's signals) answer "
             "DIFFERENT questions and diverge when a truck fails off-schedule — every failed truck here failed while "
             "the schedule still read GREEN/YELLOW; the condition signals are what catch it.",
             fontsize=7.5, color="#95A5A6", style="italic")
    fig.text(0.085, 0.008,
             "Panels 1b–4 are this truck's parametric evidence. M5 health-zone is supporting context (can flag "
             "early/transiently). n=25 data ceiling. Confidential.",
             fontsize=7.5, color="#95A5A6", style="italic")

    png = os.path.join(OUT, f"{b['dname']}_evidence_stack.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"{b['dname']}_evidence_stack.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


ALL_VINS = list(cfg.ALL_VINS)


def main():
    ok, skip = [], []
    for vin in ALL_VINS:
        try:
            build_figure(vin); ok.append(vin); print(f"  {vin} -> {display_name(vin)}: saved")
        except Exception as ex:
            skip.append((vin, str(ex))); print(f"  {vin}: SKIP {ex}")
    print(f"\nDone: {len(ok)}/25 figures. Skipped: {skip}")
    assert len(ok) >= 23, f"too many skips: {skip}"


if __name__ == "__main__":
    main()
