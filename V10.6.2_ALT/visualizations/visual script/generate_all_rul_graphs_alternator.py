#!/usr/bin/env python3
"""
Generate Alternator RUL Degradation Graphs for ALL VINs  (executive grade)
==========================================================================
Clutch-aesthetic PORT of generate_all_rul_graphs.py for the V10.6.2 alternator
fleet (n=25: 10 failed + 15 non-failed).  The visual design is intentionally
IDENTICAL to the clutch chart (same 22x12 figure, palette, typography, phase
bands, threshold markers, projection wedge, legend, footer).  Only the DATA and
the SEMANTICS change, because alternators have NO mechanical wear signal.

Honest mapping (clutch -> alternator):
  * Wear-model RUL curve     -> conditional Weibull posterior MEDIAN RUL(age)
  * Deviation band           -> p10-p90 conditional posterior band (funnel)
  * Hermite projection wedge  -> median RUL -> 0 at forecast failure date
  * Wear-mm horizontal zones  -> RUL-horizon zones 180/90/30 d
  * Lifecycle phase bands     -> fleet wear-out window 577/601/652 d (date-mapped)
  * Volatile/wear-accel boxes -> GED2 excitation-storm weeks (verified only)
  * Secondary axis (wear mm)  -> remaining distance (est. km) = RUL x km/day
  * Failure annotation        -> est. km + engine-hrs at actual/forecast failure

Honesty guardrails preserved: the primary curve is a function of AGE + the fleet
survival model (NOT per-truck telemetry), so it is not a fabricated per-truck
wear trend.  Failed trucks land at RUL=0 on their real failure date (event
observed).  GED2 overlay fires ONLY for ged_emergency-verified trucks
(VIN1_F ~21 d, VIN10_F ~1 d) -- a flat signal is honestly flat.

Run with:  py -3 "generate_all_rul_graphs_alternator.py"
"""
import os, json, warnings, importlib.util, pathlib
import numpy as np, pandas as pd
from datetime import datetime
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
warnings.filterwarnings('ignore')

# -- Datestamp for output filenames ----------------------------------------
DATESTAMP = datetime.now().strftime('%Y%m%d')

# -- Paths -----------------------------------------------------------------
#   __file__ = .../V10.6.2_ALT/visualizations/visual script/<this>.py
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
V10_6_2_ROOT = SCRIPT_DIR.parents[1]            # .../V10.6.2_ALT
SRC = V10_6_2_ROOT / 'src'
OUT = V10_6_2_ROOT / 'visualizations' / 'rul_curves_clutch_style'
OUT.mkdir(parents=True, exist_ok=True)


def _load(mod, fn):
    """Import a V10.6.2 source module by path (reuses pipeline math/config)."""
    spec = importlib.util.spec_from_file_location(mod, str(SRC / fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cfg = _load('V10_6_2_ALT_config', 'V10.6.2_ALT_config.py')
surv = _load('V10_6_2_ALT_survival', 'V10.6.2_ALT_survival.py')

# -- Alternator RUL-horizon thresholds (days) ------------------------------
#   Mirror the clutch GREEN/YELLOW/ORANGE/BLACK zoning, but on the RUL axis
#   (higher RUL = healthier).  Boundaries from the deployed 90-day short
#   horizon + service replacement bands.
H_GY = 180.0   # GREEN  / YELLOW boundary
H_YO = 90.0    # YELLOW / ORANGE boundary
H_OB = 30.0    # ORANGE / BLACK  boundary

# -- Color palette (IDENTICAL to the clutch script) ------------------------
DK  = '#1B2838'; SIL = '#C0C6CC'; BG = '#FAFBFC'; GRD = '#E8ECEF'
PRD = '#E8490F'                       # primary RUL curve (red-orange)
ZG  = '#27AE60'; ZY = '#F5A623'; ZO = '#E67E22'; ZB = '#2C3E50'
PBG_G = '#E8F5E9'; PBG_Y = '#FFF8E1'; PBG_O = '#FFF3E0'; PBG_B = '#ECEFF1'
VOL_E = '#FF9800'                     # GED2 excitation-storm outline
C_PROJ = '#B71C1C'                    # projection-to-failure (dark maroon)


# =========================================================================
# HELPERS
# =========================================================================
def fmt_km(km):
    """Abbreviated km format for compact labels."""
    if km >= 1e6:
        return f'{km/1e6:.1f}M'
    elif km >= 10000:
        return f'{km/1000:.0f}k'
    else:
        return f'{km:,.0f}'


def crossing_idx(curve, threshold):
    """First index where a DECLINING curve drops to/below threshold."""
    if threshold is None:
        return None
    below = np.where(np.asarray(curve) <= threshold)[0]
    if len(below) == 0:
        return None
    i = int(below[0])
    return i if (i > 0 or curve[0] <= threshold) else (0 if curve[0] <= threshold else None)


def conditional_rul_curve(ages, shape_s, scale_s, rng):
    """Conditional Weibull posterior RUL (median/p10/p90) at each age.

    R = T - a | T > a, sampled from the fleet posterior (epistemic + aleatoric).
    Enforces a monotone non-increasing curve (residual life of an increasing-
    hazard Weibull decreases with age) to remove Monte-Carlo jitter.
    """
    med, lo, hi = [], [], []
    for a in ages:
        draws = surv.conditional_predictive_rul(float(a), shape_s, scale_s, rng)
        med.append(float(np.median(draws)))
        lo.append(float(np.percentile(draws, 10)))
        hi.append(float(np.percentile(draws, 90)))
    med = np.minimum.accumulate(np.array(med))
    lo  = np.minimum.accumulate(np.array(lo))
    hi  = np.minimum.accumulate(np.array(hi))
    return med, lo, hi


def _project_to_failure(dates, predicted_rul, last_dt, fc_dt):
    """Hermite-cubic projection from the last data point to RUL=0 at fc_dt.

    Copied verbatim (behaviour-preserving) from the clutch generator so the
    projection wedge looks identical.  Uses the recent ~8-snapshot slope to set
    the starting velocity; physically-plausible bounds (>=1x, <=2.5x uniform);
    gentle 1/3 arrival slope.
    """
    T = (fc_dt - last_dt).days
    rul_last = float(predicted_rul[-1])
    if T <= 0 or rul_last <= 0:
        return np.array([]), np.array([])

    n = len(predicted_rul)
    n_recent = min(8, n - 1)
    if n_recent >= 2:
        dt_recent = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[-n_recent - 1])).days
        v_recent = ((float(predicted_rul[-1]) - float(predicted_rul[-n_recent - 1]))
                    / dt_recent) if dt_recent > 0 else (-rul_last / T)
    else:
        v_recent = -rul_last / T
    if v_recent >= 0:
        v_recent = -rul_last / T

    v_uniform = -rul_last / T
    v_start = min(v_recent, 1.0 * v_uniform)
    v_start = max(v_start, 2.5 * v_uniform)
    v_end = v_start / 3.0

    c0, c1 = rul_last, v_start
    A = np.array([[T**2, T**3], [2*T, 3*T**2]])
    b_vec = np.array([-c0 - c1 * T, v_end - c1])
    try:
        c2, c3 = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        c2 = 0.0
        c3 = (-c0 - c1 * T) / T**3 if T > 0 else 0.0

    t_proj = np.arange(1, T + 1, dtype=float)
    rul_proj = np.clip(c0 + c1 * t_proj + c2 * t_proj**2 + c3 * t_proj**3, 0, None)
    proj_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1),
                               periods=len(t_proj), freq='D')
    return proj_dates.values, rul_proj


def burst_regions(flags):
    """Group consecutive True weekly GED2-burst flags into (start, end) runs."""
    regions, start = [], None
    flags = list(flags)
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append((start, i - 1)); start = None
    if start is not None:
        regions.append((start, len(flags) - 1))
    return regions


def zone_legend():
    return [
        Line2D([0], [0], color=ZG, lw=0, marker='s', ms=10, alpha=.3,
               label='GREEN  RUL > 180 d'),
        Line2D([0], [0], color=ZY, lw=0, marker='s', ms=10, alpha=.3,
               label='YELLOW 90-180 d'),
        Line2D([0], [0], color=ZO, lw=0, marker='s', ms=10, alpha=.3,
               label='ORANGE 30-90 d'),
        Line2D([0], [0], color=ZB, lw=0, marker='s', ms=10, alpha=.3,
               label='BLACK  RUL < 30 d'),
    ]


# =========================================================================
# CORE GRAPH -- executive grade (alternator)
# =========================================================================
def create_graph(weeks, rul_med, rul_p10, rul_p90, vin_id,
                 *, failed, age_now, t0, last_dt, fc_dt,
                 median_rul, p10_now, p90_now,
                 fleet_med, fw_p25, fw_p75,
                 ridge_prob, tier, km_per_day, est_km,
                 failure_km, failure_ehrs,
                 ged_burst_dates, lead_days):
    """Executive-grade alternator RUL degradation visualisation."""
    weeks = pd.to_datetime(weeks)
    n = len(weeks)

    fig, ax = plt.subplots(figsize=(22, 12), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # -- X-axis extent -----------------------------------------------------
    right_bound = last_dt
    if fc_dt and fc_dt > right_bound:
        right_bound = fc_dt
    span_days = max((right_bound - weeks[0]).days, 1)
    pad_left  = max(5, int(span_days * 0.03))
    pad_right = max(10, int(span_days * 0.05))
    x_left  = weeks[0] - pd.Timedelta(days=pad_left)
    x_right = right_bound + pd.Timedelta(days=pad_right)

    # -- Y limits (RUL days) ----------------------------------------------
    #   Median curve starts near the unconditional life (~720 d) and declines.
    #   Cap to keep the actionable low-RUL region readable; the wide early band
    #   is clipped (clip_on=True).
    y_max = int(np.ceil(max(rul_med.max(), 200) / 50) * 50) + 60
    y_lo = -15
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_lo, y_max)

    # -- Date <-> age helpers ---------------------------------------------
    def age_to_date(a):
        return t0 + pd.Timedelta(days=float(a))

    # =====================================================================
    # 1. CHRONOLOGICAL PHASE BANDS  (fleet wear-out window, date-mapped)
    # =====================================================================
    pa = 0.055
    # (label, age_start, age_end, text_color, bg_color)
    phase_defs = [
        ('Young\n(<{:.0f}d)'.format(fw_p25),                 0,        fw_p25,  ZG,        PBG_G),
        ('Approaching\nwear-out',                            fw_p25,   fleet_med, '#C8850F', PBG_Y),
        ('In wear-out\nwindow',                              fleet_med, fw_p75,  ZO,        PBG_O),
        ('Past\nwindow',                                     fw_p75,   1e9,     ZB,        PBG_B),
    ]
    x_total_n = mdates.date2num(x_right) - mdates.date2num(x_left)
    legend_cutoff = mdates.date2num(x_left) + x_total_n * 0.82
    max_age = (x_right - t0).days + 1          # clamp open-ended "Past window"
    for lab, a0, a1, clr, bg_c in phase_defs:
        d0 = max(age_to_date(min(a0, max_age)), x_left)
        d1 = min(age_to_date(min(a1, max_age)), x_right)
        if d1 <= d0:
            continue
        ax.axvspan(d0, d1, alpha=pa, color=clr, zorder=0)
        x0n, x1n = mdates.date2num(d0), mdates.date2num(d1)
        if x1n - x0n < 21:
            continue
        if x1n - x0n < 40:           # skip labels on narrow bands (collision)
            continue
        mid_n = (x0n + x1n) / 2
        if mid_n > legend_cutoff:
            continue
        ax.text(mdates.num2date(mid_n), y_max * 0.97, lab,
                fontsize=9.5, fontweight='bold', color=clr,
                ha='center', va='top', alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=bg_c,
                          edgecolor=clr, alpha=0.55, lw=0.9), zorder=15)

    # =====================================================================
    # 2. HORIZONTAL RUL-HORIZON ZONE BANDS (+ right-side labels)
    # =====================================================================
    hz_a = 0.05
    ax.axhspan(H_GY, y_max + 50, color=ZG, alpha=hz_a)
    ax.axhspan(H_YO, H_GY, color=ZY, alpha=hz_a + 0.01)
    ax.axhspan(H_OB, H_YO, color=ZO, alpha=hz_a + 0.015)
    ax.axhspan(y_lo, H_OB, color=ZB, alpha=hz_a + 0.01)
    for v, c in [(H_GY, ZG), (H_YO, ZO), (H_OB, ZB)]:
        ax.axhline(y=v, color=c, lw=0.7, ls='--', alpha=0.30, zorder=1)

    trans_r = blended_transform_factory(ax.transAxes, ax.transData)
    for y_pos, lbl, clr in [((y_max + H_GY) / 2, 'GREEN\n> 180 d', ZG),
                            ((H_GY + H_YO) / 2, 'YELLOW\n90-180', '#C8850F'),
                            ((H_YO + H_OB) / 2, 'ORANGE\n30-90', ZO),
                            (max(H_OB / 2, 6),  'BLACK\n< 30 d', ZB)]:
        ax.text(0.965, y_pos, lbl, fontsize=7.5, fontweight='bold',
                color=clr, alpha=0.55, va='center', ha='left', transform=trans_r)

    # =====================================================================
    # 3. RUL CURVE (conditional posterior median + band) + fleet-clock ref
    # =====================================================================
    # 3a. 80% posterior band (funnel)
    ax.fill_between(weeks, rul_p10, rul_p90, color=PRD, alpha=0.10,
                    zorder=2, clip_on=True, label='_band')
    ax.plot(weeks, rul_p10, color=PRD, lw=0.8, ls=':', alpha=0.45, zorder=3)
    ax.plot(weeks, rul_p90, color=PRD, lw=0.8, ls=':', alpha=0.45, zorder=3)

    # 3b. Primary conditional-median curve
    ax.plot(weeks, rul_med, lw=2.5, color=PRD, zorder=6, alpha=0.95,
            solid_capstyle='round')
    _ms = np.where(rul_med < 40, 40, 22)
    ax.scatter(weeks, rul_med, s=_ms, color=PRD, zorder=7,
               edgecolors='white', linewidths=0.5, alpha=0.78, marker='D')

    # 3d. Projection wedge to forecast failure (non-failed only)
    proj_d, proj_r = (np.array([]), np.array([]))
    if (not failed) and fc_dt and fc_dt > last_dt and rul_med[-1] > 0:
        proj_d, proj_r = _project_to_failure(weeks.values, rul_med, last_dt, fc_dt)
        if len(proj_d) > 0:
            full_d = np.concatenate([[np.datetime64(last_dt)], proj_d])
            full_r = np.concatenate([[rul_med[-1]], proj_r])
            ax.plot(full_d, full_r, lw=2.0, color=C_PROJ, ls=':', alpha=0.9, zorder=7)
            ax.fill_between(full_d, full_r, 0, color=C_PROJ, alpha=0.07, zorder=2)
            n_proj = len(proj_r)
            for frac in [0.25, 0.50, 0.75]:
                qi = min(int(frac * n_proj), n_proj - 1)
                ax.scatter([proj_d[qi]], [proj_r[qi]], s=80, color=C_PROJ,
                           edgecolors='white', lw=1.2, zorder=8, marker='o')
                ax.annotate(f'{proj_r[qi]:.0f}d', xy=(proj_d[qi], proj_r[qi]),
                            xytext=(0, 14), textcoords='offset points',
                            fontsize=9, fontweight='bold', color=C_PROJ,
                            ha='center', va='bottom', zorder=9,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec=C_PROJ, alpha=0.85, lw=0.6))
            ax.scatter([fc_dt], [0], s=250, color=C_PROJ, edgecolors='black',
                       lw=2, zorder=9, marker='X')

    # =====================================================================
    # 4. RUL-HORIZON CROSSINGS (on observed median + projection) + arrows
    # =====================================================================
    # Build the full trajectory so crossings that occur in the projection are
    # still annotated.
    if len(proj_d) > 0:
        traj_d = np.concatenate([weeks.values, proj_d])
        traj_r = np.concatenate([rul_med, proj_r])
    else:
        traj_d = weeks.values; traj_r = rul_med
    rec_specs = [(H_GY, ZY, '#9C27B0', 'Review at next\ndepot visit'),
                 (H_YO, ZO, '#7B1FA2', 'Schedule\ninspection'),
                 (H_OB, ZB, '#4A148C', 'Replace alternator\n(critical)')]
    rec_items = []
    for thr, mclr, rclr, rtext in rec_specs:
        ci = crossing_idx(traj_r, thr)
        if ci is None:
            continue
        xc, yc = traj_d[ci], traj_r[ci]
        ax.scatter([xc], [yc], s=160, color=mclr, edgecolors=DK, lw=1.3,
                   zorder=8, marker='v')
        arrow_top = min(yc + y_max * 0.07, y_max * 0.92)
        ax.annotate('', xy=(xc, yc), xytext=(xc, arrow_top),
                    arrowprops=dict(arrowstyle='->', color=mclr, lw=1.2, alpha=0.45),
                    zorder=7)
        rec_items.append((xc, rclr, rtext))

    rec_y_base = y_max * 0.40; rec_y_step = y_max * 0.13
    for i, (x_pos, clr, rec_text) in enumerate(rec_items):
        ax.axvline(x=x_pos, color=clr, lw=1.3, ls=':', alpha=0.40, zorder=3)
        ax.text(x_pos, rec_y_base + i * rec_y_step, f'  {rec_text}',
                fontsize=7.5, fontweight='semibold', color=clr, va='center',
                ha='left', alpha=0.72, rotation=90, fontstyle='italic', zorder=11)

    # =====================================================================
    # 5. GED2 EXCITATION-STORM OVERLAY (verified precursors only)
    # =====================================================================
    has_ged = len(ged_burst_dates) > 0
    if has_ged:
        # map burst week-dates to nearest snapshot index for vertical placement
        week_nums = mdates.date2num(weeks)
        for (gs, ge) in burst_regions_from_dates(weeks, ged_burst_dates):
            sl = slice(max(0, gs), min(ge + 1, n))
            seg = rul_med[sl]
            if len(seg) == 0:
                continue
            y_lo_r = float(seg.min()); y_hi_r = float(seg.max())
            margin = (y_hi_r - y_lo_r) * 0.12 + 8
            x0 = week_nums[max(0, gs)]; x1 = week_nums[min(ge, n - 1)]
            if x1 <= x0:
                x1 = x0 + 5
            ax.add_patch(Rectangle((x0, y_lo_r - margin), x1 - x0,
                                   (y_hi_r - y_lo_r) + 2 * margin, fill=False,
                                   edgecolor=VOL_E, linestyle=':', linewidth=1.4,
                                   alpha=0.7, zorder=6, clip_on=True))

    # =====================================================================
    # 6. LAST DATA / FORECAST / ACTUAL-FAILURE REFERENCE LINES
    # =====================================================================
    vline_y = 3
    ax.axvline(x=last_dt, color='#5D6D7E', lw=1.0, ls=':', alpha=0.55, zorder=3)
    ld_label = f'Last Data  {last_dt.strftime("%Y-%m-%d")}  |  age {age_now:.0f}d'
    ax.text(last_dt, vline_y, f'  {ld_label}', fontsize=7, fontweight='bold',
            color='#5D6D7E', va='bottom', ha='left', alpha=0.85, rotation=90,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, edgecolor='#5D6D7E',
                      alpha=0.7, lw=0.4))

    if (not failed) and fc_dt and fc_dt > weeks[0]:
        ax.axvline(x=fc_dt, color='#C0392B', lw=1.0, ls=':', alpha=0.55, zorder=3)
        fc_label = (f'  Forecast Failure  {fc_dt.strftime("%Y-%m-%d")}  (fleet est.)\n'
                    f'  Est. {fmt_km(failure_km)} km  |  {failure_ehrs:,.0f} engine-hrs')
        ax.text(fc_dt, vline_y, fc_label, fontsize=7, fontweight='bold',
                color='#C0392B', va='bottom', ha='left', alpha=0.85, rotation=90,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF5F5',
                          edgecolor='#C0392B', alpha=0.7, lw=0.4))

    if failed:
        # Event observed: actual failure at RUL=0 on the last data date.
        ax.axvline(x=last_dt, color='#8B0000', lw=1.2, ls='-.', alpha=0.6, zorder=3)
        ax.plot([last_dt, last_dt], [rul_med[-1], 0], color='#8B0000', lw=1.6,
                ls='-', alpha=0.75, zorder=7)
        ax.scatter([last_dt], [0], s=300, color='#8B0000', edgecolors=DK, lw=2,
                   zorder=10, marker='X')
        lead_txt = (f'  ~{lead_days:.0f} d GED2 lead' if (has_ged and lead_days)
                    else '  no usable precursor')
        km_ehrs_txt = f'\n{fmt_km(failure_km)} km  |  {failure_ehrs:,.0f} engine-hrs'
        ax.annotate(f'EVENT OBSERVED\nactual RUL = 0{lead_txt}{km_ehrs_txt}',
                    xy=(last_dt, 0), xytext=(-12, 60), textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color='#8B0000', ha='right',
                    va='bottom', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#8B0000',
                              alpha=0.9, lw=0.8))

    # =====================================================================
    # 7. SECONDARY Y-AXIS -- remaining distance (est. km)
    # =====================================================================
    ax2 = ax.twinx()
    ax2.set_ylim(y_lo, y_max)
    ticks_data = [(H_GY, f'{fmt_km(H_GY * km_per_day)} km'),
                  (H_YO, f'{fmt_km(H_YO * km_per_day)} km'),
                  (H_OB, f'{fmt_km(H_OB * km_per_day)} km')]
    if not failed and median_rul > 0:
        ticks_data.append((median_rul, f'{fmt_km(median_rul * km_per_day)} km (now)'))
    valid = [(v, l) for v, l in ticks_data if v is not None and v <= y_max]
    ax2.set_yticks([v for v, _ in valid])
    ax2.set_yticklabels([l for _, l in valid], fontsize=9, color='#7F8C8D')
    ax2.tick_params(axis='y', length=4, colors='#7F8C8D')
    ax2.spines['right'].set_color(SIL); ax2.spines['right'].set_linewidth(0.6)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('Remaining distance (est. km)', fontsize=10, color='#7F8C8D',
                   labelpad=10, style='italic')

    # =====================================================================
    # 8. AXIS FORMATTING
    # =====================================================================
    # Adaptive date ticks (spans run ~2-3 yrs; monthly ticks would smear)
    _xloc = mdates.AutoDateLocator(minticks=8, maxticks=14)
    ax.xaxis.set_major_locator(_xloc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_xloc))
    tick_step = max(30, (y_max // 8) // 10 * 10)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.set_xlabel('Timeline', fontsize=13, fontweight='bold', color=DK, labelpad=12)
    ax.set_ylabel('Predicted RUL (Days)', fontsize=13, fontweight='bold',
                  color=DK, labelpad=12)
    ax.grid(True, which='major', axis='y', color=GRD, lw=0.6, alpha=0.8)
    ax.grid(True, which='major', axis='x', color=GRD, lw=0.4, alpha=0.5)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(SIL); ax.spines['bottom'].set_color(SIL)
    ax.tick_params(axis='both', which='major', labelsize=10, colors=DK, length=5)

    # =====================================================================
    # 9. LEGEND
    # =====================================================================
    entries = [
        Line2D([0], [0], color=PRD, lw=2.5, marker='D', ms=4.5,
               markeredgecolor='w', markeredgewidth=0.5,
               label='Conditional RUL - Weibull posterior median'),
        Line2D([0], [0], color=PRD, lw=6, alpha=0.18,
               label='80% posterior band (p10-p90)'),
    ]
    if len(proj_d) > 0:
        entries.append(Line2D([0], [0], color=C_PROJ, lw=2.0, ls=':',
                              label='Projection to forecast failure'))
    entries.append(Line2D([0], [0], color='#5D6D7E', lw=1, ls=':',
                          label='Last Data Date'))
    if failed:
        entries.append(Line2D([0], [0], color='#8B0000', lw=1.2, ls='-.',
                              marker='X', ms=7, label='Actual failure (RUL=0)'))
    elif fc_dt:
        entries.append(Line2D([0], [0], color='#C0392B', lw=1, ls=':',
                              label='Forecast Failure Date'))
    if has_ged:
        entries.append(Line2D([0], [0], color=VOL_E, lw=1.4, ls=':',
                              label='GED2 excitation storm (verified)'))
    entries += zone_legend()
    ax.legend(handles=entries, loc='upper right', fontsize=7.5, framealpha=0.92,
              edgecolor=SIL, fancybox=True, ncol=2,
              bbox_to_anchor=(0.99, 0.92)).get_frame().set_linewidth(0.6)

    # =====================================================================
    # 10. TITLES + FOOTER
    # =====================================================================
    status = 'FAILED' if failed else 'in-service'
    fig.suptitle(f'Alternator RUL Degradation  --  {vin_id}', fontsize=18,
                 fontweight='bold', color=DK, y=0.97)
    if failed:
        sub = (f'Age {age_now:.0f}d  |  {fmt_km(failure_km)} km  |  '
               f'{failure_ehrs:,.0f} eng-hrs  |  Risk {ridge_prob:.2f} ({tier})  |  '
               f'EVENT OBSERVED (RUL=0)  |  Fleet window {fw_p25:.0f}-{fw_p75:.0f}d '
               f'(median {fleet_med:.0f})  |  Status: {status}')
    else:
        sub = (f'Age {age_now:.0f}d  |  Risk {ridge_prob:.2f} ({tier})  |  '
               f'Conditional RUL median {median_rul:.0f}d (p10 {p10_now:.0f} - '
               f'p90 {p90_now:.0f})  |  Est. at failure: {fmt_km(failure_km)} km, '
               f'{failure_ehrs:,.0f} eng-hrs  |  Fleet window {fw_p25:.0f}-{fw_p75:.0f}d '
               f'(median {fleet_med:.0f})  |  Status: {status}')
    ax.set_title(sub, fontsize=10, color='#5D6D7E', style='italic', pad=14)

    fig.text(0.02, 0.012,
             'Daimler Alternator Failure Prediction | V10.6.2 | '
             'GREEN >180d | YELLOW 90-180d | ORANGE 30-90d | BLACK <30d | Confidential',
             fontsize=7.5, color='#95A5A6', style='italic')
    fig.text(0.98, 0.012,
             f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=8, color='#95A5A6', style='italic', ha='right')
    fig.text(0.02, 0.035,
             'RUL = fleet Weibull posterior (n=10 events); per-truck timing not '
             'predictable (backtest 142d > 50d fleet clock). Risk tier = STATIC '
             'classifier (AUROC 0.927 = which, not when). km estimated.',
             fontsize=7, color='#B0B7BD', style='italic')

    plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])
    return fig


def burst_regions_from_dates(weeks, burst_dates):
    """Map a set of GED2-burst week timestamps to (start_idx, end_idx) runs
    over the weekly snapshot index."""
    burst_set = set(pd.to_datetime(burst_dates).normalize())
    flags = [pd.Timestamp(w).normalize() in burst_set for w in weeks]
    return burst_regions(flags)


# =========================================================================
# DISPLAY-NAME MAPPING (unique 1-25 numbering for deliverable clarity)
# =========================================================================
#   Raw data reuses VIN1-VIN10 across failed and non-failed CSVs (different
#   physical trucks).  The pipeline disambiguates with _F_ALT / _NF_ALT, but
#   that confuses deliverable readers.  This map assigns each truck a unique
#   number: failed = VIN1-VIN10, non-failed = VIN11-VIN25.
_DISPLAY_NAME = {}
for _i in range(1, 11):
    _DISPLAY_NAME[f'VIN{_i}_F_ALT'] = f'VIN{_i}_F_ALT'
_NF_SORTED = ([f'VIN{i}_NF_ALT' for i in range(1, 11)]
              + [f'VIN{i}_NF_ALT' for i in range(11, 16)])
for _idx, _nf in enumerate(_NF_SORTED, start=11):
    _DISPLAY_NAME[_nf] = f'VIN{_idx}_NF_ALT'


def display_name(vin_label):
    """Return the unique deliverable-friendly display name for a VIN."""
    return _DISPLAY_NAME.get(vin_label, vin_label)


# =========================================================================
# DATA ASSEMBLY + DRIVER
# =========================================================================
def main():
    print('=' * 70)
    print('Alternator RUL Degradation Graphs -- Executive Grade (All 25 VINs)')
    print('=' * 70)

    # -- Load per-VIN tables ----------------------------------------------
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    lc['alt_t0'] = pd.to_datetime(lc['alt_t0'])
    lc['alt_t1'] = pd.to_datetime(lc['alt_t1'])
    lc = lc.set_index('vin_label')

    fr = pd.read_csv(pathlib.Path(cfg.RUL_CACHE) / 'final_rul_per_vin.csv')
    fr = fr.set_index('vin_label')

    # Fleet wear-out window
    fw = json.loads((pathlib.Path(cfg.RUL_CACHE) / 'fleet_window.json').read_text())
    fleet_med = float(fw['median_ttf_days'])
    fw_p25 = float(fw['p25_ttf_days']); fw_p75 = float(fw['p75_ttf_days'])

    # Weibull posterior samples (shared survival math)
    ps = pd.read_csv(pathlib.Path(cfg.WEIBULL_CACHE) / 'posterior_samples.csv')
    shape_s, scale_s = ps['shape'].values, ps['scale'].values
    rng = np.random.default_rng(cfg.RNG_SEED)

    weekly_dir = pathlib.Path(cfg.V10_6_WEEKLY)

    done = []
    for vin in cfg.ALL_VINS:
        r_lc = lc.loc[vin]; r_fr = fr.loc[vin]
        failed = bool(r_lc['failed_flag'])
        t0 = pd.Timestamp(r_lc['alt_t0']); t1 = pd.Timestamp(r_lc['alt_t1'])
        age_now = float(r_lc['age_days_observed'])

        # Weekly snapshots within [t0, t1]
        wk = pd.read_parquet(weekly_dir / f'{vin}.parquet')
        wk['week'] = pd.to_datetime(wk['week'])
        wk = wk[(wk['week'] >= t0) & (wk['week'] <= t1)].sort_values('week')
        if len(wk) < 3:
            print(f'  {vin}: only {len(wk)} weekly snapshots -- skipped')
            continue
        weeks = wk['week'].reset_index(drop=True)
        ages = (weeks - t0).dt.days.clip(lower=0).values.astype(float)

        # Conditional posterior RUL curve + band
        rul_med, rul_p10, rul_p90 = conditional_rul_curve(ages, shape_s, scale_s, rng)

        # Current-age numbers (authoritative, from the pipeline CSV)
        median_rul = float(r_fr['median_rul_days'])
        p10_now = float(r_fr['rul_p10_days']); p90_now = float(r_fr['rul_p90_days'])
        km_per_day = float(r_fr['km_per_day_est'])
        ehrs_per_day = float(r_fr['ehrs_per_day_est'])
        ridge_prob = float(r_fr['ridge_prob'])
        tier = (str(r_fr['risk_tier']) if pd.notna(r_fr['risk_tier'])
                and str(r_fr['risk_tier']).strip() else str(r_fr['ridge_band']))
        ged_emergency = bool(r_fr['ged_emergency'])
        wflag = r_fr.get('would_have_flagged_lead_days', None)
        try:
            lead_days = float(wflag)
        except (ValueError, TypeError):
            lead_days = None

        # Anchor the curve endpoint to the pipeline's current median RUL so the
        # chart agrees with the deliverable tables (NF only; failed = event).
        if not failed and median_rul > 0:
            rul_med[-1] = median_rul
            rul_med = np.minimum.accumulate(rul_med)
            rul_p10[-1] = min(rul_p10[-1], p10_now)
            rul_p90[-1] = max(rul_p90[-1], min(p90_now, rul_p90[-1]) if rul_p90[-1] > 0 else p90_now)

        # Forecast failure date (NF only)
        fc_dt = (t1 + pd.Timedelta(days=median_rul)) if (not failed and median_rul > 0) else None

        # Estimated km and engine-hours at failure (actual or forecast)
        current_km = float(r_lc['est_km'])
        current_ehrs = float(r_lc['est_engine_hrs'])
        if failed:
            failure_km = current_km
            failure_ehrs = current_ehrs
        else:
            failure_km = current_km + median_rul * km_per_day
            failure_ehrs = current_ehrs + median_rul * ehrs_per_day

        # GED2 storm weeks -- verified precursors only (ged_emergency gate)
        ged_burst_dates = []
        if ged_emergency and 'ged_weekly_burst_flag' in wk.columns:
            ged_burst_dates = list(wk.loc[wk['ged_weekly_burst_flag'] == True, 'week'])

        dname = display_name(vin)

        fig = create_graph(
            weeks, rul_med, rul_p10, rul_p90, dname,
            failed=failed, age_now=age_now, t0=t0, last_dt=t1, fc_dt=fc_dt,
            median_rul=median_rul, p10_now=p10_now, p90_now=p90_now,
            fleet_med=fleet_med, fw_p25=fw_p25, fw_p75=fw_p75,
            ridge_prob=ridge_prob, tier=tier, km_per_day=km_per_day,
            est_km=current_km, failure_km=failure_km, failure_ehrs=failure_ehrs,
            ged_burst_dates=ged_burst_dates, lead_days=lead_days)

        for ext in ['png', 'svg']:
            p = OUT / f'Alternator_RUL_Degradation_{dname}_{DATESTAMP}.{ext}'
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor=BG,
                        format=ext if ext == 'svg' else None)
        plt.close(fig)
        done.append(dname)
        print(f'  {vin} -> {dname}: saved ({len(wk)} weeks, '
              f'{"FAILED" if failed else f"RUL {median_rul:.0f}d"}'
              f'{", GED2 storm" if ged_burst_dates else ""})')

    print('\n' + '=' * 70)
    print(f'Done! {len(done)} VINs x 2 formats = {len(done) * 2} files')
    print(f'Output: {OUT}')
    print('=' * 70)


if __name__ == '__main__':
    main()
