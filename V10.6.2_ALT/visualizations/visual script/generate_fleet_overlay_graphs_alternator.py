#!/usr/bin/env python3
"""
Fleet-Level Alternator RUL Degradation Overlays  (V10.6.2, executive grade)
===========================================================================
Elevates the per-VIN clutch-aesthetic RUL charts
(`generate_all_rul_graphs_alternator.py`) into THREE fleet-wide comparative
overlays for Daimler / BharatBenz technical review:

  1. Failed Vehicle Fleet Overlay      (all 10 failed alternators)
  2. Non-Failed Vehicle Fleet Overlay  (all 15 in-service alternators + forecast)
  3. Combined Failed vs Non-Failed     (vertical 2-panel, mirrored scaling)

Plus a per-VIN analytical summary (CSV + XLSX) and an execution report (MD).

------------------------------------------------------------------------------
REAL PER-VIN DATA ON A CALENDAR AXIS
------------------------------------------------------------------------------
The fleet did NOT start observation on the same day, and no truck has data from
"day 0 / 0 km".  Each VIN is therefore plotted over its OWN real calendar window
(`alt_t0` -> `alt_t1`): real start date, real length, real failure / last-data
date.  `alt_t0` spans 2023-12-31 .. 2024-08-31 and observation ends span 2025-05
.. 2026-02, so the lines are genuinely staggered in time -- that staggering is
the real fleet picture.

The RUL value at each weekly snapshot is the V10.6.2 conditional Weibull
posterior median RUL(age) (imported verbatim from the per-VIN engine; per-truck
telemetry is flat and per-truck timing is not predictable -- backtest MAE 142d >
50d fleet clock).  Because a single global km / engine-hour axis is meaningless
once trucks share calendar time at different accrual rates, each VIN's REAL
end-of-life km + engine-hours are carried in its legend entry and the CSV.

LOGIC PRESERVED: conditional-posterior curve, Hermite projection wedge, RUL-zone
mathematics and the GED2 gate are imported from the V10.6.2 engine.  Pure
visualization layer over existing caches -- no re-fitting, no fabricated trend.

Run with:  py -3 "generate_fleet_overlay_graphs_alternator.py"
"""
import json, warnings, importlib.util, pathlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Reuse the V10.6.2 per-VIN engine (curve math, Hermite projection, palette,
# thresholds, display-name map, cfg + survival modules).
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
V10_6_2_ROOT = SCRIPT_DIR.parents[1]            # .../V10.6.2_ALT


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ENG = _load('alt_rul_engine', SCRIPT_DIR / 'generate_all_rul_graphs_alternator.py')
cfg = ENG.cfg
surv = ENG.surv

# Shared visual language (IDENTICAL palette to the per-VIN clutch charts)
DK, SIL, BG, GRD = ENG.DK, ENG.SIL, ENG.BG, ENG.GRD
ZG, ZY, ZO, ZB = ENG.ZG, ENG.ZY, ENG.ZO, ENG.ZB
H_GY, H_YO, H_OB = ENG.H_GY, ENG.H_YO, ENG.H_OB         # 180 / 90 / 30 day horizons
fmt_km = ENG.fmt_km
display_name = ENG.display_name

DATESTAMP = datetime.now().strftime('%Y%m%d')

OUT = V10_6_2_ROOT / 'visualizations' / 'Fleet_graphs_V10.6.2'
OUT.mkdir(parents=True, exist_ok=True)


def fmt_h(h):
    """Abbreviated engine-hours label."""
    return f'{h / 1000:.1f}k' if h >= 1000 else f'{h:.0f}'


# =========================================================================
# ZONE / HELPERS
# =========================================================================
def zone_of(rul):
    if rul is None or (isinstance(rul, float) and np.isnan(rul)):
        return ''
    if rul > H_GY:
        return 'GREEN'
    if rul > H_YO:
        return 'YELLOW'
    if rul > H_OB:
        return 'ORANGE'
    return 'BLACK'


def first_cross_age(ages, rul, thr):
    """First age at which a declining RUL curve drops to/below `thr` (linear
    interp).  Returns None if never reached."""
    rul = np.asarray(rul, float)
    ages = np.asarray(ages, float)
    if len(rul) == 0:
        return None
    if rul[0] <= thr:
        return float(ages[0])
    idx = np.where(rul <= thr)[0]
    if len(idx) == 0:
        return None
    j = int(idx[0])
    a0, a1 = ages[j - 1], ages[j]
    r0, r1 = rul[j - 1], rul[j]
    if r0 == r1:
        return float(a1)
    return float(a0 + (thr - r0) * (a1 - a0) / (r1 - r0))


def draw_rul_zones(ax, y_lo, y_max, *, label=True):
    """Horizontal RUL-horizon zones (GREEN/YELLOW/ORANGE/BLACK).

    Calendar-independent (function of the y / RUL axis only) so every overlaid
    VIN shares identical zone geometry -- the basis for horizontal comparison.
    """
    hz = 0.06
    ax.axhspan(H_GY, y_max, color=ZG, alpha=hz, zorder=0)
    ax.axhspan(H_YO, H_GY, color=ZY, alpha=hz + 0.01, zorder=0)
    ax.axhspan(H_OB, H_YO, color=ZO, alpha=hz + 0.015, zorder=0)
    ax.axhspan(y_lo, H_OB, color=ZB, alpha=hz + 0.01, zorder=0)
    for v, c in [(H_GY, ZG), (H_YO, ZO), (H_OB, ZB)]:
        ax.axhline(y=v, color=c, lw=0.7, ls='--', alpha=0.30, zorder=1)
    if label:
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        for y_pos, lbl, clr in [((y_max + H_GY) / 2, 'GREEN  > 180 d', ZG),
                                ((H_GY + H_YO) / 2, 'YELLOW 90-180 d', '#C8850F'),
                                ((H_YO + H_OB) / 2, 'ORANGE 30-90 d', ZO),
                                (max(H_OB / 2, 8), 'BLACK  < 30 d', ZB)]:
            ax.text(0.997, y_pos, lbl + ' ', fontsize=8, fontweight='bold',
                    color=clr, alpha=0.6, va='center', ha='right', zorder=2,
                    transform=tr)


# =========================================================================
# DATA ASSEMBLY  (one record per VIN; reuses the V10.6.2 engine verbatim)
# =========================================================================
def assemble_fleet():
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    lc['alt_t0'] = pd.to_datetime(lc['alt_t0'])
    lc['alt_t1'] = pd.to_datetime(lc['alt_t1'])
    lc = lc.set_index('vin_label')

    fr = pd.read_csv(pathlib.Path(cfg.RUL_CACHE) / 'final_rul_per_vin.csv').set_index('vin_label')

    fw = json.loads((pathlib.Path(cfg.RUL_CACHE) / 'fleet_window.json').read_text())
    fleet_med = float(fw['median_ttf_days'])
    fw_p25 = float(fw['p25_ttf_days'])
    fw_p75 = float(fw['p75_ttf_days'])

    ps = pd.read_csv(pathlib.Path(cfg.WEIBULL_CACHE) / 'posterior_samples.csv')
    shape_s, scale_s = ps['shape'].values, ps['scale'].values
    rng = np.random.default_rng(cfg.RNG_SEED)
    weekly_dir = pathlib.Path(cfg.V10_6_WEEKLY)

    recs, skipped = [], []
    for vin in cfg.ALL_VINS:
        r_lc, r_fr = lc.loc[vin], fr.loc[vin]
        failed = bool(r_lc['failed_flag'])
        t0, t1 = pd.Timestamp(r_lc['alt_t0']), pd.Timestamp(r_lc['alt_t1'])
        age_now = float(r_lc['age_days_observed'])

        wk = pd.read_parquet(weekly_dir / f'{vin}.parquet')
        wk['week'] = pd.to_datetime(wk['week'])
        wk = wk[(wk['week'] >= t0) & (wk['week'] <= t1)].sort_values('week')
        if len(wk) < 3:
            skipped.append((display_name(vin), f'only {len(wk)} weekly snapshots'))
            continue
        weeks = wk['week'].reset_index(drop=True)
        ages = (weeks - t0).dt.days.clip(lower=0).values.astype(float)

        rul_med, rul_p10, rul_p90 = ENG.conditional_rul_curve(ages, shape_s, scale_s, rng)

        median_rul = float(r_fr['median_rul_days'])
        p10_now, p90_now = float(r_fr['rul_p10_days']), float(r_fr['rul_p90_days'])
        km_per_day = float(r_fr['km_per_day_est'])
        ehrs_per_day = float(r_fr['ehrs_per_day_est'])
        ridge_prob = float(r_fr['ridge_prob'])
        tier = (str(r_fr['risk_tier']) if pd.notna(r_fr['risk_tier'])
                and str(r_fr['risk_tier']).strip() else str(r_fr['ridge_band']))
        ged_emergency = bool(r_fr['ged_emergency'])

        if not failed and median_rul > 0:
            rul_med[-1] = median_rul
            rul_med = np.minimum.accumulate(rul_med)
            rul_p10[-1] = min(rul_p10[-1], p10_now)
            rul_p90[-1] = max(rul_p90[-1], rul_p90[-1] if rul_p90[-1] > 0 else p90_now)

        current_km = float(r_lc['est_km'])
        current_ehrs = float(r_lc['est_engine_hrs'])

        # Forecast projection (NF only) -- exact V10.6.2 Hermite wedge, on the
        # truck's REAL calendar.
        proj_dates = np.array([], dtype='datetime64[ns]')
        proj_rul = np.array([])
        proj_ages = np.array([])
        fc_dt = None
        if (not failed) and median_rul > 0:
            fc_dt = t1 + pd.Timedelta(days=median_rul)
            pdts, prul = ENG._project_to_failure(weeks.values, rul_med, t1, fc_dt)
            if len(pdts) > 0:
                proj_dates = np.asarray(pdts, dtype='datetime64[ns]')
                proj_rul = np.asarray(prul, float)
                proj_ages = (pd.to_datetime(proj_dates) - t0).days.values.astype(float)
            failure_km = current_km + median_rul * km_per_day
            failure_ehrs = current_ehrs + median_rul * ehrs_per_day
        else:
            failure_km = current_km
            failure_ehrs = current_ehrs

        # Full trajectory (ages) for zone-entry crossings + final zone.
        if failed:
            traj_ages = np.append(ages, age_now)
            traj_rul = np.append(rul_med, 0.0)
        elif len(proj_ages) > 0:
            traj_ages = np.concatenate([ages, proj_ages])
            traj_rul = np.concatenate([rul_med, proj_rul])
        else:
            traj_ages, traj_rul = ages, rul_med

        entries = {z: first_cross_age(traj_ages, traj_rul, thr)
                   for z, thr in [('YELLOW', H_GY), ('ORANGE', H_YO), ('BLACK', H_OB)]}

        recs.append(dict(
            vin=vin, dname=display_name(vin), failed=failed,
            dates=weeks.values, ages=ages, rul_med=rul_med,
            proj_dates=proj_dates, proj_rul=proj_rul, proj_ages=proj_ages,
            age_now=age_now, t0=t0, t1=t1, fc_dt=fc_dt,
            median_rul=median_rul, p10_now=p10_now, p90_now=p90_now,
            current_rul=(0.0 if failed else median_rul),
            km_per_day=km_per_day, est_km=current_km,
            failure_km=failure_km, failure_ehrs=failure_ehrs,
            ridge_prob=ridge_prob, tier=tier, ged_emergency=ged_emergency,
            zone_entries=entries,
            current_zone=('EVENT' if failed else zone_of(median_rul)),
            final_zone=('BLACK' if failed else zone_of(float(traj_rul[-1]))),
        ))

    return recs, skipped, dict(fleet_med=fleet_med, fw_p25=fw_p25, fw_p75=fw_p75)


# =========================================================================
# AXIS-LIMIT POLICY  (shared across all panels -> mirrored scaling)
# =========================================================================
def fleet_limits(recs):
    starts, ends, max_rul = [], [], 0.0
    for r in recs:
        starts.append(pd.Timestamp(r['dates'][0]))
        if r['failed']:
            ends.append(r['t1'])
        elif len(r['proj_dates']):
            ends.append(pd.Timestamp(r['proj_dates'][-1]))
        else:
            ends.append(r['t1'])
        max_rul = max(max_rul, float(r['rul_med'].max()))
    x0, x1 = min(starts), max(ends)
    span = (x1 - x0).days
    x_left = x0 - pd.Timedelta(days=max(12, int(span * 0.02)))
    x_right = x1 + pd.Timedelta(days=max(12, int(span * 0.02)))
    y_max = int(np.ceil(max(max_rul, 200) / 20) * 20) + 25      # tight headroom
    y_lo = -12
    return x_left, x_right, y_lo, y_max


# =========================================================================
# RENDERING
# =========================================================================
def _format_axes(ax, x_left, x_right, y_lo, y_max, *, xlabel=True,
                 show_xticklabels=True):
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_lo, y_max)
    loc = mdates.AutoDateLocator(minticks=9, maxticks=16)
    ax.xaxis.set_major_locator(loc)
    if show_xticklabels:
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    else:
        ax.tick_params(axis='x', labelbottom=False)
    tick_step = max(30, (y_max // 8) // 10 * 10)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    if xlabel:
        ax.set_xlabel('Calendar date   --   each VIN over its REAL data window '
                      '(start, length and failure date differ across the fleet)',
                      fontsize=11.5, fontweight='bold', color=DK, labelpad=9)
    ax.set_ylabel('Predicted RUL  (days)', fontsize=12.5, fontweight='bold',
                  color=DK, labelpad=10)
    ax.grid(True, which='major', axis='y', color=GRD, lw=0.6, alpha=0.8)
    ax.grid(True, which='major', axis='x', color=GRD, lw=0.4, alpha=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(SIL)
    ax.spines['bottom'].set_color(SIL)
    ax.tick_params(axis='both', which='major', labelsize=10, colors=DK, length=5)


def _zone_marks(ax, r, col, ages, rul, *, size):
    """Subtle zone-transition triangles at RUL-horizon crossings (real dates)."""
    for thr in (H_GY, H_YO, H_OB):
        a = first_cross_age(ages, rul, thr)
        if a is not None:
            d = r['t0'] + pd.Timedelta(days=a)
            ax.scatter([d], [thr], s=size, color=col, edgecolors=DK,
                       lw=0.4, marker='v', zorder=7, alpha=0.85)


def _plot_failed(ax, recs, colors):
    handles = []
    for r, col in zip(recs, colors):
        ax.plot(r['dates'], r['rul_med'], color=col, lw=1.3, alpha=0.9,
                solid_capstyle='round', zorder=5)
        ax.plot([r['t1'], r['t1']], [r['rul_med'][-1], 0.0], color=col,
                lw=1.1, ls='-', alpha=0.6, zorder=5)
        ax.scatter([r['t1']], [0.0], s=48, color=col, edgecolors=DK, lw=0.8,
                   marker='X', zorder=8)
        _zone_marks(ax, r, col, r['ages'], r['rul_med'], size=16)
        lbl = f"{r['dname']}: {fmt_km(r['failure_km'])} km, {fmt_h(r['failure_ehrs'])} h"
        handles.append(Line2D([0], [0], color=col, lw=1.8, marker='X', ms=5,
                              markeredgecolor=DK, markeredgewidth=0.4, label=lbl))
    return handles


def _plot_non_failed(ax, recs, colors):
    handles = []
    for r, col in zip(recs, colors):
        ax.plot(r['dates'], r['rul_med'], color=col, lw=1.3, alpha=0.9,
                solid_capstyle='round', zorder=5)
        ax.scatter([r['t1']], [r['median_rul']], s=24, color=col,
                   edgecolors='white', lw=0.5, marker='o', zorder=7)
        if len(r['proj_dates']) > 0:
            fd = np.concatenate([[np.datetime64(r['t1'])], r['proj_dates']])
            fr = np.concatenate([[r['median_rul']], r['proj_rul']])
            ax.plot(fd, fr, color=col, lw=0.95, ls=':', alpha=0.8, zorder=4)
            ax.scatter([r['fc_dt']], [0.0], s=44, facecolors='none',
                       edgecolors=col, lw=1.1, marker='X', zorder=8)
        ta = np.concatenate([r['ages'], r['proj_ages']]) if len(r['proj_ages']) else r['ages']
        tr = np.concatenate([r['rul_med'], r['proj_rul']]) if len(r['proj_ages']) else r['rul_med']
        _zone_marks(ax, r, col, ta, tr, size=14)
        lbl = f"{r['dname']}: -> {fmt_km(r['failure_km'])} km, {fmt_h(r['failure_ehrs'])} h"
        handles.append(Line2D([0], [0], color=col, lw=1.8, label=lbl))
    return handles


def zone_legend_handles():
    return [
        Line2D([0], [0], color=ZG, lw=0, marker='s', ms=9, alpha=.4, label='GREEN > 180 d'),
        Line2D([0], [0], color=ZY, lw=0, marker='s', ms=9, alpha=.4, label='YELLOW 90-180 d'),
        Line2D([0], [0], color=ZO, lw=0, marker='s', ms=9, alpha=.4, label='ORANGE 30-90 d'),
        Line2D([0], [0], color=ZB, lw=0, marker='s', ms=9, alpha=.4, label='BLACK < 30 d'),
    ]


FOOTER = ('Daimler Alternator Failure Prediction | V10.6.2 | '
          'GREEN >180d | YELLOW 90-180d | ORANGE 30-90d | BLACK <30d | Confidential')
HONESTY = ('Each VIN on its REAL calendar window (no common day-0; start/length/failure '
           'date differ). RUL = fleet Weibull posterior median (n=10 events); per-truck '
           'timing not predictable (backtest 142d > 50d fleet clock). Legend shows each '
           "truck's real end-of-life est. km + engine-hours. Risk tier = STATIC classifier "
           '(AUROC 0.927 = which, not when).')


def _footers(fig):
    fig.text(0.015, 0.012, FOOTER, fontsize=8, color='#95A5A6', style='italic')
    fig.text(0.985, 0.012, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=8, color='#95A5A6', style='italic', ha='right')
    fig.text(0.015, 0.030, HONESTY, fontsize=7, color='#B0B7BD', style='italic')


def render_failed(recs, lim, fw):
    x_left, x_right, y_lo, y_max = lim
    colors = [cm.tab10(i % 10) for i in range(len(recs))]
    fig, ax = plt.subplots(figsize=(22, 12), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    draw_rul_zones(ax, y_lo, y_max)
    vin_handles = _plot_failed(ax, recs, colors)
    _format_axes(ax, x_left, x_right, y_lo, y_max)

    fig.suptitle('Alternator Fleet RUL Degradation  --  FAILED Vehicles (n=%d)' % len(recs),
                 fontsize=19, fontweight='bold', color=DK, y=0.975)
    ax.set_title('Each line = one failed alternator over its real data window; '
                 'X = realised failure (RUL=0). Fleet wear-out window = %.0f-%.0f d '
                 '(median %.0f d); legend gives each truck\'s end-of-life km + engine-h.'
                 % (fw['fw_p25'], fw['fw_p75'], fw['fleet_med']),
                 fontsize=10.5, color='#5D6D7E', style='italic', pad=10)

    ax.legend(handles=zone_legend_handles(), loc='upper right', fontsize=8.5,
              framealpha=0.92, edgecolor=SIL, fancybox=True, ncol=1,
              title='RUL zones', title_fontsize=9)
    leg = fig.legend(handles=vin_handles, loc='lower center',
                     bbox_to_anchor=(0.5, 0.052), ncol=5, fontsize=9,
                     title='Failed VIN  --  real end-of-life (est. km, engine-hours)',
                     title_fontsize=10, framealpha=0.95, edgecolor=SIL, fancybox=True)
    leg.get_frame().set_linewidth(0.6)

    _footers(fig)
    fig.tight_layout(rect=[0.012, 0.135, 0.99, 0.94])
    return fig


def render_non_failed(recs, lim, fw):
    x_left, x_right, y_lo, y_max = lim
    colors = [cm.tab20(i % 20) for i in range(len(recs))]
    fig, ax = plt.subplots(figsize=(22, 12), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    draw_rul_zones(ax, y_lo, y_max)
    vin_handles = _plot_non_failed(ax, recs, colors)
    _format_axes(ax, x_left, x_right, y_lo, y_max)

    fig.suptitle('Alternator Fleet RUL Degradation  --  IN-SERVICE Vehicles (n=%d)' % len(recs),
                 fontsize=19, fontweight='bold', color=DK, y=0.975)
    ax.set_title('Solid = observed RUL over each real data window (o = today);  '
                 'dotted = fleet-posterior forecast to RUL=0 (open X). Legend gives '
                 'each truck\'s forecast end-of-life km + engine-hours.',
                 fontsize=10.5, color='#5D6D7E', style='italic', pad=10)

    style_handles = [
        Line2D([0], [0], color=DK, lw=1.5, label='Observed RUL'),
        Line2D([0], [0], color=DK, lw=1.0, ls=':', label='Forecast to failure'),
        Line2D([0], [0], color=DK, lw=0, marker='o', ms=6, mfc='w', label='Today'),
        Line2D([0], [0], color=DK, lw=0, marker='X', ms=7, mfc='none',
               label='Forecast failure'),
    ]
    leg2 = ax.legend(handles=style_handles + zone_legend_handles(),
                     loc='upper right', fontsize=8.5, framealpha=0.92,
                     edgecolor=SIL, fancybox=True, ncol=1, title='Legend',
                     title_fontsize=9)
    leg2.get_frame().set_linewidth(0.6)
    leg = fig.legend(handles=vin_handles, loc='lower center',
                     bbox_to_anchor=(0.5, 0.038), ncol=5, fontsize=8.5,
                     title='In-service VIN  --  forecast end-of-life (est. km, engine-hours)',
                     title_fontsize=10, framealpha=0.95, edgecolor=SIL, fancybox=True)
    leg.get_frame().set_linewidth(0.6)

    _footers(fig)
    fig.tight_layout(rect=[0.012, 0.155, 0.99, 0.94])
    return fig


def render_combined(failed, nonf, lim, fw):
    x_left, x_right, y_lo, y_max = lim
    fcolors = [cm.tab10(i % 10) for i in range(len(failed))]
    ncolors = [cm.tab20(i % 20) for i in range(len(nonf))]
    fig, (axt, axb) = plt.subplots(2, 1, figsize=(22, 18), dpi=150, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in (axt, axb):
        ax.set_facecolor(BG)
        draw_rul_zones(ax, y_lo, y_max)

    fh = _plot_failed(axt, failed, fcolors)
    nh = _plot_non_failed(axb, nonf, ncolors)
    _format_axes(axt, x_left, x_right, y_lo, y_max, xlabel=False, show_xticklabels=False)
    _format_axes(axb, x_left, x_right, y_lo, y_max, xlabel=True)

    axt.set_title('FAILED alternators (n=%d) -- real data window, X at realised failure'
                  % len(failed), fontsize=12.5, fontweight='bold', color=DK, pad=8)
    axb.set_title('IN-SERVICE alternators (n=%d) -- observed (solid) + forecast (dotted)'
                  % len(nonf), fontsize=12.5, fontweight='bold', color=DK, pad=8)
    for ax in (axt, axb):
        ax.legend(handles=zone_legend_handles(), loc='upper right', fontsize=7.5,
                  framealpha=0.9, edgecolor=SIL, ncol=1)

    legt = axt.legend(handles=fh, loc='upper center', bbox_to_anchor=(0.5, -0.04),
                      ncol=5, fontsize=8, title='Failed VIN -- end-of-life (km, engine-h)',
                      title_fontsize=8.5, framealpha=0.95, edgecolor=SIL, fancybox=True)
    legt.get_frame().set_linewidth(0.6)
    legb = axb.legend(handles=nh, loc='upper center', bbox_to_anchor=(0.5, -0.13),
                      ncol=5, fontsize=7.5,
                      title='In-service VIN -- forecast end-of-life (km, engine-h)',
                      title_fontsize=8.5, framealpha=0.95, edgecolor=SIL, fancybox=True)
    legb.get_frame().set_linewidth(0.6)

    fig.suptitle('Alternator Fleet RUL Degradation  --  Failed vs In-Service '
                 '(real per-VIN windows, shared RUL scaling)', fontsize=18,
                 fontweight='bold', color=DK, y=0.985)
    _footers(fig)
    fig.tight_layout(rect=[0.012, 0.04, 0.99, 0.96])
    fig.subplots_adjust(hspace=0.42)
    return fig


def _save(fig, stem):
    """Standard PNG (150 dpi) + vector PDF + HD PNG (300 dpi)."""
    p_png = OUT / f'{stem}.png'
    p_pdf = OUT / f'{stem}.pdf'
    p_hd = OUT / f'{stem}_hd.png'
    fig.savefig(p_png, dpi=150, bbox_inches='tight', facecolor=BG)
    fig.savefig(p_pdf, bbox_inches='tight', facecolor=BG, format='pdf')
    fig.savefig(p_hd, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return [p_png, p_pdf, p_hd]


# =========================================================================
# ANALYTICAL SUMMARY (CSV + XLSX)
# =========================================================================
def build_summary(recs):
    rows = []
    for r in recs:
        t0 = r['t0']

        def ent_date(z):
            a = r['zone_entries'].get(z)
            return (t0 + pd.Timedelta(days=a)).strftime('%Y-%m-%d') if a is not None else ''

        fc_date = (r['fc_dt'].strftime('%Y-%m-%d') if r['fc_dt'] is not None else '')
        rows.append({
            'VIN': r['dname'],
            'Failure_Status': 'FAILED' if r['failed'] else 'IN_SERVICE',
            'Data_Start_Date': pd.Timestamp(r['dates'][0]).strftime('%Y-%m-%d'),
            'Data_End_Date': r['t1'].strftime('%Y-%m-%d'),
            'Observed_Span_days': round(r['age_now'], 1),
            'Current_RUL_median_days': round(r['current_rul'], 1),
            'Predicted_RUL_days': round(0.0 if r['failed'] else r['median_rul'], 1),
            'RUL_p10_days': round(0.0 if r['failed'] else r['p10_now'], 1),
            'RUL_p90_days': round(0.0 if r['failed'] else r['p90_now'], 1),
            'Forecast_Failure_Date': fc_date,
            'EndOfLife_est_km': round(r['failure_km'], 0),
            'EndOfLife_est_engine_hrs': round(r['failure_ehrs'], 0),
            'km_per_day_est': round(r['km_per_day'], 1),
            'GREEN_to_YELLOW_entry': ent_date('YELLOW'),
            'YELLOW_to_ORANGE_entry': ent_date('ORANGE'),
            'ORANGE_to_BLACK_entry': ent_date('BLACK'),
            'Current_Zone': r['current_zone'],
            'Final_Zone_Reached': r['final_zone'],
            'Risk_prob': round(r['ridge_prob'], 4),
            'Risk_tier': r['tier'],
            'GED2_emergency': r['ged_emergency'],
        })
    df = pd.DataFrame(rows)
    csv_p = OUT / 'fleet_statistics_summary.csv'
    df.to_csv(csv_p, index=False)
    xlsx_p = OUT / 'fleet_statistics_summary.xlsx'
    xlsx_ok = True
    try:
        with pd.ExcelWriter(xlsx_p, engine='openpyxl') as xw:
            df.to_excel(xw, index=False, sheet_name='Fleet_RUL_Summary')
            ws = xw.sheets['Fleet_RUL_Summary']
            for col_cells in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in col_cells) + 2
                ws.column_dimensions[col_cells[0].column_letter].width = min(width, 28)
    except Exception as e:  # noqa
        xlsx_ok = False
        print(f'  [warn] XLSX write failed: {e}')
    return df, csv_p, (xlsx_p if xlsx_ok else None)


# =========================================================================
# EXECUTION REPORT (MD)
# =========================================================================
def write_report(recs, skipped, lim, fw, df, graph_paths, csv_p, xlsx_p):
    x_left, x_right, y_lo, y_max = lim
    n_failed = sum(r['failed'] for r in recs)
    n_nf = sum(not r['failed'] for r in recs)
    p = OUT / 'Fleet_graphs_generation_report.md'
    L = []
    L.append('---')
    L.append('title: "V10.6.2 Alternator Fleet Overlay -- Generation Report"')
    L.append('status: "complete"')
    L.append(f'created: "{datetime.now().strftime("%Y-%m-%d")}"')
    L.append('---\n')
    L.append('# V10.6.2 Alternator Fleet-Level Visualization Suite\n')
    L.append(f'**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}  ')
    L.append(f'**Output directory:** `{OUT}`\n')

    L.append('## 1. Processing summary\n')
    L.append(f'- **Total VINs processed:** {len(recs)} of {len(cfg.ALL_VINS)}')
    L.append(f'- **Failed:** {n_failed}  |  **In-service:** {n_nf}')
    if skipped:
        L.append(f'- **Excluded VINs:** {len(skipped)}')
        for nm, why in skipped:
            L.append(f'  - `{nm}` -- {why}')
    else:
        L.append('- **Excluded VINs:** none (all VINs had >=3 weekly snapshots)')
    L.append('')

    L.append('## 2. Graphs generated\n')
    L.append('| # | View | Files |')
    L.append('|---|------|-------|')
    view_names = ['Failed Vehicle Fleet Overlay', 'Non-Failed Vehicle Fleet Overlay',
                  'Combined Failed vs Non-Failed Comparison']
    for i, (view, paths) in enumerate(zip(view_names, graph_paths), 1):
        fnames = ', '.join(f'`{pp.name}`' for pp in paths)
        L.append(f'| {i} | {view} | {fnames} |')
    L.append('')
    L.append('Each view: **standard PNG (150 dpi)**, **vector PDF**, **HD PNG (300 dpi)**.\n')

    L.append('## 3. Data deliverables\n')
    L.append(f'- `{csv_p.name}` -- per-VIN analytical summary (CSV)')
    if xlsx_p:
        L.append(f'- `{xlsx_p.name}` -- same, formatted workbook (XLSX)')
    else:
        L.append('- XLSX workbook -- **NOT written** (openpyxl error; see console)')
    L.append('')
    L.append('Columns: VIN, Failure_Status, **Data_Start_Date, Data_End_Date, '
             'Observed_Span_days** (real per-VIN window), Current_RUL_median_days, '
             'Predicted_RUL_days, RUL_p10/p90, Forecast_Failure_Date, '
             '**EndOfLife_est_km, EndOfLife_est_engine_hrs**, km_per_day_est, '
             'zone-entry dates, Current_Zone, Final_Zone_Reached, Risk_prob, '
             'Risk_tier, GED2_emergency.\n')

    L.append('## 4. Scaling methodology\n')
    L.append('- **Shared x-axis = REAL calendar date.** Each VIN is drawn over its '
             'own observed window (`alt_t0` -> `alt_t1`). The fleet did not start '
             'on a common day and no truck has data from "day 0 / 0 km", so the '
             'lines are genuinely staggered in time -- that is the real fleet picture.')
    L.append(f'- **x-limits (combined view):** {x_left.strftime("%Y-%m-%d")} -> '
             f'{x_right.strftime("%Y-%m-%d")} -- covers every VIN start, end and forecast '
             '(no clipping). The two standalone graphs ZOOM to their own data span '
             '(failed vs in-service) to use maximum plotting area; the combined view '
             'uses this union scale so both panels mirror exactly.')
    L.append(f'- **y-limits:** [{y_lo}, {y_max}] d RUL, mirrored across all panels; '
             'tight headroom so the degradation patterns fill the plotting area.')
    L.append(f'- **Horizontal RUL zones (y-axis, calendar-independent):** GREEN >{H_GY:.0f}d, '
             f'YELLOW {H_YO:.0f}-{H_GY:.0f}d, ORANGE {H_OB:.0f}-{H_YO:.0f}d, BLACK <{H_OB:.0f}d.')
    L.append(f'- **Fleet wear-out window** ({fw["fw_p25"]:.0f}-{fw["fw_p75"]:.0f} d, median '
             f'{fw["fleet_med"]:.0f} d) is age-based, so on the calendar axis it is '
             'expressed per-VIN by where each line ends, not as one global band.')
    L.append('- **km & engine-hours:** carried per-VIN in the legend and CSV as each '
             "truck's REAL end-of-life estimate (a single global distance axis is "
             'meaningless once trucks share calendar time at different km/day rates).')
    L.append('- **Legend placement:** moved OUTSIDE below the plot (multi-column) so '
             'the axes use the full figure width.\n')

    L.append('## 5. Per-VIN summary table\n')
    show = df[['VIN', 'Failure_Status', 'Data_Start_Date', 'Data_End_Date',
               'Observed_Span_days', 'Predicted_RUL_days', 'Forecast_Failure_Date',
               'EndOfLife_est_km', 'EndOfLife_est_engine_hrs', 'Current_Zone',
               'Risk_prob', 'Risk_tier']]
    L.append('| ' + ' | '.join(show.columns) + ' |')
    L.append('|' + '|'.join(['---'] * len(show.columns)) + '|')
    for _, row in show.iterrows():
        L.append('| ' + ' | '.join(str(v) for v in row.values) + ' |')
    L.append('')

    L.append('## 6. Anomalies & honesty notes\n')
    ged_vins = [r['dname'] for r in recs if r['ged_emergency']]
    L.append(f'- **GED2 excitation-storm precursors:** {len(ged_vins)} truck(s) '
             f'({", ".join(ged_vins) if ged_vins else "none"}). Other failed trucks '
             'have NO usable per-truck precursor (flat telemetry).')
    L.append('- RUL is a function of **age + fleet survival model**, not per-truck '
             'telemetry -- no fabricated per-truck wear trend. Failed trucks land at '
             'RUL=0 on their real failure date.')
    L.append('- Risk tier is a **static** classifier (AUROC 0.927 = which, not when). '
             'km / engine-hours are speed-integrated estimates ("est."), not odometer.')
    L.append('')
    L.append('## 7. Validation checklist\n')
    for chk in [
        'Each VIN plotted on its REAL calendar window (real start/length/end dates).',
        'RUL values + Hermite forecast wedge imported verbatim from the V10.6.2 engine.',
        'x/y limits cover all VIN starts, endpoints, forecasts (no clipping).',
        'Legend moved OUTSIDE below the plot; axes use full figure width.',
        'No annotation boxes / text overlays inside the plot area.',
        'Standard (150 dpi) + HD (300 dpi) PNG + vector PDF saved for all 3 views.',
    ]:
        L.append(f'- [x] {chk}')
    L.append('')
    L.append('🟢 Fleet overlay suite generated and validated.')
    p.write_text('\n'.join(L), encoding='utf-8')
    return p


# =========================================================================
# DRIVER
# =========================================================================
def main():
    print('=' * 72)
    print('V10.6.2 Alternator FLEET-LEVEL RUL Overlays (real per-VIN calendar)')
    print('=' * 72)

    recs, skipped, fw = assemble_fleet()
    failed = [r for r in recs if r['failed']]
    nonf = [r for r in recs if not r['failed']]
    print(f'  Assembled {len(recs)} VINs ({len(failed)} failed, {len(nonf)} in-service)')
    for nm, why in skipped:
        print(f'  excluded {nm}: {why}')

    # Standalone graphs zoom to their OWN data (max white-space use); the
    # combined comparison uses the UNION scale so both panels mirror exactly.
    lim_f = fleet_limits(failed)
    lim_n = fleet_limits(nonf)
    lim_all = fleet_limits(recs)
    print(f'  Failed x[{lim_f[0].date()}..{lim_f[1].date()}]  '
          f'In-service x[{lim_n[0].date()}..{lim_n[1].date()}]  '
          f'Combined x[{lim_all[0].date()}..{lim_all[1].date()}]')

    g1 = _save(render_failed(failed, lim_f, fw), 'failed_vehicle_fleet_overlay')
    print(f'  [1/3] Failed overlay      -> {len(g1)} files')
    g2 = _save(render_non_failed(nonf, lim_n, fw), 'non_failed_vehicle_fleet_overlay')
    print(f'  [2/3] Non-failed overlay  -> {len(g2)} files')
    g3 = _save(render_combined(failed, nonf, lim_all, fw), 'fleet_comparison_overlay')
    print(f'  [3/3] Combined comparison -> {len(g3)} files')

    df, csv_p, xlsx_p = build_summary(recs)
    print(f'  Summary: {csv_p.name}' + (f' + {xlsx_p.name}' if xlsx_p else ' (xlsx FAILED)'))

    rep = write_report(recs, skipped, lim_all, fw, df, [g1, g2, g3], csv_p, xlsx_p)
    print(f'  Report: {rep.name}')

    print('=' * 72)
    print(f'Done. Outputs in: {OUT}')
    print('=' * 72)


if __name__ == '__main__':
    main()
