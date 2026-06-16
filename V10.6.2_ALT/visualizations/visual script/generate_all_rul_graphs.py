#!/usr/bin/env python3
"""
Generate RUL Degradation Graphs for ALL VINs  (v4 -- executive grade)
=====================================================================
Executive-grade enhancements over v3:
  - Chronological lifecycle phase segmentation (vertical timeline bands)
  - Volatile driving behavior highlighting (dotted boundary regions)
  - Predictive recommendation annotations at prognostic reference lines
  - Early Warning Lead Time visualization band
  - Removed clutter boxes; kept arrows, markers, deviation band
  - End-state info card (clean bottom-right panel)
  - Professional typography and spacing for OEM presentations
"""
import os, warnings, numpy as np, pandas as pd
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

# -- Paths ----------------------------------------------------------------
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT    = os.path.dirname(BASE)
META    = os.path.join(BASE, 'data', 'V10_5_rul_sample_metadata.csv')
V11_EVT = os.path.join(ROOT, 'V11_RERUN_SET3_FINAL',
                       'V11_RERUN_engagement_events.parquet')
V11_XLS = os.path.join(ROOT, 'V11_RERUN_SET3_FINAL',
                       'V11_RERUN_Failure_prediction_DAIMLER_results.xlsx')
V11_SCORES = os.path.join(ROOT, 'V11_RERUN_SET3_FINAL',
                          'V11_RERUN_vin_scores.csv')
OUT     = os.path.join(BASE, 'reports', 'graphs')
os.makedirs(OUT, exist_ok=True)

# -- Daimler thresholds ---------------------------------------------------
TRIGGER = 4.50
W_GY, W_YO, W_OB = 3.15, 4.05, 4.50

# -- Color palette --------------------------------------------------------
DK  = '#1B2838'; SIL = '#C0C6CC'; BG = '#FAFBFC'; GRD = '#E8ECEF'
ACTC = '#6BAED6'; PRD = '#E8490F'; ABG = '#FFFFFF'
ZG  = '#27AE60'; ZY = '#F5A623'; ZO = '#E67E22'; ZB = '#2C3E50'

# Phase band backgrounds
PBG_G = '#E8F5E9'; PBG_Y = '#FFF8E1'; PBG_O = '#FFF3E0'; PBG_B = '#ECEFF1'
PBG_EW = '#F3E5F5'

# Volatile behavior
VOL_F = '#FFE0B2'; VOL_E = '#FF9800'; VOL_T = '#E65100'

# Wear-rate acceleration
WACC_E = '#B39DDB'  # light purple


# =========================================================================
# HELPERS
# =========================================================================
def interp_rul_at_wear(wear, rul, thresh):
    idx = np.where(wear >= thresh)[0]
    if len(idx) == 0: return None
    i = idx[0]
    if i == 0: return float(rul[0])
    f = (thresh - wear[i-1]) / (wear[i] - wear[i-1]) if wear[i] != wear[i-1] else 0
    return float(rul[i-1] + f * (rul[i] - rul[i-1]))

def zone_thresholds(wear, rul):
    return {k: interp_rul_at_wear(wear, rul, t)
            for k, t in [('gy', W_GY), ('yo', W_YO), ('ob', W_OB)]}

def crossing_idx(curve, threshold):
    """First index where curve drops to or below threshold."""
    if threshold is None:
        return None
    idx = np.argmax(curve <= threshold)
    return idx if (idx > 0 or curve[0] <= threshold) else None


def _project_to_failure(dates, predicted_rul, last_dt, fc_dt):
    """Project RUL to failure continuing the vehicle's recent trajectory.

    Uses the actual degradation slope from the last ~8 weeks of observed
    data to set the Hermite starting velocity.  The projection shape
    reflects genuine driving behaviour:
      - Steady-wear vehicles  → near-linear projection
      - Accelerating-wear     → naturally concave
      - Decelerating-wear     → clamped to at least linear

    Bounds:
      - Floor: 1.0× uniform rate (can't be slower than straight-line)
      - Cap:   2.5× uniform rate (physical plausibility ceiling)
      - Arrival slope: 1/3 of starting (natural, not extreme contrast)
    """
    T = (fc_dt - last_dt).days
    rul_last = float(predicted_rul[-1])
    if T <= 0 or rul_last <= 0:
        return np.array([]), np.array([])

    n = len(predicted_rul)

    # ── Recent slope from last ~8 weeks of actual RUL curve ───────────
    n_recent = min(8, n - 1)
    if n_recent >= 2:
        dt_recent = (pd.Timestamp(dates[-1])
                     - pd.Timestamp(dates[-n_recent - 1])).days
        if dt_recent > 0:
            v_recent = (float(predicted_rul[-1])
                        - float(predicted_rul[-n_recent - 1])) / dt_recent
        else:
            v_recent = -rul_last / T
    else:
        v_recent = -rul_last / T

    if v_recent >= 0:
        v_recent = -rul_last / T          # must be declining

    # ── Physically plausible bounds (no artificial steepening) ────────
    v_uniform = -rul_last / T
    v_start = min(v_recent, 1.0 * v_uniform)   # at least linear
    v_start = max(v_start, 2.5 * v_uniform)    # cap at 2.5×

    # Arrival slope: 1/3 of starting (gentle landing)
    v_end = v_start / 3.0

    # ── Hermite cubic: r(0)=rul_last, r'(0)=v_start, r(T)=0, r'(T)=v_end
    c0, c1 = rul_last, v_start
    A = np.array([[T**2, T**3], [2*T, 3*T**2]])
    b_vec = np.array([-c0 - c1 * T, v_end - c1])
    try:
        c2, c3 = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        c2 = 0.0
        c3 = (-c0 - c1 * T) / T**3 if T > 0 else 0.0

    t_proj = np.arange(1, T + 1, dtype=float)
    rul_proj = c0 + c1 * t_proj + c2 * t_proj**2 + c3 * t_proj**3
    rul_proj = np.clip(rul_proj, 0, None)

    proj_dates = pd.date_range(
        start=last_dt + pd.Timedelta(days=1), periods=len(t_proj), freq='D')
    return proj_dates.values, rul_proj


def _load_vin_scores():
    """Load V11 vin scores CSV for terminal RUL anchoring."""
    if os.path.exists(V11_SCORES):
        df = pd.read_csv(V11_SCORES)
        return {r['vin']: r.to_dict() for _, r in df.iterrows()}
    return {}

_VIN_SCORES = _load_vin_scores()


def _fix_terminal_rul(wear_predicted_rul, wear_final, vin):
    """Anchor terminal RUL to pipeline RHE for sub-trigger VINs.

    The wear-model formula produces RUL=0 at the last data point for ALL VINs
    because max_w = wear[-1].  For sub-trigger VINs (wear < 4.50mm) that still
    have 55-170 days remaining, this is wrong.  We taper-shift the curve so the
    endpoint matches the known RHE value while preserving the curve shape.
    """
    if wear_final >= TRIGGER:
        return wear_predicted_rul  # trigger breached -- RUL=0 is correct

    scores = _VIN_SCORES.get(vin, {})
    if not scores:
        return wear_predicted_rul

    rhe = float(scores.get('RHE', 0))
    physics_rul = float(scores.get('physics_rul_days', 0))
    ml_rul = float(scores.get('ml_rul', 0))

    # Use RHE; if suspiciously low (ML compression), prefer physics
    terminal_rul = rhe
    if terminal_rul < 30 and physics_rul > 90:
        terminal_rul = physics_rul

    if terminal_rul <= 0:
        return wear_predicted_rul

    offset = terminal_rul - wear_predicted_rul[-1]
    if offset > 0:
        taper = np.linspace(0, 1, len(wear_predicted_rul))
        wear_predicted_rul = wear_predicted_rul + offset * taper

    return wear_predicted_rul


def detect_volatile_regions(wear_mm, min_length=3):
    """Detect volatile driving patterns from raw wear accumulation data.

    Pure data-driven detection -- no model assumptions:
      1. Compute week-over-week wear rate (delta_wear = physical measurement)
      2. Apply Tukey upper fence (Q3 + 1.5 * IQR) on positive wear rates
      3. Only flag regions with >= min_length consecutive outlier weeks

    This reflects genuine accelerated-wear episodes in the measured data.
    """
    n = len(wear_mm)
    if n < 12:
        return []

    delta = np.maximum(np.diff(wear_mm), 0.0)
    pos = delta[delta > 0]
    if len(pos) < 6:
        return []

    q1, q3 = np.percentile(pos, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-10:
        return []
    fence = q3 + 1.5 * iqr

    # Flag weeks whose wear rate exceeds the Tukey upper fence
    flagged = np.zeros(n, dtype=bool)
    for i, d in enumerate(delta):
        if d > fence:
            flagged[i] = True
            flagged[i + 1] = True

    # Group contiguous flagged points (require >= min_length)
    regions, start = [], None
    for i, v in enumerate(flagged):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_length:
                regions.append((start, i - 1))
            start = None
    if start is not None and n - start >= min_length:
        regions.append((start, n - 1))
    return regions

def detect_wear_acceleration(wear_mm, min_length=2):
    """Detect periods where wear rate is sustained above the vehicle baseline.

    Pure data-driven: computes per-week wear increment, takes the vehicle's
    median rate as baseline, and flags consecutive weeks where the rate
    exceeds 1.5x that baseline -- indicating sustained aggressive driving
    or abnormal clutch loading.
    """
    n = len(wear_mm)
    if n < 8:
        return []

    delta = np.maximum(np.diff(wear_mm), 0.0)
    pos = delta[delta > 0]
    if len(pos) < 4:
        return []

    baseline = np.median(pos)
    if baseline < 1e-10:
        return []
    threshold = baseline * 1.5  # 50 % above median

    flagged = np.zeros(n, dtype=bool)
    for i, d in enumerate(delta):
        if d > threshold:
            flagged[i] = True
            flagged[min(i + 1, n - 1)] = True

    regions, start = [], None
    for i, v in enumerate(flagged):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_length:
                regions.append((start, i - 1))
            start = None
    if start is not None and n - start >= min_length:
        regions.append((start, n - 1))
    return regions

def fmt_km(km):
    """Abbreviated km format for compact labels."""
    if km >= 1e6:
        return f'{km/1e6:.1f}M'
    elif km >= 10000:
        return f'{km/1000:.0f}k'
    else:
        return f'{km:,.0f}'

def zone_legend():
    return [
        Line2D([0],[0], color=ZG, lw=0, marker='s', ms=10, alpha=.3,
               label='GREEN  <3.15mm (CWS<70%)'),
        Line2D([0],[0], color=ZY, lw=0, marker='s', ms=10, alpha=.3,
               label='YELLOW 3.15-4.05mm (CWS 70-90%)'),
        Line2D([0],[0], color=ZO, lw=0, marker='s', ms=10, alpha=.3,
               label='ORANGE 4.05-4.50mm (CWS 90-100%)'),
        Line2D([0],[0], color=ZB, lw=0, marker='s', ms=10, alpha=.3,
               label='BLACK  >=4.50mm (Trigger)'),
    ]


# =========================================================================
# CORE GRAPH -- v4 (executive grade)
# =========================================================================
def create_graph(dates, predicted_rul, wear_mm, vin_id,
                 actual_rul=None, show_actual=True,
                 subtitle='', status='', wear_final=0, rul_pred=0,
                 forecast_date='', last_data_date='',
                 wear_zone='', action='',
                 is_blind=False, n_snapshots=0,
                 failure_date_known='',
                 odo_at_dates=None, forecast_km=None,
                 show_wear_accel=False,
                 bw_mode=False):
    """Executive-grade RUL degradation visualisation."""

    # In BW mode every text element uses pure black
    _tk = '#000000' if bw_mode else None  # text-color override

    fig, ax = plt.subplots(figsize=(22, 12), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # -- Parse key dates ---------------------------------------------------
    last_dt = pd.Timestamp(last_data_date) if last_data_date else pd.Timestamp(dates[-1])
    fc_dt   = pd.Timestamp(forecast_date)  if forecast_date  else None
    fail_dt = pd.Timestamp(failure_date_known) if failure_date_known else None

    # -- X-axis extent -----------------------------------------------------
    right_bound = last_dt
    if fc_dt   and fc_dt   > right_bound: right_bound = fc_dt
    if fail_dt and fail_dt > right_bound: right_bound = fail_dt
    span_days = (right_bound - pd.Timestamp(dates[0])).days
    pad_left  = max(5, int(span_days * 0.03))
    pad_right = max(10, int(span_days * 0.05))
    x_left  = pd.Timestamp(dates[0]) - pd.Timedelta(days=pad_left)
    x_right = right_bound + pd.Timedelta(days=pad_right)

    # -- Y limits ----------------------------------------------------------
    y_vals = predicted_rul.copy()
    y_max = int(np.ceil(max(y_vals.max(), 10) / 30) * 30) + 40

    # -- Zone thresholds ---------------------------------------------------
    ref_curve = predicted_rul
    z  = zone_thresholds(wear_mm, ref_curve)
    gy, yo, ob = z.get('gy'), z.get('yo'), z.get('ob')

    # -- Crossing indices on main reference curve --------------------------
    main_curve = ref_curve
    n = len(dates)
    idx_gy = crossing_idx(main_curve, gy)
    idx_yo = crossing_idx(main_curve, yo)
    idx_ob = crossing_idx(main_curve, ob)

    # Set axis limits early so date-axis is configured before axvspan calls
    y_lo = -10 if (ob is not None and idx_ob is not None) else -5
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_lo, y_max)

    # =====================================================================
    # 1. CHRONOLOGICAL PHASE SEGMENTATION (vertical timeline bands)
    # =====================================================================
    pa = 0 if bw_mode else 0.055
    phase_items = []  # (label, x_start, x_end, text_color, bg_color)
    has_km = odo_at_dates is not None and len(odo_at_dates) == n

    def _km_tag(i_start, i_end):
        """Build km range suffix for phase labels."""
        if not has_km:
            return ''
        ks = odo_at_dates[min(i_start, n - 1)]
        if i_end is not None and i_end < n:
            ke = odo_at_dates[i_end]
        elif forecast_km:
            ke = forecast_km
        else:
            ke = odo_at_dates[-1]
        return f'\n({fmt_km(ks)} -- {fmt_km(ke)} km)'

    if idx_gy is not None:
        # Healthy phase -> Yellow crossing
        ax.axvspan(dates[0], dates[idx_gy], alpha=pa, color=ZG, zorder=0)
        phase_items.append((f'Healthy Phase{_km_tag(0, idx_gy)}',
                            dates[0], dates[idx_gy], ZG, PBG_G))

        next_b = dates[idx_yo] if idx_yo is not None else x_right
        ax.axvspan(dates[idx_gy], next_b, alpha=pa + 0.02, color=ZY, zorder=0)
        phase_items.append((f'Yellow Zone{_km_tag(idx_gy, idx_yo)}',
                            dates[idx_gy], next_b, '#C8850F', PBG_Y))

        if idx_yo is not None:
            next_b2 = dates[idx_ob] if idx_ob is not None else x_right
            ax.axvspan(dates[idx_yo], next_b2, alpha=pa + 0.03, color=ZO, zorder=0)
            phase_items.append((f'Orange Zone{_km_tag(idx_yo, idx_ob)}',
                                dates[idx_yo], next_b2, ZO, PBG_O))

            if idx_ob is not None:
                ax.axvspan(dates[idx_ob], x_right, alpha=pa + 0.02, color=ZB, zorder=0)
                phase_items.append((f'Trigger Breach{_km_tag(idx_ob, None)}',
                                    dates[idx_ob], x_right, ZB, PBG_B))
    else:
        # Entirely healthy
        ax.axvspan(dates[0], x_right, alpha=pa, color=ZG, zorder=0)
        phase_items.append((f'Healthy Phase{_km_tag(0, None)}',
                            dates[0], x_right, ZG, PBG_G))

    # Phase labels at top of graph (skip labels in rightmost 20% to avoid
    # legend overlap; also skip very narrow phases)
    x_total_n = mdates.date2num(x_right) - mdates.date2num(x_left)
    legend_cutoff = mdates.date2num(x_left) + x_total_n * 0.82
    for label, x_s, x_e, clr, bg_c in phase_items:
        x_s_n = mdates.date2num(pd.Timestamp(x_s))
        x_e_n = mdates.date2num(pd.Timestamp(x_e))
        if x_e_n - x_s_n < 21:
            continue  # too narrow for a label
        mid_n = (x_s_n + x_e_n) / 2
        if mid_n > legend_cutoff:
            continue  # would overlap with legend
        mid = mdates.num2date(mid_n)
        ax.text(mid, y_max * 0.97, label,
                fontsize=9.5, fontweight='bold', color=_tk or clr,
                ha='center', va='top', alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=bg_c,
                          edgecolor=_tk or clr, alpha=0.55, lw=0.9),
                zorder=15)

    # (Early Warning Lead Time band removed -- requires two curves)

    # =====================================================================
    # 2. SUBTLE HORIZONTAL ZONE BANDS (RUL-axis reference)
    # =====================================================================
    hz_a = 0 if bw_mode else 0.035
    if gy is not None:
        ax.axhspan(gy, y_max + 50, color=ZG, alpha=hz_a)
    else:
        ax.axhspan(0, y_max + 50, color=ZG, alpha=hz_a)
    if gy is not None and yo is not None:
        ax.axhspan(yo, gy, color=ZY, alpha=hz_a + 0.01)
    if yo is not None and ob is not None:
        ax.axhspan(ob, yo, color=ZO, alpha=hz_a + 0.015)
    if ob is not None:
        ax.axhspan(-10, ob, color=ZB, alpha=hz_a + 0.01)
    for v, c in [(gy, ZG), (yo, ZO), (ob, ZB)]:
        if v:
            ax.axhline(y=v, color=c, lw=0.7, ls='--', alpha=0.25, zorder=1)

    # Right-side zone labels
    trans_r = blended_transform_factory(ax.transAxes, ax.transData)
    zd = []
    if gy is not None:
        zd.append(((y_max + gy) / 2, 'GREEN\n<3.15 mm', ZG))
    else:
        zd.append((y_max * 0.7, 'GREEN\n<3.15 mm', ZG))
    if gy is not None and yo is not None:
        zd.append(((gy + yo) / 2, 'YELLOW\n3.15-4.05', '#C8850F'))
    if yo is not None and ob is not None:
        zd.append((max((yo + ob) / 2, 15), 'ORANGE\n4.05-4.50', ZO))
    if ob is not None:
        zd.append((max(ob / 2, 6), 'BLACK\n>=4.50', ZB))
    for y_pos, lbl, clr in zd:
        ax.text(0.965, y_pos, lbl, fontsize=7.5, fontweight='bold',
                color=_tk or clr, alpha=0.50, va='center', ha='left',
                transform=trans_r)

    # =====================================================================
    # 3. RUL CURVES (actual/forecast + predicted)
    # =====================================================================
    ax.plot(dates, predicted_rul, lw=2.5, color=PRD, zorder=5,
            alpha=0.9, solid_capstyle='round')
    _ms = np.where(predicted_rul < 30, 40, 22)  # larger markers in dense low-RUL region
    ax.scatter(dates, predicted_rul, s=_ms, color=PRD, zorder=6,
               edgecolors='white', linewidths=0.5, alpha=0.75, marker='D')

    # 3b. Curved projection from last data point to forecast failure (RUL=0)
    if fc_dt and fc_dt > last_dt and predicted_rul[-1] > 0:
        _proj_c = _tk or '#B71C1C'   # dark maroon (contrast with main curve)
        proj_d, proj_r = _project_to_failure(dates, predicted_rul, last_dt, fc_dt)
        if len(proj_d) > 0:
            full_d = np.concatenate([[last_dt], proj_d])
            full_r = np.concatenate([[predicted_rul[-1]], proj_r])
            # Thin dotted line tracing the full Hermite curve
            ax.plot(full_d, full_r, lw=2.0, color=_proj_c, ls=':',
                    alpha=0.88, zorder=7)
            # Shaded wedge under projection
            ax.fill_between(full_d, full_r, 0, color=_proj_c, alpha=0.08,
                            zorder=2)
            # Quarter markers with RUL value labels (makes decline explicit)
            n_proj = len(proj_r)
            q_dates = [last_dt]
            q_ruls  = [predicted_rul[-1]]
            for frac in [0.25, 0.50, 0.75]:
                qi = min(int(frac * n_proj), n_proj - 1)
                q_dates.append(proj_d[qi])
                q_ruls.append(proj_r[qi])
                ax.scatter([proj_d[qi]], [proj_r[qi]], s=80, color=_proj_c,
                           edgecolors='white', lw=1.2, zorder=8, marker='o')
                ax.annotate(f'{proj_r[qi]:.0f}d',
                            xy=(proj_d[qi], proj_r[qi]),
                            xytext=(0, 14), textcoords='offset points',
                            fontsize=9, fontweight='bold', color=_proj_c,
                            ha='center', va='bottom', zorder=9,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec=_proj_c, alpha=0.85, lw=0.6))
            q_dates.append(fc_dt)
            q_ruls.append(0)
            # Thin dashed line connecting start → quarters → failure
            ax.plot(q_dates, q_ruls, lw=1.8, color=_proj_c, ls='--',
                    alpha=0.75, zorder=7, solid_capstyle='round')
            ax.scatter([fc_dt], [0], s=250, color=_proj_c,
                       edgecolors='black', lw=2, zorder=9, marker='X')

    # =====================================================================
    # 4. THRESHOLD CROSSING MARKERS + DIRECTIONAL ARROWS (no text boxes)
    # =====================================================================
    for thr, clr, idx_c in [(gy, ZY, idx_gy), (yo, ZO, idx_yo),
                            (ob, ZB, idx_ob)]:
        if thr is not None and idx_c is not None:
            ax.scatter([dates[idx_c]], [main_curve[idx_c]], s=160,
                       color=clr, edgecolors=DK, lw=1.3, zorder=8,
                       marker='v')
            # Directional arrow indicator (kept, no text box)
            arrow_top = min(main_curve[idx_c] + y_max * 0.07, y_max * 0.92)
            ax.annotate('', xy=(dates[idx_c], main_curve[idx_c]),
                        xytext=(dates[idx_c], arrow_top),
                        arrowprops=dict(arrowstyle='->', color=clr,
                                        lw=1.2, alpha=0.45),
                        zorder=7)

    # =====================================================================
    # 5. VOLATILE DRIVING BEHAVIOUR HIGHLIGHTING
    #    Detection uses raw wear_mm (Tukey fence on week-over-week rate).
    #    Visual: thin dotted outline only -- no fill, no label.
    # =====================================================================
    vol_regions = detect_volatile_regions(wear_mm)
    for s, e in vol_regions:
        sl = slice(max(0, s), min(e + 1, n))
        y_lo = predicted_rul[sl].min()
        y_hi = predicted_rul[sl].max()
        margin_y = (y_hi - y_lo) * 0.12 + 3

        x0 = mdates.date2num(pd.Timestamp(dates[max(0, s)]))
        x1 = mdates.date2num(pd.Timestamp(dates[min(e, n - 1)]))
        if x1 <= x0:
            continue

        rect = Rectangle(
            (x0, y_lo - margin_y), x1 - x0, (y_hi - y_lo) + 2 * margin_y,
            fill=False, edgecolor=VOL_E, linestyle=':',
            linewidth=1.0, alpha=0.55, zorder=2, clip_on=True)
        ax.add_patch(rect)

    # =====================================================================
    # 5b. WEAR-RATE ACCELERATION HIGHLIGHTING (light purple dotted outline)
    # =====================================================================
    accel_regions = []
    if show_wear_accel:
        accel_regions = detect_wear_acceleration(wear_mm)
        for s, e in accel_regions:
            sl = slice(max(0, s), min(e + 1, n))
            y_lo = predicted_rul[sl].min()
            y_hi = predicted_rul[sl].max()
            margin_y = (y_hi - y_lo) * 0.12 + 3

            x0 = mdates.date2num(pd.Timestamp(dates[max(0, s)]))
            x1 = mdates.date2num(pd.Timestamp(dates[min(e, n - 1)]))
            if x1 <= x0:
                continue

            rect = Rectangle(
                (x0, y_lo - margin_y), x1 - x0,
                (y_hi - y_lo) + 2 * margin_y,
                fill=False, edgecolor=WACC_E, linestyle=':',
                linewidth=1.0, alpha=0.60, zorder=2, clip_on=True)
            ax.add_patch(rect)

    # =====================================================================
    # 6. VERTICAL REFERENCE LINES + PREDICTIVE RECOMMENDATIONS
    # =====================================================================
    vline_y = 3  # Y position for date labels at bottom

    # 6a. Purple PROGNOSTIC lines at threshold crossings
    rec_items = []
    if idx_gy is not None:
        rec_items.append((dates[idx_gy], '#9C27B0',
                          'Inspect clutch wear\nprogression'))
    if idx_yo is not None:
        rec_items.append((dates[idx_yo], '#7B1FA2',
                          'Preventive maintenance\nrecommended'))
    if idx_ob is not None:
        rec_items.append((dates[idx_ob], '#4A148C',
                          'Critical degradation\nreplace clutch'))

    # Stagger recommendation y-positions to avoid overlap
    rec_y_base = y_max * 0.40
    rec_y_step = y_max * 0.13

    for i, (x_pos, clr, rec_text) in enumerate(rec_items):
        _rc = _tk or clr
        ax.axvline(x=x_pos, color=_rc, lw=1.3, ls=':', alpha=0.40, zorder=3)
        rec_y = rec_y_base + i * rec_y_step
        ax.text(x_pos, rec_y, f'  {rec_text}',
                fontsize=7.5, fontweight='semibold', color=_rc,
                va='center', ha='left', alpha=0.70, rotation=90,
                fontstyle='italic', zorder=11)

    # 6b. Last Data Date
    close_dates = (fc_dt and abs((fc_dt - last_dt).days) < 10)
    _ld_c = _tk or '#5D6D7E'
    ax.axvline(x=last_dt, color=_ld_c, lw=1.0, ls=':', alpha=0.55,
               zorder=3)
    ld_label = f'Last Data  {last_dt.strftime("%Y-%m-%d")}'
    if has_km:
        ld_label += f'  |  {odo_at_dates[-1]:,.0f} km'
    if close_dates and fc_dt:
        fc_km_str = f'  |  ~{forecast_km:,.0f} km' if forecast_km else ''
        ld_label += f'\nForecast  {fc_dt.strftime("%Y-%m-%d")}{fc_km_str}'
    ax.text(last_dt, vline_y, f'  {ld_label}',
            fontsize=7, fontweight='bold', color=_ld_c,
            va='bottom', ha='left', alpha=0.8, rotation=90,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=BG,
                      edgecolor=_ld_c, alpha=0.7, lw=0.4))

    # 6c. Forecast Failure Date
    _fc_c = _tk or '#C0392B'
    _fc_bg = BG if bw_mode else '#FFF5F5'
    if fc_dt and fc_dt > pd.Timestamp(dates[0]) and not close_dates:
        ax.axvline(x=fc_dt, color=_fc_c, lw=1.0, ls=':', alpha=0.55,
                   zorder=3)
        fc_label = f'  Forecast Failure  {fc_dt.strftime("%Y-%m-%d")}'
        if forecast_km:
            fc_label += f'  |  ~{forecast_km:,.0f} km'
        ax.text(fc_dt, vline_y, fc_label,
                fontsize=7, fontweight='bold', color=_fc_c,
                va='bottom', ha='left', alpha=0.8, rotation=90,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=_fc_bg,
                          edgecolor=_fc_c, alpha=0.7, lw=0.4))

    # 6d. Actual failure date (training VINs only)
    if fail_dt and fail_dt > pd.Timestamp(dates[0]) and fail_dt != fc_dt:
        _fl_c = _tk or '#8B0000'
        ax.axvline(x=fail_dt, color=_fl_c, lw=1.2, ls='-.',
                   alpha=0.6, zorder=3)
        ax.scatter([fail_dt], [0], s=250, color=_fl_c,
                   edgecolors=DK, lw=2, zorder=9, marker='X')

    # (failure point km annotation removed per request)

    # =====================================================================
    # 8. SECONDARY Y-AXIS (wear mm at thresholds)
    # =====================================================================
    ax2 = ax.twinx()
    ax2.set_ylim(y_lo, y_max)
    ticks_data = [(ob, '4.50 mm'), (yo, '4.05 mm'), (gy, '3.15 mm')]
    if wear_final > 4.5:
        cur_rul = max(predicted_rul[-1], 0)
        ticks_data.append((cur_rul, f'{wear_final:.2f} mm'))
        # 6.0mm physical ceiling reference
        ceil_rul = interp_rul_at_wear(wear_mm, ref_curve, 6.0)
        if ceil_rul is not None and ceil_rul not in [v for v, _ in ticks_data]:
            ticks_data.append((ceil_rul, '6.00 mm (ceiling)'))
    valid = [(v, l) for v, l in ticks_data if v is not None]
    if not valid and wear_final > 0:
        cur_rul = max(predicted_rul[-1], 0)
        valid = [(cur_rul, f'{wear_final:.2f} mm')]
    _sa = _tk or '#7F8C8D'
    if valid:
        ax2.set_yticks([v for v, _ in valid])
        ax2.set_yticklabels([l for _, l in valid], fontsize=9, color=_sa)
    ax2.tick_params(axis='y', length=4, colors=_sa)
    ax2.spines['right'].set_color(SIL)
    ax2.spines['right'].set_linewidth(0.6)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('Cumulative Wear (mm)', fontsize=10,
                   color=_tk or '#7F8C8D', labelpad=10, style='italic')

    # =====================================================================
    # 9. AXIS FORMATTING
    # =====================================================================
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    tick_step = max(30, (y_max // 8) // 10 * 10)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.set_xlabel('Timeline', fontsize=13, fontweight='bold', color=DK,
                  labelpad=12)
    ax.set_ylabel('Predicted RUL (Days)', fontsize=13, fontweight='bold',
                  color=DK, labelpad=12)
    ax.grid(True, which='major', axis='y', color=GRD, lw=0.6, alpha=0.8)
    ax.grid(True, which='minor', axis='y', color=GRD, lw=0.3, alpha=0.3)
    ax.grid(True, which='major', axis='x', color=GRD, lw=0.4, alpha=0.5)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(SIL)
    ax.spines['bottom'].set_color(SIL)
    ax.tick_params(axis='both', which='major', labelsize=10, colors=DK,
                   length=5)

    # =====================================================================
    # 10. LEGEND (enhanced with new elements)
    # =====================================================================
    entries = []
    entries.append(Line2D([0],[0], color=PRD, lw=2.5, marker='D', ms=4.5,
                          markeredgecolor='w', markeredgewidth=0.5,
                          label='V10.5 Wear-Model Predicted RUL'))
    entries.append(Line2D([0],[0], color='#B71C1C', lw=2.0, ls=':',
                          alpha=0.88, label='Forecast Projection to Failure'))
    entries.append(Line2D([0],[0], color='#9C27B0', lw=1.3, ls=':',
                          label='Prognostic Reference Line'))
    entries.append(Line2D([0],[0], color='#5D6D7E', lw=1, ls=':',
                          label='Last Data Date'))
    entries.append(Line2D([0],[0], color='#C0392B', lw=1, ls=':',
                          label='Forecast Failure Date'))
    if vol_regions:
        entries.append(Line2D([0],[0], color=VOL_E, lw=1.0, ls=':',
                              alpha=0.55,
                              label='Volatile Driving Behavior'))
    if accel_regions:
        entries.append(Line2D([0],[0], color=WACC_E, lw=1.0, ls=':',
                              alpha=0.60,
                              label='Accelerated Wear Rate'))
    entries += zone_legend()
    ax.legend(handles=entries, loc='upper right', fontsize=7.5,
              framealpha=0.92, edgecolor=SIL, fancybox=True, ncol=2,
              bbox_to_anchor=(0.99, 0.92)).get_frame().set_linewidth(0.6)

    # =====================================================================
    # 11. TITLES
    # =====================================================================
    blind_tag = 'Blind Test  |  ' if is_blind else ''
    fig.suptitle(f'Clutch RUL Degradation  --  {vin_id}',
                 fontsize=18, fontweight='bold', color=DK, y=0.97)
    ax.set_title(f'{blind_tag}{subtitle}', fontsize=10,
                 color=_tk or '#5D6D7E', style='italic', pad=14)

    _ft = _tk or '#95A5A6'
    fig.text(0.02, 0.01,
             'Daimler Clutch Failure Prediction | V10.5 | '
             'GREEN <3.15mm | YELLOW 3.15-4.05mm | '
             'ORANGE 4.05-4.50mm | BLACK >=4.50mm | Confidential',
             fontsize=7.5, color=_ft, style='italic')
    fig.text(0.98, 0.01,
             f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=8, color=_ft, style='italic', ha='right')

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    return fig


# =========================================================================
# v8_vin1
# =========================================================================
def generate_v8_vin1():
    meta = pd.read_csv(META)
    v = meta[meta['vin'] == 'v8_vin1'].sort_values('lifecycle_days').reset_index(drop=True)
    dates       = pd.to_datetime(v['cutoff_timestamp']).values
    actual_rul  = v['rul_days'].values
    wear        = v['wear_mm'].values

    max_wf   = v['wear_fraction'].max()
    first_wf = v['wear_fraction'].iloc[0]
    first_rul = v['rul_days'].iloc[0]
    predicted_rul = np.clip(
        first_rul * (max_wf - v['wear_fraction'].values) / (max_wf - first_wf),
        0, None)

    total_lc = first_rul + v['lifecycle_days'].iloc[0]
    last_ts  = pd.Timestamp(dates[-1])
    fail_dt  = last_ts + pd.Timedelta(days=float(actual_rul[-1]))

    sub = (f'Lifecycle: {total_lc:.0f} days  |  {len(v)} Weekly Snapshots  |  '
           f'Wear Zones per Daimler Spec (4.50mm trigger)')

    fig = create_graph(
        dates, predicted_rul, wear, 'v8_vin1',
        subtitle=sub, status='FAILED', wear_final=wear[-1],
        rul_pred=int(actual_rul[-1]),
        forecast_date=fail_dt.strftime('%Y-%m-%d'),
        last_data_date=last_ts.strftime('%Y-%m-%d'),
        wear_zone='BLACK' if wear[-1] >= 4.5 else 'GREEN',
        action='REPLACE' if wear[-1] >= 4.5 else '',
        n_snapshots=len(v),
        failure_date_known=fail_dt.strftime('%Y-%m-%d'))
    for ext in ['png', 'svg']:
        p = os.path.join(OUT,
            f'V10_5_RUL_Degradation_v8_vin1_{DATESTAMP}_v3.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor=BG,
                    format=ext if ext == 'svg' else None)
    plt.close(fig)
    print(f'    v8_vin1 saved')
    return ['v8_vin1']


# =========================================================================
# V11 BLIND VINs
# =========================================================================
def generate_v11_vins():
    print('  Loading V11 engagement events...')
    events = pd.read_parquet(V11_EVT)
    events['timestamp'] = pd.to_datetime(events['timestamp'])

    xls = pd.read_excel(V11_XLS, sheet_name='Predictions')
    xls_map = {r['VIN']: r for _, r in xls.iterrows()}

    results = []
    for vin in sorted(events['vin_label'].unique()):
        print(f'    {vin}...')
        ve = events[events['vin_label'] == vin].sort_values('timestamp').copy()

        ve = ve.set_index('timestamp')
        weekly = ve.resample('W-MON').agg(
            wear_mm=('wear_mm', 'last'),
            odo=('ODO', 'last'),
            n_events=('wear_per_event_mm', 'count')
        ).dropna().reset_index()

        if len(weekly) < 3:
            print(f'      Skipped ({len(weekly)} weeks)')
            continue

        dates = weekly['timestamp'].values
        wear  = weekly['wear_mm'].values

        info = xls_map.get(vin, {})
        status_str    = info.get('Status', '')
        wear_final    = info.get('Wear (mm)', wear[-1])
        wear_zone_str = info.get('Wear Zone', '')
        rul_pred      = info.get('RUL Predicted (d)', 0)
        forecast_str  = str(info.get('Forecast Failure Date', ''))[:10]
        last_date_str = str(info.get('Last Data Date', ''))[:10]
        total_days    = info.get('Total Running Days', 0)
        action = 'REPLACE' if wear_final >= 4.5 else 'SCHEDULE'

        try:
            forecast_dt = pd.Timestamp(forecast_str)
        except Exception:
            forecast_dt = (pd.Timestamp(dates[-1])
                           + pd.Timedelta(days=int(rul_pred)))

        # Forecast-based RUL (linear countdown)
        forecast_rul = np.array([
            max(0, (forecast_dt - pd.Timestamp(d)).days) for d in dates
        ], dtype=float)

        # Wear-model predicted RUL (nonlinear)
        max_w = wear[-1] if wear[-1] > 0 else 1.0
        first_w = wear[0]
        first_rul_est = forecast_rul[0]
        denom = max_w - first_w if max_w != first_w else 1.0
        wear_predicted_rul = np.clip(
            first_rul_est * (max_w - wear) / denom, 0, None)
        wear_predicted_rul = _fix_terminal_rul(wear_predicted_rul, wear_final, vin)

        sub = (f'{total_days} days monitoring  |  '
               f'{len(weekly)} Weekly Snapshots  |  '
               f'Status: {status_str}  |  '
               f'Wear: {wear_final:.1f}mm ({wear_zone_str})  |  '
               f'RUL: {rul_pred}d  |  Forecast: {forecast_str}')

        fig = create_graph(
            dates, wear_predicted_rul, wear, vin,
            subtitle=sub, status=status_str,
            wear_final=wear_final,
            rul_pred=rul_pred,
            forecast_date=forecast_str,
            last_data_date=last_date_str,
            wear_zone=wear_zone_str, action=action,
            is_blind=True, n_snapshots=len(weekly))
        for ext in ['png', 'svg']:
            p = os.path.join(OUT,
                f'V10_5_RUL_Degradation_blind_{vin}_{DATESTAMP}_v3.{ext}')
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor=BG,
                        format=ext if ext == 'svg' else None)
        plt.close(fig)
        results.append(vin)
        print(f'      saved')
    return results


# =========================================================================
# V11 BLIND VINs -- Days + KMs graphs
# =========================================================================
def generate_v11_vins_km():
    """Generate km-enhanced graphs for V11 blind VINs.

    Uses authoritative km values from the Daimler deliverable Excel:
      - Pred Failure ODO (km)  ->  forecast_km
      - Last ODO (km)          ->  last ODO for vertical line label
      - First ODO (km)         ->  phase start km
    Weekly ODO from engagement events provides per-sample km for phase
    boundary interpolation.
    """
    print('  Loading V11 engagement events + Excel km data...')
    events = pd.read_parquet(V11_EVT)
    events['timestamp'] = pd.to_datetime(events['timestamp'])

    xls = pd.read_excel(V11_XLS, sheet_name='Predictions')
    xls_map = {r['VIN']: r for _, r in xls.iterrows()}

    results = []
    for vin in sorted(events['vin_label'].unique()):
        print(f'    {vin}...')
        ve = events[events['vin_label'] == vin].sort_values('timestamp').copy()
        ve = ve.set_index('timestamp')
        weekly = ve.resample('W-MON').agg(
            wear_mm=('wear_mm', 'last'),
            odo=('ODO', 'last'),
            n_events=('wear_per_event_mm', 'count')
        ).dropna().reset_index()

        if len(weekly) < 3:
            print(f'      Skipped ({len(weekly)} weeks)')
            continue

        dates = weekly['timestamp'].values
        wear  = weekly['wear_mm'].values

        info = xls_map.get(vin, {})
        status_str    = info.get('Status', '')
        wear_final    = info.get('Wear (mm)', wear[-1])
        wear_zone_str = info.get('Wear Zone', '')
        rul_pred      = info.get('RUL Predicted (d)', 0)
        forecast_str  = str(info.get('Forecast Failure Date', ''))[:10]
        last_date_str = str(info.get('Last Data Date', ''))[:10]
        total_days    = info.get('Total Running Days', 0)
        action = 'REPLACE' if wear_final >= 4.5 else 'SCHEDULE'

        # Authoritative km from Excel (Daimler deliverable)
        fc_km    = info.get('Pred Failure ODO (km)', None)
        last_km  = info.get('Last ODO (km)', None)
        first_km = info.get('First ODO (km)', None)

        # Convert to float safely
        def _safe_float(v):
            try:
                f = float(v) if v is not None and not pd.isna(v) else None
            except (ValueError, TypeError):
                f = None
            return f
        fc_km    = _safe_float(fc_km)
        last_km  = _safe_float(last_km)
        first_km = _safe_float(first_km)

        # Build per-sample ODO by linear interpolation from Excel values.
        # Raw event-level ODO is unreliable (constant / non-monotonic for
        # many VINs), so we interpolate using authoritative First/Last ODO.
        odo = None
        if first_km is not None and last_km is not None:
            first_date_ts = pd.Timestamp(dates[0])
            last_date_ts  = pd.Timestamp(dates[-1])
            span_days = max((last_date_ts - first_date_ts).days, 1)
            daily_km  = (last_km - first_km) / span_days
            odo = np.array([
                first_km + daily_km *
                (pd.Timestamp(d) - first_date_ts).days
                for d in dates
            ])

        try:
            forecast_dt = pd.Timestamp(forecast_str)
        except Exception:
            forecast_dt = (pd.Timestamp(dates[-1])
                           + pd.Timedelta(days=int(rul_pred)))

        # Forecast-based RUL (linear countdown)
        forecast_rul = np.array([
            max(0, (forecast_dt - pd.Timestamp(d)).days) for d in dates
        ], dtype=float)

        # Wear-model predicted RUL (nonlinear)
        max_w = wear[-1] if wear[-1] > 0 else 1.0
        first_w = wear[0]
        first_rul_est = forecast_rul[0]
        denom = max_w - first_w if max_w != first_w else 1.0
        wear_predicted_rul = np.clip(
            first_rul_est * (max_w - wear) / denom, 0, None)
        wear_predicted_rul = _fix_terminal_rul(wear_predicted_rul, wear_final, vin)

        # Subtitle with km info
        km_info = ''
        if last_km and not pd.isna(last_km):
            km_info = f'  |  ODO: {int(last_km):,} km'
        if fc_km:
            km_info += f'  |  Forecast ODO: {int(fc_km):,} km'

        sub = (f'{total_days} days monitoring  |  '
               f'{len(weekly)} Weekly Snapshots  |  '
               f'Status: {status_str}  |  '
               f'Wear: {wear_final:.1f}mm ({wear_zone_str})  |  '
               f'RUL: {rul_pred}d{km_info}')

        fig = create_graph(
            dates, wear_predicted_rul, wear, vin,
            subtitle=sub, status=status_str,
            wear_final=wear_final,
            rul_pred=rul_pred,
            forecast_date=forecast_str,
            last_data_date=last_date_str,
            wear_zone=wear_zone_str, action=action,
            is_blind=True, n_snapshots=len(weekly),
            odo_at_dates=odo,
            forecast_km=fc_km)
        for ext in ['png', 'svg']:
            p = os.path.join(OUT,
                f'V10_5_RUL_Degradation_Days_KMs_{vin}_{DATESTAMP}_v3.{ext}')
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor=BG,
                        format=ext if ext == 'svg' else None)
        plt.close(fig)
        results.append(vin)
        print(f'      saved (Days+KMs)')
    return results


# =========================================================================
# V11 BLIND VINs -- Days + KMs + Wear Acceleration graphs
# =========================================================================
def generate_v11_vins_wear_acc():
    """Generate km-enhanced graphs WITH wear-acceleration highlights.

    Same content as Days+KMs graphs plus light-purple dotted outlines
    around periods where weekly wear rate exceeds 1.5x the vehicle's
    median rate.  Naming includes 'wear_acc'.
    """
    print('  Loading V11 engagement events + Excel km data...')
    events = pd.read_parquet(V11_EVT)
    events['timestamp'] = pd.to_datetime(events['timestamp'])

    xls = pd.read_excel(V11_XLS, sheet_name='Predictions')
    xls_map = {r['VIN']: r for _, r in xls.iterrows()}

    results = []
    for vin in sorted(events['vin_label'].unique()):
        print(f'    {vin}...')
        ve = events[events['vin_label'] == vin].sort_values('timestamp').copy()
        ve = ve.set_index('timestamp')
        weekly = ve.resample('W-MON').agg(
            wear_mm=('wear_mm', 'last'),
            odo=('ODO', 'last'),
            n_events=('wear_per_event_mm', 'count')
        ).dropna().reset_index()

        if len(weekly) < 3:
            print(f'      Skipped ({len(weekly)} weeks)')
            continue

        dates = weekly['timestamp'].values
        wear  = weekly['wear_mm'].values

        info = xls_map.get(vin, {})
        status_str    = info.get('Status', '')
        wear_final    = info.get('Wear (mm)', wear[-1])
        wear_zone_str = info.get('Wear Zone', '')
        rul_pred      = info.get('RUL Predicted (d)', 0)
        forecast_str  = str(info.get('Forecast Failure Date', ''))[:10]
        last_date_str = str(info.get('Last Data Date', ''))[:10]
        total_days    = info.get('Total Running Days', 0)
        action = 'REPLACE' if wear_final >= 4.5 else 'SCHEDULE'

        # Authoritative km from Excel
        fc_km    = info.get('Pred Failure ODO (km)', None)
        last_km  = info.get('Last ODO (km)', None)
        first_km = info.get('First ODO (km)', None)

        def _safe_float(v):
            try:
                return float(v) if v is not None and not pd.isna(v) else None
            except (ValueError, TypeError):
                return None
        fc_km    = _safe_float(fc_km)
        last_km  = _safe_float(last_km)
        first_km = _safe_float(first_km)

        # Interpolated ODO from Excel values
        odo = None
        if first_km is not None and last_km is not None:
            first_date_ts = pd.Timestamp(dates[0])
            last_date_ts  = pd.Timestamp(dates[-1])
            span_days = max((last_date_ts - first_date_ts).days, 1)
            daily_km  = (last_km - first_km) / span_days
            odo = np.array([
                first_km + daily_km *
                (pd.Timestamp(d) - first_date_ts).days
                for d in dates
            ])

        try:
            forecast_dt = pd.Timestamp(forecast_str)
        except Exception:
            forecast_dt = (pd.Timestamp(dates[-1])
                           + pd.Timedelta(days=int(rul_pred)))

        forecast_rul = np.array([
            max(0, (forecast_dt - pd.Timestamp(d)).days) for d in dates
        ], dtype=float)

        max_w = wear[-1] if wear[-1] > 0 else 1.0
        first_w = wear[0]
        first_rul_est = forecast_rul[0]
        denom = max_w - first_w if max_w != first_w else 1.0
        wear_predicted_rul = np.clip(
            first_rul_est * (max_w - wear) / denom, 0, None)
        wear_predicted_rul = _fix_terminal_rul(wear_predicted_rul, wear_final, vin)

        km_info = ''
        if last_km and not pd.isna(last_km):
            km_info = f'  |  ODO: {int(last_km):,} km'
        if fc_km:
            km_info += f'  |  Forecast ODO: {int(fc_km):,} km'

        sub = (f'{total_days} days monitoring  |  '
               f'{len(weekly)} Weekly Snapshots  |  '
               f'Status: {status_str}  |  '
               f'Wear: {wear_final:.1f}mm ({wear_zone_str})  |  '
               f'RUL: {rul_pred}d{km_info}')

        fig = create_graph(
            dates, wear_predicted_rul, wear, vin,
            subtitle=sub, status=status_str,
            wear_final=wear_final,
            rul_pred=rul_pred,
            forecast_date=forecast_str,
            last_data_date=last_date_str,
            wear_zone=wear_zone_str, action=action,
            is_blind=True, n_snapshots=len(weekly),
            odo_at_dates=odo,
            forecast_km=fc_km,
            show_wear_accel=True)
        for ext in ['png', 'svg']:
            p = os.path.join(OUT,
                f'V10_5_RUL_Degradation_Days_KMs_wear_acc_{vin}_{DATESTAMP}_v3.{ext}')
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor=BG,
                        format=ext if ext == 'svg' else None)
        plt.close(fig)
        results.append(vin)
        print(f'      saved (wear_acc)')
    return results


# =========================================================================
# V11 BLIND VINs -- Print BW (grayscale palette, white background)
# =========================================================================
_BW_PALETTE = dict(
    DK='#000000', SIL='#AAAAAA', BG='#FFFFFF', GRD='#D0D0D0',
    ACTC='#555555', PRD='#000000', ABG='#FFFFFF',
    ZG='#000000', ZY='#000000', ZO='#000000', ZB='#000000',
    PBG_G='#FFFFFF', PBG_Y='#FFFFFF', PBG_O='#FFFFFF',
    PBG_B='#FFFFFF', PBG_EW='#FFFFFF',
    VOL_E='#000000', VOL_F='#FFFFFF', VOL_T='#000000',
    WACC_E='#000000',
)

def generate_v11_vins_print_bw():
    """Generate BW-printable graphs (Days+KMs+WearAcc, grayscale palette).

    Temporarily swaps the global colour palette to grayscale, generates
    graphs with Print_BW_ prefix, then restores the original palette.
    """
    # --- save originals & apply BW palette --------------------------------
    saved = {}
    for name, bw_val in _BW_PALETTE.items():
        saved[name] = globals()[name]
        globals()[name] = bw_val

    try:
        print('  Loading V11 engagement events + Excel km data (BW)...')
        events = pd.read_parquet(V11_EVT)
        events['timestamp'] = pd.to_datetime(events['timestamp'])

        xls = pd.read_excel(V11_XLS, sheet_name='Predictions')
        xls_map = {r['VIN']: r for _, r in xls.iterrows()}

        results = []
        for vin in sorted(events['vin_label'].unique()):
            print(f'    {vin}...')
            ve = events[events['vin_label'] == vin].sort_values(
                'timestamp').copy()
            ve = ve.set_index('timestamp')
            weekly = ve.resample('W-MON').agg(
                wear_mm=('wear_mm', 'last'),
                odo=('ODO', 'last'),
                n_events=('wear_per_event_mm', 'count')
            ).dropna().reset_index()

            if len(weekly) < 3:
                print(f'      Skipped ({len(weekly)} weeks)')
                continue

            dates = weekly['timestamp'].values
            wear  = weekly['wear_mm'].values

            info = xls_map.get(vin, {})
            status_str    = info.get('Status', '')
            wear_final    = info.get('Wear (mm)', wear[-1])
            wear_zone_str = info.get('Wear Zone', '')
            rul_pred      = info.get('RUL Predicted (d)', 0)
            forecast_str  = str(info.get('Forecast Failure Date', ''))[:10]
            last_date_str = str(info.get('Last Data Date', ''))[:10]
            total_days    = info.get('Total Running Days', 0)
            action = 'REPLACE' if wear_final >= 4.5 else 'SCHEDULE'

            fc_km    = info.get('Pred Failure ODO (km)', None)
            last_km  = info.get('Last ODO (km)', None)
            first_km = info.get('First ODO (km)', None)

            def _sf(v):
                try:
                    return float(v) if v is not None and not pd.isna(v) \
                        else None
                except (ValueError, TypeError):
                    return None
            fc_km = _sf(fc_km); last_km = _sf(last_km)
            first_km = _sf(first_km)

            odo = None
            if first_km is not None and last_km is not None:
                fts = pd.Timestamp(dates[0])
                lts = pd.Timestamp(dates[-1])
                span = max((lts - fts).days, 1)
                dkm = (last_km - first_km) / span
                odo = np.array([
                    first_km + dkm * (pd.Timestamp(d) - fts).days
                    for d in dates])

            try:
                forecast_dt = pd.Timestamp(forecast_str)
            except Exception:
                forecast_dt = (pd.Timestamp(dates[-1])
                               + pd.Timedelta(days=int(rul_pred)))

            forecast_rul = np.array([
                max(0, (forecast_dt - pd.Timestamp(d)).days)
                for d in dates], dtype=float)

            max_w = wear[-1] if wear[-1] > 0 else 1.0
            first_w = wear[0]
            denom = max_w - first_w if max_w != first_w else 1.0
            wear_predicted_rul = np.clip(
                forecast_rul[0] * (max_w - wear) / denom, 0, None)
            wear_predicted_rul = _fix_terminal_rul(wear_predicted_rul, wear_final, vin)

            km_info = ''
            if last_km:
                km_info = f'  |  ODO: {int(last_km):,} km'
            if fc_km:
                km_info += f'  |  Forecast ODO: {int(fc_km):,} km'

            sub = (f'{total_days} days monitoring  |  '
                   f'{len(weekly)} Weekly Snapshots  |  '
                   f'Status: {status_str}  |  '
                   f'Wear: {wear_final:.1f}mm ({wear_zone_str})  |  '
                   f'RUL: {rul_pred}d{km_info}')

            fig = create_graph(
                dates, wear_predicted_rul, wear, vin,
                subtitle=sub, status=status_str,
                wear_final=wear_final,
                rul_pred=rul_pred,
                forecast_date=forecast_str,
                last_data_date=last_date_str,
                wear_zone=wear_zone_str, action=action,
                is_blind=True, n_snapshots=len(weekly),
                odo_at_dates=odo,
                forecast_km=fc_km,
                show_wear_accel=True,
                bw_mode=True)
            for ext in ['png', 'svg']:
                p = os.path.join(OUT,
                    f'Print_BW_V10_5_RUL_Degradation_Days_KMs_{vin}_{DATESTAMP}_v3.{ext}')
                fig.savefig(p, dpi=300, bbox_inches='tight',
                            facecolor='#FFFFFF',
                            format=ext if ext == 'svg' else None)
            plt.close(fig)
            results.append(vin)
            print(f'      saved (Print_BW)')
        return results

    finally:
        # --- restore original palette ------------------------------------
        for name, orig_val in saved.items():
            globals()[name] = orig_val


# =========================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('RUL Degradation Graphs v4 -- Executive Grade (All VINs)')
    print('=' * 70)

    print('\n[1/5] v8_vin1...')
    generate_v8_vin1()

    print('\n[2/5] V11 blind VINs...')
    v11 = generate_v11_vins()

    print('\n[3/5] V11 blind VINs -- Days+KMs...')
    v11_km = generate_v11_vins_km()

    print('\n[4/5] V11 blind VINs -- Days+KMs+WearAcc...')
    v11_wa = generate_v11_vins_wear_acc()

    print('\n[5/5] V11 blind VINs -- Print BW (10 VINs x 2 versions)...')
    v11_bw = generate_v11_vins_print_bw()

    total = (2 + len(v11)*2 + len(v11_km)*2 + len(v11_wa)*2
             + len(v11_bw)*2) * 2
    print('\n' + '=' * 70)
    print(f'Done! {total} files total')
    print(f'  - v8_vin1: 4 files')
    print(f'  - V11 standard: {len(v11) * 2 * 2} files')
    print(f'  - V11 Days+KMs: {len(v11_km) * 2 * 2} files')
    print(f'  - V11 Days+KMs+WearAcc: {len(v11_wa) * 2 * 2} files')
    print(f'  - V11 Print_BW: {len(v11_bw) * 2 * 2} files')
    print(f'Output: {OUT}')
    print('=' * 70)
