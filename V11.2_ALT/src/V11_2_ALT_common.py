"""V11.2_ALT shared core: paths, loaders, stats, plot style. Read-only on inputs."""
from __future__ import annotations
import os, json, datetime as dt
import numpy as np
import pandas as pd

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
V112 = os.path.join(ROOT, "V11.2_ALT")
RESULTS = os.path.join(V112, "results")
VIZ = os.path.join(V112, "visualizations")
CACHE = os.path.join(V112, "cache")

FAMILY_A = ["vsi_std_ratio_30d", "vsi_dominant_freq", "vsi_spectral_entropy",
            "bat_charge_delta_trend_right", "vsi_range_trend_last30d", "progressive_drift"]
FAMILY_B = ["crank_recovery_t", "vsi_ceiling", "vsi_resid_mean", "resting_vsi_mean", "ged_churn"]

PERM_IMPORTANCE = {
    "vsi_std_ratio_30d": 0.1547, "vsi_dominant_freq": 0.1053, "vsi_range_trend_last30d": 0.0613,
    "vsi_spectral_entropy": 0.0573, "progressive_drift": 0.0480, "bat_charge_delta_trend_right": 0.0340}

JCOPENDATE = {
    "VIN1_F_ALT": "2025-11-29", "VIN2_F_ALT": "2025-12-16", "VIN3_F_ALT": "2025-12-02",
    "VIN4_F_ALT": "2025-11-25", "VIN5_F_ALT": "2025-11-22", "VIN6_F_ALT": "2025-09-30",
    "VIN7_F_ALT": "2025-12-04", "VIN8_F_ALT": "2025-11-24", "VIN9_F_ALT": "2025-12-29",
    "VIN10_F_ALT": "2025-12-16"}

# ---- pure stats (unit-tested) ----
def concordant_pairs(failed, healthy):
    f = np.asarray(failed, float); h = np.asarray(healthy, float)
    c = t = d = 0
    for fi in f:
        c += int(np.sum(fi > h)); t += int(np.sum(fi == h)); d += int(np.sum(fi < h))
    return c, t, d, f.size * h.size

def auroc_from_scores(failed, healthy):
    c, t, d, n = concordant_pairs(failed, healthy)
    return (c + 0.5 * t) / n if n else float("nan")

def cliffs_delta(failed, healthy):
    c, t, d, n = concordant_pairs(failed, healthy)
    return (c - d) / n if n else float("nan")

def jcopendate_failure_age(t0, jcopendate):
    return (jcopendate - t0).days

# ---- loaders (verify-on-load; raise if a required file/col is missing) ----
def load_ridge_spec():
    p = os.path.join(ROOT, "V5.2_ALT/models/classification/V10.5.3_20_5_ALT_ridge_spec.json")
    with open(p) as fh: return json.load(fh)

def load_v111_rul():
    p = os.path.join(ROOT, "V11.1_ALT/cache/rul/final_rul_per_vin.csv")
    return pd.read_csv(p)

def load_emergency():
    p = os.path.join(ROOT, "V11.1_ALT/cache/emergency/emergency_per_vin.csv")
    return pd.read_csv(p)

def load_feature_parquet(vin):
    import polars as pl
    p = os.path.join(ROOT, f"V5.2_ALT/features/parquets/V5.2_20_5_ALT_{vin}.parquet")
    return pl.read_parquet(p)

def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as fh: json.dump(obj, fh, indent=2, default=str)

# ---- plot style (brand-aligned with V11.1) ----
PALETTE = {"fail": "#B00020", "healthy": "#1f7a3d", "accent": "#0B5394", "grid": "#d9d9d9"}
def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, weight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, color=PALETTE["grid"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
