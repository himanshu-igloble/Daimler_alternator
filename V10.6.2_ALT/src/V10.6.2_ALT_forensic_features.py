"""
V10.6.2 Alternator — Forensic Feature Engine  (deep zero-precursor investigation)
================================================================================
Deterministic substrate for the failure-forensics workflow.  Builds an EXPANDED
daily feature panel per VIN from the raw telemetry, then for each failed VIN
computes, at multiple pre-failure horizons (90/60/45/30/14/7d), how far each
feature deviates from (a) the truck's own healthy mid-life baseline AND (b) the
healthy-fleet daily envelope (NF p05-p95).  A deviation only counts as a real
precursor if it is BOTH a within-truck change AND outside the healthy-fleet
range (discriminative) — this is the honest test the prior work skipped.

Expanded panel (beyond the frozen 6 classifier features):
  voltage level/spread/instability : vsi_mean/std/cv/min/p05/range/entropy
  undervoltage sag frequency       : vsi_sag_frac  (engine-on VSI < 24 V)
  charging-by-regime               : idle_vsi_mean, cruise_vsi_mean
  battery interaction              : resting_vsi_mean (engine off), crank_vsi_min (SMA=1)
  excitation                       : ged2_cnt, ged2_frac
  duty                             : sma_starts, rpm_mean, n_eo

Outputs (cfg.* / cache/forensics/):
  <VIN>_daily.csv                 per-VIN daily panel
  nf_baseline.csv                 healthy-fleet daily envelope (p05/p50/p95 per feature)
  failed_window_deviations.csv    failed VIN x horizon x feature deviations
  earliest_signal_per_vin.csv     earliest discriminative precursor per failed VIN
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("V10_6_2_ALT_config", str(_src / "V10.6.2_ALT_config.py"))
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

FORENSICS = pathlib.Path(cfg.V10_6_2_ROOT) / "cache" / "forensics"

# feature direction: does a HIGH value mean worse health, or a LOW value?
BAD_HIGH = {"vsi_std", "vsi_cv", "vsi_range", "vsi_sag_frac", "vsi_entropy", "ged2_frac", "ged2_cnt"}
BAD_LOW = {"vsi_mean", "vsi_min", "vsi_p05", "idle_vsi_mean", "cruise_vsi_mean",
           "resting_vsi_mean", "crank_vsi_min"}
KEY_FEATURES = ["vsi_std", "vsi_cv", "vsi_range", "vsi_sag_frac", "vsi_entropy",
                "ged2_frac", "vsi_min", "idle_vsi_mean", "resting_vsi_mean", "crank_vsi_min"]
HORIZON_BINS = [(60, 90, "90"), (45, 60, "60"), (30, 45, "45"),
                (14, 30, "30"), (7, 14, "14"), (0, 7, "7")]
# Minimum engine-on samples for a day's features to be trusted.  Without this,
# days with a handful of readings (e.g. 2 samples both during a cold crank)
# yield vsi_sag_frac=1.0 and pollute both the NF envelope and the failed
# windows with low-sample artifacts (observed: several NF trucks hit sag=1.0).
MIN_EO_SAMPLES = 200   # ~17 min of engine-on at 5 s sampling


def _entropy(vals, lo=22.0, hi=31.0, step=0.25):
    if len(vals) < 5:
        return np.nan
    bins = np.arange(lo, hi + step, step)
    h, _ = np.histogram(vals, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def _raw(vin):
    return pathlib.Path(cfg.V52_PARQUET_DIR) / f"{cfg.V52_PARQUET_PREFIX}{vin}.parquet"


def build_daily(vin: str) -> pd.DataFrame:
    df = pd.read_parquet(_raw(vin), columns=["RPM", "CSP", "VSI", "GED", "SMA",
                                             "DAYS_SINCE_SALE", "DAYS_TO_FAILURE"])
    # VSI scaling guard (raw is already volts here, but be safe) + validity
    vsi = df["VSI"].where(df["VSI"] <= 36, df["VSI"] * 0.2)
    df["vsi"] = vsi.where((vsi > 10) & (vsi <= 36))
    df["eo"] = (df["RPM"] > 0) & (df["RPM"] <= 3500)
    df["off"] = (df["RPM"] == 0)
    day = "DAYS_SINCE_SALE"

    eo = df[df["eo"] & df["vsi"].notna()]
    g = eo.groupby(day)["vsi"]
    daily = pd.DataFrame({
        "n_eo": g.size(),
        "vsi_mean": g.mean(),
        "vsi_std": g.std(),
        "vsi_min": g.min(),
        "vsi_p05": g.quantile(0.05),
        "vsi_p95": g.quantile(0.95),
    })
    daily["vsi_cv"] = daily["vsi_std"] / daily["vsi_mean"]
    daily["vsi_range"] = daily["vsi_p95"] - daily["vsi_p05"]
    daily["vsi_entropy"] = g.apply(lambda s: _entropy(s.values))
    daily["vsi_sag_frac"] = eo.assign(sag=eo["vsi"] < 24).groupby(day)["sag"].mean()

    idle = eo[(eo["RPM"].between(550, 950)) & (eo["CSP"] < 5)]
    daily["idle_vsi_mean"] = idle.groupby(day)["vsi"].mean()
    cruise = eo[eo["CSP"] > 40]
    daily["cruise_vsi_mean"] = cruise.groupby(day)["vsi"].mean()

    rest = df[df["off"] & df["vsi"].notna()]
    daily["resting_vsi_mean"] = rest.groupby(day)["vsi"].mean()
    crank = df[(df["SMA"] == 1) & df["vsi"].notna()]
    daily["crank_vsi_min"] = crank.groupby(day)["vsi"].min()

    daily["ged2_cnt"] = (df["GED"] == 2).groupby(df[day]).sum()
    daily["ged2_frac"] = (df["GED"] == 2).groupby(df[day]).mean()
    sma_rise = ((df["SMA"] == 1) & (df["SMA"].shift(1) == 0))
    daily["sma_starts"] = sma_rise.groupby(df[day]).sum()
    daily["rpm_mean"] = eo.groupby(day)["RPM"].mean()
    daily["dtf"] = df.groupby(day)["DAYS_TO_FAILURE"].median()
    daily = daily.reset_index().rename(columns={day: "day"})
    daily["vin_label"] = vin
    return daily


def main() -> None:
    FORENSICS.mkdir(parents=True, exist_ok=True)
    failed = set(cfg.FAILED_VIN_SET)
    feat_cols = ["vsi_mean", "vsi_std", "vsi_cv", "vsi_min", "vsi_p05", "vsi_range",
                 "vsi_entropy", "vsi_sag_frac", "idle_vsi_mean", "cruise_vsi_mean",
                 "resting_vsi_mean", "crank_vsi_min", "ged2_cnt", "ged2_frac",
                 "sma_starts", "rpm_mean"]

    print("[forensic_features] Building daily panels for 25 VINs ...")
    dailies = {}
    nf_rows = []
    for vin in cfg.ALL_VINS:
        d = build_daily(vin)
        d.to_csv(FORENSICS / f"{vin}_daily.csv", index=False)
        dailies[vin] = d
        if vin not in failed:
            nf_rows.append(d)
        print(f"  {vin:<16} days={len(d):<4} ged2_total={int(d['ged2_cnt'].sum()):>7} "
              f"vsi_sag_frac_max={d['vsi_sag_frac'].max():.3f}")

    # NF healthy envelope (pool NF daily rows, trusted days only)
    nf = pd.concat(nf_rows, ignore_index=True)
    nf = nf[nf["n_eo"] >= MIN_EO_SAMPLES]
    nf_base = []
    for f in feat_cols:
        s = nf[f].dropna()
        nf_base.append({"feature": f, "nf_p05": s.quantile(0.05),
                        "nf_p50": s.quantile(0.50), "nf_p95": s.quantile(0.95),
                        "nf_mean": s.mean(), "nf_std": s.std()})
    nf_base = pd.DataFrame(nf_base)
    nf_base.to_csv(FORENSICS / "nf_baseline.csv", index=False)
    nfb = nf_base.set_index("feature")
    print(f"  NF envelope built from {len(nf)} healthy daily rows")

    # Failed-VIN window deviations + earliest discriminative precursor
    dev_rows, earliest_rows = [], []
    for vin in cfg.FAILED_VIN_SET:
        d_all = dailies[vin]
        d = d_all[d_all["n_eo"] >= MIN_EO_SAMPLES]   # trusted days only
        # baseline = healthy mid-life (dtf 120-365); fallback earliest 40% of days
        base = d[(d["dtf"] >= 120) & (d["dtf"] <= 365)]
        if len(base) < 15:
            base = d.nsmallest(max(int(len(d) * 0.4), 15), "day")
        best_h, best_feat, best_z = None, None, 0.0
        for lo, hi, lbl in HORIZON_BINS:
            win = d[(d["dtf"] > lo) & (d["dtf"] <= hi)]
            n_days = len(win)
            for f in feat_cols:
                bmean, bstd = base[f].mean(), base[f].std()
                wmean = win[f].mean() if n_days else np.nan
                z = (wmean - bmean) / bstd if (bstd and bstd > 0 and not np.isnan(wmean)) else np.nan
                p05, p95 = nfb.loc[f, "nf_p05"], nfb.loc[f, "nf_p95"]
                if np.isnan(wmean):
                    disc = False
                elif f in BAD_HIGH:
                    disc = wmean > p95
                else:
                    disc = wmean < p05
                dev_rows.append({
                    "vin_label": vin, "horizon_days": lbl, "n_days": n_days, "feature": f,
                    "window_mean": round(wmean, 4) if not np.isnan(wmean) else "",
                    "baseline_mean": round(bmean, 4) if not np.isnan(bmean) else "",
                    "z_vs_baseline": round(z, 2) if not np.isnan(z) else "",
                    "nf_p05": round(p05, 4), "nf_p95": round(p95, 4),
                    "discriminative": bool(disc),
                })
                # earliest discriminative + strong key-feature signal
                if (f in KEY_FEATURES and disc and not np.isnan(z) and abs(z) >= 2.0
                        and int(lbl) >= (int(best_h) if best_h else -1)):
                    if best_h is None or int(lbl) > int(best_h) or abs(z) > abs(best_z):
                        best_h, best_feat, best_z = lbl, f, z
        # data coverage in the 30d window (gap detection)
        cov30 = len(d[(d["dtf"] >= 0) & (d["dtf"] <= 30)])
        earliest_rows.append({
            "vin_label": vin,
            "earliest_discriminative_horizon_days": (best_h if best_h else "none"),
            "feature": (best_feat if best_feat else ""),
            "z": (round(best_z, 2) if best_h else ""),
            "ged2_total": int(d["ged2_cnt"].sum()),
            "n_days_final_30d": cov30,
            "verdict": ("discriminative_precursor" if best_h else "no_discriminative_precursor"),
        })

    pd.DataFrame(dev_rows).to_csv(FORENSICS / "failed_window_deviations.csv", index=False)
    es = pd.DataFrame(earliest_rows)
    es.to_csv(FORENSICS / "earliest_signal_per_vin.csv", index=False)

    print("\n  EARLIEST DISCRIMINATIVE PRECURSOR PER FAILED VIN")
    print(f"  {'VIN':<14}{'earliest_d':>11}{'feature':>18}{'z':>7}{'ged2':>8}{'cov30':>7}")
    for _, r in es.iterrows():
        print(f"  {r['vin_label']:<14}{str(r['earliest_discriminative_horizon_days']):>11}"
              f"{str(r['feature']):>18}{str(r['z']):>7}{r['ged2_total']:>8}{r['n_days_final_30d']:>7}")
    n_detect = int((es["verdict"] == "discriminative_precursor").sum())
    print(f"\n  {n_detect}/10 failed VINs have a discriminative precursor at >=7d horizon.")
    print(f"  Saved 4 forensic artifacts to {FORENSICS}")


if __name__ == "__main__":
    main()
