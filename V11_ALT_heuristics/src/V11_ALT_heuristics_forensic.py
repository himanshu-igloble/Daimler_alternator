"""V11_ALT_heuristics — Forensic Feature Engine (extended honest-gate harness).

Builds the expanded daily panel (V10.6.2's 16 features + 19 new heuristic
features) per VIN, then runs the UNCHANGED honest gate (within-truck z>=2 AND
outside NF p05-p95, MIN_EO_SAMPLES=200, horizons 90/60/45/30/14/7) over the new
feature set. Adds an NF self-test (each NF truck scored as if failing, LOO
envelope) so multiple-comparison false-alarm risk is reported, not hidden.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
F = _load("V11_ALT_heuristics_features")
FORENSICS = cfg.FORENSICS

RAW_COLS = ["RPM", "CSP", "ANR", "VSI", "GED", "SMA",
            "DATETIME", "DAYS_SINCE_SALE", "DAYS_TO_FAILURE"]


def _entropy(vals, lo=22.0, hi=31.0, step=0.25):
    if len(vals) < 5:
        return np.nan
    bins = np.arange(lo, hi + step, step)
    h, _ = np.histogram(vals, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def _raw(vin):
    return pathlib.Path(cfg.V52_PARQUET_DIR) / f"{cfg.V52_PARQUET_PREFIX}{vin}.parquet"


def _read_prepared(vin):
    df = pd.read_parquet(_raw(vin), columns=RAW_COLS)
    return F.prepare(df, cfg)


def build_reference() -> pd.DataFrame:
    """Pool NF engine-on valid rows and build the #2 reference surface."""
    nf_eo = []
    for vin in cfg.ALL_VINS:
        if vin in cfg.FAILED_VIN_SET:
            continue
        p = _read_prepared(vin)
        nf_eo.append(p[p["eo"] & p["vsi"].notna()][["RPM", "anr", "CSP", "vsi", "day"]])
    return F.build_load_reference(pd.concat(nf_eo, ignore_index=True), cfg)


def build_daily(vin: str, ref: pd.DataFrame) -> pd.DataFrame:
    df = _read_prepared(vin)
    day = "day"
    eo = df[df["eo"] & df["vsi"].notna()]
    g = eo.groupby(day)["vsi"]
    daily = pd.DataFrame({
        "n_eo": g.size(), "vsi_mean": g.mean(), "vsi_std": g.std(),
        "vsi_min": g.min(), "vsi_p05": g.quantile(0.05), "vsi_p95": g.quantile(0.95),
    })
    daily["vsi_cv"] = daily["vsi_std"] / daily["vsi_mean"]
    daily["vsi_range"] = daily["vsi_p95"] - daily["vsi_p05"]
    daily["vsi_entropy"] = g.apply(lambda s: _entropy(s.values))
    daily["vsi_sag_frac"] = eo.assign(sag=eo["vsi"] < cfg.SAG_V).groupby(day)["sag"].mean()
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
    daily["sma_starts"] = ((df["SMA"] == 1) & (df["SMA"].shift(1) == 0)).groupby(df[day]).sum()
    daily["rpm_mean"] = eo.groupby(day)["RPM"].mean()

    # --- new heuristic features (joined on day) ---
    daily = daily.join(F.vsi_rpm_curve(eo, cfg))                         # #1
    daily = daily.join(F.load_residual(eo, ref, cfg))                    # #2
    daily = daily.join(F.crank_recovery(df, cfg))                        # #3
    daily = daily.join(F.reg_duty(eo, cfg))                             # #4
    daily = daily.join(F.crank_effort(df, eo, cfg))                     # #7
    daily = daily.join(F.ged_states(df, cfg))                           # #8
    daily = daily.join(F.idle_hunting(eo, cfg))                         # #9
    daily = daily.join(F.sag_typing(eo, cfg))                           # #10
    daily = daily.join(F.uv_dose_daily(eo, cfg))                        # #5 daily

    daily["dtf"] = df.groupby(day)["DAYS_TO_FAILURE"].median()
    daily = daily.reset_index().rename(columns={day: "day"})
    daily["vin_label"] = vin
    for col in cfg.FEAT_COLS:
        if col not in daily.columns:
            daily[col] = np.nan
    return daily


def _gate_one_vin(d_all: pd.DataFrame, nfb: pd.DataFrame):
    """Run the V10.6.2 honest gate over FEAT_COLS for one VIN's daily panel.
    Returns (best_horizon_label, best_feature, best_z, dev_rows)."""
    d = d_all[d_all["n_eo"] >= cfg.MIN_EO_SAMPLES]
    base = d[(d["dtf"] >= 120) & (d["dtf"] <= 365)]
    if len(base) < 15:
        base = d.nsmallest(max(int(len(d) * 0.4), 15), "day")
    best_h, best_feat, best_z = None, None, 0.0
    dev_rows = []
    for lo, hi, lbl in cfg.HORIZON_BINS:
        win = d[(d["dtf"] > lo) & (d["dtf"] <= hi)]
        n_days = len(win)
        for f in cfg.FEAT_COLS:
            bmean, bstd = base[f].mean(), base[f].std()
            wmean = win[f].mean() if n_days else np.nan
            z = (wmean - bmean) / bstd if (bstd and bstd > 0 and not np.isnan(wmean)) else np.nan
            p05, p95 = nfb.loc[f, "nf_p05"], nfb.loc[f, "nf_p95"]
            if np.isnan(wmean):
                disc = False
            elif f in cfg.BAD_HIGH:
                disc = wmean > p95
            else:
                disc = wmean < p05
            dev_rows.append({
                "vin_label": d_all["vin_label"].iloc[0], "horizon_days": lbl,
                "n_days": n_days, "feature": f,
                "window_mean": round(wmean, 4) if not np.isnan(wmean) else "",
                "baseline_mean": round(bmean, 4) if not np.isnan(bmean) else "",
                "z_vs_baseline": round(z, 2) if not np.isnan(z) else "",
                "nf_p05": round(p05, 4), "nf_p95": round(p95, 4),
                "discriminative": bool(disc),
            })
            if (f in cfg.KEY_FEATURES and disc and not np.isnan(z) and abs(z) >= 2.0
                    and int(lbl) >= (int(best_h) if best_h else -1)):
                if best_h is None or int(lbl) > int(best_h) or abs(z) > abs(best_z):
                    best_h, best_feat, best_z = lbl, f, z
    return best_h, best_feat, best_z, dev_rows


def _nf_baseline(nf_dailies):
    nf = pd.concat(nf_dailies, ignore_index=True)
    nf = nf[nf["n_eo"] >= cfg.MIN_EO_SAMPLES]
    rows = []
    for f in cfg.FEAT_COLS:
        s = nf[f].dropna()
        rows.append({"feature": f, "nf_p05": s.quantile(0.05), "nf_p50": s.quantile(0.50),
                     "nf_p95": s.quantile(0.95), "nf_mean": s.mean(), "nf_std": s.std()})
    return pd.DataFrame(rows)


def main() -> None:
    FORENSICS.mkdir(parents=True, exist_ok=True)
    print("[v11 forensic] building NF reference surface ...")
    ref = build_reference()

    print("[v11 forensic] building daily panels for 25 VINs ...")
    dailies = {}
    for vin in cfg.ALL_VINS:
        d = build_daily(vin, ref)
        d.to_csv(FORENSICS / f"{vin}_daily.csv", index=False)
        dailies[vin] = d
        print(f"  {vin:<16} days={len(d):<4} uv_dose_total={np.nansum(d['uv_dose_day']):.0f}")

    nf_list = [dailies[v] for v in cfg.ALL_VINS if v not in cfg.FAILED_VIN_SET]
    nf_base = _nf_baseline(nf_list)
    nf_base.to_csv(FORENSICS / "nf_baseline.csv", index=False)
    nfb = nf_base.set_index("feature")

    dev_rows, earliest_rows = [], []
    for vin in cfg.FAILED_VIN_SET:
        d_all = dailies[vin]
        best_h, best_feat, best_z, devs = _gate_one_vin(d_all, nfb)
        dev_rows.extend(devs)
        d = d_all[d_all["n_eo"] >= cfg.MIN_EO_SAMPLES]
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

    self_rows = []
    nf_vins = [v for v in cfg.ALL_VINS if v not in cfg.FAILED_VIN_SET]
    for vin in nf_vins:
        loo = [dailies[v] for v in nf_vins if v != vin]
        nfb_loo = _nf_baseline(loo).set_index("feature")
        best_h, best_feat, best_z, _ = _gate_one_vin(dailies[vin], nfb_loo)
        self_rows.append({
            "vin_label": vin,
            "false_precursor_horizon_days": (best_h if best_h else "none"),
            "feature": (best_feat if best_feat else ""),
            "z": (round(best_z, 2) if best_h else ""),
            "verdict": ("FALSE_ALARM" if best_h else "clean"),
        })
    pd.DataFrame(self_rows).to_csv(FORENSICS / "nf_self_test.csv", index=False)

    n_detect = int((es["verdict"] == "discriminative_precursor").sum())
    n_false = int(sum(r["verdict"] == "FALSE_ALARM" for r in self_rows))
    print(f"\n  V11: {n_detect}/10 failed VINs with a discriminative precursor at >=7d")
    print(f"  NF self-test false alarms: {n_false}/15")
    print(f"  Saved forensic artifacts to {FORENSICS}")


if __name__ == "__main__":
    main()
