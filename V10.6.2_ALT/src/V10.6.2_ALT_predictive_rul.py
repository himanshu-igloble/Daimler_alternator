"""
V10.6.2 Alternator — Tier A per-VIN predictive RUL band (honest)
================================================================
Per-VIN Remaining Useful Life as a SURVIVAL-CONDITIONED predictive interval
(plan D7): R = T - a | T > a, with (shape, scale) drawn from the Bayesian
posterior grid so each draw carries epistemic + aleatoric uncertainty.

This is Tier A ONLY — a single fleet wear-out curve shifted by each truck's
current age.  No idle-group / lifecycle hazard multiplier (that is exactly
what inverted the classifier in V10.6.1).  The frozen classifier enters the
DELIVERABLE as a separate risk tier (W4 decisions), not as an RUL multiplier.

km / engine-hour figures are DISPLAY conversions using each truck's own
trailing average rate; they are explicitly estimated (km_is_estimated) and
do not add accuracy beyond the days figure.

Output: cfg.RUL_CACHE / predictive_rul_per_vin.csv
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")
surv = _load("V10_6_2_ALT_survival", "V10.6.2_ALT_survival.py")


def main() -> None:
    out = pathlib.Path(cfg.RUL_CACHE)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RNG_SEED)

    print("[predictive_rul] Loading lifecycle + posterior samples ...")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    ps = pd.read_csv(pathlib.Path(cfg.WEIBULL_CACHE) / "posterior_samples.csv")
    shape_s = ps["shape"].values
    scale_s = ps["scale"].values
    print(f"  {len(lc)} VINs, {len(ps)} posterior draws")

    rows = []
    for _, r in lc.iterrows():
        vin = r["vin_label"]
        failed = bool(r["failed_flag"])
        a = float(r["age_days_observed"])

        # per-truck trailing rates for unit DISPLAY (estimated)
        km_per_day = (float(r["est_km"]) / a) if a > 0 and pd.notna(r["est_km"]) else cfg.FLEET_AVG_KM_PER_DAY
        ehrs_per_day = (float(r["est_engine_hrs"]) / a) if a > 0 and pd.notna(r["est_engine_hrs"]) else cfg.FLEET_AVG_EHRS_PER_DAY

        if failed:
            med = p10 = p90 = 0.0
        else:
            med, p10, p90 = surv.predictive_rul_summary(
                a, shape_s, scale_s, rng,
                lower_pct=cfg.PI_LOWER_PCT, upper_pct=cfg.PI_UPPER_PCT,
            )

        rows.append({
            "vin_label": vin,
            "failed_flag": int(failed),
            "lifecycle_stage": r.get("lifecycle_stage", ""),
            "current_age_days": round(a, 1),
            "median_rul_days": round(med, 1),
            "rul_p10_days": round(p10, 1),
            "rul_p90_days": round(p90, 1),
            "pi_width_days": round(p90 - p10, 1),
            "km_per_day_est": round(km_per_day, 2),
            "ehrs_per_day_est": round(ehrs_per_day, 2),
            "median_rul_km": round(med * km_per_day, 0),
            "rul_p10_km": round(p10 * km_per_day, 0),
            "rul_p90_km": round(p90 * km_per_day, 0),
            "median_rul_ehrs": round(med * ehrs_per_day, 1),
            "rul_p10_ehrs": round(p10 * ehrs_per_day, 1),
            "rul_p90_ehrs": round(p90 * ehrs_per_day, 1),
            "km_is_estimated": True,
        })

    df = pd.DataFrame(rows).sort_values(
        ["failed_flag", "median_rul_days"], ascending=[True, True]
    )
    df.to_csv(out / "predictive_rul_per_vin.csv", index=False)
    print(f"  Saved predictive_rul_per_vin.csv ({len(df)} rows)")

    nf = df[df["failed_flag"] == 0]
    print(f"\n  Non-failed RUL band (days):")
    print(f"    median range: {nf['median_rul_days'].min():.0f} - {nf['median_rul_days'].max():.0f}")
    print(f"    mean PI width: {nf['pi_width_days'].mean():.0f}d "
          f"(vs V10.6.1 params-only ~80d; honest interval is wider)")
    print(f"\n  {'VIN':<16}{'age':>6}{'RUL_d':>8}{'p10':>7}{'p90':>7}")
    for _, r in nf.iterrows():
        print(f"  {r['vin_label']:<16}{r['current_age_days']:>6.0f}"
              f"{r['median_rul_days']:>8.0f}{r['rul_p10_days']:>7.0f}{r['rul_p90_days']:>7.0f}")


if __name__ == "__main__":
    main()
