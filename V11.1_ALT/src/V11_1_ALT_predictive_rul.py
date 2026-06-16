"""
V11.1_ALT — Per-VIN predictive RUL with chosen AFT variant
===========================================================
Adapts V10.6.2's predictive_rul to the chosen backtest variant (M0/M1/M2).

Reads  cfg.BACKTEST_CACHE / "backtest_results.json" for chosen_variant.
Loads  cfg.WEIBULL_CACHE / "posterior_samples_{chosen}.csv".
Computes covariates (x1, x2) per non-failed truck at age_days_observed.

Beta column subset per variant:
  M0 -> no beta cols; beta_s shape (n, 0); x_vec = []
  M1 -> ps[["beta1"]]; x_vec = [x1]
  M2 -> ps[["beta1", "beta2"]]; x_vec = [x1, x2]

scale_i_s = S.scale_for(ps["scale0"], beta_s, x_vec)
(med, p10, p90) = S.predictive_rul_summary(a, ps["shape"], scale_i_s, rng, ...)

Failed trucks: 0.0s (same as V10.6.2).

Output columns (same schema as V10.6.2 PLUS x1, x2, variant):
  vin_label, failed_flag, lifecycle_stage, current_age_days,
  x1, x2, variant,
  median_rul_days, rul_p10_days, rul_p90_days, pi_width_days,
  km_per_day_est, ehrs_per_day_est,
  median_rul_km, rul_p10_km, rul_p90_km,
  median_rul_ehrs, rul_p10_ehrs, rul_p90_ehrs,
  km_is_estimated

Output: cfg.RUL_CACHE / "predictive_rul_per_vin.csv"
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(n: str):
    s = importlib.util.spec_from_file_location(n, str(_src / f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


cfg = _load("V11_1_ALT_config")
S = _load("V11_1_ALT_survival")
C = _load("V11_1_ALT_covariates")


def main() -> None:
    cfg.RUL_CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RNG_SEED)

    # ------------------------------------------------------------------ #
    # 1. Determine chosen variant from backtest results
    # ------------------------------------------------------------------ #
    backtest_json = cfg.BACKTEST_CACHE / "backtest_results.json"
    chosen = json.loads(backtest_json.read_text(encoding="utf-8"))["chosen_variant"]
    print(f"[predictive_rul] chosen variant: {chosen}")

    # ------------------------------------------------------------------ #
    # 2. Load posterior samples for chosen variant
    # ------------------------------------------------------------------ #
    ps_path = cfg.WEIBULL_CACHE / f"posterior_samples_{chosen}.csv"
    ps = pd.read_csv(ps_path)
    n_draws = len(ps)
    print(f"[predictive_rul] posterior draws: {n_draws}  (from {ps_path.name})")

    # ------------------------------------------------------------------ #
    # 3. Beta column subset
    # ------------------------------------------------------------------ #
    if chosen == "M0":
        beta_s = np.zeros((n_draws, 0))
    elif chosen == "M1":
        beta_s = ps[["beta1"]].to_numpy()
    else:  # M2
        beta_s = ps[["beta1", "beta2"]].to_numpy()

    shape_s = ps["shape"].to_numpy()
    scale0_s = ps["scale0"].to_numpy()

    # ------------------------------------------------------------------ #
    # 4. Covariates — loaded once per truck
    # ------------------------------------------------------------------ #
    p95 = float((cfg.COV_CACHE / "crank_p95.txt").read_text(encoding="utf-8").strip())
    nfb = pd.read_csv(cfg.V11_FORENSICS / "nf_baseline.csv").set_index("feature")
    print(f"[predictive_rul] crank p95={p95:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Lifecycle loop
    # ------------------------------------------------------------------ #
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    print(f"[predictive_rul] {len(lc)} VINs loaded from lifecycle parquet")

    rows = []
    for _, r in lc.iterrows():
        vin = r["vin_label"]
        failed = bool(r["failed_flag"])
        a = float(r["age_days_observed"])

        # Display-unit rates (per-truck trailing average; same logic as V10.6.2)
        km_per_day = (
            float(r["est_km"]) / a
            if a > 0 and pd.notna(r.get("est_km"))
            else cfg.FLEET_AVG_KM_PER_DAY
        )
        ehrs_per_day = (
            float(r["est_engine_hrs"]) / a
            if a > 0 and pd.notna(r.get("est_engine_hrs"))
            else cfg.FLEET_AVG_EHRS_PER_DAY
        )

        if failed:
            med = p10 = p90 = 0.0
            x1 = x2 = float("nan")
        else:
            x1, x2 = C.covariate_vector(vin, a, p95, nfb)

            if chosen == "M0":
                x_vec: list[float] = []
            elif chosen == "M1":
                x_vec = [x1]
            else:
                x_vec = [x1, x2]

            scale_i_s = S.scale_for(scale0_s, beta_s, x_vec)
            med, p10, p90 = S.predictive_rul_summary(
                a, shape_s, scale_i_s, rng,
                lower_pct=cfg.PI_LOWER_PCT,
                upper_pct=cfg.PI_UPPER_PCT,
            )

        rows.append({
            "vin_label": vin,
            "failed_flag": int(failed),
            "lifecycle_stage": r.get("lifecycle_stage", ""),
            "current_age_days": round(a, 1),
            "x1": round(x1, 4) if not (isinstance(x1, float) and np.isnan(x1)) else float("nan"),
            "x2": int(x2) if not (isinstance(x2, float) and np.isnan(x2)) else float("nan"),
            "variant": chosen,
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
    out_path = cfg.RUL_CACHE / "predictive_rul_per_vin.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}  ({len(df)} rows, variant={chosen})")

    nf = df[df["failed_flag"] == 0]
    print(f"\n  Non-failed RUL band (days)  [variant={chosen}]:")
    print(f"    median range : {nf['median_rul_days'].min():.0f} – "
          f"{nf['median_rul_days'].max():.0f}")
    print(f"    mean PI width: {nf['pi_width_days'].mean():.0f}d")
    print(f"\n  {'VIN':<18}{'age':>6}{'x1':>7}{'x2':>5}{'RUL_d':>8}{'p10':>7}{'p90':>7}")
    for _, row in nf.iterrows():
        print(f"  {row['vin_label']:<18}{row['current_age_days']:>6.0f}"
              f"{row['x1']:>7.3f}{row['x2']:>5}"
              f"{row['median_rul_days']:>8.0f}{row['rul_p10_days']:>7.0f}"
              f"{row['rul_p90_days']:>7.0f}")


if __name__ == "__main__":
    main()
