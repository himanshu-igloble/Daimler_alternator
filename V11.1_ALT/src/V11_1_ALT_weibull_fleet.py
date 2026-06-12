"""
V11.1_ALT — AFT Fleet Weibull Survival Fits (M0 / M1 / M2)
===========================================================
Adapts V10.6.2's weibull_fleet stage to the covariate AFT model.

Variants:
  M0 — no covariates (beta forced to 0); reproduces V10.6.2's result
  M1 — x1 covariate only  (BETA1 grid)
  M2 — x1 + x2 covariates (BETA1 x BETA2 grid)

Grid memory note: 200x200x25x18 float64 ~ 1.4 GB; M1/M2 use shape_n=120,
scale_n=120 (recorded in aft_params_<V>.json as grid_n) to stay safe.

Outputs (cfg.WEIBULL_CACHE):
  posterior_samples_<V>.csv  — cols: shape, scale0, beta1[, beta2]
  aft_params_<V>.json        — MAP values, grid sizes, n_draws
  [M0 only]
  posterior_samples.csv      — V10.6.2-compatible (cols: shape, scale)
  fleet_weibull_params.json  — V10.6.2-compatible keys
  fleet_survival_curve.csv   — V10.6.2-compatible
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from lifelines import WeibullFitter

# ---------------------------------------------------------------------------
# Config + survival-helper import (mandatory importlib pattern)
# ---------------------------------------------------------------------------
_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config", "V11_1_ALT_config.py")
S = _load("V11_1_ALT_survival", "V11_1_ALT_survival.py")


def main() -> None:
    out = pathlib.Path(cfg.WEIBULL_CACHE)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RNG_SEED)

    # =======================================================================
    # Step 1: cohort = ALL 25 VINs (inherits D3 from V10.6.2)
    # =======================================================================
    print("[Step 1] Loading lifecycle (cohort = all 25 VINs) ...")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    durations = lc["age_days_observed"].values.astype(float)
    events = lc["failed_flag"].astype(int).values
    n_cohort = len(lc)
    n_events = int(events.sum())
    print(f"  Cohort: {n_cohort} VINs, {n_events} events (failures)")
    assert n_cohort == 25 and n_events == 10, "expected 25 VINs / 10 events"

    # =======================================================================
    # Step 2: Load covariates and merge on vin_label
    # =======================================================================
    print("[Step 2] Loading covariates ...")
    cov_path = pathlib.Path(cfg.COV_CACHE) / "covariates_fit.csv"
    covs_raw = pd.read_csv(cov_path)
    # Merge lifecycle with covariates on vin_label
    lc_cov = lc.merge(covs_raw[["vin_label", "x1", "x2"]], on="vin_label", how="left")
    assert lc_cov["x1"].isna().sum() == 0, "NaN x1 after merge"
    assert lc_cov["x2"].isna().sum() == 0, "NaN x2 after merge"
    print(f"  Merged {len(lc_cov)} rows; x1 range [{lc_cov['x1'].min():.3f}, {lc_cov['x1'].max():.3f}], "
          f"x2 unique={sorted(lc_cov['x2'].unique())}")

    # =======================================================================
    # Step 3: MLE reference (not the reported estimate)
    # =======================================================================
    print("[Step 3] MLE Weibull fit (reference) ...")
    wf = WeibullFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf.fit(durations, event_observed=events)
    mle_scale = float(wf.lambda_)
    mle_shape = float(wf.rho_)
    mle_median = float(wf.median_survival_time_)
    print(f"  MLE: shape(rho)={mle_shape:.3f}  scale(lambda)={mle_scale:.1f}  median={mle_median:.1f}d")

    # =======================================================================
    # Step 4: Fit M0 / M1 / M2
    # =======================================================================
    # Grid sizes: M0 uses full 200x200; M1/M2 use 120x120 to cap memory
    GRID_FULL_N  = cfg.GRID_SHAPE_N   # 200
    GRID_FULL_SN = cfg.GRID_SCALE_N   # 200
    GRID_COVY_N  = 120                # reduced for M1/M2

    variant_results = {}

    for variant in cfg.VARIANTS:
        print(f"\n[Step 4] Fitting variant {variant} ...")

        if variant == "M0":
            x = np.zeros((n_cohort, 1))
            beta_grids = [np.array([0.0])]
            sn, scn = GRID_FULL_N, GRID_FULL_SN
        elif variant == "M1":
            x = lc_cov[["x1"]].to_numpy()
            beta_grids = [np.linspace(cfg.BETA1_LO, cfg.BETA1_HI, cfg.BETA1_N)]
            sn, scn = GRID_COVY_N, GRID_COVY_N
        else:  # M2
            x = lc_cov[["x1", "x2"]].to_numpy()
            beta_grids = [
                np.linspace(cfg.BETA1_LO, cfg.BETA1_HI, cfg.BETA1_N),
                np.linspace(cfg.BETA2_LO, cfg.BETA2_HI, cfg.BETA2_N),
            ]
            sn, scn = GRID_COVY_N, GRID_COVY_N

        post = S.fit_aft_grid_posterior(
            durations=durations,
            events=events,
            x=x,
            prior_shape=cfg.WEIBULL_PRIOR_SHAPE,
            prior_scale=cfg.WEIBULL_PRIOR_SCALE,
            prior_shape_sd=cfg.WEIBULL_PRIOR_SHAPE_SD,
            prior_scale_sd=cfg.WEIBULL_PRIOR_SCALE_SD,
            shape_lo=cfg.GRID_SHAPE_LO,
            shape_hi=cfg.GRID_SHAPE_HI,
            shape_n=sn,
            scale_lo=cfg.GRID_SCALE_LO,
            scale_hi=cfg.GRID_SCALE_HI,
            scale_n=scn,
            beta_grids=beta_grids,
            beta_prior_sd=cfg.BETA_PRIOR_SD,
        )

        map_shape  = post["map_shape"]
        map_scale0 = post["map_scale0"]
        map_beta   = post["map_beta"]
        p          = post["p"]
        print(f"  MAP: shape={map_shape:.4f}  scale0={map_scale0:.2f}  beta={map_beta}")

        # -------------------------------------------------------------------
        # Posterior samples
        # -------------------------------------------------------------------
        n_draws = cfg.N_PREDICTIVE_DRAWS
        rng_v = np.random.default_rng(cfg.RNG_SEED)
        ks, ls, bs = S.sample_aft_posterior(post, n_draws, rng_v)

        # Build samples DataFrame
        samples_df = pd.DataFrame({"shape": ks, "scale0": ls})
        for j in range(p):
            samples_df[f"beta{j+1}"] = bs[:, j]
        samples_df.to_csv(out / f"posterior_samples_{variant}.csv", index=False)
        print(f"  Saved posterior_samples_{variant}.csv ({n_draws} draws)")

        # -------------------------------------------------------------------
        # AFT params JSON
        # -------------------------------------------------------------------
        aft_params = {
            "variant": variant,
            "map_shape": round(map_shape, 4),
            "map_scale0": round(map_scale0, 2),
            "map_beta": [round(b, 4) for b in map_beta],
            "grid_n_shape": sn,
            "grid_n_scale": scn,
            "n_draws": n_draws,
            "n_cohort": n_cohort,
            "n_events": n_events,
            "cohort": "all_25",
        }
        with open(out / f"aft_params_{variant}.json", "w") as f:
            json.dump(aft_params, f, indent=2)
        print(f"  Saved aft_params_{variant}.json")

        variant_results[variant] = {
            "post": post,
            "ks": ks, "ls": ls, "bs": bs,
            "map_shape": map_shape, "map_scale0": map_scale0, "map_beta": map_beta,
        }

    # =======================================================================
    # Step 5: M0 — write V10.6.2-compatible outputs
    # =======================================================================
    print("\n[Step 5] Writing V10.6.2-compatible M0 outputs ...")
    m0 = variant_results["M0"]
    ks0, ls0 = m0["ks"], m0["ls"]
    map_shape0  = m0["map_shape"]
    map_scale0_ = m0["map_scale0"]

    # posterior_samples.csv — cols: shape, scale  (scale = scale0 for M0)
    pd.DataFrame({"shape": ks0, "scale": ls0}).to_csv(
        out / "posterior_samples.csv", index=False
    )
    print(f"  Saved posterior_samples.csv")

    # median CI from sampled draws
    map_median = float(S.weibull_median(map_shape0, map_scale0_))
    median_draws = S.weibull_median(ks0, ls0)
    ci_lo = float(np.percentile(median_draws, cfg.PI_LOWER_PCT))
    ci_hi = float(np.percentile(median_draws, cfg.PI_UPPER_PCT))
    ci_width = ci_hi - ci_lo
    print(f"  Posterior median TTF = {map_median:.1f}d "
          f"[{cfg.PI_LOWER_PCT:.0f}-{cfg.PI_UPPER_PCT:.0f}%: {ci_lo:.1f}-{ci_hi:.1f}], width={ci_width:.1f}d")

    # fleet_weibull_params.json — V10.6.2-compatible keys
    fleet_params = {
        "shape": round(map_shape0, 4),
        "scale": round(map_scale0_, 2),
        "median": round(map_median, 1),
        "median_ttf_days": round(map_median, 1),
        "ci_lower": round(ci_lo, 1),
        "ci_upper": round(ci_hi, 1),
        "ci_width_days": round(ci_width, 1),
        "ci_method": "posterior_grid_sampling",
        "n_cohort": n_cohort,
        "n_events": n_events,
        "cohort": "all_25",
    }
    with open(out / "fleet_weibull_params.json", "w") as f:
        json.dump(fleet_params, f, indent=2)
    print("  Saved fleet_weibull_params.json")

    # fleet_survival_curve.csv — t_days 0..1200 step 10, S at MAP + envelope
    t_eval = np.arange(0, 1201, 10, dtype=float)
    s_map = S.weibull_sf(t_eval, map_shape0, map_scale0_)
    n_env = min(2000, len(ks0))
    s_env = np.empty((n_env, len(t_eval)))
    for b in range(n_env):
        s_env[b, :] = S.weibull_sf(t_eval, ks0[b], ls0[b])
    s_lower = np.percentile(s_env, cfg.PI_LOWER_PCT, axis=0)
    s_upper = np.percentile(s_env, cfg.PI_UPPER_PCT, axis=0)
    pd.DataFrame({
        "t_days": t_eval.astype(int),
        "S_t": np.round(s_map, 6),
        "S_lower": np.round(s_lower, 6),
        "S_upper": np.round(s_upper, 6),
    }).to_csv(out / "fleet_survival_curve.csv", index=False)
    print("  Saved fleet_survival_curve.csv")

    # =======================================================================
    # Step 6: M0 sanity assert (must match V10.6.2 within tolerance)
    # =======================================================================
    print("\n[Step 6] M0 sanity check ...")
    print(f"  M0 map_shape  = {map_shape0:.4f}  (target 5.17 ± 0.3)")
    print(f"  M0 map_scale0 = {map_scale0_:.2f}  (target 771 ± 30)")
    if not (abs(map_shape0 - 5.17) < 0.3):
        print(f"  SANITY FAIL: map_shape {map_shape0:.4f} outside [4.87, 5.47]")
        import sys; sys.exit(1)
    if not (abs(map_scale0_ - 771) < 30):
        print(f"  SANITY FAIL: map_scale0 {map_scale0_:.2f} outside [741, 801]")
        import sys; sys.exit(1)
    print("  SANITY PASS")

    # =======================================================================
    # Step 7: Summary table
    # =======================================================================
    print("\n" + "=" * 65)
    print(f"{'Variant':<10} {'map_shape':>10} {'map_scale0':>12} {'map_beta'}")
    print("-" * 65)
    for v in cfg.VARIANTS:
        r = variant_results[v]
        beta_str = str([round(b, 4) for b in r["map_beta"]])
        print(f"{v:<10} {r['map_shape']:>10.4f} {r['map_scale0']:>12.2f}   {beta_str}")
    print("=" * 65)


if __name__ == "__main__":
    main()
