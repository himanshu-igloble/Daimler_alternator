"""
V10.6.2 Alternator — Tier A Fleet Weibull Survival (honest baseline)
====================================================================
Changes vs V10.6.1 (per plan D3/D4/D7):
  * Cohort = ALL 25 VINs (10 events + 15 censored), not TRAIN 17/6.
  * Reported point estimate = POSTERIOR MAP shape/scale (not the MLE ~7.6).
  * Fleet median CI comes from POSTERIOR-GRID sampling, not the divergent
    MLE bootstrap (which blew up to lambda=160 000).
  * Writes `median` AND `median_ttf_days` (B1 fix for the report readers).
  * Emits posterior_samples.csv so predictive_rul / backtest reuse the exact
    same epistemic draws.

Outputs (cfg.WEIBULL_CACHE):
  - fleet_weibull_params.json
  - posterior_grid.npz
  - posterior_samples.csv
  - fleet_survival_curve.csv
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


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")
surv = _load("V10_6_2_ALT_survival", "V10.6.2_ALT_survival.py")


def main() -> None:
    out = pathlib.Path(cfg.WEIBULL_CACHE)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RNG_SEED)

    # =======================================================================
    # Step 1: cohort = ALL 25 VINs (D3)
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
    # Step 2: MLE (reference only — NOT the reported point estimate)
    # =======================================================================
    print("[Step 2] MLE Weibull fit (reference) ...")
    wf = WeibullFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf.fit(durations, event_observed=events)
    mle_scale = float(wf.lambda_)   # lambda
    mle_shape = float(wf.rho_)      # rho
    mle_median = float(wf.median_survival_time_)
    print(f"  MLE: shape(rho)={mle_shape:.3f}  scale(lambda)={mle_scale:.1f}  median={mle_median:.1f}d")

    # =======================================================================
    # Step 3: Grid Bayesian posterior (reported point estimate, D4)
    # =======================================================================
    print("[Step 3] Grid Bayesian posterior (200x200) ...")
    post = surv.fit_grid_posterior(
        durations, events,
        prior_shape=cfg.WEIBULL_PRIOR_SHAPE, prior_scale=cfg.WEIBULL_PRIOR_SCALE,
        prior_shape_sd=cfg.WEIBULL_PRIOR_SHAPE_SD, prior_scale_sd=cfg.WEIBULL_PRIOR_SCALE_SD,
        shape_lo=cfg.GRID_SHAPE_LO, shape_hi=cfg.GRID_SHAPE_HI, shape_n=cfg.GRID_SHAPE_N,
        scale_lo=cfg.GRID_SCALE_LO, scale_hi=cfg.GRID_SCALE_HI, scale_n=cfg.GRID_SCALE_N,
    )
    map_shape, map_scale = post["map_shape"], post["map_scale"]
    mean_shape, mean_scale = post["mean_shape"], post["mean_scale"]
    print(f"  MAP : shape={map_shape:.3f}  scale={map_scale:.1f}")
    print(f"  Mean: shape={mean_shape:.3f}  scale={mean_scale:.1f}")

    np.savez_compressed(
        out / "posterior_grid.npz",
        shape_grid=post["shape_grid"], scale_grid=post["scale_grid"],
        posterior=post["posterior"],
    )
    print("  Saved posterior_grid.npz")

    # =======================================================================
    # Step 4: posterior samples (shared epistemic draws) + median CI
    # =======================================================================
    print("[Step 4] Posterior sampling for CI + downstream reuse ...")
    n_draws = max(cfg.N_PREDICTIVE_DRAWS, 5000)
    shape_s, scale_s = surv.sample_posterior(
        post["shape_grid"], post["scale_grid"], post["posterior"], n_draws, rng
    )
    pd.DataFrame({"shape": shape_s, "scale": scale_s}).to_csv(
        out / "posterior_samples.csv", index=False
    )
    print(f"  Saved posterior_samples.csv ({n_draws} draws)")

    median_draws = surv.weibull_median(shape_s, scale_s)
    map_median = float(surv.weibull_median(map_shape, map_scale))
    ci_lo = float(np.percentile(median_draws, cfg.PI_LOWER_PCT))
    ci_hi = float(np.percentile(median_draws, cfg.PI_UPPER_PCT))
    ci_width = ci_hi - ci_lo
    print(f"  Posterior median TTF = {map_median:.1f}d "
          f"[{cfg.PI_LOWER_PCT:.0f}-{cfg.PI_UPPER_PCT:.0f}%: {ci_lo:.1f}-{ci_hi:.1f}], width={ci_width:.1f}d")

    # =======================================================================
    # Step 5: Fleet survival curve (posterior MAP + posterior envelope)
    # =======================================================================
    print("[Step 5] Fleet survival curve (posterior) ...")
    t_eval = np.arange(0, 1201, 10, dtype=float)
    s_map = surv.weibull_sf(t_eval, map_shape, map_scale)
    # envelope from a manageable subset of posterior draws
    n_env = min(2000, n_draws)
    s_env = np.empty((n_env, len(t_eval)))
    for b in range(n_env):
        s_env[b, :] = surv.weibull_sf(t_eval, shape_s[b], scale_s[b])
    s_lower = np.percentile(s_env, cfg.PI_LOWER_PCT, axis=0)
    s_upper = np.percentile(s_env, cfg.PI_UPPER_PCT, axis=0)

    pd.DataFrame({
        "t_days": t_eval.astype(int),
        "S_t": np.round(s_map, 6),
        "S_lower": np.round(s_lower, 6),
        "S_upper": np.round(s_upper, 6),
    }).to_csv(out / "fleet_survival_curve.csv", index=False)
    print(f"  Saved fleet_survival_curve.csv")

    # =======================================================================
    # Step 6: params JSON (B1 fix: both `median` and `median_ttf_days`)
    # =======================================================================
    summary = {
        # primary (posterior MAP — D4)
        "shape": round(map_shape, 4),
        "scale": round(map_scale, 2),
        "median": round(map_median, 1),
        "median_ttf_days": round(map_median, 1),     # B1 alias for report readers
        "ci_lower": round(ci_lo, 1),
        "ci_upper": round(ci_hi, 1),
        "ci_width_days": round(ci_width, 1),
        "ci_method": "posterior_grid_sampling",
        # posterior detail
        "posterior_map_shape": round(map_shape, 4),
        "posterior_map_scale": round(map_scale, 2),
        "posterior_mean_shape": round(mean_shape, 4),
        "posterior_mean_scale": round(mean_scale, 2),
        # MLE reference (NOT used downstream)
        "mle_shape": round(mle_shape, 4),
        "mle_scale": round(mle_scale, 2),
        "mle_median": round(mle_median, 1),
        # cohort
        "n_cohort": n_cohort,
        "n_events": n_events,
        "cohort": "all_25",
    }
    with open(out / "fleet_weibull_params.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Done] fleet_weibull_params.json:")
    for k, v in summary.items():
        print(f"    {k:24s} = {v}")


if __name__ == "__main__":
    main()
