"""
V11.1_ALT — Time-rewound LOVO backtest with truncated covariates
================================================================
Decisive validation: leave-one-failed-VIN-out backtest comparing AFT variants
M0/M1/M2 against the fleet-clock dummy (Dummy-A).

Key design:
  - rewind_covariates() is the ONLY covariate entry point (G-LEAK audits it).
    All calls from the fold loop must go through this function; any direct call
    to C.covariate_vector() outside it is a leakage violation.
  - Per-fold refit: fit on the other 24 trucks; score on the held-out failed VIN
    at age = ttf_v - horizon.
  - NF trucks are NEVER held out (nf_envelope = "all_15_nf").
  - Variant selection: among variants with pi_coverage >= 0.80, choose min
    mae_model; ties -> fewer parameters (M0 < M1 < M2).  If none reach coverage,
    choose M0 with selection_rule="fallback_coverage".
  - Smoke mode: set env var V11_1_BT_SMOKE=1 to run only the first failed VIN
    and variants ["M0","M1"] for fast iteration testing.

Outputs:
  cfg.BACKTEST_CACHE / backtest_results.json
  cfg.BACKTEST_CACHE / per_fold_residuals.csv
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / f"{mod_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config")
S = _load("V11_1_ALT_survival")
C = _load("V11_1_ALT_covariates")

_FOLD_DRAWS = 2000   # posterior samples per fold (backtest speed)
_BACKTEST_SHAPE_N = 80
_BACKTEST_SCALE_N = 80


# ---------------------------------------------------------------------------
# Public contract — G-LEAK audits this as the ONLY covariate entry point
# ---------------------------------------------------------------------------

def rewind_covariates(vin, ttf, horizon, cov_fn, p95, nfb):
    """Covariates at the rewind point t = ttf - horizon. The ONLY covariate
    entry point the backtest may use (G-LEAK audits this)."""
    return cov_fn(vin, float(ttf) - float(horizon), p95, nfb)


# ---------------------------------------------------------------------------
# Variant design matrix helpers
# ---------------------------------------------------------------------------

def _variant_design(covs_df, variant):
    """Return (x_matrix [N x p], beta_grids list) for the training frame.

    M0: intercept-only (beta=0 forced; zeros covariate).
    M1: x1 only.
    M2: x1 + x2.
    """
    n = len(covs_df)
    if variant == "M0":
        x = np.zeros((n, 1))
        beta_grids = [np.array([0.0])]
    elif variant == "M1":
        x = covs_df[["x1"]].to_numpy(dtype=float)
        beta_grids = [np.linspace(cfg.BETA1_LO, cfg.BETA1_HI, cfg.BETA1_N)]
    elif variant == "M2":
        x = covs_df[["x1", "x2"]].to_numpy(dtype=float)
        beta_grids = [
            np.linspace(cfg.BETA1_LO, cfg.BETA1_HI, cfg.BETA1_N),
            np.linspace(cfg.BETA2_LO, cfg.BETA2_HI, cfg.BETA2_N),
        ]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return x, beta_grids


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def main() -> None:
    smoke = os.environ.get("V11_1_BT_SMOKE", "0").strip() == "1"
    out = pathlib.Path(cfg.BACKTEST_CACHE)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load inputs -------------------------------------------------------
    print("[backtest] Loading lifecycle cohort ...")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET).reset_index(drop=True)

    print("[backtest] Loading fit-time covariates ...")
    covs_df = pd.read_csv(cfg.COV_CACHE / "covariates_fit.csv")

    print("[backtest] Loading NF p95 + NF baseline ...")
    p95 = float((cfg.COV_CACHE / "crank_p95.txt").read_text(encoding="utf-8").strip())
    nfb = pd.read_csv(cfg.V11_FORENSICS / "nf_baseline.csv").set_index("feature")

    # ---- fleet metadata ----------------------------------------------------
    failed_lc = lc[lc["failed_flag"] == True]
    failed_ttf = dict(zip(failed_lc["vin_label"], failed_lc["ttf_days"].astype(float)))

    failed_vins_sorted = sorted(cfg.FAILED_VIN_SET)
    if smoke:
        failed_vins_sorted = failed_vins_sorted[:1]
    n_folds = len(failed_vins_sorted)

    variants_to_run = cfg.VARIANTS if not smoke else ["M0", "M1"]

    print(f"  {len(lc)} VINs total, {len(failed_ttf)} failed, LOVO folds={n_folds}")
    if smoke:
        print("  [SMOKE MODE] limited to 1 fold + M0/M1")

    # ---- per-fold loop ------------------------------------------------------
    # rows accumulate per (fold, variant, horizon) results
    rows = []

    for fold_idx, v in enumerate(failed_vins_sorted):
        ttf_v = failed_ttf[v]
        # training mask: all trucks except held-out failed VIN
        train_mask = lc["vin_label"] != v
        train_lc = lc[train_mask].reset_index(drop=True)
        train_covs = covs_df[covs_df["vin_label"] != v].reset_index(drop=True)

        # training durations + events
        durations_tr = train_lc["age_days_observed"].values.astype(float)
        events_tr = train_lc["failed_flag"].astype(int).values

        # leave-k-out fleet clock: median TTF of the OTHER failed VINs
        other_ttf = [t for w, t in failed_ttf.items() if w != v]
        dummyA_failtime = float(np.median(other_ttf))

        print(f"\n  [fold {fold_idx+1}/{n_folds}] vin={v} ttf={ttf_v:.0f}d "
              f"dummyA_ref={dummyA_failtime:.0f}d")

        for variant in variants_to_run:
            x_tr, beta_grids = _variant_design(train_covs, variant)

            # per-fold+variant RNG (seeded for reproducibility)
            rng = np.random.default_rng(cfg.RNG_SEED + fold_idx)

            # fit AFT grid posterior (once per fold × variant)
            post = S.fit_aft_grid_posterior(
                durations_tr, events_tr, x_tr,
                prior_shape=cfg.WEIBULL_PRIOR_SHAPE,
                prior_scale=cfg.WEIBULL_PRIOR_SCALE,
                prior_shape_sd=cfg.WEIBULL_PRIOR_SHAPE_SD,
                prior_scale_sd=cfg.WEIBULL_PRIOR_SCALE_SD,
                shape_lo=cfg.GRID_SHAPE_LO, shape_hi=cfg.GRID_SHAPE_HI,
                shape_n=_BACKTEST_SHAPE_N,
                scale_lo=cfg.GRID_SCALE_LO, scale_hi=cfg.GRID_SCALE_HI,
                scale_n=_BACKTEST_SCALE_N,
                beta_grids=beta_grids,
                beta_prior_sd=cfg.BETA_PRIOR_SD,
            )

            # sample 2000 posterior draws
            shape_s, scale0_s, beta_s = S.sample_aft_posterior(post, _FOLD_DRAWS, rng)

            # score at each horizon
            for h in cfg.BACKTEST_HORIZONS_DAYS:
                if ttf_v <= h:
                    # cannot rewind — skip (truck died before this horizon)
                    continue
                age_a = ttf_v - h

                # rewind covariates (ONLY entry point for held-out truck covariates)
                x1_v, x2_v = rewind_covariates(
                    v, ttf_v, h, C.covariate_vector, p95, nfb
                )

                # build x_vec for scale_for() — variant-specific
                if variant == "M0":
                    x_vec = np.array([0.0])
                    beta_s_sub = beta_s  # shape (N, 1) with zeros
                elif variant == "M1":
                    x_vec = np.array([x1_v])
                    beta_s_sub = beta_s  # shape (N, 1)
                else:  # M2
                    x_vec = np.array([x1_v, x2_v])
                    beta_s_sub = beta_s  # shape (N, 2)

                # per-draw per-truck scales
                scale_i_s = S.scale_for(scale0_s, beta_s_sub, x_vec)

                # conditional predictive RUL draws
                rul_draws = S.conditional_predictive_rul(age_a, shape_s, scale_i_s, rng)
                median_rul = float(np.median(rul_draws))
                err_model = float(abs(median_rul - h))

                pi_lo = float(np.percentile(rul_draws, cfg.PI_LOWER_PCT))
                pi_hi = float(np.percentile(rul_draws, cfg.PI_UPPER_PCT))
                covered = bool(pi_lo <= h <= pi_hi)
                width = float(pi_hi - pi_lo)

                # Dummy-A fleet clock at this age
                rul_dummy = max(dummyA_failtime - age_a, 0.0)
                err_dummy = float(abs(rul_dummy - h))

                rows.append({
                    "vin_label": v,
                    "variant": variant,
                    "horizon": h,
                    "ttf": round(ttf_v, 1),
                    "age_at_rewind": round(age_a, 1),
                    "x1": round(x1_v, 4),
                    "x2": int(x2_v),
                    "median_rul": round(median_rul, 1),
                    "err_model": round(err_model, 1),
                    "err_dummy": round(err_dummy, 1),
                    "pi_lo": round(pi_lo, 1),
                    "pi_hi": round(pi_hi, 1),
                    "covered": covered,
                    "width": round(width, 1),
                })
                print(f"    {variant} H={h:>3}d  age={age_a:.0f}  "
                      f"x1={x1_v:.3f} x2={x2_v}  "
                      f"rul_med={median_rul:.0f}  err={err_model:.0f}  "
                      f"dummy_err={err_dummy:.0f}  "
                      f"PI=[{pi_lo:.0f},{pi_hi:.0f}] cov={covered}")

    # ---- save per-fold CSV --------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_fold_residuals.csv", index=False)
    print(f"\n[backtest] Saved per_fold_residuals.csv ({len(df)} rows)")

    # ---- aggregate per variant ----------------------------------------------
    results_by_variant = {}

    for variant in variants_to_run:
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue

        mae_model = float(sub["err_model"].mean())
        mae_dummy = float(sub["err_dummy"].mean())
        pi_coverage = float(sub["covered"].mean())
        mean_pi_width = float(sub["width"].mean())

        # per-horizon MAEs
        per_horizon_mae = {}
        for h in cfg.BACKTEST_HORIZONS_DAYS:
            hdf = sub[sub["horizon"] == h]
            if not hdf.empty:
                per_horizon_mae[str(h)] = round(float(hdf["err_model"].mean()), 1)

        # wilcoxon: model vs dummy (alternative="less" → model better i.e. lower errors)
        try:
            _, p_vs_dummy = stats.wilcoxon(
                sub["err_model"].values, sub["err_dummy"].values,
                zero_method="wilcox", alternative="less"
            )
            p_vs_dummy = float(p_vs_dummy)
        except Exception:
            p_vs_dummy = 1.0

        # wilcoxon: variant vs M0 (only meaningful for M1/M2)
        p_vs_m0 = None
        if variant != "M0":
            m0_sub = df[df["variant"] == "M0"]
            # align on (vin_label, horizon) pairs that exist in both
            merged = sub.merge(m0_sub[["vin_label", "horizon", "err_model"]],
                               on=["vin_label", "horizon"], suffixes=("", "_m0"))
            if len(merged) >= 2:
                try:
                    _, p_vs_m0 = stats.wilcoxon(
                        merged["err_model"].values, merged["err_model_m0"].values,
                        zero_method="wilcox", alternative="less"
                    )
                    p_vs_m0 = float(p_vs_m0)
                except Exception:
                    p_vs_m0 = 1.0

        results_by_variant[variant] = {
            "mae_model": round(mae_model, 1),
            "mae_dummy": round(mae_dummy, 1),
            "per_horizon_mae": per_horizon_mae,
            "pi_coverage": round(pi_coverage, 3),
            "mean_pi_width": round(mean_pi_width, 1),
            "wilcoxon_p_vs_dummy": round(p_vs_dummy, 4) if p_vs_dummy is not None else None,
            "wilcoxon_p_vs_m0": round(p_vs_m0, 4) if p_vs_m0 is not None else None,
            "n_rows": int(len(sub)),
        }

    # ---- variant selection --------------------------------------------------
    # Among variants with pi_coverage >= 0.80, choose min mae_model.
    # Ties resolved by parameter count: M0 < M1 < M2.
    # If none reach coverage, choose M0 with selection_rule="fallback_coverage".
    param_order = {"M0": 0, "M1": 1, "M2": 2}
    candidates = [
        vt for vt in variants_to_run
        if results_by_variant.get(vt, {}).get("pi_coverage", 0.0) >= 0.80
    ]

    if candidates:
        chosen_variant = min(
            candidates,
            key=lambda vt: (results_by_variant[vt]["mae_model"], param_order[vt])
        )
        selection_rule = "min_mae_with_coverage"
    else:
        chosen_variant = "M0"
        selection_rule = "fallback_coverage"

    # ---- write backtest_results.json ----------------------------------------
    results = {
        "cohort": "all_25",
        "n_folds": n_folds,
        "per_fold_refit": True,
        "nf_envelope": "all_15_nf",
        "grid_note": (
            f"backtest speed grids: shape_n={_BACKTEST_SHAPE_N}, "
            f"scale_n={_BACKTEST_SCALE_N}, draws_per_fold={_FOLD_DRAWS}"
        ),
        "variants": results_by_variant,
        "chosen_variant": chosen_variant,
        "selection_rule": selection_rule,
    }
    with open(out / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary table ----------------------------------------------
    print("\n" + "=" * 72)
    print("BACKTEST SUMMARY — time-rewound LOVO with truncated covariates")
    print("=" * 72)
    header = f"  {'Variant':<8} {'MAE_model':>10} {'MAE_dummy':>10} {'Coverage':>10} {'PI_width':>10} {'p_vs_dum':>10} {'p_vs_M0':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for vt in variants_to_run:
        if vt not in results_by_variant:
            continue
        rv = results_by_variant[vt]
        marker = " <-- CHOSEN" if vt == chosen_variant else ""
        print(f"  {vt:<8} {rv['mae_model']:>10.1f} {rv['mae_dummy']:>10.1f} "
              f"{rv['pi_coverage']:>10.3f} {rv['mean_pi_width']:>10.1f} "
              f"{str(rv['wilcoxon_p_vs_dummy']):>10} "
              f"{str(rv['wilcoxon_p_vs_m0'] if rv['wilcoxon_p_vs_m0'] is not None else '—'):>10}"
              f"{marker}")
    print(f"\n  chosen_variant = {chosen_variant}  (rule: {selection_rule})")
    print(f"  Saved backtest_results.json + per_fold_residuals.csv -> {out}")


if __name__ == "__main__":
    main()
