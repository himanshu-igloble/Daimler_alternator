"""
V10.6.2 Alternator — Backtest harness  (the centerpiece, plan W2)
================================================================
The validation V10.6.1 never had: an OUT-OF-SAMPLE test of whether the
per-truck RUL beats a fleet-clock baseline.

Two tests, both LOVO over the 10 FAILED trucks with PER-FOLD REFIT (D8 — at
n=10 events one truck moves the fleet median 15-25%, so the held-out truck is
excluded from every quantity it is scored against):

  (1) LOVO total-TTF calibration + PI coverage
      For each held-out failed truck k: refit the fleet Weibull on the other
      24 VINs; predict k's total TTF (= leave-k-out fleet median) and the
      predictive interval; compare to k's actual TTF.

  (2) Time-rewound RUL at horizons T-270/-180/-90 d
      Tier A is age-only, so the calendar rewind is exact: evaluate the
      survival-conditioned RUL at age = ttf_k - H and compare the predicted
      remaining days to the true remaining H.  (Ridge is NOT used here — α's
      RUL carries no classifier signal — so there is no leakage to label.
      Weekly-feature rewind is reserved for beta, when features enter.)

Baselines (per-fold, excluding held-out — D9):
  Dummy-A (fleet clock): leave-k-out median failed TTF, minus current age.
  Dummy-B (constant):    leave-k-out median failed TTF, minus the fleet mean
                          failed age (ignores the truck's own age).

Decision gate (D9): the model must beat Dummy-A on out-of-sample day-MAE.
A "no_improvement" verdict is an ACCEPTED, documented outcome — it is the
evidence that keeps the beta tiers shelved.

Outputs: cfg.BACKTEST_CACHE / {backtest_results.json, per_fold_residuals.csv}
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")
surv = _load("V10_6_2_ALT_survival", "V10.6.2_ALT_survival.py")

_FOLD_DRAWS = 4000   # posterior draws per leave-k-out fold


def _refit_fold(durations, events, rng):
    post = surv.fit_grid_posterior(
        durations, events,
        prior_shape=cfg.WEIBULL_PRIOR_SHAPE, prior_scale=cfg.WEIBULL_PRIOR_SCALE,
        prior_shape_sd=cfg.WEIBULL_PRIOR_SHAPE_SD, prior_scale_sd=cfg.WEIBULL_PRIOR_SCALE_SD,
        shape_lo=cfg.GRID_SHAPE_LO, shape_hi=cfg.GRID_SHAPE_HI, shape_n=cfg.GRID_SHAPE_N,
        scale_lo=cfg.GRID_SCALE_LO, scale_hi=cfg.GRID_SCALE_HI, scale_n=cfg.GRID_SCALE_N,
    )
    shape_s, scale_s = surv.sample_posterior(
        post["shape_grid"], post["scale_grid"], post["posterior"], _FOLD_DRAWS, rng
    )
    map_median = float(surv.weibull_median(post["map_shape"], post["map_scale"]))
    return shape_s, scale_s, map_median


def main() -> None:
    out = pathlib.Path(cfg.BACKTEST_CACHE)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RNG_SEED)

    print("[backtest] Loading lifecycle cohort ...")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET).reset_index(drop=True)
    vins = lc["vin_label"].tolist()
    durations = lc["age_days_observed"].values.astype(float)
    events = lc["failed_flag"].astype(int).values

    failed = lc[lc["failed_flag"] == True]
    failed_ttf = dict(zip(failed["vin_label"], failed["ttf_days"].astype(float)))
    failed_vins = list(failed_ttf.keys())
    mean_failed_age = float(np.mean(list(failed_ttf.values())))
    print(f"  {len(vins)} VINs, {len(failed_vins)} failed (LOVO folds)")
    print(f"  Failed TTF range: {min(failed_ttf.values()):.0f}-{max(failed_ttf.values()):.0f}d, "
          f"mean {mean_failed_age:.0f}d")

    rows = []          # per_fold_residuals
    cov_total = 0      # PI coverage on total TTF

    # ---- per-fold refit, then both tests ---------------------------------
    for k in failed_vins:
        ttf_k = failed_ttf[k]
        keep = np.array([v != k for v in vins])
        shape_s, scale_s, map_median = _refit_fold(durations[keep], events[keep], rng)

        # leave-k-out empirical fleet clock (other 9 failed)
        other_ttf = [t for v, t in failed_ttf.items() if v != k]
        dummyA_failtime = float(np.median(other_ttf))   # predicted failure age

        # (1) total-TTF calibration + PI coverage (unconditional draws => a=0)
        pred_total = map_median
        total_draws = surv.conditional_predictive_rul(0.0, shape_s, scale_s, rng)
        p10t = float(np.percentile(total_draws, cfg.PI_LOWER_PCT))
        p90t = float(np.percentile(total_draws, cfg.PI_UPPER_PCT))
        covered_total = bool(p10t <= ttf_k <= p90t)
        cov_total += int(covered_total)
        rows.append({
            "vin_label": k, "test": "total_ttf", "horizon_days": np.nan,
            "current_age_days": 0.0, "actual_days": round(ttf_k, 1),
            "model_pred_days": round(pred_total, 1),
            "dummyA_pred_days": round(dummyA_failtime, 1),
            "dummyB_pred_days": round(dummyA_failtime, 1),
            "p10_days": round(p10t, 1), "p90_days": round(p90t, 1),
            "model_abs_err": round(abs(pred_total - ttf_k), 1),
            "dummyA_abs_err": round(abs(dummyA_failtime - ttf_k), 1),
            "pi_covered": covered_total,
        })

        # (2) time-rewound RUL at horizons
        for H in cfg.BACKTEST_HORIZONS_DAYS:
            age = ttf_k - H
            if age <= 0:
                continue
            med, p10, p90 = surv.predictive_rul_summary(
                age, shape_s, scale_s, rng,
                lower_pct=cfg.PI_LOWER_PCT, upper_pct=cfg.PI_UPPER_PCT,
            )
            actual = float(H)
            dummyA = dummyA_failtime - age          # fleet clock at this age
            dummyB = dummyA_failtime - mean_failed_age   # constant, ignores age
            rows.append({
                "vin_label": k, "test": "rewound", "horizon_days": H,
                "current_age_days": round(age, 1), "actual_days": actual,
                "model_pred_days": round(med, 1),
                "dummyA_pred_days": round(dummyA, 1),
                "dummyB_pred_days": round(dummyB, 1),
                "p10_days": round(p10, 1), "p90_days": round(p90, 1),
                "model_abs_err": round(abs(med - actual), 1),
                "dummyA_abs_err": round(abs(dummyA - actual), 1),
                "pi_covered": bool(p10 <= actual <= p90),
            })
        print(f"  fold {k:<14} ttf={ttf_k:.0f}  pred_total={pred_total:.0f}  "
              f"dummyA={dummyA_failtime:.0f}  PI[{p10t:.0f},{p90t:.0f}] cov={covered_total}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_fold_residuals.csv", index=False)

    # ---- aggregate -------------------------------------------------------
    def _agg(sub):
        m = float(sub["model_abs_err"].mean())
        med_ae = float(sub["model_abs_err"].median())
        a = float(sub["dummyA_abs_err"].mean())
        cov = float(sub["pi_covered"].mean())
        # paired signed-rank: model vs dummyA absolute errors
        try:
            stat, p = stats.wilcoxon(sub["model_abs_err"], sub["dummyA_abs_err"])
            p = float(p)
        except Exception:
            p = None
        return {"n": int(len(sub)), "mae_model": round(m, 1), "median_ae_model": round(med_ae, 1),
                "mae_dummyA": round(a, 1), "pi_coverage": round(cov, 3),
                "signed_rank_p": (round(p, 4) if p is not None else None)}

    total = df[df["test"] == "total_ttf"]
    rew = df[df["test"] == "rewound"]

    total_agg = _agg(total)
    rew_agg = _agg(rew)
    by_h = {str(H): _agg(rew[rew["horizon_days"] == H]) for H in cfg.BACKTEST_HORIZONS_DAYS
            if (rew["horizon_days"] == H).any()}

    verdict = "model_wins" if rew_agg["mae_model"] < rew_agg["mae_dummyA"] else "no_improvement"
    pi_cov_total = f"{cov_total}/{len(failed_vins)}"
    pi_gate_pass = cov_total >= int(np.ceil(cfg.PI_COVERAGE_TARGET * len(failed_vins)))

    results = {
        "cohort": "all_25", "n_events": len(failed_vins), "per_fold_refit": True,
        "lovo_total_ttf": {**total_agg, "pi_coverage_k_of_n": pi_cov_total,
                            "pi_gate_pass": bool(pi_gate_pass)},
        "time_rewound": {"overall": rew_agg, "by_horizon": by_h, "verdict": verdict},
        "interpretation": (
            "Tier-A RUL is age-only; the honest question is whether the survival "
            "shape beats a linear fleet-clock subtraction. Verdict + coverage below."
        ),
    }
    with open(out / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary -------------------------------------------------
    print("\n" + "=" * 64)
    print("BACKTEST SUMMARY (out-of-sample, per-fold refit)")
    print("=" * 64)
    print(f"  Total-TTF : MAE model={total_agg['mae_model']:.0f}d  "
          f"dummyA={total_agg['mae_dummyA']:.0f}d   "
          f"PI coverage={pi_cov_total} (gate >= {int(np.ceil(cfg.PI_COVERAGE_TARGET*len(failed_vins)))}/10: "
          f"{'PASS' if pi_gate_pass else 'FAIL'})")
    print(f"  Rewound   : MAE model={rew_agg['mae_model']:.0f}d  "
          f"dummyA={rew_agg['mae_dummyA']:.0f}d  "
          f"signed-rank p={rew_agg['signed_rank_p']}  PI cov={rew_agg['pi_coverage']:.2f}")
    for H, a in by_h.items():
        print(f"     T-{H:>3}d : MAE model={a['mae_model']:.0f}d  dummyA={a['mae_dummyA']:.0f}d  "
              f"cov={a['pi_coverage']:.2f}  n={a['n']}")
    print(f"\n  VERDICT: {verdict.upper()}  "
          f"({'model beats fleet clock' if verdict=='model_wins' else 'no day-MAE improvement over fleet clock'})")
    print(f"  Saved backtest_results.json + per_fold_residuals.csv")


if __name__ == "__main__":
    main()
