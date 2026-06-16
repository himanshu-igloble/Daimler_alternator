"""
V10.6.2 Alternator — Shared Survival Math (helper module, no main())
====================================================================
Single source of truth for the Weibull survival math so the fleet fit
(weibull_fleet), the per-VIN predictive interval (predictive_rul), and the
backtest harness (backtest) all use *identical* formulas.  This avoids the
class of bug where the conditional-survival RUL is implemented three slightly
different ways.

Conventions: Weibull S(t) = exp(-(t/scale)**shape).  `scale` is lifelines'
lambda_, `shape` is lifelines' rho_.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core Weibull functions (vectorised over shape/scale grids or sample arrays)
# ---------------------------------------------------------------------------

def weibull_log_sf(t, shape, scale):
    """log S(t) = -(t/scale)**shape."""
    return -np.power(t / scale, shape)


def weibull_sf(t, shape, scale):
    """Survival S(t)."""
    return np.exp(weibull_log_sf(t, shape, scale))


def weibull_log_pdf(t, shape, scale):
    """log f(t)."""
    return (
        np.log(shape / scale)
        + (shape - 1.0) * np.log(t / scale)
        + weibull_log_sf(t, shape, scale)
    )


def weibull_median(shape, scale):
    """Median = scale * ln(2)**(1/shape)."""
    return scale * np.log(2) ** (1.0 / shape)


# ---------------------------------------------------------------------------
# Grid Bayesian posterior over (shape, scale)
# ---------------------------------------------------------------------------

def fit_grid_posterior(durations, events, *,
                       prior_shape, prior_scale, prior_shape_sd, prior_scale_sd,
                       shape_lo, shape_hi, shape_n,
                       scale_lo, scale_hi, scale_n):
    """Compute a normalised 2-D posterior over (shape, scale).

    durations : array of observed times (days)
    events    : array of 0/1 (1 = failure observed, 0 = right-censored)

    Returns dict with shape_grid, scale_grid, posterior (shape_n x scale_n),
    map_shape, map_scale, mean_shape, mean_scale.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    shape_grid = np.linspace(shape_lo, shape_hi, shape_n)
    scale_grid = np.linspace(scale_lo, scale_hi, scale_n)
    SHAPE, SCALE = np.meshgrid(shape_grid, scale_grid, indexing="ij")

    log_prior = (
        stats.norm.logpdf(SHAPE, prior_shape, prior_shape_sd)
        + stats.norm.logpdf(SCALE, prior_scale, prior_scale_sd)
    )

    log_lik = np.zeros_like(SHAPE)
    for t_i, e_i in zip(durations, events):
        if e_i == 1:
            log_lik += weibull_log_pdf(t_i, SHAPE, SCALE)
        else:
            log_lik += weibull_log_sf(t_i, SHAPE, SCALE)

    log_post = log_prior + log_lik
    log_post -= log_post.max()
    posterior = np.exp(log_post)
    posterior /= posterior.sum()

    map_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
    return {
        "shape_grid": shape_grid,
        "scale_grid": scale_grid,
        "posterior": posterior,
        "map_shape": float(shape_grid[map_idx[0]]),
        "map_scale": float(scale_grid[map_idx[1]]),
        "mean_shape": float(np.sum(SHAPE * posterior)),
        "mean_scale": float(np.sum(SCALE * posterior)),
    }


def sample_posterior(shape_grid, scale_grid, posterior, n, rng):
    """Draw n (shape, scale) pairs from the grid posterior, jittered within
    each cell so the draws are continuous rather than snapped to grid nodes.
    """
    flat = posterior.ravel()
    flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, replace=True, p=flat)
    i, j = np.unravel_index(idx, posterior.shape)

    d_shape = (shape_grid[1] - shape_grid[0]) if len(shape_grid) > 1 else 0.0
    d_scale = (scale_grid[1] - scale_grid[0]) if len(scale_grid) > 1 else 0.0
    shape_s = shape_grid[i] + rng.uniform(-0.5, 0.5, size=n) * d_shape
    scale_s = scale_grid[j] + rng.uniform(-0.5, 0.5, size=n) * d_scale

    # keep strictly positive
    shape_s = np.clip(shape_s, 1e-3, None)
    scale_s = np.clip(scale_s, 1e-3, None)
    return shape_s, scale_s


# ---------------------------------------------------------------------------
# Conditional predictive RUL  (D7)
# ---------------------------------------------------------------------------

def conditional_predictive_rul(a, shape_samples, scale_samples, rng):
    """Sample R = T - a | T > a for a truck of current age `a`.

    Derivation: P(T > a+r | T > a) = S(a+r)/S(a) = U  ~Uniform(0,1)
        ((a+r)/scale)^shape = (a/scale)^shape - ln(U)
        r = scale * [ (a/scale)^shape - ln(U) ]^(1/shape) - a

    Returns an array of RUL draws (days), clipped at >= 0.  This single draw
    carries BOTH epistemic uncertainty (shape/scale vary across the posterior
    samples) and aleatoric uncertainty (U).
    """
    shape_samples = np.asarray(shape_samples, dtype=float)
    scale_samples = np.asarray(scale_samples, dtype=float)
    U = rng.uniform(size=shape_samples.shape)
    inner = np.power(a / scale_samples, shape_samples) - np.log(U)
    inner = np.clip(inner, 0.0, None)
    t_fail = scale_samples * np.power(inner, 1.0 / shape_samples)
    return np.clip(t_fail - a, 0.0, None)


def predictive_rul_summary(a, shape_samples, scale_samples, rng,
                           lower_pct=10.0, upper_pct=90.0):
    """Return (median, p_lower, p_upper) of the conditional predictive RUL."""
    draws = conditional_predictive_rul(a, shape_samples, scale_samples, rng)
    return (
        float(np.median(draws)),
        float(np.percentile(draws, lower_pct)),
        float(np.percentile(draws, upper_pct)),
    )
