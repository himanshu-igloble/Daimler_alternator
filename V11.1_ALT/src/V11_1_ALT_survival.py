"""
V11.1_ALT — Survival Math
=========================
Extension of V10.6.2's shared survival math module.  This module copies the
core Weibull functions verbatim from V10.6.2_ALT_survival.py and adds the
Accelerated Failure Time (AFT) grid posterior and sampling functions for the
M0/M1/M2 covariate model variants.

Convention: Weibull S(t) = exp(-(t/scale)**shape).  In the AFT model,
the per-truck scale is lambda_i = scale0 * exp(-(x_i @ beta)), so at
beta=0 the model reduces exactly to the plain Weibull of V10.6.2.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core Weibull functions (vectorised over shape/scale grids or sample arrays)
# Copied VERBATIM from V10.6.2_ALT/src/V10.6.2_ALT_survival.py
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
# Conditional predictive RUL  (D7)
# Copied VERBATIM from V10.6.2_ALT/src/V10.6.2_ALT_survival.py
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


# ---------------------------------------------------------------------------
# AFT grid posterior (V11.1 extension)
# ---------------------------------------------------------------------------

def fit_aft_grid_posterior(durations, events, x, *, prior_shape, prior_scale,
                           prior_shape_sd, prior_scale_sd, shape_lo, shape_hi, shape_n,
                           scale_lo, scale_hi, scale_n, beta_grids,
                           beta_prior_sd=0.5):
    """Grid posterior over (shape, scale0, beta_1..beta_p).
    lambda_i = scale0 * exp(-(x_i @ beta)). beta_grids: list of 1-D arrays.
    Returns dict with axes, posterior (ndim = 2+p), MAP values, and meta."""
    from scipy import stats
    durations = np.asarray(durations, float); events = np.asarray(events, int)
    x = np.asarray(x, float)
    if x.ndim == 1: x = x.reshape(-1, 1)
    p = x.shape[1]
    assert len(beta_grids) == p
    shape_grid = np.linspace(shape_lo, shape_hi, shape_n)
    scale_grid = np.linspace(scale_lo, scale_hi, scale_n)
    axes = [shape_grid, scale_grid] + list(beta_grids)
    mesh = np.meshgrid(*axes, indexing="ij")
    SHAPE, SCALE0 = mesh[0], mesh[1]
    BETAS = mesh[2:]
    log_prior = (stats.norm.logpdf(SHAPE, prior_shape, prior_shape_sd)
                 + stats.norm.logpdf(SCALE0, prior_scale, prior_scale_sd))
    for B in BETAS:
        log_prior = log_prior + stats.norm.logpdf(B, 0.0, beta_prior_sd)
    log_lik = np.zeros_like(SHAPE)
    for t_i, e_i, x_i in zip(durations, events, x):
        eta = np.zeros_like(SHAPE)
        for j, B in enumerate(BETAS):
            eta = eta + B * x_i[j]
        lam_i = SCALE0 * np.exp(-eta)
        if e_i == 1:
            log_lik += weibull_log_pdf(t_i, SHAPE, lam_i)
        else:
            log_lik += weibull_log_sf(t_i, SHAPE, lam_i)
    log_post = log_prior + log_lik
    log_post -= log_post.max()
    post = np.exp(log_post); post /= post.sum()
    midx = np.unravel_index(np.argmax(post), post.shape)
    return {"axes": axes, "posterior": post,
            "map_shape": float(shape_grid[midx[0]]),
            "map_scale0": float(scale_grid[midx[1]]),
            "map_beta": [float(beta_grids[j][midx[2 + j]]) for j in range(p)],
            "p": p}

def sample_aft_posterior(post, n, rng):
    """Draw n tuples (shape, scale0, betas[n,p]) from the AFT grid posterior,
    cell-jittered like V10.6.2's sample_posterior."""
    axes, posterior, p = post["axes"], post["posterior"], post["p"]
    flat = posterior.ravel(); flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, replace=True, p=flat)
    multi = np.unravel_index(idx, posterior.shape)
    def jit(grid, ii):
        d = (grid[1] - grid[0]) if len(grid) > 1 else 0.0
        return grid[ii] + rng.uniform(-0.5, 0.5, size=n) * d
    ks = np.clip(jit(axes[0], multi[0]), 1e-3, None)
    ls = np.clip(jit(axes[1], multi[1]), 1e-3, None)
    bs = np.column_stack([jit(axes[2 + j], multi[2 + j]) for j in range(p)]) if p else np.zeros((n, 0))
    return ks, ls, bs

def scale_for(scale0_s, beta_s, x_vec):
    """Per-draw per-truck scale: scale0 * exp(-(x @ beta)). x_vec shape (p,)."""
    eta = beta_s @ np.asarray(x_vec, float) if beta_s.size else 0.0
    return scale0_s * np.exp(-eta)
