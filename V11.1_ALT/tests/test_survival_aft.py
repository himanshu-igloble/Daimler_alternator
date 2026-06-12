import importlib.util, pathlib
import numpy as np
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
S = _load("V11_1_ALT_survival")

def test_aft_beta_zero_matches_plain_grid():
    rng = np.random.default_rng(0)
    t = rng.weibull(4.0, 40) * 700; e = np.ones(40, dtype=int)
    x = np.zeros((40, 1))
    post = S.fit_aft_grid_posterior(t, e, x, prior_shape=3.5, prior_scale=650,
        prior_shape_sd=1.5, prior_scale_sd=100, shape_lo=2, shape_hi=12, shape_n=40,
        scale_lo=500, scale_hi=1100, scale_n=40, beta_grids=[np.array([0.0])])
    assert 3.0 < post["map_shape"] < 6.0 and 600 < post["map_scale0"] < 800
    assert post["map_beta"] == [0.0]

def test_aft_recovers_positive_beta():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.integers(0, 2, size=n).astype(float).reshape(-1, 1)
    true_scale = 700 * np.exp(-0.6 * x[:, 0])
    t = rng.weibull(4.0, n) * true_scale; e = np.ones(n, dtype=int)
    post = S.fit_aft_grid_posterior(t, e, x, prior_shape=3.5, prior_scale=650,
        prior_shape_sd=1.5, prior_scale_sd=150, shape_lo=2, shape_hi=12, shape_n=40,
        scale_lo=400, scale_hi=1100, scale_n=50,
        beta_grids=[np.linspace(-0.2, 1.2, 29)])
    assert 0.3 < post["map_beta"][0] < 0.9

def test_sample_aft_posterior_shapes():
    rng = np.random.default_rng(2)
    t = rng.weibull(4.0, 30) * 700; e = np.ones(30, dtype=int)
    x = np.zeros((30, 1))
    post = S.fit_aft_grid_posterior(t, e, x, prior_shape=3.5, prior_scale=650,
        prior_shape_sd=1.5, prior_scale_sd=100, shape_lo=2, shape_hi=12, shape_n=20,
        scale_lo=500, scale_hi=1100, scale_n=20, beta_grids=[np.linspace(-0.2, 1.0, 7)])
    ks, ls, bs = S.sample_aft_posterior(post, 500, rng)
    assert ks.shape == (500,) and ls.shape == (500,) and bs.shape == (500, 1)
