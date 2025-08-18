import math

import numpy as np

from src.utils.math_tools import (
    OnlineStats,
    ema,
    logit,
    logsumexp,
    normalize_minmax,
    percentile,
    quantile_clip,
    rng,
    robust_zscore,
    rolling_mean,
    safe_div,
    safe_prob,
    set_global_seed,
    sigmoid,
    softmax,
    stable_sum,
    zscore,
)


def test_safe_div():
    assert safe_div(4, 2) == 2
    assert math.isnan(safe_div(1, 0))
    assert math.isnan(safe_div(1, float("nan")))


def test_sigmoid_logit_roundtrip():
    x = np.array([-5.0, 0.0, 5.0])
    p = sigmoid(x)
    x2 = logit(p)
    assert np.allclose(x, x2, atol=1e-8)


def test_softmax_temperature():
    x = np.array([1.0, 2.0, 3.0])
    s1 = softmax(x, temperature=1.0)
    s2 = softmax(x, temperature=0.5)
    assert np.isclose(s1.sum(), 1.0)
    assert np.isclose(s2.sum(), 1.0)
    # lower temperature => peakier distribution
    assert s2[-1] > s1[-1]


def test_stable_sum_matches_numpy():
    x = np.linspace(0, 1, 10000)
    assert np.isclose(stable_sum(x), x.sum(), rtol=1e-12, atol=1e-12)


def test_percentile_and_normalize():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    assert np.allclose(percentile(x, [0, 50, 100]), [1, 3, 5])
    nm = normalize_minmax(x)
    assert np.allclose(nm, [0, 0.25, 0.5, 0.75, 1.0])


def test_robust_and_std_zscores():
    x = np.array([1, 1, 1, 100], dtype=float)
    rz = robust_zscore(x)
    zz = zscore(x)
    # robust z shouldn’t explode on outlier
    assert np.isfinite(rz).all()
    assert np.isfinite(zz).all()


def test_quantile_clip():
    x = np.array([0, 1, 2, 100], float)
    qc = quantile_clip(x, 0.25, 0.75)
    assert qc.min() >= np.quantile(x, 0.25) - 1e-12
    assert qc.max() <= np.quantile(x, 0.75) + 1e-12


def test_online_stats():
    stats = OnlineStats().update([1, 2, 3, 4, 5])
    n, mean, std = stats.snapshot()
    assert n == 5
    assert np.isclose(mean, 3.0)
    assert np.isclose(std, np.std([1, 2, 3, 4, 5], ddof=1))


def test_ema_and_rolling_mean():
    x = np.array([1, 2, 3, 4, 5], float)
    out = ema(x, 0.5)
    assert np.allclose(out, [1.0, 1.5, 2.25, 3.125, 4.0625])
    rm = rolling_mean(x, 3)
    assert np.allclose(rm[-3:], [2.0, 3.0, 4.0])


def test_prob_and_logsumexp():
    p = safe_prob([0.0, 0.5, 1.0])
    assert (p > 0).all() and (p < 1).all()
    x = np.array([1000.0, 1001.0])
    lse = logsumexp(x)
    # Compare to direct stable expression
    m = x.max()
    assert np.isclose(lse, m + np.log(np.sum(np.exp(x - m))))


def test_rng_seed_reproducible():
    set_global_seed(123)
    a = rng().normal(size=3)
    set_global_seed(123)
    b = rng().normal(size=3)
    assert np.allclose(a, b)
