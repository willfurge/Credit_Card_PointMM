# src/utils/math_tools.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]
Number = Union[int, float, np.number]

# ---- RNG utilities --------------------------------------------------------- #

# Singleton random generator (fast, modern bit generator)
_rng: np.random.Generator = np.random.default_rng()


def set_global_seed(seed: Optional[int] = None) -> np.random.Generator:
    """
    Set (or reset) the module-level RNG for reproducibility.
    If seed is None, uses an entropy source.
    """
    global _rng
    _rng = np.random.default_rng(seed)
    return _rng


def rng() -> np.random.Generator:
    """Return the module-level RNG."""
    return _rng


# ---- Numeric safety helpers ------------------------------------------------ #


def safe_div(numer: Number, denom: Number, default: Number = np.nan) -> Number:
    """
    Divide with safety against zero / nan denominators.
    """
    try:
        if denom == 0 or np.isnan(denom):
            return default
        return numer / denom
    except Exception:
        return default


def clip(x: ArrayLike, low: Number, high: Number) -> np.ndarray:
    """Vectorized clipping with float64 stability."""
    return np.clip(np.asarray(x, dtype=np.float64), low, high)


def clip01(x: ArrayLike) -> np.ndarray:
    """Clip values to [0, 1]."""
    return clip(x, 0.0, 1.0)


# ---- Transformations ------------------------------------------------------- #


def sigmoid(x: ArrayLike) -> np.ndarray:
    """
    Numerically stable logistic function.
    """
    x = np.asarray(x, dtype=np.float64)
    # Prevent overflow for large negative values
    out = np.empty_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    ex = np.exp(x[neg_mask])
    out[neg_mask] = ex / (1 + ex)
    return out


def logit(p: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """
    Inverse of sigmoid; clamps to avoid inf.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def softmax(x: ArrayLike, temperature: float = 1.0, axis: Optional[int] = None) -> np.ndarray:
    """
    Temperature-scaled softmax with numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = x / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


# ---- Descriptive stats ----------------------------------------------------- #


def stable_sum(x: ArrayLike) -> float:
    """
    High-precision sum (float64) with Kahan-style compensation.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    total = 0.0
    c = 0.0
    for v in arr:
        y = v - c
        t = total + y
        c = (t - total) - y
        total = t
    return float(total)


def percentile(
    x: ArrayLike, q: Union[float, Sequence[float]], method: str = "linear"
) -> np.ndarray:
    """
    Wrapper for np.percentile with consistent dtype/empty-handling.
    Returns np.nan for empty inputs.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return np.full_like(np.asarray(q, dtype=np.float64), np.nan, dtype=np.float64)
    return np.percentile(arr, q, method=method)


def robust_zscore(x: ArrayLike, eps: float = 1e-9) -> np.ndarray:
    """
    Robust z-score using median and MAD.
    """
    arr = np.asarray(x, dtype=np.float64)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + eps
    return 0.6744897501960817 * (arr - med) / mad  # 0.67449 = Phi^{-1}(0.75)


def zscore(x: ArrayLike, ddof: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Standard z-score. Adds eps to denom for numerical safety.
    """
    arr = np.asarray(x, dtype=np.float64)
    mu = arr.mean() if arr.size else 0.0
    sd = arr.std(ddof=ddof) + eps
    return (arr - mu) / sd


def normalize_minmax(x: ArrayLike) -> np.ndarray:
    """
    Min-max normalization to [0,1]. Returns zeros if constant.
    """
    arr = np.asarray(x, dtype=np.float64)
    lo, hi = np.min(arr), np.max(arr)
    if hi - lo == 0:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def quantile_clip(x: ArrayLike, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """
    Winsorize values to [q_lower, q_upper].
    """
    arr = np.asarray(x, dtype=np.float64)
    ql, qu = np.quantile(arr, [lower, upper])
    return np.clip(arr, ql, qu)


# ---- Online / streaming stats --------------------------------------------- #


@dataclass
class OnlineStats:
    """
    Welford's algorithm for streaming mean/variance.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations from the mean

    def update(self, x: ArrayLike) -> "OnlineStats":
        arr = np.asarray(x, dtype=np.float64).ravel()
        for v in arr:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.m2 += delta * delta2
        return self

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def snapshot(self) -> Tuple[int, float, float]:
        """Return (n, mean, std)."""
        return (self.n, self.mean, self.std)


# ---- Smoothing ------------------------------------------------------------- #


def ema(x: ArrayLike, alpha: float) -> np.ndarray:
    """
    Exponential moving average (1D). alpha in (0,1].
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def rolling_mean(x: ArrayLike, window: int) -> np.ndarray:
    """
    Simple rolling mean with 'same' length; pads with NaN at start.
    """
    if window <= 0:
        raise ValueError("window must be positive.")
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < window:
        return np.full_like(arr, np.nan)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    vals = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, vals])


# ---- Probability helpers --------------------------------------------------- #


def safe_prob(p: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """
    Clamp probabilities to (eps, 1-eps) to avoid log(0) issues.
    """
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)


def logsumexp(x: ArrayLike, axis: Optional[int] = None) -> np.ndarray:
    """
    Numerically stable log-sum-exp.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze()
