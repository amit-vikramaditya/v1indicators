import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _fracdiff_kernel(x: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(k - 1, n):
        acc = 0.0
        bad = False
        for j in range(k):
            v = x[i - j]
            if np.isnan(v):
                bad = True
                break
            acc += weights[j] * v
        if not bad:
            out[i] = acc
    return out


def _fracdiff_weights(d: float, min_weight: float, max_weights: int) -> np.ndarray:
    """Binomial fractional-difference weights, truncated at ``min_weight``.

    w[0] = 1; w[k] = -w[k-1] * (d - k + 1) / k  (Lopez de Prado, 2018).
    """
    weights = np.zeros(max_weights, dtype=np.float64)
    weights[0] = 1.0
    k = 1
    while k < max_weights:
        weights[k] = -weights[k - 1] * (d - k + 1) / k
        if abs(weights[k]) < min_weight:
            break
        k += 1
    return weights[:k].copy()


def fractional_difference(
    close: pd.Series,
    d: float = 0.5,
    min_weight: float = 1e-4,
    max_weights: int = 1000,
) -> pd.Series:
    """Fractional difference of a price series (Lopez de Prado, 2018).

    A memory-preserving stationarity transform: instead of the full first
    difference (d=1, which erases all long-run structure) or raw levels
    (d=0, non-stationary), fractional differencing with 0 < d < 1 removes
    the unit root while retaining most of the original signal.

    y[i] = sum_j w[j] * x[i - j],  w[0] = 1,
    w[k] = -w[k-1] * (d - k + 1) / k

    The weight sequence is truncated when |w[k]| drops below ``min_weight``
    or at ``max_weights`` terms. Computation is strictly causal: bar i uses
    only bars i, i-1, ..., i-k+1. Windows containing NaN are skipped.

    Parameters
    ----------
    close : pd.Series
        Input series (price levels by convention).
    d : float
        Differencing order, 0 <= d <= 1.
    min_weight : float
        Weight-magnitude cutoff for truncation.
    max_weights : int
        Hard cap on the number of weights.

    Returns
    -------
    pd.Series
        Fractionally differenced series named ``FRACDIFF_{d}``; NaN during
        the warmup of the first (k-1) bars.
    """
    if not 0.0 <= d <= 1.0:
        raise ValueError("d must be in [0, 1]")
    if min_weight <= 0.0:
        raise ValueError("min_weight must be > 0")
    if max_weights <= 0:
        raise ValueError("max_weights must be > 0")

    close_s = check_series(close, "close")

    weights = _fracdiff_weights(d, min_weight, max_weights)
    out = _fracdiff_kernel(close_s.to_numpy(dtype=np.float64), weights, weights.shape[0])
    return pd.Series(out, index=close_s.index, name=f"FRACDIFF_{d}")
