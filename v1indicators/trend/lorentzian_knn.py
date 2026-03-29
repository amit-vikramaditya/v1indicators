import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _lorentzian_knn_kernel(
    f1: np.ndarray,
    f2: np.ndarray,
    close_v: np.ndarray,
    neighbors_count: int,
    max_bars_back: int,
    horizon: int,
    stride: int,
):
    n = close_v.shape[0]
    pred = np.full(n, np.nan, dtype=np.float64)
    signal = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if i < horizon or i >= n - horizon:
            continue

        start = i - max_bars_back
        if start < horizon:
            start = horizon

        # Keep a small sorted list of nearest neighbors.
        dists = np.full(neighbors_count, np.inf, dtype=np.float64)
        labels = np.zeros(neighbors_count, dtype=np.int8)

        for j in range(start, i, stride):
            if np.isnan(f1[i]) or np.isnan(f2[i]) or np.isnan(f1[j]) or np.isnan(f2[j]):
                continue

            d = np.log(1.0 + abs(f1[i] - f1[j])) + np.log(1.0 + abs(f2[i] - f2[j]))

            # Label from future move at j.
            future_idx = j + horizon
            if future_idx >= n:
                continue

            lbl = 0
            if close_v[future_idx] > close_v[j]:
                lbl = 1
            elif close_v[future_idx] < close_v[j]:
                lbl = -1

            # Insert if better than worst neighbor.
            worst_idx = 0
            worst_dist = dists[0]
            for k in range(1, neighbors_count):
                if dists[k] > worst_dist:
                    worst_dist = dists[k]
                    worst_idx = k

            if d < worst_dist:
                dists[worst_idx] = d
                labels[worst_idx] = lbl

        score = 0.0
        cnt = 0
        for k in range(neighbors_count):
            if dists[k] < np.inf:
                score += labels[k]
                cnt += 1

        if cnt > 0:
            pred[i] = score
            if score > 0:
                signal[i] = 1
            elif score < 0:
                signal[i] = -1

    return pred, signal


def lorentzian_knn(
    close: pd.Series,
    neighbors_count: int = 8,
    max_bars_back: int = 2000,
    horizon: int = 4,
    stride: int = 4,
    feature_fast: int = 5,
    feature_slow: int = 14,
) -> pd.DataFrame:
    """
    Lorentzian-distance KNN directional classifier.

    Features:
    - F1: normalized return over `feature_fast`
    - F2: smoothed return over `feature_slow`
    """
    if neighbors_count <= 0:
        raise ValueError("neighbors_count must be > 0")
    if max_bars_back <= 0:
        raise ValueError("max_bars_back must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if feature_fast <= 0 or feature_slow <= 0:
        raise ValueError("feature_fast and feature_slow must be > 0")

    close_s = check_series(close, "close")

    f1 = close_s.pct_change(feature_fast)
    f2 = close_s.pct_change().rolling(feature_slow).mean()

    pred, sig = _lorentzian_knn_kernel(
        f1.to_numpy(dtype=np.float64),
        f2.to_numpy(dtype=np.float64),
        close_s.to_numpy(dtype=np.float64),
        int(neighbors_count),
        int(max_bars_back),
        int(horizon),
        int(stride),
    )

    return pd.DataFrame(
        {
            "LKNN_F1": f1,
            "LKNN_F2": f2,
            "LKNN_PRED": pred,
            "LKNN_SIGNAL": sig,
        },
        index=close_s.index,
    )
