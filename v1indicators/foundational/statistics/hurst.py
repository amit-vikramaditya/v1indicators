import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _hurst_rs_kernel(x: np.ndarray, window: int, sizes: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    ns = sizes.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    logsz = np.zeros(ns, dtype=np.float64)
    for j in range(ns):
        logsz[j] = np.log(sizes[j])

    for i in range(window - 1, n):
        nan_in_window = False
        for b in range(i - window + 1, i + 1):
            if np.isnan(x[b]):
                nan_in_window = True
                break
        if nan_in_window:
            continue

        logrs = np.zeros(ns, dtype=np.float64)
        valid = 0
        for j in range(ns):
            sz = sizes[j]
            nb = window // sz
            if nb < 1:
                continue
            rs = 0.0
            for b in range(nb):
                # block b of the trailing window: [b*sz, (b+1)*sz)
                mean = 0.0
                for t in range(sz):
                    mean += x[i - window + 1 + b * sz + t]
                mean /= sz

                cumdev = 0.0
                max_dev = -1e300
                min_dev = 1e300
                ss = 0.0
                for t in range(sz):
                    dev = x[i - window + 1 + b * sz + t] - mean
                    cumdev += dev
                    if cumdev > max_dev:
                        max_dev = cumdev
                    if cumdev < min_dev:
                        min_dev = cumdev
                    ss += dev * dev
                r = max_dev - min_dev
                s = np.sqrt(ss / sz)
                if s > 1e-10:
                    rs += r / s
            avg_rs = rs / nb
            if avg_rs > 1e-10:
                logrs[valid] = np.log(avg_rs)
                valid += 1

        if valid >= 2:
            # least-squares slope of log(R/S) vs log(size)
            sx = 0.0
            sy = 0.0
            sxy = 0.0
            sxx = 0.0
            for k in range(valid):
                lx = logsz[k]
                ly = logrs[k]
                sx += lx
                sy += ly
                sxy += lx * ly
                sxx += lx * lx
            denom = valid * sxx - sx * sx
            if np.abs(denom) > 1e-12:
                h = (valid * sxy - sx * sy) / denom
                out[i] = min(1.0, max(0.0, h))

    return out


def hurst(close: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range (R/S) analysis.

    Estimates the self-similarity parameter H over a trailing window using
    nested sub-window sizes (window/4, window/3, window/2, window) and the
    slope of log(R/S) against log(size):

        H > 0.5  -> trending / persistent
        H = 0.5  -> geometric-random-walk-like
        H < 0.5  -> mean-reverting / anti-persistent

    Strictly causal: bar i uses only bars [i-window+1, i]. Windows
    containing NaN are skipped. Estimation is noisy on short windows;
    100+ bars is recommended (20 is the enforced minimum).

    Parameters
    ----------
    close : pd.Series
        Input series.
    window : int
        Trailing analysis window (>= 20).

    Returns
    -------
    pd.Series
        Hurst exponent in [0, 1] named ``HURST_{window}``; NaN during warmup.
    """
    if window < 20:
        raise ValueError("window must be >= 20 (R/S estimation is meaningless below that)")

    close_s = check_series(close, "close")

    sizes = np.array(
        [window // 4, window // 3, window // 2, window], dtype=np.int64
    )
    out = _hurst_rs_kernel(close_s.to_numpy(dtype=np.float64), int(window), sizes)
    return pd.Series(out, index=close_s.index, name=f"HURST_{window}")
