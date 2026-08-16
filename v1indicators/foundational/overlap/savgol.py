import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _weighted_window_kernel(x: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
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


def savgol_weights(window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing weights, evaluated at the window centre.

    Least-squares polynomial of order ``polyorder`` over ``window`` equally
    spaced points, evaluated at the centre of the fit. The weights are
    applied to a TRAILING window ending at the current bar, so the output is
    the classic centred Savitzky-Golay value delayed by (window-1)/2 bars —
    strictly causal, with the smoothing properties of centred SG.
    """
    if window <= polyorder:
        raise ValueError("window must be > polyorder")
    if window < 2:
        raise ValueError("window must be >= 2")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")

    t = np.arange(window, dtype=np.float64) - (window - 1) / 2.0
    vander = np.vander(t, polyorder + 1, increasing=True)
    # coefficients of the LS fit: c = pinv(V) @ y; the centre evaluation is
    # the constant term c0 = weights @ y with weights = pinv(V)[0, :].
    weights = np.linalg.pinv(vander)[0, :]
    return weights


def savgol(
    close: pd.Series,
    window: int = 7,
    polyorder: int = 2,
) -> pd.Series:
    """One-sided (causal) Savitzky-Golay smoother.

    Classic Savitzky-Golay smoothing fits a polynomial over a CENTRED window
    and is therefore look-ahead by construction. This implementation applies
    the same least-squares weights to a TRAILING window ending at the current
    bar: the output equals the centred SG smooth delayed by (window-1)/2
    bars, preserving the noise-suppression/polyline-preserving trade-off
    without using a single future bar.

    Parameters
    ----------
    close : pd.Series
        Input series.
    window : int
        Trailing window length (must exceed ``polyorder``).
    polyorder : int
        Polynomial order of the local fit.

    Returns
    -------
    pd.Series
        Smoothed series named ``SAVGOL_{window}``; NaN during warmup and for
        windows containing NaN.
    """
    close_s = check_series(close, "close")

    # The kernel consumes weights in x[i], x[i-1], ... order.
    weights = savgol_weights(window, polyorder)[::-1].copy()
    out = _weighted_window_kernel(
        close_s.to_numpy(dtype=np.float64), weights, weights.shape[0]
    )
    return pd.Series(out, index=close_s.index, name=f"SAVGOL_{window}")
