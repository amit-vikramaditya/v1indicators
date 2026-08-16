import numpy as np
import pandas as pd

from ..._utils import check_series


def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility estimator.

    Per-bar variance estimate:

        sigma^2 = 0.5 * (ln(high / low))^2 - (2 * ln(2) - 1) * (ln(close / open))^2

    The single-bar estimator can be negative (both terms are squared log
    ratios with different coefficients); those bars return NaN rather than a
    clipped, silently optimistic zero. When ``window`` is given, variances
    are averaged over a trailing window first — the standard multi-sample
    form, which is both more efficient and almost surely positive.

    Parameters
    ----------
    open_, high, low, close : pd.Series
        OHLC prices.
    window : int | None
        Optional trailing window for averaging the variance estimate before
        the square root. ``None`` returns the raw per-bar estimate.

    Returns
    -------
    pd.Series
        Volatility (standard deviation) estimate named ``GARMAN_KLASS``.
    """
    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    log_hl = np.log(high_s / low_s.replace(0.0, np.nan))
    log_co = np.log(close_s / open_s.replace(0.0, np.nan))

    variance = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    variance = variance.where(variance > 0.0)

    if window is not None:
        if window <= 0:
            raise ValueError("window must be > 0")
        variance = variance.rolling(window).mean()
        variance = variance.where(variance > 0.0)

    out = np.sqrt(variance)
    out.name = "GARMAN_KLASS" if window is None else f"GARMAN_KLASS_{window}"
    return out
