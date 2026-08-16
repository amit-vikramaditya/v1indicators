import numpy as np
import pandas as pd

from ..._utils import check_series


def parkinson(
    high: pd.Series,
    low: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """Parkinson (1980) high-low range volatility estimator.

    Per-bar variance estimate from the intra-bar range:

        sigma^2 = (ln(high / low))^2 / (4 * ln(2))

    The single-bar estimator only uses the bar's high/low and is causal by
    construction. When ``window`` is given, the per-bar variances are averaged
    over a trailing window (the standard multi-sample Parkinson estimator)
    before taking the square root.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    window : int | None
        Optional trailing window for averaging the variance estimate before
        the square root. ``None`` returns the raw per-bar estimate.

    Returns
    -------
    pd.Series
        Volatility (standard deviation) estimate named ``PARKINSON``.
    """
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    log_hl = np.log(high_s / low_s.replace(0.0, np.nan))
    variance = (log_hl**2) / (4.0 * np.log(2.0))

    if window is not None:
        if window <= 0:
            raise ValueError("window must be > 0")
        variance = variance.rolling(window).mean()

    out = np.sqrt(variance)
    out.name = "PARKINSON" if window is None else f"PARKINSON_{window}"
    return out
