import numpy as np
import pandas as pd

from ..._utils import check_series


def _skew_window(x: np.ndarray) -> float:
    """Adjusted Fisher-Pearson skewness (G1) of one complete window.

    Computed per window (not via an online accumulator): the result depends
    only on the window's contents, so it is identical whether the series is
    full or truncated — bit-for-bit prefix-invariant on every pandas
    version. pandas' own rolling.skew uses an online moments accumulator
    whose floating-point drift depends on the series start, which made it
    fail exact prefix-invariance on pandas 2.x.
    """
    n = x.size
    if n < 3:
        return np.nan
    d = x - x.mean()
    m2 = (d * d).mean()
    if m2 <= 0.0:
        return np.nan
    m3 = (d * d * d).mean()
    g1 = m3 / m2**1.5
    return g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0)


def skew(close: pd.Series, length: int = 30) -> pd.Series:
    """Rolling skewness (adjusted Fisher-Pearson G1).

    Same estimator as ``pandas.Series.rolling().skew()`` (values agree to
    ~1e-12), but computed per window for exact prefix invariance.
    NaN until `length` bars have elapsed.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    close_s = check_series(close, "close")
    out = close_s.rolling(length).apply(_skew_window, raw=True)
    out.name = f"SKEW_{length}"
    return out
