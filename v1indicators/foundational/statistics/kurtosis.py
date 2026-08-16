import numpy as np
import pandas as pd

from ..._utils import check_series


def _kurt_window(x: np.ndarray) -> float:
    """Adjusted excess kurtosis (G2, Fisher) of one complete window.

    Computed per window (not via an online accumulator): the result depends
    only on the window's contents, so it is identical whether the series is
    full or truncated — bit-for-bit prefix-invariant on every pandas
    version. pandas' own rolling.kurt uses an online moments accumulator
    whose floating-point drift depends on the series start, which made it
    fail exact prefix-invariance on pandas 2.x.
    """
    n = x.size
    if n < 4:
        return np.nan
    d = x - x.mean()
    m2 = (d * d).mean()
    if m2 <= 0.0:
        return np.nan
    m4 = (d * d * d * d).mean()
    g2 = m4 / (m2 * m2) - 3.0
    return ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)


def kurtosis(close: pd.Series, length: int = 30) -> pd.Series:
    """Rolling kurtosis (adjusted excess kurtosis, G2 / Fisher).

    Same estimator as ``pandas.Series.rolling().kurt()`` (values agree to
    ~1e-12), but computed per window for exact prefix invariance.
    NaN until `length` bars have elapsed.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    close_s = check_series(close, "close")
    out = close_s.rolling(length).apply(_kurt_window, raw=True)
    out.name = f"KURTOSIS_{length}"
    return out
