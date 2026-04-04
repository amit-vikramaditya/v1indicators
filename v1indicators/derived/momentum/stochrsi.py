import numpy as np
import pandas as pd

from .._utils import check_series
from ...foundational.momentum.rsi import rsi


def stochrsi(
    close: pd.Series,
    rsi_length: int = 14,
    stoch_length: int = 14,
    k: int = 3,
    d: int = 3,
) -> pd.DataFrame:
    """
    Stochastic RSI.

    Returns raw StochRSI, %K, and %D lines.
    """
    if min(rsi_length, stoch_length, k, d) <= 0:
        raise ValueError("rsi_length, stoch_length, k, and d must be > 0")

    close_s = check_series(close, "close")
    rsi_s = rsi(close_s, rsi_length)

    ll = rsi_s.rolling(stoch_length).min()
    hh = rsi_s.rolling(stoch_length).max()
    denom = (hh - ll).replace(0.0, np.nan)

    raw = (rsi_s - ll) / denom
    k_line = raw.rolling(k).mean()
    d_line = k_line.rolling(d).mean()

    return pd.DataFrame(
        {
            "STOCHRSI": raw,
            "STOCHRSI_K": k_line,
            "STOCHRSI_D": d_line,
        }
    )
