import pandas as pd
from ..overlap.ema import ema


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD)."""

    if not isinstance(close, pd.Series):
        raise TypeError("close must be pandas Series")

    if min(fast, slow, signal) <= 0:
        raise ValueError("periods must be > 0")

    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })

