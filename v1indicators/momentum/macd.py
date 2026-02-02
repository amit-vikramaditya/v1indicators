import pandas as pd
from ..overlap.ema import ema
from .._utils import check_series

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    A trend-following momentum indicator that shows the relationship between
    two moving averages of a security’s price.

    Formula:
        MACD Line = EMA(close, fast) - EMA(close, slow)
        Signal Line = EMA(MACD Line, signal)
        Histogram = MACD Line - Signal Line

    Args:
        close: Pandas Series of prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        Pandas DataFrame with columns: ['MACD', 'MACD_SIGNAL', 'MACD_HIST'].
    """
    
    if min(fast, slow, signal) <= 0:
        raise ValueError("periods must be > 0")

    # Ensure input is series
    close = check_series(close, "close")

    # EMA handles its own validation but we already did it
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)

    macd_line = fast_ema - slow_ema
    
    # Signal line is EMA of MACD line
    signal_line = ema(macd_line, signal)

    hist = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_SIGNAL": signal_line,
        "MACD_HIST": hist,
    })

