import pandas as pd
from .._utils import check_series

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    mamode: str = "ema",
    drift: int = 1,
) -> pd.Series:
    """
    Average True Range (ATR).

    A market volatility indicator derived from the greatest of three values (True Range).
    Uses Wilder's Smoothing (RMA).

    Formula:
        TR = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = RMA(TR, length)

    Args:
        high: Pandas Series of high prices.
        low: Pandas Series of low prices.
        close: Pandas Series of close prices.
        length: Smoothing period (default 14).

    Returns:
        Pandas Series named 'ATR_{length}'.
    """
    
    if length <= 0:
        raise ValueError("length must be > 0")
    if drift <= 0:
        raise ValueError("drift must be > 0")
        
    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    prev_close = close.shift(drift)

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    min_periods = length
    mode = mamode.lower() if mamode else "ema"
    if mode == "sma":
        result = tr.rolling(length, min_periods=min_periods).mean()
    else:
        result = tr.ewm(span=length, min_periods=min_periods, adjust=True).mean()

    result.name = f"ATR_{length}"
    return result

