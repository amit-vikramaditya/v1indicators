import pandas as pd
from .._utils import check_series

def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku Cloud."""

    if min(tenkan, kijun, senkou_b) <= 0:
        raise ValueError("periods must be > 0")

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    # Midpoints
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2

    # Senkou Span A (Leading Span A)
    # Average of Tenkan and Kijun, shifted forward by Kijun length
    senkou_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B)
    # 52-period midpoint, shifted forward by Kijun length
    senkou_b_line = (
        (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    ).shift(kijun)

    # Chikou Span (Lagging Span)
    # Current close, shifted back by Kijun length
    chikou = close.shift(-kijun)

    return pd.DataFrame({
        "ICHIMOKU_TENKAN": tenkan_line,
        "ICHIMOKU_KIJUN": kijun_line,
        "ICHIMOKU_SPAN_A": senkou_a,
        "ICHIMOKU_SPAN_B": senkou_b_line,
        "ICHIMOKU_CHIKOU": chikou,
    })

