import pandas as pd

from .._utils import check_series


def apo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Absolute Price Oscillator = EMA(close, fast) - EMA(close, slow)."""
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be > 0")
    if slow < fast:
        fast, slow = slow, fast

    close_s = check_series(close, "close")
    min_periods = fast
    ema_fast = close_s.ewm(span=fast, min_periods=min_periods, adjust=True).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=min_periods, adjust=True).mean()

    out = ema_fast - ema_slow
    out.name = f"APO_{fast}_{slow}"
    return out
