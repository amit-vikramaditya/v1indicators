import pandas as pd

from ..._utils import check_series


def massi(high: pd.Series, low: pd.Series, fast: int = 9, slow: int = 25) -> pd.Series:
    """Mass Index."""
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    if slow < fast:
        fast, slow = slow, fast

    hl_range = (high_s - low_s).abs()
    ema1 = hl_range.ewm(span=fast, adjust=False).mean()
    ema2 = ema1.ewm(span=fast, adjust=False).mean()

    out = (ema1 / ema2).rolling(slow).sum()
    out.name = f"MASSI_{fast}_{slow}"
    return out
