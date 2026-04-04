import pandas as pd

from .._utils import check_series


def ao(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
    """Awesome Oscillator = SMA(median, fast) - SMA(median, slow)."""
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    median = 0.5 * (high_s + low_s)
    out = median.rolling(fast).mean() - median.rolling(slow).mean()
    out.name = f"AO_{fast}_{slow}"
    return out
