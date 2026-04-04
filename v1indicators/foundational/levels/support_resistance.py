import pandas as pd

from ..._utils import check_series


def support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    left: int = 15,
    right: int = 15,
) -> pd.DataFrame:
    """
    Pivot-based support/resistance levels with breakout flags.

    A pivot high at bar t is the maximum over [t-left, t+right],
    and similarly for pivot lows.
    """
    if left <= 0 or right <= 0:
        raise ValueError("left and right must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    window = left + right + 1

    roll_max = high_s.rolling(window).max().shift(-right)
    roll_min = low_s.rolling(window).min().shift(-right)

    pivot_high = high_s.where(high_s == roll_max)
    pivot_low = low_s.where(low_s == roll_min)

    resistance = pivot_high.ffill()
    support = pivot_low.ffill()

    prev_resistance = resistance.shift(1)
    prev_support = support.shift(1)
    prev_close = close_s.shift(1)

    break_resistance = (close_s > prev_resistance) & (prev_close <= prev_resistance)
    break_support = (close_s < prev_support) & (prev_close >= prev_support)

    return pd.DataFrame(
        {
            "PIVOT_HIGH": pivot_high,
            "PIVOT_LOW": pivot_low,
            "RESISTANCE": resistance,
            "SUPPORT": support,
            "BREAK_RESISTANCE": break_resistance,
            "BREAK_SUPPORT": break_support,
        }
    )
