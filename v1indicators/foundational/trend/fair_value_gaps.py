import pandas as pd

from .._utils import check_series


def fair_value_gaps(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Fair Value Gap (FVG) detector.

    Bullish FVG: current low > high[t-2] and close[t-1] > high[t-2].
    Bearish FVG: current high < low[t-2] and close[t-1] < low[t-2].
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    high2 = high_s.shift(2)
    low2 = low_s.shift(2)
    close1 = close_s.shift(1)

    bullish = (low_s > high2) & (close1 > high2)
    bearish = (high_s < low2) & (close1 < low2)

    bull_top = low_s.where(bullish)
    bull_bottom = high2.where(bullish)
    bear_top = low2.where(bearish)
    bear_bottom = high_s.where(bearish)

    if threshold > 0:
        bull_size = (bull_top - bull_bottom).abs()
        bear_size = (bear_top - bear_bottom).abs()
        bullish = bullish & (bull_size >= threshold)
        bearish = bearish & (bear_size >= threshold)
        bull_top = bull_top.where(bullish)
        bull_bottom = bull_bottom.where(bullish)
        bear_top = bear_top.where(bearish)
        bear_bottom = bear_bottom.where(bearish)

    return pd.DataFrame(
        {
            "BULLISH_FVG": bullish.fillna(False),
            "BULLISH_FVG_TOP": bull_top,
            "BULLISH_FVG_BOTTOM": bull_bottom,
            "BEARISH_FVG": bearish.fillna(False),
            "BEARISH_FVG_TOP": bear_top,
            "BEARISH_FVG_BOTTOM": bear_bottom,
        }
    )
