import numpy as np
import pandas as pd

from .._utils import check_series


def order_blocks(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    periods: int = 5,
    threshold: float = 0.0,
    use_wicks: bool = False,
) -> pd.DataFrame:
    """
    Candle-sequence order block detector.

    A bullish order block is the last bearish candle before `periods`
    consecutive bullish candles and sufficient displacement.
    A bearish order block is the inverse.
    """
    if periods <= 0:
        raise ValueError("periods must be > 0")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    ob_period = periods + 1

    bull_candles = close_s > open_s
    bear_candles = close_s < open_s

    bullish_seq = bull_candles.rolling(periods).sum().shift(1) == periods
    bearish_seq = bear_candles.rolling(periods).sum().shift(1) == periods

    base_close = close_s.shift(ob_period).replace(0.0, np.nan)
    displacement = ((close_s.shift(ob_period) - close_s.shift(1)).abs() / base_close) * 100.0
    relmove = displacement >= threshold

    bullish_ob = (close_s.shift(ob_period) < open_s.shift(ob_period)) & bullish_seq & relmove
    bearish_ob = (close_s.shift(ob_period) > open_s.shift(ob_period)) & bearish_seq & relmove

    bullish_upper_base = high_s.shift(ob_period) if use_wicks else open_s.shift(ob_period)
    bullish_lower_base = low_s.shift(ob_period)

    bearish_upper_base = high_s.shift(ob_period)
    bearish_lower_base = low_s.shift(ob_period) if use_wicks else open_s.shift(ob_period)

    bullish_upper = bullish_upper_base.where(bullish_ob)
    bullish_lower = bullish_lower_base.where(bullish_ob)
    bearish_upper = bearish_upper_base.where(bearish_ob)
    bearish_lower = bearish_lower_base.where(bearish_ob)

    return pd.DataFrame(
        {
            "BULLISH_OB": bullish_ob,
            "BULLISH_OB_HIGH": bullish_upper,
            "BULLISH_OB_LOW": bullish_lower,
            "BULLISH_OB_MID": (bullish_upper + bullish_lower) / 2.0,
            "BEARISH_OB": bearish_ob,
            "BEARISH_OB_HIGH": bearish_upper,
            "BEARISH_OB_LOW": bearish_lower,
            "BEARISH_OB_MID": (bearish_upper + bearish_lower) / 2.0,
        },
        index=close_s.index,
    )
