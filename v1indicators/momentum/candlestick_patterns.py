import numpy as np
import pandas as pd

from .._utils import check_series


def candlestick_patterns(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    doji_size: float = 0.05,
) -> pd.DataFrame:
    """
    Candlestick pattern detector.

    Returns common bullish/bearish and neutral pattern flags.
    """
    if doji_size <= 0:
        raise ValueError("doji_size must be > 0")

    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    prev_open = open_s.shift(1)
    prev_close = close_s.shift(1)

    body = (open_s - close_s).abs()
    full_range = (high_s - low_s).replace(0.0, np.nan)

    doji = body <= full_range * doji_size

    bullish_engulfing = (
        (prev_open > prev_close)
        & (close_s > open_s)
        & (close_s >= prev_open)
        & (open_s <= prev_close)
    )
    bearish_engulfing = (
        (prev_close > prev_open)
        & (open_s > close_s)
        & (open_s >= prev_close)
        & (close_s <= prev_open)
    )

    hammer = (
        (high_s - low_s > 3.0 * body)
        & (((close_s - low_s) / (high_s - low_s + 1e-12)) > 0.6)
        & (((open_s - low_s) / (high_s - low_s + 1e-12)) > 0.6)
    )
    inverted_hammer = (
        (high_s - low_s > 3.0 * body)
        & (((high_s - close_s) / (high_s - low_s + 1e-12)) > 0.6)
        & (((high_s - open_s) / (high_s - low_s + 1e-12)) > 0.6)
    )

    morning_star = (
        (close_s.shift(2) < open_s.shift(2))
        & (pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).max(axis=1) < close_s.shift(2))
        & (open_s > pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).max(axis=1))
        & (close_s > open_s)
    )
    evening_star = (
        (close_s.shift(2) > open_s.shift(2))
        & (pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).min(axis=1) > close_s.shift(2))
        & (open_s < pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).min(axis=1))
        & (close_s < open_s)
    )

    return pd.DataFrame(
        {
            "DOJI": doji,
            "BULLISH_ENGULFING": bullish_engulfing,
            "BEARISH_ENGULFING": bearish_engulfing,
            "HAMMER": hammer,
            "INVERTED_HAMMER": inverted_hammer,
            "MORNING_STAR": morning_star,
            "EVENING_STAR": evening_star,
        }
    )
