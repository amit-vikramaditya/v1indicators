import pandas as pd

from .._utils import check_series


def candlestick_patterns_extended(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    doji_size: float = 0.05,
    trend_bars: int = 5,
) -> pd.DataFrame:
    """
    Extended candlestick pattern detector.

    Includes common bullish/bearish reversal and continuation patterns.
    """
    if doji_size <= 0:
        raise ValueError("doji_size must be > 0")
    if trend_bars <= 0:
        raise ValueError("trend_bars must be > 0")

    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    prev_open = open_s.shift(1)
    prev_close = close_s.shift(1)

    full_range = high_s - low_s
    body = (open_s - close_s).abs()

    uptrend = open_s.shift(trend_bars) < open_s
    downtrend = open_s.shift(trend_bars) > open_s

    doji = body <= full_range * doji_size

    bullish_harami = (
        (prev_open > prev_close)
        & (close_s > open_s)
        & (close_s <= prev_open)
        & (prev_close <= open_s)
        & ((close_s - open_s) < (prev_open - prev_close))
        & downtrend
    )
    bearish_harami = (
        (prev_close > prev_open)
        & (open_s > close_s)
        & (open_s <= prev_close)
        & (prev_open <= close_s)
        & ((open_s - close_s) < (prev_close - prev_open))
        & uptrend
    )

    bullish_engulfing = (
        (prev_open > prev_close)
        & (close_s > open_s)
        & (close_s >= prev_open)
        & (prev_close >= open_s)
        & ((close_s - open_s) > (prev_open - prev_close))
        & downtrend
    )
    bearish_engulfing = (
        (prev_close > prev_open)
        & (open_s > close_s)
        & (open_s >= prev_close)
        & (prev_open >= close_s)
        & ((open_s - close_s) > (prev_close - prev_open))
        & uptrend
    )

    piercing_line = (
        (prev_close < prev_open)
        & (open_s < low_s.shift(1))
        & (close_s > (prev_close + (prev_open - prev_close) / 2.0))
        & (close_s < prev_open)
        & downtrend
    )

    lower10 = low_s.rolling(10).min().shift(1)
    bullish_belt_hold = (
        (low_s == open_s)
        & (open_s < lower10)
        & (open_s < close_s)
        & (close_s > ((high_s.shift(1) - low_s.shift(1)) / 2.0 + low_s.shift(1)))
        & downtrend
    )

    bullish_kicker = (prev_open > prev_close) & (open_s >= prev_open) & (close_s > open_s) & downtrend
    bearish_kicker = (prev_open < prev_close) & (open_s <= prev_open) & (close_s <= open_s) & uptrend

    hanging_man = (
        (full_range > 4.0 * body)
        & (((close_s - low_s) / (0.001 + full_range)) >= 0.75)
        & (((open_s - low_s) / (0.001 + full_range)) >= 0.75)
        & uptrend
        & (high_s.shift(1) < open_s)
        & (high_s.shift(2) < open_s)
    )

    evening_star = (
        (close_s.shift(2) > open_s.shift(2))
        & (pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).min(axis=1) > close_s.shift(2))
        & (open_s < pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).min(axis=1))
        & (close_s < open_s)
    )
    morning_star = (
        (close_s.shift(2) < open_s.shift(2))
        & (pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).max(axis=1) < close_s.shift(2))
        & (open_s > pd.concat([open_s.shift(1), close_s.shift(1)], axis=1).max(axis=1))
        & (close_s > open_s)
    )

    shooting_star = (
        (open_s.shift(1) < close_s.shift(1))
        & (open_s > close_s.shift(1))
        & ((high_s - pd.concat([open_s, close_s], axis=1).max(axis=1)) >= body * 3.0)
        & ((pd.concat([open_s, close_s], axis=1).min(axis=1) - low_s) <= body)
    )

    hammer = (
        (full_range > 3.0 * body)
        & (((close_s - low_s) / (0.001 + full_range)) > 0.6)
        & (((open_s - low_s) / (0.001 + full_range)) > 0.6)
    )
    inverted_hammer = (
        (full_range > 3.0 * body)
        & (((high_s - close_s) / (0.001 + full_range)) > 0.6)
        & (((high_s - open_s) / (0.001 + full_range)) > 0.6)
    )

    return pd.DataFrame(
        {
            "DOJI": doji,
            "BULLISH_HARAMI": bullish_harami,
            "BEARISH_HARAMI": bearish_harami,
            "BULLISH_ENGULFING": bullish_engulfing,
            "BEARISH_ENGULFING": bearish_engulfing,
            "PIERCING_LINE": piercing_line,
            "BULLISH_BELT_HOLD": bullish_belt_hold,
            "BULLISH_KICKER": bullish_kicker,
            "BEARISH_KICKER": bearish_kicker,
            "HANGING_MAN": hanging_man,
            "MORNING_STAR": morning_star,
            "EVENING_STAR": evening_star,
            "SHOOTING_STAR": shooting_star,
            "HAMMER": hammer,
            "INVERTED_HAMMER": inverted_hammer,
        }
    )
