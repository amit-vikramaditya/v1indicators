import pandas as pd

from .._utils import check_series


def three_line_strike(open_: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Three Line Strike candlestick pattern.

    Bullish: 3 bearish candles then bullish close above prior open.
    Bearish: 3 bullish candles then bearish close below prior open.
    """
    open_s = check_series(open_, "open_")
    close_s = check_series(close, "close")

    bull = (
        (close_s.shift(3) < open_s.shift(3))
        & (close_s.shift(2) < open_s.shift(2))
        & (close_s.shift(1) < open_s.shift(1))
        & (close_s > open_s.shift(1))
    )
    bear = (
        (close_s.shift(3) > open_s.shift(3))
        & (close_s.shift(2) > open_s.shift(2))
        & (close_s.shift(1) > open_s.shift(1))
        & (close_s < open_s.shift(1))
    )

    return pd.DataFrame(
        {
            "BULLISH_THREE_LINE_STRIKE": bull.fillna(False),
            "BEARISH_THREE_LINE_STRIKE": bear.fillna(False),
        }
    )
