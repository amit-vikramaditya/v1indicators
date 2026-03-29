import pandas as pd

from .._utils import check_series
from .rsi import rsi


def rsi_bbands_signal(
    close: pd.Series,
    rsi_length: int = 6,
    rsi_oversold: float = 50.0,
    rsi_overbought: float = 50.0,
    bb_length: int = 200,
    bb_mult: float = 2.0,
) -> pd.DataFrame:
    """
    RSI + Bollinger Bands signal model.

    Long setup: RSI crosses above oversold and price crosses above lower band.
    Short setup: RSI crosses below overbought and price crosses below upper band.
    """
    if rsi_length <= 0 or bb_length <= 0:
        raise ValueError("rsi_length and bb_length must be > 0")
    if bb_mult <= 0:
        raise ValueError("bb_mult must be > 0")

    close_s = check_series(close, "close")
    rsi_line = rsi(close_s, rsi_length)

    bb_basis = close_s.rolling(bb_length).mean()
    bb_dev = bb_mult * close_s.rolling(bb_length).std(ddof=0)
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev

    rsi_cross_up = (rsi_line.shift(1) <= rsi_oversold) & (rsi_line > rsi_oversold)
    rsi_cross_down = (rsi_line.shift(1) >= rsi_overbought) & (rsi_line < rsi_overbought)

    price_cross_up = (close_s.shift(1) <= bb_lower.shift(1)) & (close_s > bb_lower)
    price_cross_down = (close_s.shift(1) >= bb_upper.shift(1)) & (close_s < bb_upper)

    long_signal = rsi_cross_up & price_cross_up
    short_signal = rsi_cross_down & price_cross_down

    return pd.DataFrame(
        {
            "RSI": rsi_line,
            "BB_BASIS": bb_basis,
            "BB_UPPER": bb_upper,
            "BB_LOWER": bb_lower,
            "RSI_CROSS_UP": rsi_cross_up.fillna(False),
            "RSI_CROSS_DOWN": rsi_cross_down.fillna(False),
            "PRICE_CROSS_BB_LOWER": price_cross_up.fillna(False),
            "PRICE_CROSS_BB_UPPER": price_cross_down.fillna(False),
            "LONG_SIGNAL": long_signal.fillna(False),
            "SHORT_SIGNAL": short_signal.fillna(False),
        }
    )
