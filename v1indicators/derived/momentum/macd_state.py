import pandas as pd

from .._utils import check_series
from ...foundational.overlap.ema import ema


def macd_state(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    signal_ma: str = "sma",
) -> pd.DataFrame:
    """
    MACD with crossover and histogram-state flags.

    Returns MACD line, signal line, histogram, and directional states similar
    to common multi-timeframe MACD overlays.
    """
    if min(fast, slow, signal) <= 0:
        raise ValueError("fast, slow, and signal must be > 0")

    ma_kind = signal_ma.lower()
    if ma_kind not in {"sma", "ema"}:
        raise ValueError("signal_ma must be either 'sma' or 'ema'")

    close_s = check_series(close, "close")

    macd_line = ema(close_s, fast) - ema(close_s, slow)
    if ma_kind == "sma":
        signal_line = macd_line.rolling(signal).mean()
    else:
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    hist = macd_line - signal_line
    prev_hist = hist.shift(1)

    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    hist_up_pos = (hist > 0) & (hist > prev_hist)
    hist_down_pos = (hist > 0) & (hist < prev_hist)
    hist_down_neg = (hist <= 0) & (hist < prev_hist)
    hist_up_neg = (hist <= 0) & (hist > prev_hist)

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_SIGNAL": signal_line,
            "MACD_HIST": hist,
            "MACD_ABOVE_SIGNAL": macd_line >= signal_line,
            "MACD_CROSS_UP": cross_up,
            "MACD_CROSS_DOWN": cross_down,
            "HIST_UP_POS": hist_up_pos,
            "HIST_DOWN_POS": hist_down_pos,
            "HIST_DOWN_NEG": hist_down_neg,
            "HIST_UP_NEG": hist_up_neg,
        }
    )
