import pandas as pd

from .._utils import check_series


def tsi(close: pd.Series, fast: int = 13, slow: int = 25, signal: int = 13) -> pd.DataFrame:
    """True Strength Index and signal."""
    if min(fast, slow, signal) <= 0:
        raise ValueError("fast, slow, and signal must be > 0")

    close_s = check_series(close, "close")
    m = close_s.diff()
    ema1 = m.ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
    ema2 = m.abs().ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()

    tsi_line = 100.0 * ema1 / ema2.replace(0.0, pd.NA)
    sig = tsi_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"TSI": tsi_line, "TSI_SIGNAL": sig})
