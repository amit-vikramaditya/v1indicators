import pandas as pd

from .._utils import check_series


def hilo(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """HiLo Activator."""
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    hma = high_s.rolling(length).mean()
    lma = low_s.rolling(length).mean()

    dir_ = pd.Series(index=close_s.index, dtype=float)
    dir_[close_s > hma] = 1.0
    dir_[close_s < lma] = -1.0
    dir_ = dir_.ffill().fillna(0.0)

    hilo_line = pd.Series(index=close_s.index, dtype=float)
    hilo_line[dir_ >= 0.0] = lma[dir_ >= 0.0]
    hilo_line[dir_ < 0.0] = hma[dir_ < 0.0]

    return pd.DataFrame({f"HILO_{length}": hilo_line, f"HILO_DIR_{length}": dir_})
