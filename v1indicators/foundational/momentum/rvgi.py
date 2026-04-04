import pandas as pd

from .._utils import check_series


def rvgi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10) -> pd.DataFrame:
    """Relative Vigor Index and signal."""
    if length <= 1:
        raise ValueError("length must be > 1")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    num = (close_s - open_s).rolling(length).mean()
    den = (high_s - low_s).rolling(length).mean().replace(0.0, pd.NA)
    rvgi_line = num / den
    sig = rvgi_line.rolling(4).mean()

    return pd.DataFrame({f"RVGI_{length}": rvgi_line, f"RVGIs_{length}": sig})
