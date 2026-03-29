import pandas as pd

from .._utils import check_series


def t3(close: pd.Series, length: int = 10, factor: float = 0.7) -> pd.Series:
    """
    Tilson T3 moving average.

    The T3 is a smooth moving average based on cascaded EMAs and
    a volume factor ("factor").
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if factor <= 0:
        raise ValueError("factor must be > 0")

    close_s = check_series(close, "close")

    e1 = close_s.ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    e4 = e3.ewm(span=length, adjust=False).mean()
    e5 = e4.ewm(span=length, adjust=False).mean()
    e6 = e5.ewm(span=length, adjust=False).mean()

    a = float(factor)
    c1 = -a**3
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = 1 + 3 * a + a**3 + 3 * a**2

    out = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    out.name = f"T3_{length}_{factor}"
    return out
