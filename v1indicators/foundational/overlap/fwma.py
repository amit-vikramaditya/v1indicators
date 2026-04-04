import numpy as np
import pandas as pd

from .._utils import check_series


def fwma(close: pd.Series, length: int = 10) -> pd.Series:
    """Fibonacci Weighted Moving Average."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    fib = [1.0, 1.0]
    while len(fib) < length:
        fib.append(fib[-1] + fib[-2])
    w = np.array(fib[-length:], dtype=np.float64)
    w /= w.sum()

    out = close_s.rolling(length).apply(lambda x: np.dot(x, w), raw=True)
    out.name = f"FWMA_{length}"
    return out
