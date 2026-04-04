import numpy as np
import pandas as pd

from .._utils import check_series


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Vortex Indicator (VI+ and VI-)."""
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    vm_plus = (high_s - low_s.shift(1)).abs()
    vm_minus = (low_s - high_s.shift(1)).abs()

    prev_close = close_s.shift(1)
    tr = pd.concat(
        [high_s - low_s, (high_s - prev_close).abs(), (low_s - prev_close).abs()],
        axis=1,
    ).max(axis=1).replace(0.0, np.nan)

    vi_plus = vm_plus.rolling(length).sum() / tr.rolling(length).sum()
    vi_minus = vm_minus.rolling(length).sum() / tr.rolling(length).sum()

    return pd.DataFrame({f"VI_PLUS_{length}": vi_plus, f"VI_MINUS_{length}": vi_minus})
