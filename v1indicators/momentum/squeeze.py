import pandas as pd

from .._utils import check_series
from .squeeze_momentum import squeeze_momentum


def squeeze(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
) -> pd.DataFrame:
    """Alias for squeeze_momentum."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    return squeeze_momentum(
        high_s,
        low_s,
        close_s,
        bb_length=bb_length,
        bb_mult=bb_mult,
        kc_length=kc_length,
        kc_mult=kc_mult,
    )
