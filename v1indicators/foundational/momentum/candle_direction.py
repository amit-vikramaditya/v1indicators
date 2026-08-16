import numpy as np
import pandas as pd

from .._utils import check_series


def candle_direction(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Per-bar candle direction.

    +1 for a bullish bar (close > open), -1 for a bearish bar
    (close < open), 0 for a doji (close == open). A causal OHLC primitive
    commonly used as an execution filter or categorical feature.

    Returns
    -------
    pd.Series
        int8 direction series named ``CANDLE_DIR``.
    """
    open_s = check_series(open_, "open_")
    close_s = check_series(close, "close")

    direction = pd.Series(0, index=close_s.index, dtype=np.int8)
    direction[close_s > open_s] = 1
    direction[close_s < open_s] = -1
    direction.name = "CANDLE_DIR"
    return direction
