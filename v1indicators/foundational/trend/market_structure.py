import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _market_structure_kernel(
    close_v: np.ndarray,
    resistance_v: np.ndarray,
    support_v: np.ndarray,
):
    n = close_v.shape[0]

    trend = np.zeros(n, dtype=np.int8)
    bullish_bos = np.zeros(n, dtype=np.bool_)
    bearish_bos = np.zeros(n, dtype=np.bool_)
    bullish_choch = np.zeros(n, dtype=np.bool_)
    bearish_choch = np.zeros(n, dtype=np.bool_)

    cur_trend = 0
    if n == 0:
        return trend, bullish_bos, bearish_bos, bullish_choch, bearish_choch

    for i in range(1, n):
        c = close_v[i]
        p = close_v[i - 1]
        r_prev = resistance_v[i - 1]
        s_prev = support_v[i - 1]

        cross_up = not np.isnan(r_prev) and p <= r_prev and c > r_prev
        cross_down = not np.isnan(s_prev) and p >= s_prev and c < s_prev

        if cross_up:
            if cur_trend == -1:
                bullish_choch[i] = True
            else:
                bullish_bos[i] = True
            cur_trend = 1
        elif cross_down:
            if cur_trend == 1:
                bearish_choch[i] = True
            else:
                bearish_bos[i] = True
            cur_trend = -1

        trend[i] = cur_trend

    return trend, bullish_bos, bearish_bos, bullish_choch, bearish_choch


def market_structure(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    left: int = 9,
    right: int = 9,
) -> pd.DataFrame:
    """
    Market structure from pivot-level breaks.

    Identifies swing highs/lows, tracks active resistance/support,
    then classifies breaks as BOS (continuation) or CHoCH (trend change).

    This is a retrospective structure indicator: pivot highs/lows are
    confirmed with a symmetric left/right window, so the swing markers are
    only known after `right` future bars have printed.
    """
    if left <= 0 or right <= 0:
        raise ValueError("left and right must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    window = left + right + 1
    roll_max = high_s.rolling(window).max().shift(-right)
    roll_min = low_s.rolling(window).min().shift(-right)

    pivot_high = high_s.where(high_s == roll_max)
    pivot_low = low_s.where(low_s == roll_min)

    resistance = pivot_high.ffill()
    support = pivot_low.ffill()

    trend, bullish_bos, bearish_bos, bullish_choch, bearish_choch = _market_structure_kernel(
        close_s.to_numpy(dtype=np.float64),
        resistance.to_numpy(dtype=np.float64),
        support.to_numpy(dtype=np.float64),
    )

    return pd.DataFrame(
        {
            "SWING_HIGH": pivot_high,
            "SWING_LOW": pivot_low,
            "RESISTANCE": resistance,
            "SUPPORT": support,
            "MARKET_TREND": trend,
            "BULLISH_BOS": bullish_bos,
            "BEARISH_BOS": bearish_bos,
            "BULLISH_CHOCH": bullish_choch,
            "BEARISH_CHOCH": bearish_choch,
        },
        index=close_s.index,
    )
