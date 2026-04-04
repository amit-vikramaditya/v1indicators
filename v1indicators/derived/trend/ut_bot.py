import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ...foundational.volatility.atr import atr


@njit
def _ut_bot_kernel(src_v: np.ndarray, nloss_v: np.ndarray):
    length = src_v.shape[0]
    stop = np.full(length, np.nan, dtype=np.float64)
    direction = np.zeros(length, dtype=np.int8)
    buy = np.zeros(length, dtype=np.bool_)
    sell = np.zeros(length, dtype=np.bool_)

    if length == 0:
        return stop, direction, buy, sell

    stop[0] = src_v[0]

    for i in range(1, length):
        prev_stop = stop[i - 1]
        cur_src = src_v[i]
        prev_src = src_v[i - 1]
        cur_loss = nloss_v[i]

        if np.isnan(prev_stop) or np.isnan(cur_src) or np.isnan(prev_src) or np.isnan(cur_loss):
            stop[i] = np.nan
            direction[i] = direction[i - 1]
            continue

        if cur_src > prev_stop and prev_src > prev_stop:
            cur_stop = max(prev_stop, cur_src - cur_loss)
        elif cur_src < prev_stop and prev_src < prev_stop:
            cur_stop = min(prev_stop, cur_src + cur_loss)
        elif cur_src > prev_stop:
            cur_stop = cur_src - cur_loss
        else:
            cur_stop = cur_src + cur_loss

        stop[i] = cur_stop

        if prev_src < prev_stop and cur_src > prev_stop:
            direction[i] = 1
        elif prev_src > prev_stop and cur_src < prev_stop:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        above = prev_src <= prev_stop and cur_src > cur_stop
        below = prev_stop <= prev_src and cur_stop > cur_src
        buy[i] = cur_src > cur_stop and above
        sell[i] = cur_src < cur_stop and below

    return stop, direction, buy, sell


def ut_bot(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    key_value: float = 1.0,
    atr_period: int = 10,
) -> pd.DataFrame:
    """
    UT Bot trend indicator.

    Returns trailing stop, direction, and buy/sell trigger flags.
    """
    if key_value <= 0:
        raise ValueError("key_value must be > 0")
    if atr_period <= 0:
        raise ValueError("atr_period must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    atr_v = atr(high_s, low_s, close_s, length=atr_period)
    nloss = key_value * atr_v

    stop, direction, buy, sell = _ut_bot_kernel(
        close_s.to_numpy(dtype=np.float64),
        nloss.to_numpy(dtype=np.float64),
    )

    return pd.DataFrame(
        {
            "UT_STOP": stop,
            "UT_DIR": direction,
            "UT_BUY": buy,
            "UT_SELL": sell,
        },
        index=close_s.index,
    )
