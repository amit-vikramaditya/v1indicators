import numpy as np
import pandas as pd

from .._utils import check_series


def delta_volume(
    open_: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """
    Candle-directional delta volume.

    If close > open, volume is counted positive; if close < open, negative.
    """
    open_s = check_series(open_, "open_")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    sign = np.where(close_s > open_s, 1.0, np.where(close_s < open_s, -1.0, 0.0))
    delta = pd.Series(sign * volume_s.to_numpy(dtype=np.float64), index=close_s.index)

    buy_vol = pd.Series(np.where(sign > 0, volume_s, 0.0), index=close_s.index)
    sell_vol = pd.Series(np.where(sign < 0, volume_s, 0.0), index=close_s.index)

    return pd.DataFrame(
        {
            "BUY_VOLUME": buy_vol,
            "SELL_VOLUME": sell_vol,
            "DELTA_VOLUME": delta,
            "CUM_DELTA_VOLUME": delta.cumsum(),
        }
    )
