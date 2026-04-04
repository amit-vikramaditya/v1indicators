import numpy as np
import pandas as pd

from ..._utils import check_series


def williams_vix_fix(
    close: pd.Series,
    low: pd.Series,
    pd_length: int = 22,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    lb: int = 50,
    ph: float = 0.85,
    pl: float = 1.01,
) -> pd.DataFrame:
    """
    Williams Vix Fix (WVF).

    Returns WVF along with Bollinger/range reference levels.
    """
    if pd_length <= 0 or bb_length <= 0 or lb <= 0:
        raise ValueError("pd_length, bb_length, and lb must be > 0")
    if bb_mult <= 0:
        raise ValueError("bb_mult must be > 0")
    if ph <= 0 or pl <= 0:
        raise ValueError("ph and pl must be > 0")

    close_s = check_series(close, "close")
    low_s = check_series(low, "low")

    highest_close = close_s.rolling(pd_length).max().replace(0.0, np.nan)
    wvf = ((highest_close - low_s) / highest_close) * 100.0

    mid = wvf.rolling(bb_length).mean()
    sdev = bb_mult * wvf.rolling(bb_length).std(ddof=0)
    lower_band = mid - sdev
    upper_band = mid + sdev

    range_high = wvf.rolling(lb).max() * ph
    range_low = wvf.rolling(lb).min() * pl

    spike = (wvf >= upper_band) | (wvf >= range_high)

    return pd.DataFrame(
        {
            "WVF": wvf,
            "WVF_MID": mid,
            "WVF_LOWER": lower_band,
            "WVF_UPPER": upper_band,
            "WVF_RANGE_HIGH": range_high,
            "WVF_RANGE_LOW": range_low,
            "WVF_SPIKE": spike,
        }
    )
