import numpy as np
import pandas as pd

from .._utils import check_series


def _rolling_linreg_current(values: pd.Series, length: int) -> pd.Series:
    """
    Rolling linear regression evaluated at the current bar.

    For each rolling window, this returns the fitted y-value at x = length - 1.
    """
    arr = values.to_numpy(dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)

    if arr.size < length:
        return pd.Series(out, index=values.index)

    x = np.arange(length, dtype=np.float64)
    x_mean = x.mean()
    var_x = np.sum((x - x_mean) ** 2)

    window_sum = np.convolve(arr, np.ones(length, dtype=np.float64), mode="valid")
    weighted_sum = np.correlate(arr, x, mode="valid")

    slope = (weighted_sum - x_mean * window_sum) / var_x
    intercept = (window_sum / length) - slope * x_mean
    out[length - 1 :] = intercept + slope * (length - 1)

    return pd.Series(out, index=values.index)


def squeeze_momentum(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    use_true_range: bool = True,
) -> pd.DataFrame:
    """
    Squeeze Momentum oscillator.

    Combines Bollinger Bands and Keltner Channels to flag squeeze regimes,
    and computes momentum using rolling linear regression on de-meaned price.
    """
    if bb_length <= 1 or kc_length <= 1:
        raise ValueError("bb_length and kc_length must be > 1")
    if bb_mult <= 0 or kc_mult <= 0:
        raise ValueError("bb_mult and kc_mult must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    bb_mid = close_s.rolling(bb_length).mean()
    bb_std = close_s.rolling(bb_length).std(ddof=0)
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    if use_true_range:
        prev_close = close_s.shift(1)
        tr = pd.concat(
            [
                high_s - low_s,
                (high_s - prev_close).abs(),
                (low_s - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        kc_range = tr
    else:
        kc_range = high_s - low_s

    kc_mid = close_s.rolling(kc_length).mean()
    kc_range_ma = kc_range.rolling(kc_length).mean()
    kc_upper = kc_mid + kc_mult * kc_range_ma
    kc_lower = kc_mid - kc_mult * kc_range_ma

    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    sqz_no = ~(sqz_on | sqz_off)

    highest_h = high_s.rolling(kc_length).max()
    lowest_l = low_s.rolling(kc_length).min()
    center = ((highest_h + lowest_l) / 2.0 + close_s.rolling(kc_length).mean()) / 2.0
    deviation = close_s - center

    momentum = _rolling_linreg_current(deviation, kc_length)

    return pd.DataFrame(
        {
            "SQZ_MOM": momentum,
            "SQZ_ON": sqz_on,
            "SQZ_OFF": sqz_off,
            "SQZ_NO": sqz_no,
        }
    )
