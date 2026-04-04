import pandas as pd
import pytest

from v1indicators.volatility import williams_vix_fix


def test_williams_vix_fix_basic():
    close = pd.Series([10.0, 10.4, 10.8, 10.2, 10.6, 11.0, 11.3, 11.1])
    low = pd.Series([9.8, 10.1, 10.4, 9.9, 10.2, 10.6, 10.9, 10.8])

    result = williams_vix_fix(
        close,
        low,
        pd_length=3,
        bb_length=3,
        bb_mult=2.0,
        lb=4,
        ph=0.85,
        pl=1.01,
    )

    highest_close = close.rolling(3).max().replace(0.0, pd.NA)
    wvf = ((highest_close - low) / highest_close) * 100.0
    mid = wvf.rolling(3).mean()
    sdev = 2.0 * wvf.rolling(3).std(ddof=0)
    lower = mid - sdev
    upper = mid + sdev
    range_high = wvf.rolling(4).max() * 0.85
    range_low = wvf.rolling(4).min() * 1.01
    spike = (wvf >= upper) | (wvf >= range_high)

    expected = pd.DataFrame(
        {
            "WVF": wvf,
            "WVF_MID": mid,
            "WVF_LOWER": lower,
            "WVF_UPPER": upper,
            "WVF_RANGE_HIGH": range_high,
            "WVF_RANGE_LOW": range_low,
            "WVF_SPIKE": spike,
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_williams_vix_fix_input_validation():
    with pytest.raises(ValueError):
        williams_vix_fix(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd_length=0)

    with pytest.raises(ValueError):
        williams_vix_fix(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), bb_mult=0.0)

    with pytest.raises(TypeError):
        williams_vix_fix([1.0, 2.0], pd.Series([1.0, 2.0]))
