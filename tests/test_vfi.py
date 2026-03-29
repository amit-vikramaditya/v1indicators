import numpy as np
import pandas as pd
import pytest

from v1indicators.volume.vfi import vfi


def test_vfi_basic():
    high = pd.Series([10.5, 10.8, 11.2, 11.0, 11.5, 11.8, 12.0, 12.2, 12.4, 12.1])
    low = pd.Series([9.8, 10.0, 10.3, 10.2, 10.7, 11.0, 11.2, 11.4, 11.6, 11.5])
    close = pd.Series([10.1, 10.6, 10.9, 10.6, 11.2, 11.5, 11.8, 12.0, 12.1, 11.9])
    volume = pd.Series([100.0, 130.0, 120.0, 140.0, 160.0, 150.0, 170.0, 180.0, 165.0, 155.0])

    result = vfi(
        high,
        low,
        close,
        volume,
        length=4,
        coef=0.2,
        vcoef=2.5,
        signal=3,
        smooth=False,
    )

    typical = (high + low + close) / 3.0
    safe_typical = typical.where(typical > 0.0)
    inter = np.log(safe_typical) - np.log(safe_typical.shift(1))
    vinter = inter.rolling(30).std(ddof=0)
    cutoff = 0.2 * vinter * close

    vave = volume.rolling(4).mean().shift(1)
    vmax = vave * 2.5
    vc = volume.where(volume < vmax, vmax)

    mf = typical - typical.shift(1)
    vcp = pd.Series(np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0)), index=close.index)
    expected_vfi = vcp.rolling(4).sum() / vave.replace(0.0, np.nan)
    expected_signal = expected_vfi.ewm(span=3, adjust=False).mean()
    expected_hist = expected_vfi - expected_signal

    expected = pd.DataFrame(
        {
            "VFI": expected_vfi,
            "VFI_SIGNAL": expected_signal,
            "VFI_HIST": expected_hist,
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_vfi_smooth_option():
    high = pd.Series([10.0, 10.3, 10.6, 10.8, 11.0, 11.1, 11.4])
    low = pd.Series([9.6, 9.9, 10.1, 10.2, 10.4, 10.6, 10.9])
    close = pd.Series([9.8, 10.1, 10.4, 10.6, 10.8, 10.9, 11.2])
    volume = pd.Series([100.0, 110.0, 120.0, 130.0, 125.0, 140.0, 150.0])

    result_raw = vfi(high, low, close, volume, length=3, signal=2, smooth=False)
    result_smooth = vfi(high, low, close, volume, length=3, signal=2, smooth=True)

    assert len(result_raw) == len(close)
    assert len(result_smooth) == len(close)
    assert not result_raw.equals(result_smooth)


def test_vfi_zero_volume_nan():
    high = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    low = pd.Series([9.0, 9.0, 9.0, 9.0, 9.0])
    close = pd.Series([9.5, 9.5, 9.5, 9.5, 9.5])
    volume = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

    result = vfi(high, low, close, volume, length=2)
    assert result["VFI"].isna().all()


def test_vfi_input_validation():
    with pytest.raises(ValueError):
        vfi(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            length=0,
        )

    with pytest.raises(ValueError):
        vfi(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            vcoef=0.0,
        )

    with pytest.raises(TypeError):
        vfi([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
