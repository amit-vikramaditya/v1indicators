import pandas as pd
import pytest
from typing import Any, cast

from v1indicators.overlap.fibonacci_bbands import fibonacci_bbands
from v1indicators.overlap.hlc3 import hlc3
from v1indicators.overlap.vwma import vwma
from v1indicators.statistics.stdev import stdev


def test_fibonacci_bbands_basic_math():
    high = pd.Series([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    volume = pd.Series([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])

    length = 3
    mult = 2.0

    result = fibonacci_bbands(high, low, close, volume, length=length, mult=mult)

    src = hlc3(high, low, close)
    basis = vwma(src, volume, length=length)
    dev = stdev(src, length=length) * mult

    expected_columns = {
        "FBB_BASIS",
        "FBB_UPPER_236",
        "FBB_UPPER_382",
        "FBB_UPPER_500",
        "FBB_UPPER_618",
        "FBB_UPPER_764",
        "FBB_UPPER_1000",
        "FBB_LOWER_236",
        "FBB_LOWER_382",
        "FBB_LOWER_500",
        "FBB_LOWER_618",
        "FBB_LOWER_764",
        "FBB_LOWER_1000",
    }
    assert set(result.columns) == expected_columns

    basis_expected = basis.copy()
    basis_expected.name = "FBB_BASIS"
    pd.testing.assert_series_equal(result["FBB_BASIS"], basis_expected)

    for ratio, tag in ((0.236, 236), (0.382, 382), (0.5, 500), (0.618, 618), (0.764, 764), (1.0, 1000)):
        up_expected = (basis + ratio * dev).copy()
        up_expected.name = f"FBB_UPPER_{tag}"

        down_expected = (basis - ratio * dev).copy()
        down_expected.name = f"FBB_LOWER_{tag}"

        pd.testing.assert_series_equal(result[f"FBB_UPPER_{tag}"], up_expected)
        pd.testing.assert_series_equal(result[f"FBB_LOWER_{tag}"], down_expected)


def test_fibonacci_bbands_custom_source():
    high = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0])
    close = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5])
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    custom_source = close * 2.0

    result = fibonacci_bbands(high, low, close, volume, length=2, mult=1.0, source=custom_source)

    basis_expected = vwma(custom_source, volume, length=2)
    basis_expected.name = "FBB_BASIS"
    pd.testing.assert_series_equal(result["FBB_BASIS"], basis_expected)


def test_fibonacci_bbands_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        fibonacci_bbands(s, s, s, s, length=0)

    with pytest.raises(ValueError):
        fibonacci_bbands(s, s, s, s, mult=0)

    with pytest.raises(TypeError):
        fibonacci_bbands(cast(Any, [1.0, 2.0]), s, s, s)

    with pytest.raises(TypeError):
        fibonacci_bbands(s, s, s, s, source=cast(Any, [1.0, 2.0]))
