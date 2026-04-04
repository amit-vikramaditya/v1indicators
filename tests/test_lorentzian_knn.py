import pandas as pd
import pytest

from v1indicators.trend import lorentzian_knn


def test_lorentzian_knn_basic_shape():
    close = pd.Series(
        [10.0, 10.1, 10.3, 10.2, 10.5, 10.7, 10.6, 10.9, 11.1, 11.0, 11.3, 11.5]
    )

    result = lorentzian_knn(
        close,
        neighbors_count=3,
        max_bars_back=10,
        horizon=2,
        stride=1,
        feature_fast=1,
        feature_slow=3,
    )

    assert list(result.columns) == ["LKNN_F1", "LKNN_F2", "LKNN_PRED", "LKNN_SIGNAL"]
    assert len(result) == len(close)
    assert set(result["LKNN_SIGNAL"].dropna().unique()).issubset({-1, 0, 1})


def test_lorentzian_knn_input_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        lorentzian_knn(s, neighbors_count=0)

    with pytest.raises(ValueError):
        lorentzian_knn(s, stride=0)

    with pytest.raises(TypeError):
        lorentzian_knn([1.0, 2.0, 3.0])
