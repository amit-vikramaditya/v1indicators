from v1indicators.derived.momentum import macd, stochrsi
from v1indicators.derived.overlap import fibonacci_bbands, keltner
from v1indicators.derived.trend import adx, ema_rsi_signal, supertrend
from v1indicators.derived.volatility import hwc, natr
from v1indicators.derived.volume import aobv, swing_leg_profile
from v1indicators.foundational.levels import fibonacci, pivot_points
from v1indicators.foundational.momentum import rsi, stochastic
from v1indicators.foundational.overlap import ema, sma
from v1indicators.foundational.performance import drawdown, log_return
from v1indicators.foundational.statistics import stdev, variance
from v1indicators.foundational.trend import aroon, market_structure, psar
from v1indicators.foundational.volatility import atr, chop
from v1indicators.foundational.volume import obv, vwap


def test_foundational_namespace_exports():
    assert callable(ema)
    assert callable(sma)
    assert callable(rsi)
    assert callable(stochastic)
    assert callable(aroon)
    assert callable(psar)
    assert callable(market_structure)
    assert callable(atr)
    assert callable(chop)
    assert callable(obv)
    assert callable(vwap)
    assert callable(fibonacci)
    assert callable(pivot_points)
    assert callable(stdev)
    assert callable(variance)
    assert callable(log_return)
    assert callable(drawdown)


def test_derived_namespace_exports():
    assert callable(macd)
    assert callable(stochrsi)
    assert callable(keltner)
    assert callable(fibonacci_bbands)
    assert callable(adx)
    assert callable(supertrend)
    assert callable(ema_rsi_signal)
    assert callable(natr)
    assert callable(hwc)
    assert callable(aobv)
    assert callable(swing_leg_profile)
