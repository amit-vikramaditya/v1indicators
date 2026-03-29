import numpy as np
import pandas as pd

from .._utils import check_series
from ..trend.adx import adx


def _logistic_prob(series: pd.Series, mean_lookback: int, slope: float, smooth_len: int) -> pd.Series:
    mean = series.rolling(mean_lookback).mean()
    z = (series - mean) * slope
    z_clipped = z.clip(-60.0, 60.0)
    raw = pd.Series(
        1.0 / (1.0 + np.exp(-z_clipped.to_numpy())),
        index=series.index,
    )
    return raw.ewm(span=smooth_len, adjust=False).mean()


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def directional_logistic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    di_length: int = 14,
    mean_lookback: int = 360,
    slope: float = 0.18,
    probability_smoothing: int = 3,
    oscillator_scale: float = 2.5,
    oscillator_smoothing: int = 7,
) -> pd.DataFrame:
    """Directional Logistic Oscillator (DLO).

    Uses DMI streams (+DI, -DI, ADX), transforms them through logistic
    probabilities, then builds a bounded oscillator with percentile-based
    signal thresholds.
    """
    if di_length <= 0:
        raise ValueError("di_length must be > 0")
    if mean_lookback <= 1:
        raise ValueError("mean_lookback must be > 1")
    if slope <= 0:
        raise ValueError("slope must be > 0")
    if probability_smoothing <= 0:
        raise ValueError("probability_smoothing must be > 0")
    if oscillator_scale <= 0:
        raise ValueError("oscillator_scale must be > 0")
    if oscillator_smoothing <= 0:
        raise ValueError("oscillator_smoothing must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    dmi = adx(high_s, low_s, close_s, length=di_length)
    plus_di = dmi[f"DMP_{di_length}"]
    minus_di = dmi[f"DMN_{di_length}"]
    adx_val = dmi[f"ADX_{di_length}"]

    prob_plus = _logistic_prob(plus_di, mean_lookback, slope, probability_smoothing)
    prob_minus = _logistic_prob(minus_di, mean_lookback, slope, probability_smoothing)
    prob_adx = _logistic_prob(adx_val, mean_lookback, slope, probability_smoothing)

    net_direction = prob_plus - prob_minus
    strength_raw = net_direction * prob_adx * oscillator_scale
    strength_bound = pd.Series(np.tanh(strength_raw.to_numpy()), index=strength_raw.index)
    strength = strength_bound.ewm(span=probability_smoothing, adjust=False).mean()

    s_sma = strength.rolling(oscillator_smoothing).mean()
    s_ema = strength.ewm(span=oscillator_smoothing, adjust=False).mean()
    s_sma_cycle = s_sma.ewm(span=int(np.ceil(oscillator_smoothing / 2.0)), adjust=False).mean()

    lower_sma = s_sma.rolling(mean_lookback).quantile(0.10, interpolation="nearest")
    upper_sma = s_sma.rolling(mean_lookback).quantile(0.90, interpolation="nearest")
    lower_ema = s_ema.rolling(mean_lookback).quantile(0.05, interpolation="nearest")
    upper_ema = s_ema.rolling(mean_lookback).quantile(0.95, interpolation="nearest")

    mr_buy = _crossover(s_sma, lower_sma) | _crossover(s_ema, lower_ema)
    mr_sell = _crossunder(s_sma, upper_sma) | _crossunder(s_ema, upper_ema)

    rev_up = (s_sma_cycle > s_sma_cycle.shift(1)) & ~(s_sma_cycle.shift(1) > s_sma_cycle.shift(2))
    rev_down = (s_sma_cycle < s_sma_cycle.shift(1)) & ~(s_sma_cycle.shift(1) < s_sma_cycle.shift(2))

    return pd.DataFrame(
        {
            "DLO_STRENGTH": strength,
            "DLO_SMA": s_sma,
            "DLO_EMA": s_ema,
            "DLO_SMA_CYCLE": s_sma_cycle,
            "DLO_LOWER_SMA": lower_sma,
            "DLO_UPPER_SMA": upper_sma,
            "DLO_LOWER_EMA": lower_ema,
            "DLO_UPPER_EMA": upper_ema,
            "DLO_MR_BUY": mr_buy.fillna(False),
            "DLO_MR_SELL": mr_sell.fillna(False),
            "DLO_REV_UP": rev_up.fillna(False),
            "DLO_REV_DOWN": rev_down.fillna(False),
        },
        index=close_s.index,
    )
