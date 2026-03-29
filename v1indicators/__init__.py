"""Public API for v1indicators.

This module uses lazy symbol loading to avoid importing all indicator families at
interpreter startup. The public API remains available from the package root.
"""

from importlib import import_module

from ._version import __version__

OVERLAP_SYMBOLS = (
    "sma",
    "ema",
    "wma",
    "rma",
    "hma",
    "vwma",
    "dema",
    "tema",
    "kama",
    "zlema",
    "t3",
    "tma",
    "kalman_filter",
    "smma",
    "multi_ma",
    "bbands",
    "fibonacci_bbands",
    "donchian",
    "keltner",
    "ichimoku",
    "alma",
    "fwma",
    "hl2",
    "hlc3",
    "ohlc4",
    "midpoint",
    "midprice",
    "pwma",
    "sinwma",
    "zlma",
    "kc",
    "trima",
    "wcp",
    "swma",
    "ha",
    "hilo",
    "vidya",
    "accbands",
    "linreg",
    "mcgd",
    "hwma",
    "aberration",
    "ma",
    "ssf",
)

MOMENTUM_SYMBOLS = (
    "rsi",
    "macd",
    "stochastic",
    "roc",
    "mfi",
    "cci",
    "williams_r",
    "trix",
    "ppo",
    "ultimate_oscillator",
    "squeeze_momentum",
    "wavetrend",
    "candlestick_patterns",
    "macd_state",
    "candlestick_patterns_extended",
    "three_line_strike",
    "rsi_bbands_signal",
    "mom",
    "cmo",
    "stochrsi",
    "ao",
    "apo",
    "bias",
    "bop",
    "coppock",
    "er",
    "pgo",
    "psl",
    "dpo",
    "slope",
    "cg",
    "cfo",
    "eri",
    "kst",
    "qstick",
    "uo",
    "willr",
    "stoch",
    "squeeze",
    "fisher",
    "tsi",
    "smi",
    "rvgi",
    "brar",
    "kdj",
    "cdl_doji",
    "cdl_inside",
    "ebsw",
    "inertia",
    "qqe",
    "rsx",
    "directional_logistic_oscillator",
)

VOLATILITY_SYMBOLS = (
    "atr",
    "williams_vix_fix",
    "natr",
    "true_range",
    "ui",
    "massi",
    "rvi",
    "pdist",
    "thermo",
    "chop",
    "hwc",
)

TREND_SYMBOLS = (
    "adx",
    "supertrend",
    "psar",
    "aroon",
    "aroon_up",
    "aroon_down",
    "aroon_osc",
    "ut_bot",
    "trendline_breaks",
    "market_structure",
    "order_blocks",
    "support_resistance_breaks",
    "direction_regime",
    "swing_trend_entry",
    "ema_rsi_signal",
    "zigzag_swings",
    "fair_value_gaps",
    "session_killzones",
    "day_week_month_levels",
    "high_volume_levels",
    "session_range",
    "support_resistance_channels",
    "lorentzian_knn",
    "vortex",
    "cksp",
    "increasing",
    "decreasing",
    "decay",
    "long_run",
    "short_run",
    "amat",
    "td_seq",
    "ttm_trend",
)

VOLUME_SYMBOLS = (
    "obv",
    "vwap",
    "cmf",
    "vpt",
    "adl",
    "vfi",
    "delta_volume",
    "pvo",
    "adosc",
    "ad",
    "efi",
    "eom",
    "nvi",
    "pvi",
    "pvol",
    "pvr",
    "pvt",
    "vp",
    "aobv",
)

LEVELS_SYMBOLS = (
    "fibonacci",
    "pivot_points",
    "support_resistance",
    "equal_highs_lows",
)

STATISTICS_SYMBOLS = (
    "stdev",
    "variance",
    "zscore",
    "median",
    "quantile",
    "mad",
    "kurtosis",
    "skew",
    "entropy",
)

PERFORMANCE_SYMBOLS = (
    "log_return",
    "percent_return",
    "drawdown",
    "trend_return",
)

_SYMBOL_TO_PACKAGE = {}
_SYMBOL_TO_PACKAGE.update({name: ".overlap" for name in OVERLAP_SYMBOLS})
_SYMBOL_TO_PACKAGE.update({name: ".momentum" for name in MOMENTUM_SYMBOLS})
_SYMBOL_TO_PACKAGE.update({name: ".volatility" for name in VOLATILITY_SYMBOLS})
_SYMBOL_TO_PACKAGE.update({name: ".trend" for name in TREND_SYMBOLS})
_SYMBOL_TO_PACKAGE.update({name: ".volume" for name in VOLUME_SYMBOLS})
_SYMBOL_TO_PACKAGE.update({name: ".levels" for name in LEVELS_SYMBOLS})
_SYMBOL_TO_PACKAGE.update({name: ".statistics" for name in STATISTICS_SYMBOLS})
_SYMBOL_TO_PACKAGE.update({name: ".performance" for name in PERFORMANCE_SYMBOLS})

__all__ = [
    "__version__",
    *OVERLAP_SYMBOLS,
    *MOMENTUM_SYMBOLS,
    *VOLATILITY_SYMBOLS,
    *TREND_SYMBOLS,
    *VOLUME_SYMBOLS,
    *LEVELS_SYMBOLS,
    *STATISTICS_SYMBOLS,
    *PERFORMANCE_SYMBOLS,
]


def __getattr__(name: str):
    package = _SYMBOL_TO_PACKAGE.get(name)
    if package is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(package, __name__)
    value = getattr(module, name)

    # Cache resolved attribute so subsequent access has no import/module lookup.
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
