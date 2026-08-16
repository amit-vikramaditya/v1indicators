# v1indicators

![Tests](https://github.com/Vatthu/v1indicators/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Technical analysis indicators for Python. Pandas Series in, pandas Series or
DataFrame out — named, typed, and indexed like the input.

The scope is deliberate: indicator math only. No charting, no broker
integrations, no strategy execution framework.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Correctness guarantees](#correctness-guarantees)
- [What is *not* verified](#what-is-not-verified)
- [Conventions](#conventions)
- [API layout](#api-layout)
- [Development](#development)

---

## Installation

```bash
pip install .            # from a checkout
pip install -e ".[dev]"  # development install
```

Requires Python 3.10+, numpy, pandas, numba.

## Quick start

```python
import pandas as pd
from v1indicators import rsi, macd, supertrend

df = pd.read_csv("data.csv")

# single output -> named Series
df["RSI_14"] = rsi(df["close"], length=14)

# multiple outputs -> DataFrame with stable column names
macd_df = macd(df["close"], fast=12, slow=26, signal=9)
st_df = supertrend(df["high"], df["low"], df["close"], length=10, mult=3.0)
```

| Output kind | Return type | Example columns |
|---|---|---|
| Single value | `pd.Series` | `RSI_14` |
| Multiple values | `pd.DataFrame` | `MACD`, `MACD_SIGNAL`, `MACD_HIST` |

Every output preserves the input index and length.

## Correctness guarantees

These properties are enforced by tests and CI (Python 3.10 / 3.12 / 3.14).
A regression in any of them fails the build.

| Property | Enforced by |
|---|---|
| No look-ahead bias | `tests/test_causality.py`, `tests/test_causality_params.py` |
| Textbook formula parity | `tests/test_parity_core.py`, `tests/test_reference_math.py` |
| NaN warmup boundaries | `tests/test_warmup_contract.py` |
| Calendar/session correctness | `tests/test_calendar_sessions.py` |
| Cross-indicator interoperability | `tests/test_interoperability_matrix.py` |

### No look-ahead bias

An indicator is causal when its values on the first *K* bars are identical
whether it is computed on the full series or on a truncated *K*-bar prefix.
If appending future bars changes a past value, the indicator repaints and
any backtest built on it is wrong.

- Every public function is checked this way on several synthetic datasets,
  and again with parameters pushed away from their defaults.
- Pivot-based indicators (`support_resistance`, `market_structure`,
  `zigzag_swings`, ...) emit levels only after the pivot is confirmed —
  a pivot needs `right` bars of future evidence by definition.
- `ichimoku` follows the same rule: its Chikou column is emitted as
  `close - close.shift(kijun)`, carrying the same information without
  future data.
- Indicators with a `causal` parameter accept `causal=False` for
  retrospective textbook placement (plotting only). That mode repaints by
  definition and emits a `UserWarning`.
- `dpo` and `vp` were removed: no causal form of either exists.

### Textbook formula parity

About thirty core indicators — the moving-average family, Bollinger Bands,
Donchian, MACD, Stochastic, Williams %R, CMO, Ultimate Oscillator, Aroon,
ADX, PSAR, ATR, Parkinson, Garman-Klass, Choppiness, OBV, AD, CMF, VWAP,
and rolling statistics — are compared against plain-Python reference loops
written from their textbook formulas, at 1e-9 tolerance.

### NaN warmup

Rolling-window and exponential-family outputs (`ema`, `rma`, `smma`,
`zlema`, `dema`, `tema`, `t3`, and the ewm-smoothed oscillators) are NaN
until they have `length` valid observations. Nested chains add their
stages: TEMA is NaN for `3*(length-1)` bars.

Exception: the kernel-based adaptive averages (`kama`, `vidya`, `mcgd`,
`ssf`, `hwma`, `kalman_filter`) and `psar` seed their recursion from the
first bars, as their algorithms define. Feed warmup history before the
region you care about.

### Calendar and session correctness

Level and session tools are tested on weekday-only indices with weekend
gaps and holidays — the shape real exchange data has.

### Interoperability

Every public symbol runs on nine synthetic scenarios (trend up/down,
sideways, volatile, gapped, flat, low volume, NaN streaks, weekday gaps);
outputs must align on index and length.

## What is *not* verified

`range_filter_confluence`, `precision_confluence`, `dual_score_signals`,
`htf_reversal_divergence`, `swing_trend_entry` and `swing_leg_profile`
compose the indicators above into buy/sell signals.

Their outputs are causal by construction. That is all this library claims
about them — nothing here measures whether their signals predict anything.
Validate on your own data before trading them.

## Conventions

- **Sorted input required** — an unsorted index raises `ValueError`; bar
  *i+1* is always assumed to follow bar *i*.
- **Boolean flags** are `False` where inputs are NaN or insufficient;
  "no signal" and "not enough data" are not distinguished.
- **Session/calendar tools** (`day_week_month_levels`, `session_range`,
  `session_killzones`) read timestamps in the index's own clock — no
  timezone conversion. Empty calendar periods are skipped, so "prior day"
  means the prior *trading* day.
- **`vwap`** accumulates from the first bar of the input; slice the input
  to a session for a session VWAP.
- **Aliases**: `kc` = `keltner`, `squeeze` = `squeeze_momentum`,
  `uo` = `ultimate_oscillator`, `willr` = `williams_r`.

## API layout

Import from the root:

```python
from v1indicators import ema, sma, rsi, atr, obv
```

or from a family module:

```python
from v1indicators.momentum import rsi, stoch
```

| Family | Symbols | Examples |
|---|---:|---|
| overlap | 43 | `ema`, `sma`, `bbands`, `donchian`, `keltner` |
| momentum | 52 | `rsi`, `macd`, `stochastic`, `williams_r` |
| trend | 38 | `supertrend`, `adx`, `psar`, `aroon` |
| volatility | 13 | `atr`, `parkinson`, `garman_klass`, `chop` |
| volume | 17 | `obv`, `ad`, `cmf`, `vwap`, `pvt` |
| statistics | 11 | `stdev`, `zscore`, `hurst`, `entropy` |
| levels | 4 | `support_resistance`, `pivot_points` |
| performance | 4 | `log_return`, `drawdown` |

A second namespace mirrors how indicators are built:

```python
from v1indicators.foundational.overlap import ema   # direct from price/volume
from v1indicators.derived.trend import supertrend   # built on other indicators
```

Family modules re-export both layers and remain the compatibility surface.
New work incubates in `experimental/`, which is not shipped to PyPI.

Most indicators take close only; range-based ones take high/low/close;
volume indicators add volume. Series must share an index.

## Development

```bash
pytest                                     # full suite (~7 s)
pytest -q tests/test_causality.py          # prefix invariance, all functions
pytest -q tests/test_causality_params.py   # same, non-default parameters
pytest -q tests/test_parity_core.py        # textbook reference parity
pytest -q tests/test_interoperability_matrix.py
```

The causality gates discover public functions automatically: a new
repainting indicator fails the suite without a test being written for it.

See [CHANGELOG.md](CHANGELOG.md) for release history. License: MIT.
