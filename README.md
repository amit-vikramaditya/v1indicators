# v1indicators

Technical analysis indicators for Python. pandas Series in, pandas Series
or DataFrame out, named and indexed like the input.

The library does indicator math and nothing else. No charting, no broker
integrations, no strategy execution framework.

Requires Python 3.10+, numpy, pandas, numba. MIT licensed.

## Installation

From source:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Example

```python
import pandas as pd
from v1indicators import rsi, macd, supertrend

df = pd.read_csv("data.csv")

df["RSI_14"] = rsi(df["close"], length=14)

macd_df = macd(df["close"], fast=12, slow=26, signal=9)
df = pd.concat([df, macd_df], axis=1)

st_df = supertrend(df["high"], df["low"], df["close"], length=10, mult=3.0)
```

Single-output indicators return a named Series. Multi-output indicators
return a DataFrame with stable uppercase column names
(`MACD`, `MACD_SIGNAL`, `SUPERTREND_DIR`, ...). Outputs keep the input
index and length.

## What the test suite enforces

These properties are checked by tests. A change that breaks one of them
fails the build, including in CI (Python 3.10/3.12/3.14).

No look-ahead bias. An indicator is causal when its values on the first K
bars are identical whether it is computed on the full series or on a
truncated prefix of K bars. If appending future bars changes a past value,
the indicator repaints and any backtest built on it is wrong. Every public
function is checked this way on several synthetic datasets, and again with
parameter values pushed away from their defaults
(`tests/test_causality.py`, `tests/test_causality_params.py`).

Pivot-based indicators (`support_resistance`, `market_structure`,
`zigzag_swings`, `trendline_breaks`, ...) emit levels only after the pivot
bar is confirmed, because a pivot needs `right` future bars of evidence by
definition. `ichimoku` follows the same rule: the Chikou column is emitted
as `close - close.shift(kijun)`, which is the information the displaced
line carries, without future data. Every indicator with a `causal`
parameter accepts `causal=False` to restore the retrospective textbook
placement for plotting; that mode repaints by definition, emits a
`UserWarning` when used, and must not be used for backtests.

`dpo` and `vp` were removed: both require future bars or whole-series
snapshots by definition, so no causal form exists.

Correct formulas. About thirty core indicators (the moving-average family,
Bollinger Bands, Donchian, MACD, Stochastic, Williams %R, CMO, Ultimate,
Aroon, ADX, PSAR, ATR, Parkinson, Garman-Klass, Choppiness,
OBV, AD, CMF, VWAP, rolling statistics) are compared against plain-Python
reference loops written from their textbook formulas at 1e-9 tolerance
(`tests/test_parity_core.py`, `tests/test_reference_math.py`). The
seeding and warmup conventions these tests pin are documented in the test
files.

NaN warmup. Rolling-window and exponential-family outputs (`ema`, `rma`,
`smma`, `zlema`, `dema`, `tema`, `t3`, and the ewm-smoothed oscillators)
are NaN until they have `length` valid observations; nested chains add
their stages (TEMA is NaN for `3*(length-1)` bars). This is pinned in
`tests/test_warmup_contract.py`. Exception: the kernel-based adaptive
averages (`kama`, `vidya`, `mcgd`, `ssf`, `hwma`, `kalman_filter`) and
`psar` seed their recursion from the first bars, which is part of how
those algorithms are defined; feed warmup history before the region you
care about.

Calendar correctness. Levels and session tools are tested on weekday-only
indices with weekend gaps and holidays (`tests/test_calendar_sessions.py`),
which is the shape real exchange data has.

Interoperability. Every public symbol runs on nine synthetic scenarios
(trend up/down, sideways, volatile, gapped, flat, low volume, NaN streaks,
weekday gaps) and all outputs must align on index and length
(`tests/test_interoperability_matrix.py`).

## What the tests do not tell you

`range_filter_confluence`, `precision_confluence`, `dual_score_signals`,
`htf_reversal_divergence`, `swing_trend_entry` and `swing_leg_profile`
compose the indicators above into buy/sell signals. Their outputs are
causal, and that is all this library claims about them. Nothing here
measures whether their signals predict anything. Validate on your own
data before trading them.

## Conventions

- Warmup: NaN until enough history, as described above.
- Input index must be sorted ascending; otherwise `ValueError`. Bar i+1
  is assumed to follow bar i.
- Boolean flag columns are `False` where inputs are NaN or insufficient.
  "No signal" and "not enough data" are not distinguished.
- Session and calendar tools (`day_week_month_levels`, `session_range`,
  `session_killzones`) read timestamps in the index's own clock; there is
  no timezone conversion. Calendar periods without data are skipped, so
  "prior day" means the prior trading day.
- `vwap` accumulates from the first bar of the input. Slice the input to
  a session if you want a session VWAP.
- Aliases: `kc` = `keltner`, `squeeze` = `squeeze_momentum`,
  `uo` = `ultimate_oscillator`, `willr` = `williams_r`.

## API layout

Eight families: overlap, momentum, trend, volatility, volume, statistics,
levels, performance. Import from the root:

```python
from v1indicators import ema, sma, rsi, atr, obv
```

or from a family module:

```python
from v1indicators.momentum import rsi, stoch
```

There is also a two-layer namespace reflecting how indicators are built:
`v1indicators.foundational.*` (computed directly from price/volume) and
`v1indicators.derived.*` (built on other indicators):

```python
from v1indicators.foundational.overlap import ema
from v1indicators.derived.trend import supertrend
```

The family modules are the compatibility surface and re-export both
layers. New work incubates in `experimental/`, which is not shipped to
PyPI.

Most indicators take close only; range-based ones take high/low/close;
volume indicators add volume. Series must share an index.

## Development

Run the tests:

```bash
pytest
```

Individual gates:

```bash
pytest -q tests/test_causality.py          # prefix invariance, all functions
pytest -q tests/test_causality_params.py   # same, non-default parameters
pytest -q tests/test_parity_core.py        # textbook reference parity
pytest -q tests/test_interoperability_matrix.py
```

New indicators are expected to pass all of the above without exceptions;
the causality gates discover public functions automatically, so a new
repainting indicator fails without any test being written for it.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT
