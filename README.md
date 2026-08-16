# v1indicators

v1indicators is a fast, production-focused technical analysis library for Python.

It provides a clean functional API for indicator calculations and keeps scope intentionally narrow:
- no charting
- no broker integrations
- no strategy execution framework

The goal is simple: reliable indicator math on top of pandas Series/DataFrame inputs.

## Highlights

- **Causality-verified**: an automated prefix-invariance harness sweeps every
  public function, so indicators are free of look-ahead bias / repainting by
  construction (see "No look-ahead, verified" below).
- Vectorized implementations for performance-critical paths.
- Numba-accelerated kernels for recursive/stateful indicators where appropriate.
- Consistent indicator signatures across categories.
- Broad indicator coverage across overlap, momentum, trend, volatility, volume, statistics, levels, and performance.

## No look-ahead, verified

Every per-bar indicator is tested for **prefix invariance**: its values on the
first K bars are identical whether computed on the full series or a truncated
prefix. Repainting is therefore a test failure, not a surprise.

- Pivot-family indicators (`support_resistance`, `market_structure`,
  `zigzag_swings`, ...) are **causal by default**: levels and signals activate
  only once the pivot is confirmed. `causal=False` restores retrospective
  placement for plotting.
- Documented exception: `ichimoku`'s Chikou span (look-ahead by textbook
  definition). `dpo` and `vp` were removed in 1.0.0 (look-ahead by
  definition / snapshot semantics).

## Installation

From source:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from v1indicators import rsi, macd, supertrend

df = pd.read_csv("data.csv")

# Single-series output
df["RSI_14"] = rsi(df["close"], length=14)

# Multi-column output
macd_df = macd(df["close"], fast=12, slow=26, signal=9)
df = pd.concat([df, macd_df], axis=1)

st_df = supertrend(df["high"], df["low"], df["close"], length=10, mult=3.0)
df = pd.concat([df, st_df], axis=1)
```

## API Organization

The package is organized by indicator families:
- overlap
- momentum
- trend
- volatility
- volume
- statistics
- levels
- performance

You can import from the package root for common indicators:

```python
from v1indicators import ema, sma, rsi, atr, obv
```

Or from family modules for explicit namespacing:

```python
from v1indicators.momentum import rsi, stoch
from v1indicators.overlap import ema, bbands
```

The package also exposes dependency-layer namespaces:
- `v1indicators.foundational.*` for indicators built directly from price/volume/math inputs
- `v1indicators.derived.*` for indicators built on top of one or more existing indicators

Examples:

```python
from v1indicators.foundational.overlap import ema
from v1indicators.derived.trend import ema_rsi_signal, supertrend
```

The original family imports remain the compatibility surface:

```python
from v1indicators.trend import ema_rsi_signal, supertrend
```

Repo-only incubator work belongs in `experimental/`. It is kept in GitHub but stays out of the published PyPI package until promoted.

## Data Requirements

Most indicators expect pandas Series aligned on the same index. Common field expectations:
- close-only indicators: close
- range-based indicators: high, low, close
- volume indicators: close, volume (sometimes high/low/open as needed)

## Testing

Run the full test suite:

```bash
pytest
```

Run the cross-indicator interoperability quality gate:

```bash
pytest -q tests/test_interoperability_matrix.py
```

Run the causality (no look-ahead) gate over every public function:

```bash
pytest -q tests/test_causality.py
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) — 1.0.0 is the causality release.

## License

MIT
