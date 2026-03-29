# v1indicators

v1indicators is a fast, production-focused technical analysis library for Python.

It provides a clean functional API for indicator calculations and keeps scope intentionally narrow:
- no charting
- no broker integrations
- no strategy execution framework

The goal is simple: reliable indicator math on top of pandas Series/DataFrame inputs.

## Highlights

- Vectorized implementations for performance-critical paths.
- Numba-accelerated kernels for recursive/stateful indicators where appropriate.
- Consistent indicator signatures across categories.
- Broad indicator coverage across overlap, momentum, trend, volatility, volume, statistics, levels, and performance.

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

## License

MIT
