# v1indicators

**Engineering over Hype.**

`v1indicators` is a professional-grade, high-performance technical analysis library for Python. It is designed to be the "Standard Library" for financial indicators—prioritizing mathematical accuracy, system stability, and zero external bloat.

It is a pure calculation engine. It does not trade, it does not plot, and it does not make promises. It just does the math.

## The Philosophy

Existing libraries (`ta-lib`, `pandas-ta`, `ta`) often suffer from one of three problems: dependency hell (C-compilers), abandoned maintenance, or bloated "black box" object hierarchies.

**v1indicators** solves this by adhering to four rules:
1.  **Math > Magic:** Implementations are based on standard textbook formulas (Wilder, Lane, Bollinger).
2.  **Simple > Fancy:** A pure functional API. Input arrays, output arrays. No complex classes.
3.  **NumPy Core:** Heavy lifting is done in NumPy and optimized Pandas C-extensions for maximum speed.
4.  **Reliability:** 100% test coverage with strict type validation.

## Installation

Requires Python 3.9+, NumPy, and Pandas.

```bash
pip install .
```

## Quick Start

The API is standardized. All functions accept Pandas Series and return Pandas Series (for single-value indicators) or DataFrames (for multi-value indicators).

```python
import pandas as pd
from v1indicators import rsi, macd, supertrend

# 1. Load your data (Engineering Standard: Lowercase columns)
df = pd.read_csv("data.csv")  # Must have 'open', 'high', 'low', 'close', 'volume'

# 2. Simple Indicator (RSI)
# Returns a Series named 'RSI_14'
df['RSI'] = rsi(df['close'], length=14)

# 3. Complex Indicator (MACD)
# Returns a DataFrame with 'MACD', 'MACD_SIGNAL', 'MACD_HIST'
macd_df = macd(df['close'], fast=12, slow=26, signal=9)
df = pd.concat([df, macd_df], axis=1)

# 4. Pro Indicator (Supertrend)
# Returns 'SUPERTREND' and 'SUPERTREND_DIR'
st_df = supertrend(df['high'], df['low'], df['close'], length=10, mult=3.0)
df = pd.concat([df, st_df], axis=1)

print(df.tail())
```

## Available Indicators

We currently support the "Essential 20"—the indicators used by 90% of professional systematic traders.

### Momentum
*   **RSI** (Relative Strength Index)
*   **MACD** (Moving Average Convergence Divergence)
*   **Stochastic** (Stochastic Oscillator)
*   **ROC** (Rate of Change)
*   **MFI** (Money Flow Index)
*   **CCI** (Commodity Channel Index)

### Overlap (Trend Filters)
*   **SMA** (Simple Moving Average)
*   **EMA** (Exponential Moving Average)
*   **WMA** (Weighted Moving Average)
*   **RMA** (Wilder's Smoothing / Running Moving Average)
*   **Bollinger Bands**
*   **Keltner Channels**
*   **Donchian Channels**
*   **Ichimoku Cloud**

### Trend
*   **Supertrend**
*   **Parabolic SAR** (PSAR)
*   **ADX** (Average Directional Index)

### Volatility
*   **ATR** (Average True Range)

### Volume
*   **OBV** (On-Balance Volume)
*   **VWAP** (Volume Weighted Average Price)

### Levels
*   **Fibonacci** Retracements

## Development

We enforce strict engineering standards.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the test suite (100% Pass Rate Required)
pytest
```

## License

MIT
