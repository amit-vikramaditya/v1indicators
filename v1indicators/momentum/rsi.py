import pandas as pd

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder RSI)."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

