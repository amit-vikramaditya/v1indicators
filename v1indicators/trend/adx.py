import pandas as pd


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame:
    """Average Directional Index (ADX) using Wilder's method."""

    if not all(isinstance(x, pd.Series) for x in (high, low, close)):
        raise TypeError("high, low, close must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    prev_close = close.shift()

    # --- True Range ---
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False).mean()

    # --- Directional Movement ---
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    plus_dm_smoothed = plus_dm.ewm(alpha=1 / length, adjust=False).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=1 / length, adjust=False).mean()

    atr_safe = atr.replace(0, pd.NA)

    plus_di = 100 * (plus_dm_smoothed / atr_safe)
    minus_di = 100 * (minus_dm_smoothed / atr_safe)

    denom = (plus_di + minus_di).replace(0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / denom

    adx = dx.ewm(alpha=1 / length, adjust=False).mean()

    return pd.DataFrame({
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    })

