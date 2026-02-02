import pandas as pd
import numpy as np
from .._utils import check_series
from ..overlap.rma import rma

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame:
    """Average Directional Index (ADX) using Wilder's method."""
    
    if length <= 0:
        raise ValueError("length must be > 0")

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    prev_close = close.shift()

    # --- True Range ---
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder ATR (RMA)
    atr_val = rma(tr, length)

    # --- Directional Movement ---
    up_move = high.diff()
    down_move = -low.diff()

    # Determine Plus DM and Minus DM
    # If up_move > down_move and up_move > 0 -> +DM = up_move, else 0
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    
    # If down_move > up_move and down_move > 0 -> -DM = down_move, else 0
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smooth DM
    plus_dm_smoothed = rma(plus_dm, length)
    minus_dm_smoothed = rma(minus_dm, length)

    # --- Directional Indicators ---
    # Safe division
    atr_safe = atr_val.replace(0.0, np.nan)

    plus_di = 100.0 * (plus_dm_smoothed / atr_safe)
    minus_di = 100.0 * (minus_dm_smoothed / atr_safe)

    # --- DX ---
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom

    # --- ADX ---
    # ADX is the RMA of DX (smoothed DX)
    adx_val = rma(dx, length)

    return pd.DataFrame(
        {
            f"ADX_{length}": adx_val,
            f"DMP_{length}": plus_di,
            f"DMN_{length}": minus_di,
        },
        index=close.index
    )

