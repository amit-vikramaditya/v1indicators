import pandas as pd
from .._utils import check_series
from ..overlap.rma import rma

def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    """
    Money Flow Index (MFI).
    A volume-weighted momentum oscillator.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")
    volume = check_series(volume, "volume")

    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume

    # Get price change to determine positive/negative money flow
    # tp_diff > 0 -> positive, tp_diff < 0 -> negative
    tp_diff = typical_price.diff()

    pos_flow = money_flow.where(tp_diff > 0, 0.0)
    neg_flow = money_flow.where(tp_diff < 0, 0.0)

    # Wilder's Smoothing (RMA) is often used for MFI (standard)
    # Some implementations use simple sum (SMA logic), but RMA is more robust.
    # We'll use RMA to be consistent with RSI.
    avg_pos_flow = rma(pos_flow, length)
    avg_neg_flow = rma(neg_flow, length)

    mfr = avg_pos_flow / avg_neg_flow
    
    mfi_val = 100 - (100 / (1 + mfr))
    
    # Handle infinities (zero negative flow)
    import numpy as np
    mfi_val = mfi_val.replace([np.inf], 100.0)
    
    mfi_val.name = f"MFI_{length}"
    return mfi_val
