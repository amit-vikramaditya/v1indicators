import pandas as pd
from ..volatility.atr import atr


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    mult: float = 3.0,
) -> pd.DataFrame:
    """Supertrend indicator (ATR-based trend bands)."""

    if not all(isinstance(x, pd.Series) for x in (high, low, close)):
        raise TypeError("high, low, close must be pandas Series")

    if length <= 0 or mult <= 0:
        raise ValueError("length and mult must be > 0")

    atr_v = atr(high, low, close, length)
    hl2 = (high + low) / 2

    upper_band = hl2 + mult * atr_v
    lower_band = hl2 - mult * atr_v

    st = pd.Series(index=close.index, dtype="float64")
    direction = pd.Series(index=close.index, dtype="int8")

    st.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i - 1])
            else:
                upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i - 1])

        st.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return pd.DataFrame({
        "supertrend": st,
        "supertrend_direction": direction,
    })

