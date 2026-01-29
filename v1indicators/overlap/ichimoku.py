import pandas as pd


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku Cloud."""

    if not all(isinstance(x, pd.Series) for x in (high, low, close)):
        raise TypeError("high, low, close must be pandas Series")

    if min(tenkan, kijun, senkou_b) <= 0:
        raise ValueError("periods must be > 0")

    # Midpoints
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2

    senkou_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
    senkou_b_line = (
        (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    ).shift(kijun)

    chikou = close.shift(-kijun)

    return pd.DataFrame({
        "tenkan": tenkan_line,
        "kijun": kijun_line,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
        "chikou": chikou,
    })

