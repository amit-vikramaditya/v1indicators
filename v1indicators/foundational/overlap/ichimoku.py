import pandas as pd

from .._utils import check_series
from ..._causal import warn_if_non_causal


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    causal: bool = True,
) -> pd.DataFrame:
    """Ichimoku Cloud.

    Components:
    - Tenkan-sen: midpoint of the highest high / lowest low over ``tenkan``.
    - Kijun-sen: same midpoint over ``kijun``.
    - Senkou Span A/B: midpoints displaced FORWARD by ``kijun`` bars. The
      value at row t is computed only from data up to t, so forward
      displacement is causal (it projects the cloud; it does not peek).
    - Chikou Span: the lagging span.

    When ``causal=True`` (default), the Chikou output is expressed as the
    spread it actually encodes — ``close - close.shift(kijun)`` — i.e.
    today's close relative to price ``kijun`` bars ago. This carries the
    identical signal information with no future data.

    ``causal=False`` restores the textbook displaced construction
    (``close.shift(-kijun)``), which paints today's close ``kijun`` bars
    into the past and is look-ahead by definition — suitable only for
    plotting the classic chart, never for backtests.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC inputs.
    tenkan, kijun, senkou_b : int
        Component periods.
    causal : bool
        See above; default True.

    Returns
    -------
    pd.DataFrame
        ``ICHIMOKU_TENKAN``, ``ICHIMOKU_KIJUN``, ``ICHIMOKU_SPAN_A``,
        ``ICHIMOKU_SPAN_B``, ``ICHIMOKU_CHIKOU``.
    """
    if min(tenkan, kijun, senkou_b) <= 0:
        raise ValueError("periods must be > 0")

    high_s = warn_if_non_causal("ichimoku", causal)
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    # Midpoints
    tenkan_line = (high_s.rolling(tenkan).max() + low_s.rolling(tenkan).min()) / 2
    kijun_line = (high_s.rolling(kijun).max() + low_s.rolling(kijun).min()) / 2

    # Senkou spans displaced forward: values at row t use data <= t only.
    senkou_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
    senkou_b_line = (
        (high_s.rolling(senkou_b).max() + low_s.rolling(senkou_b).min()) / 2
    ).shift(kijun)

    if causal:
        # Signal-equivalent, causal form of the lagging span.
        chikou = close_s - close_s.shift(kijun)
    else:
        # Textbook displaced construction (look-ahead by definition).
        chikou = close_s.shift(-kijun)

    return pd.DataFrame(
        {
            "ICHIMOKU_TENKAN": tenkan_line,
            "ICHIMOKU_KIJUN": kijun_line,
            "ICHIMOKU_SPAN_A": senkou_a,
            "ICHIMOKU_SPAN_B": senkou_b_line,
            "ICHIMOKU_CHIKOU": chikou,
        }
    )
