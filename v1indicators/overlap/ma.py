import pandas as pd

from .._utils import check_series
from .dema import dema
from .ema import ema
from .fwma import fwma
from .hma import hma
from .rma import rma
from .sma import sma
from .swma import swma
from .tema import tema
from .trima import trima
from .vidya import vidya
from .wma import wma
from .zlma import zlma


def ma(close: pd.Series, length: int = 10, mamode: str = "sma") -> pd.Series:
    """Unified moving-average dispatcher."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    mode = mamode.lower()

    table = {
        "sma": lambda x: sma(x, length=length),
        "ema": lambda x: ema(x, length=length),
        "wma": lambda x: wma(x, length=length),
        "rma": lambda x: rma(x, length=length),
        "hma": lambda x: hma(x, length=length),
        "dema": lambda x: dema(x, length=length),
        "tema": lambda x: tema(x, length=length),
        "trima": lambda x: trima(x, length=length),
        "fwma": lambda x: fwma(x, length=length),
        "swma": lambda x: swma(x, length=length),
        "zlma": lambda x: zlma(x, length=length),
        "vidya": lambda x: vidya(x, length=length),
    }
    if mode not in table:
        raise ValueError(f"unsupported mamode: {mamode}")

    out = table[mode](close_s)
    out.name = f"MA_{mode.upper()}_{length}"
    return out
