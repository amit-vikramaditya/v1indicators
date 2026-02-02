from typing import Tuple, Dict, Union, Optional

DEFAULT_RATIOS = (
    0.0,
    0.236,
    0.382,
    0.5,
    0.618,
    0.786,
    1.0,
    1.272,
    1.618,
)

def fibonacci(
    high: Union[float, int],
    low: Union[float, int],
    ratios: Optional[Tuple[float, ...]] = None,
) -> Dict[float, float]:
    """Fibonacci price levels between two anchors."""

    if ratios is None:
        ratios = DEFAULT_RATIOS

    diff = high - low

    return {r: high - diff * r for r in ratios}

