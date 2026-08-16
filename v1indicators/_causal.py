"""Shared guard for retrospective (non-causal) output modes.

Every pivot-family indicator accepts ``causal=False`` to restore the legacy
retrospective placement for PLOTTING. That mode repaints by definition:
values depend on bars that arrive after the output bar, so any backtest or
live signal built on it is inflated or impossible. The guard below makes the
opt-in audible at runtime instead of relying on docstrings being read.
"""

import warnings

_NON_CAUSAL_MESSAGE = (
    "{name}: causal=False produces retrospective placement that repaints as "
    "future bars arrive (levels/signals activate before they were "
    "confirmable). This mode is for plotting historical charts only — never "
    "for backtests or live signals."
)


def warn_if_non_causal(name: str, causal: bool) -> None:
    if not causal:
        warnings.warn(_NON_CAUSAL_MESSAGE.format(name=name), UserWarning, stacklevel=3)
