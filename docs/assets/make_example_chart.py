"""Generate docs/assets/quickstart.png — supertrend + rsi on synthetic data."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v1indicators import rsi, supertrend

rng = np.random.default_rng(7)
n = 260
drift = np.concatenate([np.full(90, 0.12), np.full(80, -0.10), np.full(90, 0.14)])
close = 100 + np.cumsum(drift + rng.normal(0, 0.9, n))
spread = np.abs(rng.normal(0, 0.6, n))
high = close + spread
low = close - spread
idx = pd.bdate_range("2025-01-01", periods=n)

df = pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)
st = supertrend(df["high"], df["low"], df["close"], length=10, mult=3.0)
rsi_14 = rsi(df["close"], length=14)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
fig.patch.set_facecolor("white")

ax1.plot(idx, df["close"], color="#333333", lw=1.1, label="close")
ax1.plot(idx, st["SUPERTREND"], color="#c44e52", lw=1.4, label="supertrend (10, 3.0)")
ax1.set_ylabel("price")
ax1.legend(loc="upper left", frameon=False, fontsize=9)
ax1.grid(alpha=0.25, lw=0.5)
ax1.set_title("v1indicators — supertrend / rsi on synthetic data", fontsize=11, loc="left")

ax2.plot(idx, rsi_14, color="#4c72b0", lw=1.1, label="RSI_14")
ax2.axhline(70, color="#999999", lw=0.7, ls="--")
ax2.axhline(30, color="#999999", lw=0.7, ls="--")
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.grid(alpha=0.25, lw=0.5)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig("docs/assets/quickstart.png", dpi=150)
print("wrote docs/assets/quickstart.png")
