"""Naive textbook references for the core indicator set.

Every reference here is an intentionally slow, unambiguous, plain-Python
loop written directly from the textbook formula — no pandas rolling/ewm, no
library code. The tests pin EXACT parity (atol 1e-9) between the library and
the reference on the valid (post-warmup) region.

Where the library has a documented seeding/warmup convention (EMA-family
recursions seed at the first bar and are masked NaN until `length`
observations; nested chains compose their warmups), the reference mirrors
and states the convention explicitly — these tests pin the convention so it
cannot drift silently.

Dependency-free by design: no ta-lib / pandas_ta required.
"""

import numpy as np
import pandas as pd
import pytest

from v1indicators import (
    ad, aroon_down, aroon_osc, aroon_up, bbands, chop, cmf, cmo, dema,
    donchian, garman_klass, hma, macd, mom, obv, parkinson, psar, roc,
    stdev, stochastic, t3, tema, true_range, trima,
    uo, vwma, vwap, williams_r, wma, zlema, zscore,
)

N = 400


def _close(seed=5):
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 + rng.normal(0, 1.0, N).cumsum())


def _ohlcv(seed=5):
    rng = np.random.default_rng(seed)
    close = pd.Series(100.0 + rng.normal(0, 1.0, N).cumsum())
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + np.abs(rng.normal(0.4, 0.1, N))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.abs(rng.normal(0.4, 0.1, N))
    volume = pd.Series(rng.integers(100, 5000, N).astype(float))
    return open_, high, low, close, volume


def _assert_valid_close(a, b, atol=1e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    assert mask.sum() > 0.5 * len(a), "reference produced too few valid points"
    diff = np.abs(a[mask] - b[mask])
    assert diff.max() < atol, f"max |diff| = {diff.max():.3e}"


# ---------------------------------------------------------------------------
# Overlap family
# ---------------------------------------------------------------------------

def test_parity_sma_and_wma():
    close = _close()
    n = 20
    x = close.to_numpy()

    sma_ref = np.full(N, np.nan)
    wma_ref = np.full(N, np.nan)
    w = np.arange(1, n + 1, dtype=float)
    for i in range(n - 1, N):
        sma_ref[i] = x[i - n + 1 : i + 1].mean()
        wma_ref[i] = np.dot(x[i - n + 1 : i + 1], w) / w.sum()

    from v1indicators import sma
    _assert_valid_close(sma(close, length=n).to_numpy(), sma_ref)
    _assert_valid_close(wma(close, length=n).to_numpy(), wma_ref)


def test_parity_hma():
    close = _close()
    length = 16
    half, sq = max(length // 2, 1), max(int(np.sqrt(length)), 1)
    x = close.to_numpy()

    def wma_win(a, n):
        out = np.full(len(a), np.nan)
        w = np.arange(1, n + 1, dtype=float)
        for i in range(n - 1, len(a)):
            out[i] = np.dot(a[i - n + 1 : i + 1], w) / w.sum()
        return out

    raw = 2.0 * wma_win(x, half) - wma_win(x, length)
    ref = wma_win(np.nan_to_num(raw, nan=0.0), sq)  # library convolve treats NaN as 0
    # compare only where every underlying window was fully valid
    ok = np.zeros(N, dtype=bool)
    ok[length - 1 + sq - 1 :] = True
    ref = np.where(ok, ref, np.nan)
    _assert_valid_close(hma(close, length=length).to_numpy(), ref, atol=1e-8)


def test_parity_trima():
    close = _close()
    n = 10
    x = close.to_numpy()
    n1, n2 = (n + 1) // 2, n - (n + 1) // 2 + 1
    s1 = np.full(N, np.nan)
    s2 = np.full(N, np.nan)
    for i in range(n1 - 1, N):
        s1[i] = x[i - n1 + 1 : i + 1].mean()
    for i in range(n1 + n2 - 2, N):
        window = s1[i - n2 + 1 : i + 1]
        s2[i] = np.nanmean(window) if not np.isnan(window).any() else np.nan
    _assert_valid_close(trima(close, length=n).to_numpy(), s2)


def _ema_ref(x, span):
    """EMA recursion seeded at the first value; NaN until span observations."""
    alpha = 2.0 / (span + 1)
    out = np.full(len(x), np.nan)
    e = x[0]
    seen = 1
    out[0] = e
    for i in range(1, len(x)):
        e = alpha * x[i] + (1 - alpha) * e
        seen += 1
        if seen >= span:
            out[i] = e
    # library masks the first span-1 values to NaN
    out[: span - 1] = np.nan
    return out


def test_parity_dema_tema_t3():
    close = _close().to_numpy()
    n = 10

    def stage(x):
        vals = x[~np.isnan(x)]
        return np.concatenate([np.full(len(x) - len(vals), np.nan), _ema_ref(vals, n)])

    e1 = stage(close)
    e2 = stage(e1)
    e3 = stage(e2)
    dema_ref = 2 * e1 - e2
    tema_ref = 3 * e1 - 3 * e2 + e3
    _assert_valid_close(dema(_close(), length=n).to_numpy(), dema_ref, atol=1e-8)
    _assert_valid_close(tema(_close(), length=n).to_numpy(), tema_ref, atol=1e-8)

    a = 0.7
    c1, c2 = -a**3, 3 * a**2 + 3 * a**3
    c3, c4 = -6 * a**2 - 3 * a - 3 * a**3, 1 + 3 * a + a**3 + 3 * a**2
    stages = [close]
    for _ in range(6):
        stages.append(stage(stages[-1]))
    t3_ref = c1 * stages[6] + c2 * stages[5] + c3 * stages[4] + c4 * stages[3]
    _assert_valid_close(t3(_close(), length=n, factor=0.7).to_numpy(), t3_ref, atol=1e-8)


def test_parity_zlema():
    close = _close()
    n = 10
    lag = (n - 1) // 2
    x = close.to_numpy()
    adjusted = np.full(N, np.nan)
    for i in range(N):
        if i >= lag:
            adjusted[i] = x[i] + (x[i] - x[i - lag])
    vals = adjusted[~np.isnan(adjusted)]
    ref = np.concatenate([np.full(N - len(vals), np.nan), _ema_ref(vals, n)])
    _assert_valid_close(zlema(close, length=n).to_numpy(), ref, atol=1e-8)


def test_parity_bbands_donchian_vwma():
    open_, high, low, close, volume = _ohlcv()
    n, mult = 20, 2.0
    x = close.to_numpy()
    v = volume.to_numpy()

    mid = np.full(N, np.nan); up = np.full(N, np.nan); lo = np.full(N, np.nan)
    for i in range(n - 1, N):
        w = x[i - n + 1 : i + 1]
        mid[i] = w.mean()
        s = np.sqrt(((w - w.mean()) ** 2).sum() / (n - 1))  # sample std, ddof=1
        up[i] = mid[i] + mult * s
        lo[i] = mid[i] - mult * s
    out = bbands(close, length=n, mult=mult)
    _assert_valid_close(out["BB_MID"].to_numpy(), mid)
    _assert_valid_close(out["BB_UPPER"].to_numpy(), up)
    _assert_valid_close(out["BB_LOWER"].to_numpy(), lo)

    h = high.to_numpy(); l = low.to_numpy()
    du = np.full(N, np.nan); dl = np.full(N, np.nan)
    for i in range(n - 1, N):
        du[i] = h[i - n + 1 : i + 1].max()
        dl[i] = l[i - n + 1 : i + 1].min()
    d = donchian(high, low, length=n)
    _assert_valid_close(d["DONCHIAN_UPPER"].to_numpy(), du)
    _assert_valid_close(d["DONCHIAN_LOWER"].to_numpy(), dl)
    _assert_valid_close(d["DONCHIAN_MID"].to_numpy(), (du + dl) / 2)

    vw = np.full(N, np.nan)
    for i in range(n - 1, N):
        vw[i] = np.dot(x[i - n + 1 : i + 1], v[i - n + 1 : i + 1]) / v[i - n + 1 : i + 1].sum()
    _assert_valid_close(vwma(close, volume, length=n).to_numpy(), vw)


# ---------------------------------------------------------------------------
# Momentum family
# ---------------------------------------------------------------------------

def test_parity_mom_roc():
    close = _close()
    n = 10
    x = close.to_numpy()
    m = np.full(N, np.nan); r = np.full(N, np.nan)
    for i in range(n, N):
        m[i] = x[i] - x[i - n]
        r[i] = 100.0 * (x[i] / x[i - n] - 1.0)
    _assert_valid_close(mom(close, length=n).to_numpy(), m)
    _assert_valid_close(roc(close, length=n).to_numpy(), r)


def test_parity_stochastic_williams_r():
    open_, high, low, close, volume = _ohlcv()
    n, smooth = 14, 3
    h = high.to_numpy(); l = low.to_numpy(); c = close.to_numpy()
    k = np.full(N, np.nan)
    for i in range(n - 1, N):
        hh = h[i - n + 1 : i + 1].max()
        ll = l[i - n + 1 : i + 1].min()
        k[i] = 100.0 * (c[i] - ll) / (hh - ll) if hh != ll else np.nan
    d = np.full(N, np.nan)
    for i in range(n + smooth - 2, N):
        w = k[i - smooth + 1 : i + 1]
        d[i] = np.nanmean(w) if not np.isnan(w).any() else np.nan
    st = stochastic(high, low, close, length=n, smooth=smooth)
    _assert_valid_close(st["STOCH_K"].to_numpy(), k)
    _assert_valid_close(st["STOCH_D"].to_numpy(), d)

    wr = np.full(N, np.nan)
    for i in range(n - 1, N):
        hh = h[i - n + 1 : i + 1].max()
        ll = l[i - n + 1 : i + 1].min()
        wr[i] = -100.0 * (hh - c[i]) / (hh - ll) if hh != ll else np.nan
    _assert_valid_close(williams_r(high, low, close, length=n).to_numpy(), wr)


def test_parity_cmo():
    close = _close()
    n = 9
    x = close.to_numpy()
    ref = np.full(N, np.nan)
    for i in range(n, N):
        gains = losses = 0.0
        for j in range(i - n + 1, i + 1):
            dv = x[j] - x[j - 1]
            gains += max(dv, 0.0)
            losses += max(-dv, 0.0)
        if gains + losses > 0:
            ref[i] = 100.0 * (gains - losses) / (gains + losses)
    _assert_valid_close(cmo(close, length=n).to_numpy(), ref)


def test_parity_ultimate_oscillator():
    open_, high, low, close, volume = _ohlcv()
    s, m, lng = 7, 14, 28
    h, l, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    ref = np.full(N, np.nan)

    def avg(window):
        num = den = 0.0
        for i in range(window):
            pc = c[j - 1 - i] if j - 1 - i >= 0 else np.nan
            tl = min(l[j - i], pc) if not np.isnan(pc) else l[j - i]
            th = max(h[j - i], pc) if not np.isnan(pc) else h[j - i]
            num += c[j - i] - tl
            den += th - tl
        return num / den if den > 0 else np.nan

    for j in range(lng, N):
        a7, a14, a28 = avg(s), avg(m), avg(lng)
        if not any(np.isnan(v) for v in (a7, a14, a28)):
            ref[j] = 100.0 * (4 * a7 + 2 * a14 + 1 * a28) / 7.0
    _assert_valid_close(uo(high, low, close).to_numpy(), ref)


def test_parity_macd():
    close = _close().to_numpy()
    fast, slow, sig = 12, 26, 9
    ef = _ema_ref(close, fast)
    es = _ema_ref(close, slow)
    line = np.where(np.isnan(ef) | np.isnan(es), np.nan, ef - es)
    vals = line[~np.isnan(line)]
    signal = np.concatenate([np.full(len(line) - len(vals), np.nan), _ema_ref(vals, sig)])
    out = macd(_close(), fast=fast, slow=slow, signal=sig)
    _assert_valid_close(out["MACD"].to_numpy(), line, atol=1e-8)
    _assert_valid_close(out["MACD_SIGNAL"].to_numpy(), signal, atol=1e-8)
    _assert_valid_close(out["MACD_HIST"].to_numpy(), line - signal, atol=1e-8)


# ---------------------------------------------------------------------------
# Trend family
# ---------------------------------------------------------------------------

def test_parity_aroon():
    open_, high, low, close, volume = _ohlcv()
    n = 25
    h, l = high.to_numpy(), low.to_numpy()
    up = np.full(N, np.nan); dn = np.full(N, np.nan)
    for i in range(n - 1, N):
        hw = h[i - n + 1 : i + 1]
        lw = l[i - n + 1 : i + 1]
        # periods since the MOST RECENT occurrence of the extreme (library
        # convention: reversed-window argmax; ties resolve to the latest bar)
        up[i] = 100.0 * (n - hw[::-1].argmax()) / n
        dn[i] = 100.0 * (n - lw[::-1].argmin()) / n
    _assert_valid_close(aroon_up(high, length=n).to_numpy(), up)
    _assert_valid_close(aroon_down(low, length=n).to_numpy(), dn)
    _assert_valid_close(aroon_osc(high, low, length=n).to_numpy(), up - dn)


def _rma_ref(x, length):
    """Wilder recursion seeded at first value; NaN until `length` obs."""
    alpha = 1.0 / length
    out = np.full(len(x), np.nan)
    e = x[0]
    out[0] = e
    for i in range(1, len(x)):
        e = alpha * x[i] + (1 - alpha) * e
        if i >= length - 1:
            out[i] = e
    out[: length - 1] = np.nan
    return out


def test_parity_adx():
    """Naive textbook TR/+DM/-DM loops, smoothed by the library's `rma`
    (itself pinned exactly in test_reference_math.py), then naive DI/DX/ADX
    formulas — so this test validates the directional-movement logic, not
    the smoothing implementation."""
    from v1indicators import adx, rma as _rma
    open_, high, low, close, volume = _ohlcv()
    n = 14
    h, l, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    tr = np.full(N, np.nan)
    tr[0] = h[0] - l[0]  # library convention: bar 0 has no prev close, TR = range
    pdm = np.zeros(N); mdm = np.zeros(N)
    for i in range(1, N):
        pc = c[i - 1]
        tr[i] = max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0

    atr_s = _rma(pd.Series(tr), length=n).to_numpy()
    pdm_s = _rma(pd.Series(pdm), length=n).to_numpy()
    mdm_s = _rma(pd.Series(mdm), length=n).to_numpy()

    pdi = np.full(N, np.nan); mdi = np.full(N, np.nan); dx = np.full(N, np.nan)
    for i in range(N):
        if not np.isnan(atr_s[i]) and atr_s[i] != 0:
            pdi[i] = 100 * pdm_s[i] / atr_s[i]
            mdi[i] = 100 * mdm_s[i] / atr_s[i]
        if not np.isnan(pdi[i]) and not np.isnan(mdi[i]) and pdi[i] + mdi[i] != 0:
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / (pdi[i] + mdi[i])
    adx_ref = _rma(pd.Series(dx), length=n).to_numpy()

    out = adx(high, low, close, length=n)
    _assert_valid_close(out[f"ADX_{n}"].to_numpy(), adx_ref, atol=1e-8)
    _assert_valid_close(out[f"DMP_{n}"].to_numpy(), pdi, atol=1e-8)
    _assert_valid_close(out[f"DMN_{n}"].to_numpy(), mdi, atol=1e-8)


def test_parity_psar():
    """Pins the library's PSAR variant (behavioral reference, looped from the
    same rules the kernel implements): initial direction from bar-1 movement,
    EP/AF updates, two-bar clamp, reversal to prior EP."""
    open_, high, low, close, volume = _ohlcv(seed=3)
    h, l = high.to_numpy(), low.to_numpy()
    af0, af_max = 0.02, 0.2
    psar_ref = np.full(N, np.nan)
    direction = np.zeros(N, dtype=np.int8)

    bull = (h[1] > h[0]) or (l[1] > l[0])
    psar_ref[1] = l[0] if bull else h[0]
    ep = h[1] if bull else l[1]
    af = af0
    direction[1] = 1 if bull else -1

    for i in range(2, N):
        p = psar_ref[i - 1] + af * (ep - psar_ref[i - 1])
        if bull:
            if p > l[i] or p > l[i - 1]:
                bull = False
                p = ep
                ep = l[i]
                af = af0
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af0, af_max)
                p = min(p, l[i - 1], l[i - 2])
        else:
            if p < h[i] or p < h[i - 1]:
                bull = True
                p = ep
                ep = h[i]
                af = af0
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af0, af_max)
                p = max(p, h[i - 1], h[i - 2])
        psar_ref[i] = p
        direction[i] = 1 if bull else -1

    out = psar(high, low)
    col_psar = "PSAR" if "PSAR" in out.columns else out.columns[0]
    col_dir = "PSAR_DIR" if "PSAR_DIR" in out.columns else out.columns[1]
    _assert_valid_close(out[col_psar].to_numpy(), psar_ref, atol=1e-10)
    assert (out[col_dir].to_numpy() == direction).all()


# ---------------------------------------------------------------------------
# Volatility family
# ---------------------------------------------------------------------------

def test_parity_true_range():
    open_, high, low, close, volume = _ohlcv()
    h, l, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    ref = np.full(N, np.nan)
    for i in range(1, N):
        ref[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    _assert_valid_close(true_range(high, low, close).to_numpy(), ref)


def test_parity_parkinson_garman_klass():
    open_, high, low, close, volume = _ohlcv()
    o, h, l, c = (s.to_numpy() for s in (open_, high, low, close))
    pk = np.full(N, np.nan)
    gk = np.full(N, np.nan)
    for i in range(N):
        hl = np.log(h[i] / l[i])
        pk[i] = np.sqrt(hl**2 / (4.0 * np.log(2.0)))
        co = np.log(c[i] / o[i])
        v = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
        gk[i] = np.sqrt(v) if v > 0 else np.nan
    _assert_valid_close(parkinson(high, low).to_numpy(), pk)
    _assert_valid_close(garman_klass(open_, high, low, close).to_numpy(), gk)


def test_parity_chop():
    open_, high, low, close, volume = _ohlcv()
    n = 14
    h, l, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    ref = np.full(N, np.nan)
    for i in range(n, N):
        tr_sum = 0.0
        for j in range(i - n + 1, i + 1):
            pc = c[j - 1] if j - 1 >= 0 else np.nan
            tr_sum += max(h[j] - l[j], abs(h[j] - pc), abs(l[j] - pc))
        hh = h[i - n + 1 : i + 1].max()
        ll = l[i - n + 1 : i + 1].min()
        if hh > ll:
            ref[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(n)
    _assert_valid_close(chop(high, low, close, length=n).to_numpy(), ref)


def test_parity_atr_rma_mode():
    from v1indicators import atr
    open_, high, low, close, volume = _ohlcv()
    n = 14
    h, l, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    tr = np.full(N, np.nan)
    tr[0] = h[0] - l[0]  # library convention: bar 0 has no prev close, TR = range
    for i in range(1, N):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    _assert_valid_close(atr(high, low, close, length=n, mamode="rma").to_numpy(),
                        _rma_ref(tr, n), atol=1e-8)


# ---------------------------------------------------------------------------
# Volume family
# ---------------------------------------------------------------------------

def test_parity_obv_ad_cmf_vwap():
    open_, high, low, close, volume = _ohlcv()
    o, h, l, c, v = (s.to_numpy() for s in (open_, high, low, close, volume))

    obv_ref = np.cumsum(np.where(np.diff(c, prepend=c[0]) > 0, v,
                          np.where(np.diff(c, prepend=c[0]) < 0, -v, 0.0)))
    _assert_valid_close(obv(close, volume).to_numpy(), obv_ref)

    ad_ref = np.cumsum(np.where(h - l != 0, (2 * c - h - l) / (h - l) * v, 0.0))
    _assert_valid_close(ad(high, low, close, volume).to_numpy(), ad_ref)

    n = 20
    cmf_ref = np.full(N, np.nan)
    for i in range(n - 1, N):
        num = den = 0.0
        for j in range(i - n + 1, i + 1):
            if h[j] != l[j]:
                num += ((c[j] - l[j]) - (h[j] - c[j])) / (h[j] - l[j]) * v[j]
            den += v[j]
        if den > 0:
            cmf_ref[i] = num / den
    _assert_valid_close(cmf(high, low, close, volume, length=n).to_numpy(), cmf_ref)

    tp = (h + l + c) / 3.0
    cv = np.cumsum(v)
    cpv = np.cumsum(tp * v)
    vwap_ref = np.where(cv > 0, cpv / np.where(cv == 0, 1, cv), np.nan)
    _assert_valid_close(vwap(high, low, close, volume).to_numpy(), vwap_ref)


# ---------------------------------------------------------------------------
# Statistics family
# ---------------------------------------------------------------------------

def test_parity_stdev_zscore():
    close = _close()
    n = 30
    x = close.to_numpy()
    mean = np.full(N, np.nan)
    sd0 = np.full(N, np.nan)
    for i in range(n - 1, N):
        w = x[i - n + 1 : i + 1]
        mean[i] = w.mean()
        sd0[i] = np.sqrt(((w - w.mean()) ** 2).mean())  # ddof=0
    z = (x - mean) / sd0
    _assert_valid_close(stdev(close, length=n, ddof=0).to_numpy(), sd0)
    _assert_valid_close(zscore(close, length=n, ddof=0).to_numpy(), np.where(sd0 != 0, z, np.nan))
