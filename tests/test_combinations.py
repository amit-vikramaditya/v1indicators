import pandas as pd
import numpy as np
import pytest
from v1indicators import (
    sma, ema, rsi, macd, supertrend, bbands, keltner, 
    adx, psar, mfi, cci, vwap
)

@pytest.fixture
def ohlcv_data():
    """Generates synthetic OHLCV data for testing combinations."""
    np.random.seed(42)
    periods = 200
    
    # Generate a random walk
    returns = np.random.normal(0, 0.01, periods)
    price_path = 100 * np.cumprod(1 + returns)
    
    # Create OHLCV structure
    close = pd.Series(price_path, name="close")
    high = close * (1 + np.abs(np.random.normal(0, 0.005, periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, periods)))
    open_ = close.shift(1).fillna(100)
    volume = pd.Series(np.random.randint(100, 1000, periods), name="volume")
    
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })

def test_trend_strategy_combo(ohlcv_data):
    """
    Scenario: Trend Following.
    Combine EMA (Trend Filter) + Supertrend (Entry/Exit) + ADX (Strength).
    """
    df = ohlcv_data.copy()
    
    # 1. Calculate Indicators
    df['EMA_200'] = ema(df['close'], length=50) # Using 50 for shorter data
    st_df = supertrend(df['high'], df['low'], df['close'], length=10, mult=3.0)
    adx_df = adx(df['high'], df['low'], df['close'], length=14)
    
    # 2. Merge into main DataFrame
    # v1indicators returns indices aligned with input, so direct assignment or join works
    df = pd.concat([df, st_df, adx_df], axis=1)
    
    # 3. Validation
    assert 'SUPERTREND' in df.columns
    assert 'ADX_14' in df.columns
    
    # 4. Logic Check (Simulation)
    # Ensure we can create a signal mask without errors
    # Buy Signal: Close > EMA AND Supertrend is Bullish (1) AND Trend Strength > 20
    buy_signal = (
        (df['close'] > df['EMA_200']) & 
        (df['SUPERTREND_DIR'] == 1) & 
        (df['ADX_14'] > 20)
    )
    
    # Just ensure it's a boolean series and has length
    assert isinstance(buy_signal, pd.Series)
    assert len(buy_signal) == 200

def test_oscillator_divergence_combo(ohlcv_data):
    """
    Scenario: Momentum/Reversion.
    Combine RSI + MACD + CCI.
    """
    df = ohlcv_data.copy()
    
    # 1. Calculate
    df['RSI'] = rsi(df['close'], length=14)
    df['CCI'] = cci(df['high'], df['low'], df['close'], length=20)
    macd_df = macd(df['close'], fast=12, slow=26, signal=9)
    
    # 2. Join
    df = pd.concat([df, macd_df], axis=1)
    
    # 3. Check consistency
    # Ensure MACD Histogram exists alongside RSI
    valid_rows = df.dropna()
    assert not valid_rows.empty
    
    # 4. Complex Condition
    # Oversold: RSI < 30 AND CCI < -100 AND MACD Hist increasing
    condition = (
        (df['RSI'] < 30) &
        (df['CCI'] < -100) &
        (df['MACD_HIST'] > df['MACD_HIST'].shift(1))
    )
    assert len(condition) == 200

def test_volatility_squeeze_combo(ohlcv_data):
    """
    Scenario: TTM Squeeze style logic.
    Combine Bollinger Bands + Keltner Channels.
    """
    df = ohlcv_data.copy()
    
    # 1. Calculate
    bb = bbands(df['close'], length=20, mult=2.0)
    kc = keltner(df['high'], df['low'], df['close'], length=20, atr_length=10, mult=1.5)
    
    df = pd.concat([df, bb, kc], axis=1)
    
    # 2. Logic: Squeeze is ON when BB Upper < KC Upper AND BB Lower > KC Lower
    # (Bollinger bands are inside Keltner channels)
    squeeze_on = (
        (df['BB_UPPER'] < df['KELTNER_UPPER']) &
        (df['BB_LOWER'] > df['KELTNER_LOWER'])
    )
    
    # 3. Check integrity
    assert squeeze_on.dtype == bool
    # There might be no squeeze in random data, but the calculation should hold
    assert len(squeeze_on) == 200

def test_volume_confirmation_combo(ohlcv_data):
    """
    Scenario: Volume + Price action.
    Combine MFI + VWAP + PSAR.
    """
    df = ohlcv_data.copy()
    
    # 1. Calculate
    # VWAP usually requires intraday, but works math-wise on daily
    df['MFI'] = mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    # Note: vwap in v1indicators is cumulative from start of series
    df['VWAP'] = vwap(df['high'], df['low'], df['close'], df['volume']) 
    psar_df = psar(df['high'], df['low'])
    
    df = pd.concat([df, psar_df], axis=1)
    
    # 2. Logic: Price above VWAP AND PSAR is Bullish AND MFI not overbought
    long_condition = (
        (df['close'] > df['VWAP']) &
        (df['PSAR_DIR'] == 1) &
        (df['MFI'] < 80)
    )
    
    assert len(long_condition) == 200
