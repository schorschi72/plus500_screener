
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def _to_1d(x):
    """Sichere Umwandlung in 1D-Pandas-Series mit float-Werten."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Erwarte 1 Spalte, bekam {x.shape[1]}")
        x = x.iloc[:, 0]
    elif not isinstance(x, pd.Series):
        x = pd.Series(x)
    values = np.asarray(x).reshape(-1)
    return pd.Series(values, index=x.index).astype(float)

def rsi(series, period=14, method="sma"):
    """
    RSI mit SMA- oder EMA-Glättung (formstabil, ohne np.where).
    method='sma' (klassisch) oder 'ema' (Wilder/EMA-Variante).
    """
    s = _to_1d(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    if method == "ema":
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    else:
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r

def ema(series, period=12):
    s = _to_1d(series)
    return s.ewm(span=period, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    s = _to_1d(series)
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(high, low, close, period=14):
    h = _to_1d(high); l = _to_1d(low); c = _to_1d(close)
    hl = (h - l).abs()
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def volume_zscore(volume, window=20):
    v = _to_1d(volume)
    m = v.rolling(window).mean()
    s = v.rolling(window).std()
    return (v - m) / s

def roc(series, period=3):
    s = _to_1d(series)
    return s.pct_change(period)
