"""
indicators.py
---------------
Technical indicator calculations, kept separate from the app/UI code.
"""

import pandas as pd


def add_sma(data: pd.DataFrame, close_prices: pd.Series, short=20, long=50):
    """Adds Simple Moving Average columns to the dataframe."""
    data[f"SMA_{short}"] = close_prices.rolling(window=short).mean()
    data[f"SMA_{long}"] = close_prices.rolling(window=long).mean()
    return data


def calculate_support_resistance(data: pd.DataFrame, lookback=30):
    """
    Calculates a single, static Support and Resistance level - the way it's
    traditionally taught in technical analysis (a flat horizontal line the
    price has bounced off recently), NOT a rolling line that changes every day.

    Support    = lowest Low price in the last `lookback` days
    Resistance = highest High price in the last `lookback` days

    Returns a single (support, resistance) pair to be drawn as a flat
    horizontal line across the whole chart.
    """
    recent = data.tail(lookback)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    return support, resistance


def calculate_pivot_levels(data: pd.DataFrame):
    """
    Calculates classic Pivot Point trading levels using the most recent
    completed day's High, Low, and Close. This is the standard technique
    used across trading platforms to get MULTIPLE support/resistance
    levels (not just one line each).

    Returns a dict with: Pivot, R1, R2, S1, S2
    """
    last_row = data.iloc[-1]
    high = float(last_row["High"])
    low = float(last_row["Low"])
    close = float(last_row["Close"])

    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)

    return {
        "Pivot": pivot,
        "R1": r1,
        "R2": r2,
        "S1": s1,
        "S2": s2,
    }


def add_rsi(data: pd.DataFrame, close_prices: pd.Series, period=14):
    """Adds a Relative Strength Index (RSI) column to the dataframe."""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data


def calculate_atr(data: pd.DataFrame, period=14):
    """
    Calculates the Average True Range (ATR) - a measure of how much a stock
    typically moves per day. Used here to size Target Price / Stop Loss
    distances relative to the stock's own volatility, instead of a fixed
    rupee amount.
    """
    high = data["High"]
    low = data["Low"]
    prev_close = data["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    data["ATR"] = true_range.rolling(window=period).mean()
    return data


def detect_engulfing_signal(data: pd.DataFrame, rsi_threshold=70, candle_delta_length=4,
                             stability_index=0.5, rrr=2, tp_sl_multiplier=1.0):
    """
    Detects Bullish/Bearish Engulfing candlestick patterns combined with RSI
    extremes - re-implemented in Python from a TradingView Pine Script
    indicator's logic (candle engulfing + RSI filter + ATR-based TP/SL).

    A signal only fires when ALL of these line up together:
    - An engulfing candle pattern occurred
    - The candle is "stable" (not just a tiny wick/doji)
    - RSI confirms oversold (for bullish) / overbought (for bearish)
    - Price has moved enough over the last few candles to matter

    Returns a dict with the latest signal ('BUY', 'SELL', or None) plus
    suggested Target Price (TP) and Stop Loss (SL) levels based on ATR.
    """
    if "ATR" not in data.columns:
        data = calculate_atr(data)
    if "RSI" not in data.columns:
        data = add_rsi(data, data["Close"])

    open_ = data["Open"]
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    prev_close_val = data["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close_val).abs()
    tr3 = (low - prev_close_val).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    stable_candle = (close - open_).abs() / true_range > stability_index

    bullish_engulfing = (close.shift(1) < open_.shift(1)) & (close > open_) & (close > open_.shift(1))
    bearish_engulfing = (close.shift(1) > open_.shift(1)) & (close < open_) & (close < open_.shift(1))

    rsi_below = data["RSI"] < rsi_threshold
    rsi_above = data["RSI"] > (100 - rsi_threshold)

    decrease_over = close < close.shift(candle_delta_length)
    increase_over = close > close.shift(candle_delta_length)

    bull_signal = bullish_engulfing & stable_candle & rsi_below & decrease_over
    bear_signal = bearish_engulfing & stable_candle & rsi_above & increase_over

    data["Bull_Signal"] = bull_signal.fillna(False)
    data["Bear_Signal"] = bear_signal.fillna(False)

    # Calculate TP/SL for EVERY signal occurrence (not just the latest), so
    # each historical signal on the chart can show its own TP/SL box.
    dist_all = data["ATR"] * tp_sl_multiplier
    data["Signal_TP"] = None
    data["Signal_SL"] = None

    bull_mask = data["Bull_Signal"] & data["ATR"].notna()
    data.loc[bull_mask, "Signal_TP"] = (close[bull_mask] + dist_all[bull_mask] * rrr).round(2)
    data.loc[bull_mask, "Signal_SL"] = (close[bull_mask] - dist_all[bull_mask]).round(2)

    bear_mask = data["Bear_Signal"] & data["ATR"].notna()
    data.loc[bear_mask, "Signal_TP"] = (close[bear_mask] - dist_all[bear_mask] * rrr).round(2)
    data.loc[bear_mask, "Signal_SL"] = (close[bear_mask] + dist_all[bear_mask]).round(2)

    # Determine the LATEST signal (for a simple "what should I do today" summary)
    latest_close = float(close.iloc[-1])
    latest_atr = data["ATR"].iloc[-1]

    result = {"signal": None, "tp": None, "sl": None}

    if pd.notna(latest_atr):
        dist = float(latest_atr) * tp_sl_multiplier

        if bool(bull_signal.iloc[-1]):
            result["signal"] = "BUY"
            result["tp"] = round(latest_close + dist * rrr, 2)
            result["sl"] = round(latest_close - dist, 2)
        elif bool(bear_signal.iloc[-1]):
            result["signal"] = "SELL"
            result["tp"] = round(latest_close - dist * rrr, 2)
            result["sl"] = round(latest_close + dist, 2)

    return data, result


def calculate_risk_metrics(close_prices: pd.Series):
    """Returns total return %, volatility %, max price, min price."""
    import numpy as np

    current_price = float(close_prices.iloc[-1])
    start_price = float(close_prices.iloc[0])
    total_return = ((current_price - start_price) / start_price) * 100

    max_price = float(close_prices.max())
    min_price = float(close_prices.min())

    daily_returns = close_prices.pct_change().dropna()
    volatility = float(daily_returns.std() * np.sqrt(252) * 100)

    return {
        "current_price": current_price,
        "start_price": start_price,
        "total_return": total_return,
        "max_price": max_price,
        "min_price": min_price,
        "volatility": volatility,
    }