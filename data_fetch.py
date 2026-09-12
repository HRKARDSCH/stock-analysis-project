"""
data_fetch.py
---------------
Handles all calls to Yahoo Finance (via yfinance), with:
- Proper error handling (no more raw crash screens on bad ticker / no internet)
- Caching (@st.cache_data) so repeated clicks don't re-download the same data
"""

import streamlit as st
import yfinance as yf
import pandas as pd


@st.cache_data(ttl=300, show_spinner=False)  # cache for 5 minutes
def fetch_stock_data(ticker: str, period: str, interval: str = "1d"):
    """
    Fetches historical price data for a ticker.
    `interval` controls candle size, `period` controls how far back to load.
    These are independent settings, matching how real trading platforms work.

    Special case: "4h" is not a native yfinance interval (yfinance's largest
    intraday interval is "1h"), so when interval="4h" is requested, this
    fetches 1-hour candles and resamples them into 4-hour candles instead.

    Returns (data, error_message). If error_message is not None, data is None.
    """
    try:
        stock = yf.Ticker(ticker)

        if interval == "4h":
            data = stock.history(period=period, interval="1h")
            if not data.empty:
                data = resample_ohlc(data, "4h")
        else:
            data = stock.history(period=period, interval=interval)

        if data.empty:
            return None, f"No data found for ticker '{ticker}'. Please check the symbol and try again."

        return data, None

    except Exception as e:
        return None, f"Failed to fetch data for '{ticker}'. Error: {str(e)}"


def resample_ohlc(data: pd.DataFrame, rule: str):
    """
    Combines smaller candles into bigger ones (e.g. four 1-hour candles into
    one 4-hour candle). Open = first candle's open, High = highest high,
    Low = lowest low, Close = last candle's close, Volume = summed.
    """
    agg_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    available_cols = {k: v for k, v in agg_rules.items() if k in data.columns}
    resampled = data.resample(rule).agg(available_cols).dropna(how="any")
    return resampled


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(ticker: str):
    """
    Fetches recent news for a ticker.
    Returns (news_list, error_message).
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news, None
    except Exception as e:
        return [], f"Could not fetch news: {str(e)}"


def extract_ohlc(data: pd.DataFrame):
    """
    Safely extracts Open/High/Low/Close as 1D Series, handling the case
    where yfinance sometimes returns a MultiIndex DataFrame.
    """
    def safe_col(col_name):
        col = data[col_name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col

    return {
        "open": safe_col("Open"),
        "high": safe_col("High"),
        "low": safe_col("Low"),
        "close": safe_col("Close"),
    }