"""
db.py
--------
Handles all SQLite database operations for the Stock Analytics Dashboard.

Why SQLite?
- No server setup needed (unlike MySQL/PostgreSQL)
- Built into Python (the 'sqlite3' module is part of the standard library)
- Creates a single file (stock_app.db) automatically the first time you run the app

This file stores 4 things:
1. watchlist        -> tickers the user has saved to track
2. portfolio_history -> a snapshot of portfolio value every time analysis is run
3. predictions       -> every ML prediction made, so accuracy can be checked later
4. trades            -> real Buy/Sell (Long/Short) actions, tied to the signal
"""

import sqlite3
from datetime import datetime
import pandas as pd

DB_NAME = "stock_app.db"


def get_connection():
    """Create (if needed) and return a connection to the SQLite database file."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    return conn


def init_db():
    """
    Creates all required tables if they don't already exist.
    Call this once when the app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            added_on TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            buy_price REAL NOT NULL,
            current_price REAL NOT NULL,
            current_value REAL NOT NULL,
            pnl REAL NOT NULL,
            pnl_pct REAL NOT NULL,
            recorded_on TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            current_price REAL NOT NULL,
            predicted_price REAL NOT NULL,
            r2_score REAL,
            predicted_on TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'LONG',
            shares REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            entry_signal TEXT,
            exit_price REAL,
            exit_time TEXT,
            profit REAL,
            profit_pct REAL,
            status TEXT NOT NULL DEFAULT 'OPEN'
        )
    """)

    conn.commit()
    conn.close()


# ---------- WATCHLIST FUNCTIONS ----------

def add_to_watchlist(ticker: str):
    """Save a ticker to the watchlist. Ignores duplicates."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO watchlist (ticker, added_on) VALUES (?, ?)",
            (ticker.upper(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # already in watchlist, no problem
    finally:
        conn.close()


def remove_from_watchlist(ticker: str):
    """Remove a ticker from the watchlist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
    conn.commit()
    conn.close()


def get_watchlist():
    """Returns a list of saved tickers, most recently added first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT ticker, added_on FROM watchlist ORDER BY id DESC", conn
    )
    conn.close()
    return df


# ---------- PORTFOLIO HISTORY FUNCTIONS ----------

def save_portfolio_snapshot(ticker, shares, buy_price, current_price,
                             current_value, pnl, pnl_pct):
    """Saves one snapshot of the portfolio's state (called every time analysis runs)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO portfolio_history
        (ticker, shares, buy_price, current_price, current_value, pnl, pnl_pct, recorded_on)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker.upper(), shares, buy_price, current_price,
        current_value, pnl, pnl_pct,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


def get_portfolio_history(ticker: str):
    """Returns full portfolio history for a given ticker, oldest first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM portfolio_history WHERE ticker = ? ORDER BY id ASC",
        conn, params=(ticker.upper(),)
    )
    conn.close()
    return df


# ---------- PREDICTION LOG FUNCTIONS ----------

def save_prediction(ticker, current_price, predicted_price, r2_score):
    """Logs a prediction so you can later compare it to what actually happened."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (ticker, current_price, predicted_price, r2_score, predicted_on)
        VALUES (?, ?, ?, ?, ?)
    """, (
        ticker.upper(), current_price, predicted_price, r2_score,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


def get_predictions(ticker: str):
    """Returns full prediction history for a given ticker, newest first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE ticker = ? ORDER BY id DESC",
        conn, params=(ticker.upper(),)
    )
    conn.close()
    return df


# ---------- TRADE SIMULATOR FUNCTIONS (real Buy/Sell, tied to signals) ----------

def get_open_trade(ticker: str):
    """
    Returns the currently OPEN trade for this ticker, if any (as a dict), else None.
    Only one open trade per ticker is allowed at a time - you must exit before
    you can enter a new position for the same ticker.
    """
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE ticker = ? AND status = 'OPEN' ORDER BY id DESC LIMIT 1",
        conn, params=(ticker.upper(),)
    )
    conn.close()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def open_trade(ticker, shares, entry_price, entry_signal, direction="LONG"):
    """
    Records an entry action.
    - direction="LONG": a normal Buy - profits if price goes UP later
    - direction="SHORT": a short-sell - profits if price goes DOWN later
      (you're borrowing shares to sell now, planning to buy them back cheaper)

    `entry_signal` stores what the signal said at the moment of entry
    (e.g. "BUY" or "SELL") - so later you can honestly show whether the
    trade was taken WITH or AGAINST the signal.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trades (ticker, direction, shares, entry_price, entry_time, entry_signal, status)
        VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
    """, (
        ticker.upper(), direction, shares, entry_price,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), entry_signal,
    ))
    conn.commit()
    conn.close()


def close_trade(trade_id, exit_price):
    """
    Records an exit action: closes an open trade at the current market price,
    calculating realized profit/loss for that specific trade.

    LONG trades profit when exit_price > entry_price (sell higher than you bought).
    SHORT trades profit when exit_price < entry_price (buy back cheaper than you sold).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT shares, entry_price, direction FROM trades WHERE id = ?", (trade_id,))
    shares, entry_price, direction = cursor.fetchone()

    if direction == "SHORT":
        # Sold high first, buying back now - profit if price fell
        profit = (entry_price - exit_price) * shares
    else:
        # Bought first, selling now - profit if price rose
        profit = (exit_price - entry_price) * shares

    investment = entry_price * shares
    profit_pct = (profit / investment * 100) if investment > 0 else 0.0

    cursor.execute("""
        UPDATE trades
        SET exit_price = ?, exit_time = ?, profit = ?, profit_pct = ?, status = 'CLOSED'
        WHERE id = ?
    """, (
        exit_price, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        profit, profit_pct, trade_id,
    ))
    conn.commit()
    conn.close()
    return profit, profit_pct


def get_trade_history(ticker: str):
    """Returns all trades (open and closed) for this ticker, newest first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE ticker = ? ORDER BY id DESC",
        conn, params=(ticker.upper(),)
    )
    conn.close()
    return df
