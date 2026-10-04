# AI-Based Stock Market Trend, Sentiment & Risk Analytics System

A comprehensive Streamlit dashboard for stock analysis: technical indicators,
candlestick pattern detection, portfolio tracking with a real Buy/Sell
(Long & Short) trade simulator, machine learning price prediction, and
news sentiment analysis — all with persistent history stored in SQLite.

## Project Structure

```
stock_project/
├── app.py            # Main Streamlit app (UI only)
├── db.py              # SQLite: watchlist, portfolio history, predictions, trades
├── data_fetch.py       # Yahoo Finance data fetching (cached, multi-timeframe, 4h resampling)
├── indicators.py        # SMA, RSI, Support/Resistance, Pivot Points, ATR, Engulfing Pattern
├── ml_model.py          # Linear Regression price predictor (train/test split + R²)
├── sentiment.py          # VADER-based news headline sentiment analysis
├── requirements.txt      # Python dependencies
└── stock_app.db          # SQLite database file (auto-created on first run)
```

## Setup Instructions

1. **Install Python 3.9+** if you don't already have it.

2. **Create a virtual environment** (recommended, keeps things clean):
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

4. **Run the app:**
```bash
   streamlit run app.py
```

5. Your browser will open automatically at `http://localhost:8501`.

## About the Database (SQLite)

No separate database server (like MySQL/XAMPP) is needed. SQLite is
file-based and built into Python. The first time you run `app.py`, a file
called `stock_app.db` is automatically created in the project folder. It
stores four tables:

- **watchlist** — tickers you've saved via the sidebar
- **portfolio_history** — a snapshot of your portfolio's value every time
  you click "Fetch & Analyze Data", so you can see value change over time
- **predictions** — every ML prediction made, logged with a timestamp, so
  you can later compare predicted vs. actual prices
- **trades** — every real Buy (Long) or Sell (Short) action taken in the
  Trade Simulator, with entry/exit price, the signal active at entry
  time, and realized profit or loss

You can inspect this file directly with any SQLite browser (e.g.,
[DB Browser for SQLite](https://sqlitebrowser.org/), or the SQLite Viewer
extension in VS Code) if you want to show the raw tables during your
project demo/viva.

## Features

- **Technical Analysis:** SMA (20/50), RSI(14), Support/Resistance (flat,
  fixed levels), Pivot Point levels (P, R1, R2, S1, S2), candlestick and
  line charts
- **Multi-Timeframe Support:** 1 Minute to 1 Month, including a custom
  4-Hour timeframe built via resampling, since yfinance does not natively
  support 4h candles
- **Candlestick Pattern Detection:** Engulfing pattern detector (ported
  from a TradingView Pine Script indicator), combined with RSI and
  price-movement filters, with ATR-based Target Price / Stop Loss
  suggestions shown directly on the chart
- **Portfolio Tracker:** A hypothetical "what-if" P&L calculator based on
  shares owned and average buy price
- **Trade Simulator:** Real timestamped Buy (Long) and Sell (Short)
  actions, tied to the signal active at entry time, with realized
  profit/loss calculated on exit — Long profits when price rises, Short
  profits when price falls
- **ML Price Predictor:** Linear Regression trained with an 80/20
  train-test split, reporting an R² score (not just a raw guess)
- **News Sentiment:** Real NLP sentiment scoring via VADER on recent
  headlines (not simple keyword matching)
- **Persistent History:** Watchlist, portfolio value over time, past
  predictions, and full trade history are all saved to SQLite and
  survive app restarts

## Libraries Used (requirements.txt)

| Library | Purpose |
|---|---|
| `streamlit` | Web app framework (UI, buttons, tabs, charts) |
| `yfinance` | Fetches live and historical stock data from Yahoo Finance |
| `pandas` | Data manipulation (tables, rolling calculations) |
| `numpy` | Mathematical operations (volatility, ATR, etc.) |
| `plotly` | Interactive candlestick and line charts |
| `scikit-learn` | Machine Learning (Linear Regression, train/test split, R²) |
| `vaderSentiment` | NLP-based sentiment analysis on news headlines |

Note: `sqlite3` and `datetime` are also used but are built into Python's
standard library, so they don't need to be installed separately. Running
`pip install -r requirements.txt` will also pull in each library's own
dependencies automatically (e.g. `pandas` brings in `pytz`) — those don't
need to be listed here manually.

## Notes for Viva / Report

- The ML model is intentionally simple (Linear Regression on a day index)
  — it is a **trend predictor**, not a guaranteed price forecast. The R²
  score shown honestly reflects how well a straight line fits the recent
  price action; a low or negative R² means the stock is too volatile for
  this simple model, which is worth mentioning as a known limitation.
- Sentiment analysis uses VADER, a lexicon-based NLP sentiment model
  commonly used for short text (headlines, tweets) — this is real NLP,
  not a keyword-count approach.
- If asked "why SQLite and not MySQL/PostgreSQL?" — SQLite requires no
  separate server, is bundled with Python, and is well suited to a
  single-user desktop/demo application like this one. For a multi-user
  production app, migrating to a cloud database like PostgreSQL would be
  the natural next step, since Streamlit Community Cloud's free tier
  storage is temporary and resets on app restart.
- Signal disagreement between indicators (e.g. SMA Signal vs. Engulfing
  Pattern) is expected and normal — they measure different things
  (long-term trend vs. short-term event), similar to how real traders
  combine multiple indicators rather than relying on just one.
- The Trade Simulator is a simulation, not a live trading platform — no
  real broker or real money is involved. It demonstrates the realized
  outcome of following, or going against, a signal, directly supporting
  the project's risk analysis objective.
- A signal can still result in a loss even when followed correctly,
  since technical indicators are probability-based tools, not
  guarantees — this is directly demonstrable using the Trade Simulator's
  trade history.
- If asked "why does `pip list` show ~50 packages when requirements.txt
  only has 7?" — only 7 were installed directly; the rest are automatic
  dependencies of those 7 (e.g. `scikit-learn` needs `scipy`, `pandas`
  needs `pytz`), which `pip` resolves and installs on its own.
