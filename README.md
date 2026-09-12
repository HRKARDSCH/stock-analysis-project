# AI-Based Stock Market Trend, Sentiment & Risk Analytics System

A Streamlit dashboard for stock analysis: technical indicators, portfolio
tracking, machine learning price prediction, and news sentiment analysis —
with persistent history stored in SQLite.

## Project Structure

```
stock_project/
├── app.py            # Main Streamlit app (UI only)
├── db.py              # SQLite database: watchlist, portfolio history, predictions
├── data_fetch.py       # Yahoo Finance data fetching (cached + error-handled)
├── indicators.py        # SMA, RSI, risk metric calculations
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
stores three tables:

- **watchlist** — tickers you've saved via the sidebar
- **portfolio_history** — a snapshot of your portfolio's value every time
  you click "Fetch & Analyze Data", so you can see value change over time
- **predictions** — every ML prediction made, logged with a timestamp, so
  you can later compare predicted vs. actual prices

You can inspect this file directly with any SQLite browser (e.g.,
[DB Browser for SQLite](https://sqlitebrowser.org/)) if you want to show
the raw tables during your project demo/viva.

## Features

- **Technical Analysis:** SMA (20/50), RSI(14), Buy/Sell signal, candlestick
  and line charts
- **Portfolio Tracker:** Live P&L calculation based on shares owned and
  average buy price
- **ML Price Predictor:** Linear Regression trained with an 80/20
  train-test split, reporting an R² score (not just a raw guess)
- **News Sentiment:** Real NLP sentiment scoring via VADER on recent
  headlines (not simple keyword matching)
- **Persistent History:** Watchlist, portfolio value over time, and past
  predictions are all saved to SQLite and survive app restarts

## Notes for Viva / Report

- The ML model is intentionally simple (Linear Regression on a day index)
  — it is a **trend predictor**, not a guaranteed price forecast. The R²
  score shown honestly reflects how well a straight line fits the recent
  price action; a low or negative R² means the stock is too volatile for
  this simple model, which is worth mentioning as a known limitation.
- Sentiment analysis uses VADER, a lexicon-based NLP sentiment model
  commonly used for short text (headlines, tweets) — this is real NLP,
  not the earlier keyword-count approach.
- If asked "why SQLite and not MySQL/PostgreSQL?" — SQLite requires no
  separate server, is bundled with Python, and is well suited to a
  single-user desktop/demo application like this one. For a
  multi-user production app you'd typically move to PostgreSQL/MySQL.
