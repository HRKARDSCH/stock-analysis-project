"""
app.py
--------
AI-Based Stock Market Trend, Sentiment & Risk Analytics System
Main Streamlit application - handles UI/layout only.
All logic lives in separate modules: data_fetch.py, indicators.py,
ml_model.py, sentiment.py, db.py
"""

import streamlit as st
import plotly.graph_objects as go

import db
import data_fetch
import indicators
import ml_model
import sentiment

# ---------- APP SETUP ----------
st.set_page_config(page_title="Advanced Stock Analytics Dashboard", layout="wide")
db.init_db()  # creates stock_app.db and its tables on first run

st.title("AI-Based Stock Market Trend, Sentiment & Risk Analytics System")
st.markdown(
    "A comprehensive financial analytics platform featuring Technical Indicators, "
    "Portfolio Tracking, Machine Learning Price Prediction, News Sentiment Analysis, "
    "Interactive Charting, and persistent history (SQLite)."
)

# ---------- SIDEBAR ----------
st.sidebar.header("Dashboard Configuration")

# Use session_state so clicking a watchlist item below can update this box
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = "RELIANCE.NS"

ticker = st.sidebar.text_input("Enter Stock Ticker", key="active_ticker")

# Timeframe = candle size (1min, 15min, 4hr, 1day...) - DIFFERENT from Period
# (how much history to load). These are ALL the intervals yfinance supports,
# plus "4 Hours" which we build ourselves by combining 1-hour candles
# (yfinance itself doesn't offer 4h directly).
timeframe_options = {
    "1 Minute": "1m",
    "2 Minutes": "2m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "4 Hours": "4h",
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo",
}
timeframe_label = st.sidebar.selectbox(
    "Select Timeframe (candle size)", list(timeframe_options.keys()), index=7,
    help="How long each candle represents. Smaller timeframes only support a "
         "short lookback period - Yahoo Finance limits how far back intraday "
         "data goes (1-minute: last 7 days only; other intraday: last 60 days)."
)
interval = timeframe_options[timeframe_label]

# Period options change based on chosen timeframe, since Yahoo Finance limits
# how much history is available for small-candle (intraday) timeframes.
if interval == "1m":
    period_options = ["1d", "5d"]
elif interval in ["2m", "5m", "15m", "30m", "90m"]:
    period_options = ["5d", "1mo"]
elif interval in ["1h", "4h"]:
    period_options = ["1mo", "6mo", "1y", "2y"]
else:  # 1d, 1wk, 1mo
    period_options = ["1mo", "6mo", "1y", "5y"]

period = st.sidebar.selectbox("Select Time Period (how far back)", period_options)
st.sidebar.caption(
    "💡 Multi-timeframe tip: check a higher timeframe (e.g. 4 Hours or 1 Day) "
    "first for the overall trend, then switch to a lower timeframe (e.g. 15 "
    "Minutes) to time your entry — this is standard 'top-down analysis'."
)
chart_type = st.sidebar.selectbox("Select Chart Type", ["Candlestick Chart", "Line Chart"])

st.sidebar.subheader("Portfolio Tracker")
shares = st.sidebar.number_input("Shares Owned", min_value=0, value=10)
buy_price = st.sidebar.number_input("Average Purchase Price (₹)", min_value=0.0, value=2000.0)

# --- Watchlist controls (new: uses SQLite) ---
st.sidebar.subheader("⭐ Watchlist")
wl_col1, wl_col2 = st.sidebar.columns(2)
if wl_col1.button("Add to Watchlist"):
    db.add_to_watchlist(ticker)
    st.sidebar.success(f"{ticker.upper()} added!")
if wl_col2.button("Remove"):
    db.remove_from_watchlist(ticker)
    st.sidebar.info(f"{ticker.upper()} removed.")

watchlist_df = db.get_watchlist()
if not watchlist_df.empty:
    st.sidebar.write("Saved tickers (click to load):")

    def _load_ticker(t):
        # This runs BEFORE the script reruns, so it's safe to update session_state here
        st.session_state.active_ticker = t

    for _, row in watchlist_df.iterrows():
        saved_ticker = row["ticker"]
        st.sidebar.button(
            f"📌 {saved_ticker}",
            key=f"load_{saved_ticker}",
            on_click=_load_ticker,
            args=(saved_ticker,),
        )

run_analysis = st.sidebar.button("Fetch & Analyze Data", type="primary")

# ---------- MAIN LOGIC ----------
if run_analysis:
    with st.spinner("Fetching market data, running ML model and sentiment analysis..."):

        data, err = data_fetch.fetch_stock_data(ticker, period, interval)

        if err:
            st.error(err)
        else:
            st.success("Data fetched successfully!")

            ohlc = data_fetch.extract_ohlc(data)
            close_prices = ohlc["close"]

            # --- Risk metrics ---
            risk = indicators.calculate_risk_metrics(close_prices)
            current_price = risk["current_price"]

            # --- Portfolio math ---
            investment_value = shares * buy_price
            current_val = shares * current_price
            pnl = current_val - investment_value
            pnl_pct = (pnl / investment_value * 100) if investment_value > 0 else 0.0

            # Save this run to portfolio history (SQLite)
            db.save_portfolio_snapshot(
                ticker, shares, buy_price, current_price, current_val, pnl, pnl_pct
            )

            # --- Technical indicators ---
            data = indicators.add_sma(data, close_prices)
            data = indicators.add_rsi(data, close_prices)
            support_level, resistance_level = indicators.calculate_support_resistance(data)
            pivot_levels = indicators.calculate_pivot_levels(data)
            data, candle_signal = indicators.detect_engulfing_signal(data)

            # --- ML Prediction (with train/test split + R2) ---
            ml_result = ml_model.train_predict(close_prices)
            predicted_price = ml_result["predicted_price"]
            r2 = ml_result["r2_score"]

            # Log the prediction (SQLite)
            db.save_prediction(ticker, current_price, predicted_price, r2)

            # --- Sentiment (real NLP via VADER) ---
            news_list, news_err = data_fetch.fetch_news(ticker)
            sentiment_result = sentiment.analyze_headlines(news_list)

            # ---------- SIDEBAR PORTFOLIO SUMMARY ----------
            st.sidebar.subheader("Portfolio Status")
            st.sidebar.metric("Current Portfolio Value", f"₹{current_val:,.2f}", f"{pnl_pct:+.2f}%")
            st.sidebar.metric("Total Profit / Loss", f"₹{pnl:,.2f}")

            # ---------- MAIN TABS ----------
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Technical Chart & Signals",
                "💼 Risk & Portfolio",
                "🤖 AI Price Predictor",
                "📰 News Sentiment",
                "🗂️ History (Database)",
                "🕯️ Candlestick Pattern Signal",
            ])

            # --- TAB 1: Technical Chart ---
            with tab1:
                col_sig, col_chart = st.columns([1, 3])

                with col_sig:
                    st.subheader("Strategy Signals")
                    if data["SMA_20"].iloc[-1] > data["SMA_50"].iloc[-1]:
                        st.success("SMA Signal: BUY")
                    else:
                        st.error("SMA Signal: SELL")

                    rsi_value = data["RSI"].iloc[-1]
                    if rsi_value < 30:
                        st.info(f"RSI ({rsi_value:.1f}): OVERSOLD")
                    elif rsi_value > 70:
                        st.warning(f"RSI ({rsi_value:.1f}): OVERBOUGHT")
                    else:
                        st.write(f"RSI ({rsi_value:.1f}): NEUTRAL")

                    st.write("---")
                    st.metric("Support (last 30 days)", f"₹{support_level:.2f}")
                    st.metric("Resistance (last 30 days)", f"₹{resistance_level:.2f}")
                    st.caption(
                        "Support = price floor stock has bounced up from recently. "
                        "Resistance = price ceiling stock has fallen back from recently. "
                        "Shown as flat lines since these are fixed levels, not values "
                        "that change every day."
                    )

                    st.write("---")
                    st.subheader("Pivot Point Levels")
                    st.write(f"**Pivot:** ₹{pivot_levels['Pivot']:.2f}")
                    st.write(f"**R1:** ₹{pivot_levels['R1']:.2f}  |  **R2:** ₹{pivot_levels['R2']:.2f}")
                    st.write(f"**S1:** ₹{pivot_levels['S1']:.2f}  |  **S2:** ₹{pivot_levels['S2']:.2f}")
                    st.caption(
                        "Calculated from the most recent day's High, Low, Close using "
                        "the standard Pivot Point formula - gives multiple potential "
                        "reversal levels, not just one."
                    )

                with col_chart:
                    fig = go.Figure()
                    if chart_type == "Candlestick Chart":
                        fig.add_trace(go.Candlestick(
                            x=data.index, open=ohlc["open"], high=ohlc["high"],
                            low=ohlc["low"], close=close_prices, name="Candlestick"
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=data.index, y=close_prices,
                            mode="lines", name="Close Price", line=dict(color="cyan", width=2)
                        ))

                    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], line=dict(color="orange", width=1.5), name="SMA 20"))
                    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], line=dict(color="blue", width=1.5), name="SMA 50"))

                    # Support/Resistance as flat horizontal lines (traditional style)
                    fig.add_hline(
                        y=resistance_level, line_dash="dot", line_color="red",
                        annotation_text="Resistance", annotation_position="top left"
                    )
                    fig.add_hline(
                        y=support_level, line_dash="dot", line_color="green",
                        annotation_text="Support", annotation_position="bottom left"
                    )

                    # Pivot Point levels (multiple lines)
                    fig.add_hline(y=pivot_levels["Pivot"], line_dash="dash", line_color="yellow",
                                  annotation_text="Pivot", annotation_position="top right")
                    fig.add_hline(y=pivot_levels["R1"], line_dash="dash", line_color="salmon",
                                  annotation_text="R1", annotation_position="top right")
                    fig.add_hline(y=pivot_levels["R2"], line_dash="dash", line_color="salmon",
                                  annotation_text="R2", annotation_position="top right")
                    fig.add_hline(y=pivot_levels["S1"], line_dash="dash", line_color="lightgreen",
                                  annotation_text="S1", annotation_position="bottom right")
                    fig.add_hline(y=pivot_levels["S2"], line_dash="dash", line_color="lightgreen",
                                  annotation_text="S2", annotation_position="bottom right")

                    fig.update_layout(
                        title=f"{ticker.upper()} Price Action & Indicators",
                        template="plotly_dark", height=500, xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Historical Stock Data & Volume (Last 10 Days)")
                table_cols = ["Open", "High", "Low", "Close", "Volume"]
                available_cols = [c for c in table_cols if c in data.columns]
                st.dataframe(data[available_cols].tail(10), use_container_width=True)

            # --- TAB 2: Risk & Portfolio ---
            with tab2:
                st.subheader("Risk & Performance Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{risk['total_return']:.2f}%")
                m2.metric("Annual Volatility", f"{risk['volatility']:.2f}%")
                m3.metric("Period High", f"{risk['max_price']:.2f}")
                m4.metric("Period Low", f"{risk['min_price']:.2f}")

                st.write("---")
                st.subheader("Detailed Portfolio Breakdown")
                st.json({
                    "Stock Ticker": ticker.upper(),
                    "Shares Held": shares,
                    "Average Buy Price": f"₹{buy_price}",
                    "Current Market Price": f"₹{current_price:.2f}",
                    "Total Investment": f"₹{investment_value:,.2f}",
                    "Current Holding Value": f"₹{current_val:,.2f}",
                    "Net Profit / Loss": f"₹{pnl:,.2f}",
                })

            # --- TAB 3: ML Predictor ---
            with tab3:
                st.subheader("Machine Learning Trend Predictor")
                st.markdown(
                    "Linear Regression trained on historical price movement, "
                    "evaluated with an 80/20 train-test split (not just fit-and-guess)."
                )

                pred_col1, pred_col2, pred_col3 = st.columns(3)
                pred_col1.metric("Current Price", f"₹{current_price:.2f}")
                pred_col2.metric(
                    "ML Projected Price", f"₹{predicted_price:.2f}",
                    f"{((predicted_price - current_price)/current_price)*100:+.2f}%"
                )
                if r2 is not None:
                    pred_col3.metric("Model R² Score", f"{r2:.3f}")
                else:
                    pred_col3.metric("Model R² Score", "N/A (not enough data)")

                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(
                    x=ml_result["df_ml"]["Day"], y=ml_result["df_ml"]["Close"],
                    mode="lines", name="Actual Price"
                ))
                fig_ml.add_trace(go.Scatter(
                    x=ml_result["df_ml"]["Day"], y=ml_result["fitted_line"],
                    mode="lines", name="ML Trend Regression", line=dict(color="red", dash="dash")
                ))
                fig_ml.update_layout(title="Machine Learning Trend Fit", template="plotly_dark", height=400)
                st.plotly_chart(fig_ml, use_container_width=True)

                st.caption(
                    "R² closer to 1.0 means the straight-line trend explains the price "
                    "movement well. Closer to 0 (or negative) means the stock is too "
                    "volatile for a simple linear trend to predict well — worth mentioning "
                    "honestly in your report."
                )

            # --- TAB 4: Sentiment ---
            with tab4:
                st.subheader("AI News Sentiment Analysis (VADER NLP)")
                st.markdown(f"**Overall Market Sentiment:** {sentiment_result['label']}  "
                            f"(avg. compound score: {sentiment_result['avg_score']:.3f})")

                if news_err:
                    st.warning(news_err)

                if sentiment_result["details"]:
                    st.write("### Recent Headlines Analyzed:")
                    for i, article in enumerate(sentiment_result["details"], 1):
                        st.markdown(
                            f"**{i}. {article['title']}** *({article['publisher']})* "
                            f"— {article['label']} (score: {article['score']:.2f})"
                        )
                else:
                    st.info("No news feed currently accessible for this ticker.")

            # --- TAB 5: History from Database ---
            with tab5:
                st.subheader("📈 Portfolio Value Over Time (from SQLite)")
                history_df = db.get_portfolio_history(ticker)
                if not history_df.empty:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=history_df["recorded_on"], y=history_df["current_value"],
                        mode="lines+markers", name="Portfolio Value"
                    ))
                    fig_hist.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No history yet for this ticker — run analysis a few times on different days to build a trend.")

                st.subheader("🤖 Past Predictions vs Reality")
                pred_df = db.get_predictions(ticker)
                if not pred_df.empty:
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "Compare 'predicted_price' from past rows to 'current_price' in "
                        "later rows to check how accurate the model has been over time."
                    )
                else:
                    st.info("No predictions logged yet for this ticker.")

            # --- TAB 6: Candlestick Pattern Signal (Engulfing + RSI + ATR) ---
            with tab6:
                st.subheader("🕯️ Engulfing Candlestick Pattern Signal")
                st.markdown(
                    "Detects Bullish/Bearish **Engulfing candlestick patterns**, filtered by "
                    "RSI and recent price movement, with Target Price (TP) and Stop Loss (SL) "
                    "suggested using ATR (Average True Range) - a measure of the stock's own "
                    "typical daily movement. This logic was originally written as a TradingView "
                    "Pine Script indicator and re-implemented here in Python."
                )

                if candle_signal["signal"] == "BUY":
                    st.success(f"📈 Latest Signal: **BUY**")
                    c1, c2 = st.columns(2)
                    c1.metric("Suggested Target Price (TP)", f"₹{candle_signal['tp']:.2f}")
                    c2.metric("Suggested Stop Loss (SL)", f"₹{candle_signal['sl']:.2f}")
                elif candle_signal["signal"] == "SELL":
                    st.error(f"📉 Latest Signal: **SELL**")
                    c1, c2 = st.columns(2)
                    c1.metric("Suggested Target Price (TP)", f"₹{candle_signal['tp']:.2f}")
                    c2.metric("Suggested Stop Loss (SL)", f"₹{candle_signal['sl']:.2f}")
                else:
                    st.info(
                        "No Engulfing signal on the most recent candle. This pattern is "
                        "intentionally strict (needs engulfing shape + RSI extreme + recent "
                        "price move all together), so it won't fire every day - that's expected."
                    )

                st.write("---")
                st.caption(
                    f"Total Bullish signals found in this period: {int(data['Bull_Signal'].sum())}  |  "
                    f"Total Bearish signals found in this period: {int(data['Bear_Signal'].sum())}"
                )

                # Chart with markers + BUY/SELL + TP/SL boxes at each signal point
                fig_candle = go.Figure()
                fig_candle.add_trace(go.Candlestick(
                    x=data.index, open=ohlc["open"], high=ohlc["high"],
                    low=ohlc["low"], close=close_prices, name="Price"
                ))

                bull_points = data[data["Bull_Signal"]]
                bear_points = data[data["Bear_Signal"]]

                if not bull_points.empty:
                    fig_candle.add_trace(go.Scatter(
                        x=bull_points.index, y=bull_points["Low"] * 0.995,
                        mode="markers", name="Bullish Signal",
                        marker=dict(symbol="triangle-up", color="lime", size=10)
                    ))
                if not bear_points.empty:
                    fig_candle.add_trace(go.Scatter(
                        x=bear_points.index, y=bear_points["High"] * 1.005,
                        mode="markers", name="Bearish Signal",
                        marker=dict(symbol="triangle-down", color="red", size=10)
                    ))

                # BUY/SELL + TP/SL boxes, similar style to TradingView labels
                for idx, row in bull_points.iterrows():
                    fig_candle.add_annotation(
                        x=idx, y=row["Low"] * 0.99,
                        text=f"<b>BUY</b><br>TP: {row['Signal_TP']}<br>SL: {row['Signal_SL']}",
                        showarrow=True, arrowhead=2, arrowcolor="lime", ax=0, ay=40,
                        bgcolor="rgba(0,60,0,0.85)", bordercolor="lime", borderwidth=1,
                        font=dict(color="white", size=10), align="left"
                    )
                for idx, row in bear_points.iterrows():
                    fig_candle.add_annotation(
                        x=idx, y=row["High"] * 1.01,
                        text=f"<b>SELL</b><br>TP: {row['Signal_TP']}<br>SL: {row['Signal_SL']}",
                        showarrow=True, arrowhead=2, arrowcolor="red", ax=0, ay=-40,
                        bgcolor="rgba(60,0,0,0.85)", bordercolor="red", borderwidth=1,
                        font=dict(color="white", size=10), align="left"
                    )

                fig_candle.update_layout(
                    title=f"{ticker.upper()} - Engulfing Pattern Signals",
                    template="plotly_dark", height=550, xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig_candle, use_container_width=True)

                st.caption(
                    "TP/SL are suggestions based on ATR and a fixed risk-reward ratio (1:2 by "
                    "default) - not guaranteed outcomes. Always apply your own risk management."
                )

else:
    st.info("👈 Please enter your parameters in the sidebar, select chart type, and click **'Fetch & Analyze Data'** to start.")
    st.caption(
        "Tip: run analysis on the same ticker across a few different days — your "
        "Portfolio History and Prediction tabs will start showing real trends from "
        "the database."
    )