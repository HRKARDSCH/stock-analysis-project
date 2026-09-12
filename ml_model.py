"""
ml_model.py
-------------
Machine Learning price trend predictor.

Improvement over a plain "fit a line to all the data":
- Splits data into train/test sets (80/20) so we can measure how well the
  model actually generalizes, instead of just fitting a line and trusting it.
- Reports R2 score, so you have a real number to show/defend in your viva.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def train_predict(close_prices: pd.Series):
    """
    Trains a Linear Regression model on Day-index vs Close price.
    Returns a dict with the model, predictions, next-day forecast, and
    evaluation metrics (R2, MAE).
    """
    df_ml = pd.DataFrame({"Close": close_prices}).dropna().reset_index(drop=True)
    df_ml["Day"] = np.arange(len(df_ml))

    X = df_ml[["Day"]]
    y = df_ml["Close"]

    # 80% train, 20% test - keeps time order (no shuffling) since this is a time series
    split_idx = int(len(df_ml) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on the held-out test portion
    if len(X_test) > 0:
        y_pred_test = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred_test))
        mae = float(mean_absolute_error(y_test, y_pred_test))
    else:
        r2, mae = None, None

    # Refit on ALL data for the actual next-day forecast (best use of available info)
    final_model = LinearRegression()
    final_model.fit(X, y)
    next_day_index = pd.DataFrame({"Day": [len(df_ml)]})
    predicted_price = float(final_model.predict(next_day_index)[0])

    return {
        "model": final_model,
        "df_ml": df_ml,
        "fitted_line": final_model.predict(X),
        "predicted_price": predicted_price,
        "r2_score": r2,
        "mae": mae,
    }
