import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import time
import logging


logging.basicConfig(level=logging.DEBUG)

def create_lag_features(df, lags, target_col):
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)
    return df

def short_term_forecast(df, plant, forecast_horizon=48):
    lags = [1, 2, 3, 24, 48, 72]
    return forecast(df, plant, lags, forecast_horizon)

def long_term_forecast(df, plant, forecast_horizon=720):
    lags = [1, 2, 3, 24, 48, 72, 144, 288, 576]
    return forecast(df, plant, lags, forecast_horizon)

def forecast(df, plant, lags, forecast_horizon):
    start_time = time.time()
    df = create_lag_features(df, lags, plant)
    X = df.drop(columns=[plant])
    y = df[plant]

    train_size = int(len(y) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logging.debug(f"Starting model training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=10),
        lgb.log_evaluation(period=1)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=20000,
        valid_sets=[train_data, test_data],
        callbacks=callbacks
    )
    logging.debug(f"Model training completed in {time.time() - start_time:.2f} seconds")

    logging.debug(f"Making predictions for test set...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    logging.debug(f"MAE calculated: {mae}")

    logging.debug(f"Forecasting for the next {forecast_horizon} hours...")
    last_known_data = df.iloc[-forecast_horizon:].drop(columns=[plant])
    forecast_hours = model.predict(last_known_data)
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='h')
    forecast_df = pd.DataFrame({'DateTime': forecast_index, f'{forecast_horizon}_Hour_Forecast': forecast_hours})

    return forecast_df, mae
