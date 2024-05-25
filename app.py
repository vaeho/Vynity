import os
import logging
from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import io
import time

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'supersecretkey'

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(basedir, 'outputs')
app.config['PLOT_FOLDER'] = os.path.join(basedir, 'static', 'plots')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)

def create_lag_features(df, lags, target_col):
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)
    return df

def process_data(file_path, sheet, plant):
    start_time = time.time()

    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('DateTime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.ffill(inplace=True)

    if plant not in df.columns:
        raise ValueError(f"Error: '{plant}' not found in DataFrame columns")

    y = df[plant]

    lags = [1, 2, 3, 24, 48, 72]
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

    logging.debug(f"Forecasting for the next 48 hours...")
    last_known_data = df.iloc[-48:].drop(columns=[plant])
    forecast_48_hours = model.predict(last_known_data)
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(hours=1), periods=48, freq='H')
    forecast_df = pd.DataFrame({'DateTime': forecast_index, '48_Hour_Forecast': forecast_48_hours})

    output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'hydropower_forecast.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        df[[plant]].to_excel(writer, sheet_name='Original_Data')
        forecast_df.to_excel(writer, sheet_name='48_Hour_Forecast', index=False)
        mae_df = pd.DataFrame({'Metric': ['Mean Absolute Error'], 'Value': [mae]})
        mae_df.to_excel(writer, sheet_name='Metrics', index=False)

    fig = px.line(forecast_df, x='DateTime', y='48_Hour_Forecast', title='48 Hour Forecast', labels={'DateTime': 'DateTime', '48_Hour_Forecast': 'Power Output'})
    fig.update_xaxes(rangeslider_visible=True)
    plot_html_path = os.path.join(app.config['PLOT_FOLDER'], '48_hour_forecast.html')
    fig.write_html(plot_html_path)

    logging.debug(f"Total processing time: {time.time() - start_time:.2f} seconds")

    return output_file, plot_html_path, forecast_df

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect('/')
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logging.debug(f"File path to save: {file_path}")
        file.save(file_path)
        xl = pd.ExcelFile(file_path)
        sheets = xl.sheet_names
        return render_template('select_sheet.html', file_path=file_path, sheets=sheets)
    return redirect('/')

@app.route('/select_plant', methods=['POST'])
def select_plant():
    file_path = request.form['file_path']
    sheet = request.form['sheet']
    df = pd.read_excel(file_path, sheet_name=sheet)
    columns = df.columns.str.strip().tolist()
    return render_template('select_plant.html', file_path=file_path, sheet=sheet, columns=columns)

@app.route('/process', methods=['POST'])
def process_file():
    file_path = request.form['file_path']
    sheet = request.form['sheet']
    plant = request.form['plant']
    try:
        output_file, plot_html_path, forecast_df = process_data(file_path, sheet, plant)
        with open(plot_html_path, 'r', encoding='utf-8') as f:
            plot_html = f.read()
        forecast_preview = forecast_df.head(10).to_html(classes='table table-zebra', index=False, justify='left')
        return render_template('result.html', plot_html=plot_html, forecast_preview=forecast_preview, filename=os.path.basename(output_file))
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return str(e)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
