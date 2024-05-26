from flask import Flask, request, render_template, send_file, redirect, url_for, flash, session
import pandas as pd
import os
import logging
from ml_models import short_term_forecast, long_term_forecast
from config import Config
import plotly.express as px

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/short_term')
def upload_form():
    return render_template('short_term_upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    return handle_file_upload(request, 'select_sheet.html', 'short_term')

@app.route('/long_term')
def long_term_upload_form():
    return render_template('long_term_upload.html')

@app.route('/long_term_upload', methods=['POST'])
def long_term_upload_file():
    return handle_file_upload(request, 'select_sheet.html', 'long_term')

@app.route('/real_time', methods=['GET'])
def real_time():
    return render_template('real_time.html')



@app.route('/admin', methods=['GET', 'POST'])
def admin():

    #admin credentials
    ADMIN_USERNAME = "vynity"
    ADMIN_PASSWORD = "unicorn"

    if request.method == 'POST':
    
        if request.referrer != url_for('admin', _external=True):
            lead_name = request.form.get('name')
            lead_mail = request.form.get('mail')
            if lead_name and lead_mail:
                lead_string = f"{lead_name} ({lead_mail})"
                session.setdefault('leads', []).append(lead_string)
            return redirect(request.referrer)

        username = request.form.get('user')
        password = request.form.get('password')

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))

    if session.get('admin_logged_in'):
        leads = session.get('leads', [])
        return render_template('admin.html', leads=leads)
    else:
        return render_template('admin.html')
        
    

@app.route('/select_plant', methods=['POST'])
def select_plant():
    prediction_type = request.form['prediction_type']
    return handle_select_plant(request, 'select_plant.html', prediction_type)

@app.route('/process', methods=['POST'])
def process_file():
    prediction_type = request.form['prediction_type']
    if prediction_type == 'short_term':
        forecast_func = short_term_forecast
        template = 'short_term_result.html'
    else:
        forecast_func = long_term_forecast
        template = 'long_term_result.html'
    return handle_process_file(request, forecast_func, template)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

def handle_file_upload(request, template, prediction_type):
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
        return render_template(template, file_path=file_path, sheets=sheets, prediction_type=prediction_type)
    return redirect('/')

def handle_select_plant(request, template, prediction_type):
    file_path = request.form['file_path']
    sheet = request.form['sheet']
    df = pd.read_excel(file_path, sheet_name=sheet)
    columns = df.columns.str.strip().tolist()
    return render_template(template, file_path=file_path, sheet=sheet, columns=columns, prediction_type=prediction_type)

def handle_process_file(request, forecast_func, template):
    file_path = request.form['file_path']
    sheet = request.form['sheet']
    plant = request.form['plant']
    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df.columns = df.columns.str.strip()
        df['Date'] = df['Date'].astype(str)
        df['Time'] = df['Time'].astype(str)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.set_index('DateTime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.ffill(inplace=True)
        
        forecast_df, mae = forecast_func(df, plant)
        
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f'{forecast_func.__name__}_forecast.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            df[[plant]].to_excel(writer, sheet_name='Original_Data')
            forecast_df.to_excel(writer, sheet_name=f'{forecast_df.columns[1]}', index=False)
            mae_df = pd.DataFrame({'Metric': ['Mean Absolute Error'], 'Value': [mae]})
            mae_df.to_excel(writer, sheet_name='Metrics', index=False)

        plot_html_path = os.path.join(app.config['PLOT_FOLDER'], f'{forecast_func.__name__}_forecast.html')
        fig = px.line(forecast_df, x='DateTime', y=forecast_df.columns[1], title=f'{forecast_df.columns[1]} Forecast', labels={'DateTime': 'DateTime', forecast_df.columns[1]: 'Power Output'})
        fig.update_xaxes(rangeslider_visible=True)
        fig.write_html(plot_html_path)

        with open(plot_html_path, 'r', encoding='utf-8') as f:
            plot_html = f.read()
        forecast_preview = forecast_df.head(10).to_html(classes='table table-zebra', index=False, justify='left')
        return render_template(template, plot_html=plot_html, forecast_preview=forecast_preview, filename=os.path.basename(output_file))
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
