import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'supersecretkey'
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    OUTPUT_FOLDER = os.path.join(basedir, 'outputs')
    PLOT_FOLDER = os.path.join(basedir, 'static', 'plots')

    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)
