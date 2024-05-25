import logging
from waitress import serve
from app import app

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5000)
