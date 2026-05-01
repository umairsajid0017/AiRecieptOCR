import logging
from flask import Flask
from dotenv import load_dotenv
from .routes import api_bp
from .worker import start_worker

def create_app():
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    
    app.register_blueprint(api_bp)
    
    # Start the background worker
    start_worker()
    
    return app
