import logging
import os
from flask import Flask
from dotenv import load_dotenv
from .routes import api_bp

def create_app():
    # Find .env in the parent directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        dotenv_path = os.path.join(current_dir, ".env")
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            break
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
    else:
        load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    
    app.register_blueprint(api_bp)
    
    # Start the background worker lazily before the first request
    @app.before_request
    def lazy_start_worker():
        from .worker import ensure_worker_started
        ensure_worker_started()
    
    return app

