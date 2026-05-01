from flask import Blueprint
from .controllers import health_check, process_receipt

api_bp = Blueprint("api", __name__)

api_bp.route("/health", methods=["GET"])(health_check)
api_bp.route("/api/process", methods=["POST"])(process_receipt)
