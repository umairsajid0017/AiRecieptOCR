import io
import os
import json
import logging
import requests
import tempfile
from PIL import Image, ImageOps
from flask import request

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CALLBACK_TIMEOUT_SEC = 30
CALLBACK_RETRIES = 2
CATEGORIES_API_URL = os.environ.get(
    "CATEGORIES_API_URL", 
    "http://localhost:9000/api/receipts/categories?&minify=true"
)

def fetch_categories(account_type="EXPENSE"):
    """Fetch categories from external API for a specific accountType. Return list of objects or None."""
    try:
        # Base URL from env or default
        url = CATEGORIES_API_URL
        
        # Ensure we append the accountType parameter correctly
        separator = "&" if "?" in url else "?"
        final_url = f"{url}{separator}accountType={account_type}"
        
        r = requests.get(final_url, timeout=5)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return None
    except Exception as e:
        logger.warning("Could not fetch categories from API (%s): %s", account_type, e)
        return None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_from_request():
    """Load PIL Image from request: uploaded file, form path, or JSON body."""
    file = request.files.get("image") or request.files.get("file")
    if file and getattr(file, "filename", None) and file.filename.strip():
        if not allowed_file(file.filename):
            return None, "Invalid image type; use PNG or JPEG"
        try:
            image = Image.open(io.BytesIO(file.read()))
            image = ImageOps.exif_transpose(image) or image
            image = image.convert("RGB")
            return image, None
        except Exception as e:
            return None, f"Invalid image: {e!s}"

    path = (
        request.form.get("image")
        or request.form.get("file")
        or request.form.get("image_path")
    )
    if not path and request.form:
        for key in ("image", "file", "image_path", "path"):
            v = request.form.get(key)
            if v and isinstance(v, str) and (v.strip().startswith("/") or ":\\" in v or (len(v) >= 2 and v[1] == ":")):
                path = v
                break
        if not path:
            for _, v in request.form.items():
                if isinstance(v, str) and (":\\" in v or (v.startswith("/") and os.path.sep in v)):
                    path = v
                    break

    if not path and request.is_json:
        try:
            data = request.get_json(silent=True) or {}
            path = data.get("image_path") or data.get("image") or data.get("file")
        except Exception:
            pass

    if path and isinstance(path, str):
        path = path.strip().strip('"').replace("/", os.path.sep)
        if path and os.path.isfile(path):
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if ext not in ALLOWED_EXTENSIONS:
                return None, "Invalid image type; use PNG or JPEG"
            try:
                image = Image.open(path)
                image = ImageOps.exif_transpose(image) or image
                image = image.convert("RGB")
                return image, None
            except Exception as e:
                return None, f"Invalid image: {e!s}"
        return None, f"File not found or not a file: {path!r}"
    return None, "Missing image: use image=@path (file upload), image=path (form), or JSON {\"image_path\": \"C:\\\\path\"}"

def save_image_to_temp(image, job_id, temp_dir):
    """Save PIL Image to temp file; return path. Caller/worker must delete when done."""
    path = os.path.join(temp_dir, f"{job_id}.png")
    image.save(path)
    return path

def send_callback(job_id, payload):
    """POST payload to CALLBACK_URL with retries. Log and return on failure."""
    callback_url = os.environ.get("CALLBACK_URL", "").strip()
    if not callback_url:
        logger.warning("CALLBACK_URL not set; skipping callback for job_id=%s", job_id)
        return
    try:
        logger.info("Outgoing callback payload job_id=%s: %s", job_id, json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.info("Outgoing callback payload job_id=%s (non-JSON-serializable payload)", job_id)
    for attempt in range(CALLBACK_RETRIES + 1):
        try:
            r = requests.post(
                callback_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=CALLBACK_TIMEOUT_SEC,
            )
            r.raise_for_status()
            logger.info("Callback succeeded for job_id=%s", job_id)
            return
        except requests.RequestException as e:
            logger.warning("Callback attempt %s failed for job_id=%s: %s", attempt + 1, job_id, e)
    logger.error("Callback failed after %s attempts for job_id=%s", CALLBACK_RETRIES + 1, job_id)

def build_receipt_response(result):
    """Build JSON response dict from pipeline result (for sync mode)."""
    response = result["receipt"].copy()
    if result.get("receipt_meta"):
        response["receipt_meta"] = result["receipt_meta"]
    return response

def is_async_mode():
    """True if API_MODE is async (default); False if sync."""
    mode = os.environ.get("API_MODE", "async").strip().lower()
    return mode in ("async", "1", "true", "yes")
