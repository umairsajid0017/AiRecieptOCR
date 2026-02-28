"""
Flask API for receipt processing. Uses shared pipeline.process_receipt_image().
POST /api/process: image (file or path) + optional questions → receipt JSON.
Single-token throttle: only one request runs the pipeline at a time; others wait in line.
"""
import json
import os
import io
import threading

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from PIL import Image

from pipeline import process_receipt_image, DEFAULT_QUESTIONS

app = Flask(__name__)

# One token: only one /api/process request runs the pipeline at a time; others block (FIFO).
_api_pipeline_semaphore = threading.Semaphore(1)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_image_from_request():
    """Load PIL Image from request: uploaded file, form path, or JSON body."""
    file = request.files.get("image") or request.files.get("file")
    if file and getattr(file, "filename", None) and file.filename.strip():
        if not _allowed_file(file.filename):
            return None, "Invalid image type; use PNG or JPEG"
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
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
                image = Image.open(path).convert("RGB")
                return image, None
            except Exception as e:
                return None, f"Invalid image: {e!s}"
        return None, f"File not found or not a file: {path!r}"
    return None, "Missing image: use image=@path (file upload), image=path (form), or JSON {\"image_path\": \"C:\\\\path\"}"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/process", methods=["POST"])
def process():
    image, err = _load_image_from_request()
    if err:
        return jsonify({"error": err}), 400

    questions = DEFAULT_QUESTIONS
    if request.form.get("questions"):
        try:
            q = json.loads(request.form["questions"])
            if isinstance(q, list) and len(q) > 0:
                questions = [str(x) for x in q]
        except (json.JSONDecodeError, TypeError):
            pass

    # Single-token throttle: wait in line (optional timeout)
    timeout_sec = None
    try:
        raw = os.environ.get("PIPELINE_THROTTLE_TIMEOUT", "").strip()
        if raw:
            timeout_sec = int(raw)
    except (TypeError, ValueError):
        pass

    if timeout_sec and timeout_sec > 0:
        acquired = _api_pipeline_semaphore.acquire(blocking=True, timeout=timeout_sec)
        if not acquired:
            return jsonify({"error": "Queue wait timeout"}), 503
    else:
        _api_pipeline_semaphore.acquire()

    try:
        result = process_receipt_image(image, questions=questions)
    finally:
        _api_pipeline_semaphore.release()

    response = {"receipt": result["receipt"]}
    if os.environ.get("INCLUDE_RAW", "true").strip().lower() in ("1", "true", "yes"):
        response["raw"] = {
            "layoutlm": result["layoutlm_results"],
            "donut": result["donut_data"],
        }
    if result["receipt_meta"]:
        response["receipt_meta"] = result["receipt_meta"]

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
