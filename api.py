"""
Flask API for receipt processing. Uses shared pipeline.process_receipt_image().
POST /api/process: image (file or path) + optional questions.
- API_MODE=async (default): returns 202 + job_id; worker processes in background and POSTs to CALLBACK_URL.
- API_MODE=sync: blocks and returns 200 with receipt JSON. One pipeline at a time (sync and async share a semaphore).
"""
import io
import json
import logging
import os
import queue
import tempfile
import threading
import uuid

import requests
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from PIL import Image

from pipeline import process_receipt_image, DEFAULT_QUESTIONS

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Job queue and single worker (one process at a time)
_job_queue = queue.Queue()
_temp_dir = tempfile.mkdtemp(prefix="receipt_ocr_")
_pipeline_semaphore = threading.Semaphore(1)  # shared by sync and async: only one pipeline run at a time
CALLBACK_TIMEOUT_SEC = 30
CALLBACK_RETRIES = 2


def _is_async_mode():
    """True if API_MODE is async (default); False if sync."""
    mode = os.environ.get("API_MODE", "async").strip().lower()
    return mode in ("async", "1", "true", "yes")

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


def _save_image_to_temp(image, job_id):
    """Save PIL Image to temp file; return path. Caller/worker must delete when done."""
    path = os.path.join(_temp_dir, f"{job_id}.png")
    image.save(path)
    return path


def _send_callback(job_id, payload):
    """POST payload to CALLBACK_URL with retries. Log and return on failure."""
    callback_url = os.environ.get("CALLBACK_URL", "").strip()
    if not callback_url:
        logger.warning("CALLBACK_URL not set; skipping callback for job_id=%s", job_id)
        return
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


def _include_raw():
    return os.environ.get("INCLUDE_RAW", "true").strip().lower() in ("1", "true", "yes")


def _build_receipt_response(result):
    """Build JSON response dict from pipeline result (for sync mode)."""
    response = {"receipt": result["receipt"]}
    if _include_raw():
        response["raw"] = {
            "layoutlm": result["layoutlm_results"],
            "donut": result["donut_data"],
        }
    if result["receipt_meta"]:
        response["receipt_meta"] = result["receipt_meta"]
    return response


def _worker():
    """Single worker: get job from queue, run pipeline, POST to callback, delete temp file."""
    while True:
        job = _job_queue.get()
        if job is None:
            break
        job_id = job["job_id"]
        image_path = job["image_path"]
        questions = job["questions"]
        try:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                _send_callback(job_id, {"job_id": job_id, "status": "failed", "error": f"Failed to load image: {e!s}"})
                if os.path.isfile(image_path):
                    try:
                        os.remove(image_path)
                    except OSError:
                        pass
                continue
            try:
                _pipeline_semaphore.acquire()
                try:
                    result = process_receipt_image(image, questions=questions)
                finally:
                    _pipeline_semaphore.release()
            except Exception as e:
                _send_callback(job_id, {"job_id": job_id, "status": "failed", "error": str(e)})
                if os.path.isfile(image_path):
                    try:
                        os.remove(image_path)
                    except OSError:
                        pass
                continue
            payload = {
                "job_id": job_id,
                "status": "completed",
                "receipt": result["receipt"],
            }
            if _include_raw():
                payload["raw"] = {
                    "layoutlm": result["layoutlm_results"],
                    "donut": result["donut_data"],
                }
            if result["receipt_meta"]:
                payload["receipt_meta"] = result["receipt_meta"]
            _send_callback(job_id, payload)
        finally:
            if os.path.isfile(image_path):
                try:
                    os.remove(image_path)
                except OSError as e:
                    logger.warning("Could not delete temp file %s: %s", image_path, e)
        _job_queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


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

    if _is_async_mode():
        job_id = str(uuid.uuid4())
        image_path = _save_image_to_temp(image, job_id)
        _job_queue.put({"job_id": job_id, "image_path": image_path, "questions": questions})
        return jsonify({"job_id": job_id}), 202

    # Sync mode: run pipeline in request thread (one at a time via semaphore)
    _pipeline_semaphore.acquire()
    try:
        result = process_receipt_image(image, questions=questions)
        return jsonify(_build_receipt_response(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        _pipeline_semaphore.release()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
