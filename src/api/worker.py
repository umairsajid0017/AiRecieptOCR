import os
import queue
import threading
import logging
import tempfile
from PIL import Image
from pipeline import process_receipt_image
from .utils import send_callback

logger = logging.getLogger(__name__)

# Job queue and single worker (one process at a time)
job_queue = queue.Queue()
temp_dir = tempfile.mkdtemp(prefix="receipt_ocr_")
pipeline_semaphore = threading.Semaphore(1)  # shared by sync and async: only one pipeline run at a time

def worker_loop():
    """Single worker: get job from queue, run pipeline, POST to callback, delete temp file."""
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id = job["job_id"]
        image_path = job["image_path"]
        questions = job["questions"]
        try:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                send_callback(job_id, {"job_id": job_id, "status": "failed", "error": f"Failed to load image: {e!s}"})
                if os.path.isfile(image_path):
                    try:
                        os.remove(image_path)
                    except OSError:
                        pass
                continue
            try:
                pipeline_semaphore.acquire()
                try:
                    result = process_receipt_image(image, questions=questions)
                finally:
                    pipeline_semaphore.release()
            except Exception as e:
                send_callback(job_id, {"job_id": job_id, "status": "failed", "error": str(e)})
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
                "category": result["receipt"].get("category"),
                "document_type": result["receipt"].get("document_type"),
            }
            if result.get("receipt_meta"):
                payload["receipt_meta"] = result["receipt_meta"]
            send_callback(job_id, payload)
        finally:
            if os.path.isfile(image_path):
                try:
                    os.remove(image_path)
                except OSError as e:
                    logger.warning("Could not delete temp file %s: %s", image_path, e)
        job_queue.task_done()

def start_worker():
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    return worker_thread
