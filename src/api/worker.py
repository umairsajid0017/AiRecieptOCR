import os
import queue
import threading
import logging
import tempfile
from PIL import Image, ImageOps
from pipeline import process_receipt_image
from .utils import send_callback, fetch_categories

logger = logging.getLogger(__name__)

# Job queue and worker pool concurrency configurations
concurrency_limit = int(os.environ.get("CONCURRENCY_LIMIT", "10"))
job_queue = queue.Queue()
temp_dir = tempfile.mkdtemp(prefix="receipt_ocr_")
pipeline_semaphore = threading.Semaphore(concurrency_limit)  # shared by sync and async: process up to concurrency_limit pipelines at a time

def worker_loop():
    """Worker loop: get job from queue, run pipeline, POST to callback, delete temp file."""
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id = job["job_id"]
        image_path = job["image_path"]
        questions = job["questions"]
        account_type = job.get("account_type", "EXPENSE")
        try:
            try:
                image = Image.open(image_path)
                image = ImageOps.exif_transpose(image) or image
                image = image.convert("RGB")
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
                    categories = fetch_categories(account_type=account_type)
                    result = process_receipt_image(image, questions=questions, categories=categories)
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
            payload = result["receipt"].copy()
            payload.update({
                "job_id": job_id,
                "status": "completed"
            })
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
    """Start concurrency_limit of daemon worker threads."""
    threads = []
    logger.info("Starting %d background OCR worker threads...", concurrency_limit)
    for i in range(concurrency_limit):
        worker_thread = threading.Thread(
            target=worker_loop, 
            name=f"OCRWorker-{i}", 
            daemon=True
        )
        worker_thread.start()
        threads.append(worker_thread)
    return threads

_worker_started = False
_worker_lock = threading.Lock()

def ensure_worker_started():
    """Ensure that the background worker threads are started, but only in async mode."""
    global _worker_started
    if not _worker_started:
        with _worker_lock:
            if not _worker_started:
                from .utils import is_async_mode
                if is_async_mode():
                    start_worker()
                _worker_started = True


