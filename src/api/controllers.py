import uuid
import json
import logging
import time
from flask import jsonify, request
from pipeline import process_receipt_image
from .utils import (
    load_image_from_request,
    is_async_mode,
    save_image_to_temp,
    build_receipt_response,
    fetch_categories
)
from .worker import job_queue, temp_dir, pipeline_semaphore

logger = logging.getLogger(__name__)

def health_check():
    return jsonify({"status": "ok"})

def process_receipt():
    started_at = time.time()
    image, err = load_image_from_request()
    if err:
        logger.info("Outgoing /api/process error response: %s", err)
        return jsonify({"error": err}), 400

    questions = []
    account_type = request.form.get("accountType") or request.form.get("type") or "EXPENSE"
    
    if request.is_json:
        try:
            data = request.get_json(silent=True) or {}
            account_type = data.get("accountType") or data.get("type") or account_type
            if data.get("questions"):
                questions = data["questions"]
        except Exception:
            pass

    if request.form.get("questions"):
        try:
            q = json.loads(request.form["questions"])
            if isinstance(q, list) and len(q) > 0:
                questions = [str(x) for x in q]
        except (json.JSONDecodeError, TypeError):
            pass

    if is_async_mode():
        job_id = str(uuid.uuid4())
        image_path = save_image_to_temp(image, job_id, temp_dir)
        job_queue.put({
            "job_id": job_id, 
            "image_path": image_path, 
            "questions": questions,
            "account_type": account_type
        })
        logger.info("Outgoing /api/process async response: %s", json.dumps({"job_id": job_id}, ensure_ascii=False))
        return jsonify({"job_id": job_id}), 202

    # Sync mode: run pipeline in request thread (one at a time via semaphore)
    pipeline_semaphore.acquire()
    try:
        categories = fetch_categories(account_type=account_type)
        logger.info(
            "OCR pipeline starting: accountType=%s categories=%s",
            account_type,
            len(categories) if isinstance(categories, list) else 0,
        )
        result = process_receipt_image(image, questions=questions, categories=categories)
        response = build_receipt_response(result)
        logger.info("OCR pipeline finished in %.2fs", time.time() - started_at)
        logger.info("Outgoing /api/process sync response: %s", json.dumps(response, ensure_ascii=False))
        return jsonify(response)
    except Exception as e:
        logger.exception("Outgoing /api/process 500 error")
        return jsonify({"error": str(e)}), 500
    finally:
        pipeline_semaphore.release()
