import uuid
import json
import logging
from flask import jsonify, request
from pipeline import process_receipt_image
from .utils import (
    load_image_from_request,
    is_async_mode,
    save_image_to_temp,
    build_receipt_response
)
from .worker import job_queue, temp_dir, pipeline_semaphore

logger = logging.getLogger(__name__)

def health_check():
    return jsonify({"status": "ok"})

def process_receipt():
    image, err = load_image_from_request()
    if err:
        logger.info("Outgoing /api/process error response: %s", err)
        return jsonify({"error": err}), 400

    questions = []
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
        job_queue.put({"job_id": job_id, "image_path": image_path, "questions": questions})
        logger.info("Outgoing /api/process async response: %s", json.dumps({"job_id": job_id}, ensure_ascii=False))
        return jsonify({"job_id": job_id}), 202

    # Sync mode: run pipeline in request thread (one at a time via semaphore)
    pipeline_semaphore.acquire()
    try:
        result = process_receipt_image(image, questions=questions)
        response = build_receipt_response(result)
        logger.info("Outgoing /api/process sync response: %s", json.dumps(response, ensure_ascii=False))
        return jsonify(response)
    except Exception as e:
        logger.exception("Outgoing /api/process 500 error")
        return jsonify({"error": str(e)}), 500
    finally:
        pipeline_semaphore.release()
