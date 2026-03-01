"""
Shared receipt pipeline: image → vision model (Ollama API) → receipt JSON.
Used by both api.py (Flask) and app.py (Gradio). Single source of truth.
"""
import logging
import os

from llm_normalize import extract_receipt_from_image, RECEIPT_KEYS

logging.getLogger(__name__)


def ensure_receipt_schema(receipt: dict) -> dict:
    """Ensure receipt has exactly RECEIPT_KEYS; strip _error/_raw for clean output."""
    out = {}
    for key in RECEIPT_KEYS:
        out[key] = receipt.get(key) if key in receipt else None
    return out


def process_receipt_image(image, questions=None):
    """
    Run the receipt pipeline on an image: vision model → JSON.

    Args:
        image: PIL Image (RGB).
        questions: Unused (kept for API compatibility).

    Returns:
        dict with:
            receipt: normalized dict (RECEIPT_KEYS only).
            receipt_meta: None or dict with _error/_raw if extraction failed.
            layoutlm_results: [] (kept for API compatibility).
            donut_data: {} (kept for API compatibility).
    """
    receipt = extract_receipt_from_image(image)
    receipt_clean = ensure_receipt_schema(receipt)
    has_error = "_error" in receipt or "_raw" in receipt
    receipt_meta = {k: v for k, v in receipt.items() if k in ("_error", "_raw")} if has_error else None
    return {
        "receipt": receipt_clean,
        "receipt_meta": receipt_meta,
        "layoutlm_results": [],
        "donut_data": {},
    }
