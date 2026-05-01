"""
Shared document pipeline: image → vision model (Ollama API) → structured JSON.
Handles both receipts and invoices. Used by api.py (Flask) and app.py (Gradio).
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


def process_receipt_image(image, questions=None, categories=None):
    """
    Run the document pipeline on an image (receipt or invoice): vision model → JSON.

    Args:
        image: PIL Image (RGB).
        questions: Unused.
        categories: List of valid category names to choose from.

    Returns:
        dict with:
            receipt: normalized dict (RECEIPT_KEYS only). Works for both receipts and invoices.
            receipt_meta: None or dict with _error/_raw if extraction failed.
    """
    receipt = extract_receipt_from_image(image, categories=categories)
    receipt_clean = ensure_receipt_schema(receipt)

    # Lookup category_id if categories list was provided
    if categories and receipt_clean.get("category"):
        picked_name = receipt_clean["category"]
        for c in categories:
            if isinstance(c, dict) and c.get("name") == picked_name:
                receipt_clean["category_id"] = c.get("id")
                break

    has_error = "_error" in receipt or "_raw" in receipt
    receipt_meta = {k: v for k, v in receipt.items() if k in ("_error", "_raw")} if has_error else None
    return {
        "receipt": receipt_clean,
        "receipt_meta": receipt_meta,
    }
