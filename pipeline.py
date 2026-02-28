"""
Shared receipt pipeline: LayoutLM (QA) → Donut (extraction) → LLM normalize → receipt JSON.
Used by both api.py (Flask) and app.py (Gradio). Single source of truth.
"""
import json
import logging
import os
import shutil

import pytesseract

from models.layoutlm import LayoutLMQA

# Set Tesseract binary path so it works under systemd (minimal PATH).
# LayoutLM document-question-answering pipeline uses pytesseract for OCR.
_tesseract_cmd = (
    os.environ.get("TESSERACT_CMD")
    or shutil.which("tesseract")
    or "/usr/bin/tesseract"
)
pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
logging.getLogger(__name__).debug("Tesseract cmd: %s", _tesseract_cmd)
from models.donut import DonutExtractor
from llm_normalize import normalize_receipt, RECEIPT_KEYS

# Default questions for LayoutLM (same schema as receipt output)
DEFAULT_QUESTIONS = [
    "What is the store or business name?",
    "What is the shop name?",
    "What is the date on the receipt?",
    "What is the total amount?",
    "What is the tax amount?",
    "What is the GST amount?",
    "What is the sales tax?",
    "What is the amount received?",
    "What is the amount payable?",
]

_layoutlm = None
_donut = None


def get_layoutlm():
    global _layoutlm
    if _layoutlm is None:
        _layoutlm = LayoutLMQA()
    return _layoutlm


def get_donut():
    global _donut
    if _donut is None:
        _donut = DonutExtractor()
    return _donut


def parse_layoutlm_answer(raw: str) -> str:
    """Extract answer text from LayoutLM output (strips ' (Confidence: 0.xx)')."""
    if not raw or not isinstance(raw, str):
        return raw or ""
    if "\n\n(Confidence:" in raw:
        raw = raw.split("\n\n(Confidence:")[0].strip()
    return raw.strip()


def ensure_receipt_schema(receipt: dict) -> dict:
    """Ensure receipt has exactly RECEIPT_KEYS; strip _error/_raw for clean output."""
    out = {}
    for key in RECEIPT_KEYS:
        out[key] = receipt.get(key) if key in receipt else None
    return out


def process_receipt_image(image, questions=None):
    """
    Run the full pipeline on a receipt image. Single function used by API and UI.

    Args:
        image: PIL Image (RGB).
        questions: Optional list of questions for LayoutLM; defaults to DEFAULT_QUESTIONS.

    Returns:
        dict with:
            receipt: normalized dict (RECEIPT_KEYS only).
            receipt_meta: None or dict with _error/_raw if LLM failed.
            layoutlm_results: list of {question, answer}.
            donut_data: dict from Donut (or {_error: ...}).
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS

    layoutlm_results = []
    layoutlm = get_layoutlm()
    for q in questions:
        try:
            raw = layoutlm.process(image, q)
            answer = parse_layoutlm_answer(raw)
            layoutlm_results.append({"question": q, "answer": answer})
        except Exception as e:
            layoutlm_results.append({"question": q, "answer": f"[Error: {e!s}]"})

    donut = get_donut()
    donut_data = {}
    try:
        donut_raw = donut.process(image)
        if isinstance(donut_raw, str):
            try:
                donut_data = json.loads(donut_raw)
            except json.JSONDecodeError:
                donut_data = {"_raw_text": donut_raw}
        elif isinstance(donut_raw, dict):
            donut_data = donut_raw
        else:
            donut_data = {"_raw": str(donut_raw)}
    except Exception as e:
        donut_data = {"_error": str(e)}

    receipt = normalize_receipt(layoutlm_results, donut_data)
    receipt_clean = ensure_receipt_schema(receipt)
    has_error = "_error" in receipt or "_raw" in receipt
    receipt_meta = {k: v for k, v in receipt.items() if k in ("_error", "_raw")} if has_error else None

    return {
        "receipt": receipt_clean,
        "receipt_meta": receipt_meta,
        "layoutlm_results": layoutlm_results,
        "donut_data": donut_data,
    }
