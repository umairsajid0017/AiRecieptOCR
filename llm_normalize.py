"""
LLM normalization: merge LayoutLM + Donut raw outputs into a fixed receipt JSON schema.
Uses Ollama only. PROCESSING_MODE=API: send receipt image directly to Ollama (vision), return JSON only.
"""
import json
import os
import re
import tempfile

RECEIPT_KEYS = [
    "store_name", "shop_name", "date", "total_amount", "tax_amount",
    "gst_amount", "sales_tax", "received", "payable",
]

SYSTEM_PROMPT = """You are a receipt data extractor. You will receive two raw outputs from different OCR/QA models (LayoutLM Q&A and Donut extraction). Your task is to merge them into a single JSON object with exactly these keys (use null for any missing value):
- store_name (string): store or business name
- shop_name (string): shop name, can be same as store_name
- date (string): transaction date
- total_amount (number or string): total amount
- tax_amount (number or string): tax amount
- gst_amount (number or string): GST amount
- sales_tax (number or string): sales tax
- received (number or string): amount received
- payable (number or string): amount payable

Output ONLY valid JSON with these keys. No markdown, no explanation. Prefer numbers for amount fields when possible."""

# When PROCESSING_MODE=API: image is sent to LLM; instruct it to return only this JSON.
API_VISION_PROMPT = """Look at this receipt image. Extract the following fields and return ONLY a JSON object with exactly these keys (use null for any missing value). No markdown, no explanation, no other text—only the JSON.
Keys: store_name, shop_name, date, total_amount, tax_amount, gst_amount, sales_tax, received, payable.
Prefer numbers for amount fields when possible."""


def _parse_ollama_response(text: str) -> dict:
    """Parse LLM response into receipt dict; ensure all RECEIPT_KEYS exist."""
    text = text.strip()
    # Strip markdown code blocks if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text, "_error": "Invalid JSON from LLM"}
    if not isinstance(data, dict):
        return {"_raw": text, "_error": "LLM did not return a JSON object"}
    # Normalize to exact schema
    receipt = {}
    for key in RECEIPT_KEYS:
        receipt[key] = data.get(key) if data.get(key) is not None else None
    return receipt


def _normalize_via_ollama(layoutlm_results: list, donut_data: dict) -> dict:
    from ollama import chat, ResponseError

    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    user_content = (
        "LayoutLM (question-answering) results:\n"
        + json.dumps(layoutlm_results, indent=2, ensure_ascii=False)
        + "\n\nDonut (full extraction) result:\n"
        + json.dumps(donut_data, indent=2, ensure_ascii=False)
    )
    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            format="json",
        )
    except ResponseError as e:
        msg = str(e).strip()
        if "404" in msg or "not found" in msg.lower():
            return {"_error": f"Ollama model {model!r} not found. Set OLLAMA_MODEL in .env to a model you have (run 'ollama list')."}
        return {"_error": f"Ollama error: {msg}"}
    text = response.message.content if response and response.message else ""
    if not text:
        return {"_error": "Empty response from Ollama"}
    return _parse_ollama_response(text)


def _extract_via_ollama_vision(image) -> dict:
    """Send receipt image to Ollama vision model; return receipt dict (RECEIPT_KEYS or _error/_raw)."""
    from ollama import chat, ResponseError

    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    fd, path = None, None
    try:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        image.save(path)
        try:
            response = chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You extract receipt data. You must respond with only valid JSON, nothing else."},
                    {"role": "user", "content": API_VISION_PROMPT, "images": [path]},
                ],
                format="json",
            )
        except ResponseError as e:
            msg = str(e).strip()
            if "404" in msg or "not found" in msg.lower():
                return {"_error": f"Ollama model {model!r} not found. Set OLLAMA_MODEL to a vision-capable model (e.g. llava)."}
            return {"_error": f"Ollama error: {msg}"}
        text = response.message.content if response and response.message else ""
        if not text:
            return {"_error": "Empty response from Ollama"}
        return _parse_ollama_response(text)
    finally:
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def extract_receipt_from_image(image) -> dict:
    """
    PROCESSING_MODE=API: send receipt image directly to Ollama (vision). Return receipt dict
    with RECEIPT_KEYS; may include _error or _raw on failure.
    image: PIL Image (RGB).
    """
    return _extract_via_ollama_vision(image)


def normalize_receipt(layoutlm_results: list, donut_data: dict) -> dict:
    """
    Merge LayoutLM Q&A list and Donut extraction dict into a single receipt object
    with keys: store_name, shop_name, date, total_amount, tax_amount, gst_amount,
    sales_tax, received, payable.

    layoutlm_results: list of {"question": str, "answer": str}
    donut_data: dict from Donut model (may be raw extraction or parsed JSON)

    Returns dict with RECEIPT_KEYS; may include _error or _raw on failure.
    """
    return _normalize_via_ollama(layoutlm_results, donut_data)
