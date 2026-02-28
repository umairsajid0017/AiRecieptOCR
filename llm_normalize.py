"""
LLM normalization: merge LayoutLM + Donut raw outputs into a fixed receipt JSON schema.
Supports Ollama (local, ollama package) and Minimax M2.5 (cloud API).
"""
import json
import os
import re

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


def _normalize_via_minimax(layoutlm_results: list, donut_data: dict) -> dict:
    import requests

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        return {"_error": "MINIMAX_API_KEY not set"}
    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io").rstrip("/")
    url = f"{base_url}/v1/chat/completions"
    user_content = (
        "LayoutLM (question-answering) results:\n"
        + json.dumps(layoutlm_results, indent=2, ensure_ascii=False)
        + "\n\nDonut (full extraction) result:\n"
        + json.dumps(donut_data, indent=2, ensure_ascii=False)
    )
    payload = {
        "model": os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return {"_error": "No choices in Minimax response"}
        text = choices[0].get("message", {}).get("content", "")
        if not text:
            return {"_error": "Empty content from Minimax"}
        return _parse_ollama_response(text)
    except Exception as e:
        return {"_error": str(e)}


def normalize_receipt(layoutlm_results: list, donut_data: dict) -> dict:
    """
    Merge LayoutLM Q&A list and Donut extraction dict into a single receipt object
    with keys: store_name, shop_name, date, total_amount, tax_amount, gst_amount,
    sales_tax, received, payable.

    layoutlm_results: list of {"question": str, "answer": str}
    donut_data: dict from Donut model (may be raw extraction or parsed JSON)

    Returns dict with RECEIPT_KEYS; may include _error or _raw on failure.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()
    if provider == "minimax":
        return _normalize_via_minimax(layoutlm_results, donut_data)
    return _normalize_via_ollama(layoutlm_results, donut_data)
