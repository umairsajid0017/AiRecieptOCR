"""
Vision-based receipt extraction via Ollama API. Image → vision model → receipt JSON.
"""
import json
import os
import re
import tempfile

RECEIPT_KEYS = [
    "shop_name",
    "date",
    "total_amount",
    "tax_amount",
    "tax_percentage",
    "category",
    "vendor_tax_id",
    "invoice_number",
    "reference",
    "vendor_address",
    "line_items",
    "payment_method",
    "card_last_4",
    "currency_code",
    "exchange_rate",
    "net_amount",
    "document_type",
]

def get_vision_prompt(categories=None):
    """Build the vision prompt. If categories provided, instruct LLM to pick one."""
    category_rule = "auto-detect from contents (e.g. Food, Travel, Shopping, Supplies, Utilities)."
    if categories and isinstance(categories, list) and len(categories) > 0:
        # Extract names if they are objects, otherwise assume they are strings
        names = []
        for c in categories:
            if isinstance(c, dict) and "name" in c:
                names.append(c["name"])
            elif isinstance(c, str):
                names.append(c)
        
        if names:
            category_rule = f"MUST be one of these exact values: {', '.join(names)}."

    return f"""Look at this receipt OR invoice image and extract data.
Return ONLY a JSON object with exactly these keys (use null for missing values):
shop_name, date, total_amount, tax_amount, tax_percentage, category, vendor_tax_id, invoice_number, reference, vendor_address, line_items, payment_method, card_last_4, currency_code, exchange_rate, net_amount, document_type.

Field rules:
- shop_name: for receipts use the store name; for invoices use the vendor/supplier/company name shown on the document header.
- date: ISO format YYYY-MM-DD when possible.
- total_amount, tax_amount, tax_percentage, exchange_rate, net_amount: numeric.
- currency_code: 3-letter ISO code like GBP, EUR, USD.
- payment_method: one of CARD, CASH, ONLINE.
- card_last_4: string containing exactly 4 digits when available.
- line_items: array of objects with keys {{description, quantity, unit_price, total, tax_amount}}. Use empty array [] if nothing can be extracted.
- document_type: one of INVOICE or RECEIPT.
- category: {category_rule}

No markdown, no explanation, no extra keys, only JSON."""


def _to_float(value):
    """Best-effort numeric coercion for amount/confidence fields."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        # Keep digits, sign, decimal separators; normalize commas in numbers.
        cleaned = cleaned.replace(",", "")
        cleaned = re.sub(r"[^0-9.\-+]", "", cleaned)
        if cleaned in {"", "-", "+", ".", "-.", "+."}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _normalize_payment_method(value):
    """Normalize payment method to one of CARD/CASH/ONLINE."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    raw = value.strip().upper().replace(" ", "_").replace("-", "_")
    if not raw:
        return None
    if raw in {"CARD", "CASH", "ONLINE"}:
        return raw
    aliases = {
        "CREDIT_CARD": "CARD",
        "DEBIT_CARD": "CARD",
        "CREDIT": "CARD",
        "DEBIT": "CARD",
        "VISA": "CARD",
        "MASTERCARD": "CARD",
        "AMEX": "CARD",
        "CHEQUE": "ONLINE",
        "CHECK": "ONLINE",
        "DIRECT_DEBIT": "ONLINE",
        "STANDING_ORDER": "ONLINE",
        "NET_30": "ONLINE",
        "NET_60": "ONLINE",
        "NET_90": "ONLINE",
        "UPI": "ONLINE",
        "BANK_TRANSFER": "ONLINE",
        "TRANSFER": "ONLINE",
        "BANK": "ONLINE",
        "WIRE": "ONLINE",
        "ONLINE_PAYMENT": "ONLINE",
        "E_WALLET": "ONLINE",
        "WALLET": "ONLINE",
    }
    return aliases.get(raw)


def _normalize_currency_code(value):
    """Normalize currency to 3-letter uppercase ISO-like code."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    code = re.sub(r"[^A-Za-z]", "", value).upper()
    return code[:3] if code else None


def _normalize_last_4(value):
    """Extract the last 4 digits from card details if present."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    digits = "".join(re.findall(r"\d", value))
    if len(digits) < 4:
        return None
    return digits[-4:]


def _normalize_confidence_score(value):
    """Clamp confidence scores to [0, 1]."""
    number = _to_float(value)
    if number is None:
        return None
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def _normalize_document_type(value):
    """Normalize document type to INVOICE or RECEIPT."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    raw = value.strip().upper().replace("-", "_").replace(" ", "_")
    if not raw:
        return None
    if raw in {"INVOICE", "RECEIPT"}:
        return raw
    aliases = {
        "TAX_INVOICE": "INVOICE",
        "PROFORMA_INVOICE": "INVOICE",
        "PROFORMA": "INVOICE",
        "CREDIT_NOTE": "INVOICE",
        "DEBIT_NOTE": "INVOICE",
        "PURCHASE_ORDER": "INVOICE",
        "STATEMENT": "INVOICE",
        "INVO": "INVOICE",
        "BILL": "INVOICE",
        "SIMPLIFIED_RECEIPT": "RECEIPT",
        "SALES_RECEIPT": "RECEIPT",
        "TILL_RECEIPT": "RECEIPT",
        "SLIP": "RECEIPT",
    }
    return aliases.get(raw)


def _normalize_line_items(value):
    """Normalize line items into a stable list schema."""
    if not isinstance(value, list):
        return []
    normalized_items = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized_items.append(
            {
                "description": (item.get("description") or item.get("desc")
                                or item.get("item") or item.get("name")
                                or item.get("product") or None),
                "quantity": _to_float(item.get("quantity") or item.get("qty")),
                "unit_price": _to_float(item.get("unit_price") or item.get("price") or item.get("rate")),
                "total": _to_float(item.get("total") or item.get("amount") or item.get("line_total")),
                "tax_amount": _to_float(item.get("tax_amount") or item.get("tax") or item.get("vat")),
            }
        )
    return normalized_items


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
    # Normalize to exact schema and expected value types
    receipt = {}
    for key in RECEIPT_KEYS:
        receipt[key] = data.get(key) if data.get(key) is not None else None

    for numeric_key in ("total_amount", "tax_amount", "tax_percentage", "exchange_rate", "net_amount"):
        receipt[numeric_key] = _to_float(receipt.get(numeric_key))
    if receipt.get("net_amount") is None and receipt.get("total_amount") is not None and receipt.get("tax_amount") is not None:
        receipt["net_amount"] = receipt["total_amount"] - receipt["tax_amount"]
    receipt["payment_method"] = _normalize_payment_method(receipt.get("payment_method"))
    receipt["currency_code"] = _normalize_currency_code(receipt.get("currency_code"))
    receipt["card_last_4"] = _normalize_last_4(receipt.get("card_last_4"))
    receipt["line_items"] = _normalize_line_items(receipt.get("line_items"))

    receipt["document_type"] = _normalize_document_type(receipt.get("document_type"))

    if isinstance(receipt.get("vendor_tax_id"), str):
        receipt["vendor_tax_id"] = receipt["vendor_tax_id"].strip() or None
    if isinstance(receipt.get("invoice_number"), str):
        receipt["invoice_number"] = receipt["invoice_number"].strip() or None
    if isinstance(receipt.get("reference"), str):
        receipt["reference"] = receipt["reference"].strip() or None
    if isinstance(receipt.get("vendor_address"), str):
        receipt["vendor_address"] = receipt["vendor_address"].strip() or None
    if isinstance(receipt.get("shop_name"), str):
        receipt["shop_name"] = receipt["shop_name"].strip() or None
    if isinstance(receipt.get("category"), str):
        receipt["category"] = receipt["category"].strip() or None
    return receipt


# Max dimension for vision uploads (Ollama has request body limits). Env: OLLAMA_VISION_MAX_PIXELS (default 2048).
_VISION_MAX_PIXELS = int(os.environ.get("OLLAMA_VISION_MAX_PIXELS", "2048"))
_VISION_JPEG_QUALITY = int(os.environ.get("OLLAMA_VISION_JPEG_QUALITY", "88"))


def _get_vision_model() -> str:
    """Vision model for receipt extraction. Requires OLLAMA_VISION_MODEL to be set."""
    return os.environ.get("OLLAMA_VISION_MODEL", "").strip()


def _prepare_image_for_vision(image):
    """
    Resize image if needed and save as JPEG to stay under Ollama request body limits.
    image: PIL Image (RGB). Returns path to temp .jpg; caller must delete when done.
    """
    from PIL import Image as PILImage
    w, h = image.size
    max_p = _VISION_MAX_PIXELS
    if w > max_p or h > max_p:
        ratio = min(max_p / w, max_p / h)
        nw, nh = int(w * ratio), int(h * ratio)
        image = image.resize((nw, nh), getattr(PILImage, "Resampling", PILImage).LANCZOS)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(path, "JPEG", quality=_VISION_JPEG_QUALITY, optimize=True)
    return path


def _extract_via_ollama_vision(image, categories=None) -> dict:
    """Send receipt image to Ollama vision model; return receipt dict (RECEIPT_KEYS or _error/_raw)."""
    from ollama import chat, ResponseError

    model = _get_vision_model()
    if not model:
        return {"_error": "OLLAMA_VISION_MODEL is not set. Set it in .env (e.g. qwen3-vl:8b, llava)."}
    path = None
    try:
        path = _prepare_image_for_vision(image)
        try:
            response = chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You extract structured data from receipts and invoices. You must respond with only valid JSON, nothing else."},
                    {"role": "user", "content": get_vision_prompt(categories), "images": [path]},
                ],
                format="json",
            )
        except ResponseError as e:
            msg = str(e).strip()
            if "404" in msg or "not found" in msg.lower():
                return {"_error": f"Ollama vision model {model!r} not found. Set OLLAMA_VISION_MODEL in .env (e.g. llava, qwen3-vl:8b)."}
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


def extract_receipt_from_image(image, categories=None) -> dict:
    """
    Send receipt image to Ollama vision model; return receipt dict with RECEIPT_KEYS.
    May include _error or _raw on failure. image: PIL Image (RGB).
    """
    return _extract_via_ollama_vision(image, categories=categories)
