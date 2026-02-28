"""
Flask API: receipt image -> LayoutLM + Donut -> LLM merge -> fixed JSON receipt.
POST /api/process: multipart/form-data with `image` and optional `questions`.
"""
import json
import os

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from PIL import Image
import io

from models.layoutlm import LayoutLMQA
from models.donut import DonutExtractor
from llm_normalize import normalize_receipt, RECEIPT_KEYS

app = Flask(__name__)

# Lazy-initialized model wrappers
_layoutlm = None
_donut = None

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

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


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


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_layoutlm_answer(raw: str) -> str:
    """Extract answer text from LayoutLM output (strips ' (Confidence: 0.xx)')."""
    if not raw or not isinstance(raw, str):
        return raw or ""
    # Format: "answer\n\n(Confidence: 0.92)" or just "answer"
    if "\n\n(Confidence:" in raw:
        raw = raw.split("\n\n(Confidence:")[0].strip()
    return raw.strip()


def ensure_receipt_schema(receipt: dict) -> dict:
    """Ensure receipt has exactly RECEIPT_KEYS; remove _error/_raw for final output."""
    out = {}
    for key in RECEIPT_KEYS:
        out[key] = receipt.get(key) if key in receipt else None
    return out


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def _load_image_from_request():
    """Load PIL Image from request: uploaded file (image=@path), form field (image=path), or JSON body (image_path or image)."""
    # 1. Uploaded file (curl -F "image=@path" or multipart file)
    file = request.files.get("image") or request.files.get("file")
    if file and getattr(file, "filename", None) and file.filename.strip():
        if not allowed_file(file.filename):
            return None, "Invalid image type; use PNG or JPEG"
        try:
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image, None
        except Exception as e:
            return None, f"Invalid image: {e!s}"

    # 2. Form field: image=path or file=path (curl -F "image=C:\path" or -F "file=C:\path")
    path = (
        request.form.get("image")
        or request.form.get("file")
        or request.form.get("image_path")
    )
    if not path and request.form:
        # Any form value that looks like a path (Windows or Unix)
        for key in ("image", "file", "image_path", "path"):
            v = request.form.get(key)
            if v and isinstance(v, str) and (v.strip().startswith("/") or ":\\" in v or (len(v) >= 2 and v[1] == ":")):
                path = v
                break
        if not path:
            for _, v in request.form.items():
                if isinstance(v, str) and (":\\" in v or (v.startswith("/") and os.path.sep in v)):
                    path = v
                    break

    # 3. JSON body: {"image_path": "C:\\path"} or {"image": "C:\\path"}
    if not path and request.is_json:
        try:
            data = request.get_json(silent=True) or {}
            path = data.get("image_path") or data.get("image") or data.get("file")
        except Exception:
            pass

    if path and isinstance(path, str):
        path = path.strip().strip('"').replace("/", os.path.sep)
        if path and os.path.isfile(path):
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if ext not in ALLOWED_EXTENSIONS:
                return None, "Invalid image type; use PNG or JPEG"
            try:
                image = Image.open(path).convert("RGB")
                return image, None
            except Exception as e:
                return None, f"Invalid image: {e!s}"
        return None, f"File not found or not a file: {path!r}"
    return None, "Missing image: use image=@path (file upload), image=path (form), or JSON {\"image_path\": \"C:\\\\path\"}"


@app.route("/api/process", methods=["POST"])
def process():
    image, err = _load_image_from_request()
    if err:
        return jsonify({"error": err}), 400

    questions = DEFAULT_QUESTIONS
    if request.form.get("questions"):
        try:
            q = json.loads(request.form["questions"])
            if isinstance(q, list) and len(q) > 0:
                questions = [str(x) for x in q]
        except (json.JSONDecodeError, TypeError):
            pass

    layoutlm_results = []
    layoutlm_model = get_layoutlm()
    for q in questions:
        try:
            raw = layoutlm_model.process(image, q)
            answer = parse_layoutlm_answer(raw)
            layoutlm_results.append({"question": q, "answer": answer})
        except Exception as e:
            layoutlm_results.append({"question": q, "answer": f"[Error: {e!s}]"})

    donut_raw = None
    donut_data = {}
    try:
        donut_model = get_donut()
        donut_raw = donut_model.process(image)
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

    # If LLM returned an error key, still return 200 with receipt + raw
    receipt_clean = ensure_receipt_schema(receipt)
    has_error = "_error" in receipt or "_raw" in receipt

    include_raw = os.environ.get("INCLUDE_RAW", "true").strip().lower() in ("1", "true", "yes")
    response = {"receipt": receipt_clean}
    if include_raw:
        response["raw"] = {
            "layoutlm": layoutlm_results,
            "donut": donut_data if isinstance(donut_data, dict) else {"_raw": donut_raw},
        }
    if has_error:
        response["receipt_meta"] = {k: v for k, v in receipt.items() if k in ("_error", "_raw")}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
