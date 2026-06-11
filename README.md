# 🧾 Receipt/Invoice OCR

> Extract structured data from **receipt or invoice** images using a **vision model** (Ollama API) — with a shared pipeline for both API and UI.

---

## Overview

**Receipt OCR** sends **receipt or invoice** images to an **Ollama vision model** and returns structured JSON:

- **Vision model** (`AI_TASK_RECEIPT_VISION_MODEL`) — image → document fields in one step.

The same pipeline powers a **Flask REST API** and a **Gradio** web UI.

---

## ✨ Features

- **Dual interfaces**: REST API (`src/api_server.py`) and Gradio UI (`src/gradio_ui.py`) using one pipeline.
- **Modular Design**: Separated API into Routes, Controllers, Worker, and Utilities for better maintainability.
- **Multi-provider vision**: Ollama, Groq, OpenRouter, or Gemini — selected via `AI_TASK_RECEIPT_VISION_PROVIDER` in `.env`.
- **Structured output**: Fixed receipt schema with accounting-ready metadata, including VAT, invoice reference, payment details, line items, currency, and confidence scores.

---

## Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Document    │ ──► │ Vision model     │ ──► │ Normalized      │
│ Image       │     │ (Ollama API)     │     │ JSON            │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

---

## Prerequisites

- **Python** 3.10+
- **AI provider**: Configure one of Ollama, Groq, OpenRouter, or Gemini in `.env` (see `.env.example`).

---

## Installation

### 1. Clone and enter the repo

```bash
git clone https://github.com/YOUR_USERNAME/receiptOcr.git
cd receiptOcr
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: Automated setup & systemd service

If you'd like an automated way to install system deps, create the virtual environment, and run the app as a systemd service, use the included `setup_ai_ocr.sh` script. The script will:

- Install OS packages (when a supported package manager is detected).
- Create a Python virtual environment at `myenv` in the project root.
- Install `requirements.txt` into the virtualenv.
- Write a systemd unit at `/etc/systemd/system/ai_ocr.service` using the invoking sudo user (or the current user) and the project path.
- Enable and start the service, serving the Flask API on port 5050.

Run the script from the project root:

```bash
sudo ./setup_ai_ocr.sh
```

Service management (common commands):

```bash
# Check status
sudo systemctl status ai_ocr.service

# Follow logs
sudo journalctl -u ai_ocr.service -f

# Restart after changes
sudo systemctl restart ai_ocr.service

# Stop the service
sudo systemctl stop ai_ocr.service
```

If you need to change the service user, port, or other options, edit `/etc/systemd/system/ai_ocr.service`, then run:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ai_ocr.service
```

Note: The script creates a virtualenv at `myenv` and the service runs the app with the venv Python and `gunicorn` on port 5050 by default.

### 5. Environment configuration

Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

**Minimal `.env`:**

```env
AI_TASK_RECEIPT_VISION_PROVIDER=ollama
AI_TASK_RECEIPT_VISION_MODEL=ministral-3:8b-cloud
OLLAMA_URL=http://localhost:11434/api/generate
```

Supported providers (set `AI_TASK_RECEIPT_VISION_PROVIDER`): `ollama`, `groq`, `gemini`, `openrouter`. See `.env.example` for all credential variables.

**API behavior:**

- `API_MODE=async` (default) — `POST /api/process` returns **202** with `job_id`; processing runs in the background and results are sent to `CALLBACK_URL`.
- `API_MODE=sync` — `POST /api/process` blocks until done and returns **200** with receipt JSON (no callback).
- `INCLUDE_RAW` is ignored; responses contain only `receipt` (and `receipt_meta` if there was an error).
- `CALLBACK_URL` — URL to POST results to when a job completes (async mode only). Required for receiving results in async; see [Async API and callback](#async-api-and-callback) below.

---

## How to Use

### Option A — Gradio UI (easiest)

1. Set `AI_TASK_RECEIPT_VISION_PROVIDER` and `AI_TASK_RECEIPT_VISION_MODEL` in `.env` (see `.env.example`).
2. Run the app:

```bash
python src/gradio_ui.py
```

3. Open the URL in your browser (e.g. http://127.0.0.1:7860), upload a receipt or invoice image, and click **Process**. You’ll see the structured JSON from the vision model.

---

### Option B — Flask API

1. Start the API server:

```bash
python src/api_server.py
```

By default it runs at **http://0.0.0.0:5050**.

2. **Health check**

```bash
curl http://localhost:5050/health
```

3. **Process a receipt or invoice**

With `API_MODE=async` (default), the API returns immediately with a job ID; processing runs in the background and results are POSTed to `CALLBACK_URL`. With `API_MODE=sync`, the request blocks and the response is the structured JSON directly.

**Upload a file (multipart):**

```bash
curl -X POST http://localhost:5050/api/process -F "image=@/path/to/receipt.jpg"
```

**JSON with server file path:**

```bash
curl -X POST http://localhost:5050/api/process \
  -H "Content-Type: application/json" \
  -d '{"image_path": "C:\\path\\to\\receipt.png"}'
```

**Example response (async, 202 Accepted):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Results are sent to your callback URL when processing finishes. See [Async API and callback](#async-api-and-callback) for the callback payload format.

**Example response (sync, 200 OK):** same structure as the callback success payload (see below), including a top-level `document_type`.

---

#### Async API and callback

Set `CALLBACK_URL` in your `.env` (e.g. `CALLBACK_URL=https://your-server.com/receipt-callback`). When a job finishes, the API POSTs JSON to that URL.

**Success payload:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "document_type": "RECEIPT",
  "receipt": {
    "shop_name": "Coffee Shop",
    "date": "2024-01-15",
    "total_amount": 12.50,
    "tax_amount": 1.25,
    "tax_percentage": 10.00,
    "category": "Food",
    "vendor_tax_id": "GB123456789",
    "invoice_number": "INV-99821",
    "reference": "RCP-001",
    "vendor_address": "235 Regent St., London W1B 2EL",
    "line_items": [
      {
        "description": "Coffee",
        "quantity": 1.0,
        "unit_price": 11.25,
        "total": 11.25,
        "tax_amount": 1.25
      }
    ],
    "payment_method": "CARD",
    "card_last_4": "4242",
    "currency_code": "GBP",
    "exchange_rate": 1.0,
    "net_amount": 11.25,
    "confidence_scores": {
      "total_amount": 0.99,
      "date": 0.95
    },
    "document_type": "RECEIPT",
    "document_type_confidence": 0.97
  }
}
```

If `INCLUDE_RAW=false`, the payload omits `raw`. If extraction had issues, `receipt_meta` may be present with `_error` or `_raw`.

**Failure payload:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": "Error message"
}
```

Jobs are processed **one at a time** by a single background worker. If `CALLBACK_URL` is not set, the worker still runs the pipeline but does not send any HTTP callback (it only logs).

---

### Option C — Use the pipeline in your own code

```python
from PIL import Image
from src.pipeline import process_receipt_image

image = Image.open("receipt.jpg").convert("RGB")
result = process_receipt_image(image)

print(result["receipt"])           # Normalized receipt (RECEIPT_KEYS only)
print(result["receipt_meta"])      # None or {_error, _raw} if extraction failed
```

---

## Output schema

The merged **receipt** object uses these keys (values may be `null` if not found). This schema is used for **both receipts and invoices**:

| Key           | Description                |
|---------------|----------------------------|
| `shop_name`   | Shop name (or vendor name on invoices) |
| `date`        | Transaction date           |
| `total_amount`| Total amount               |
| `tax_amount`  | Tax amount                 |
| `tax_percentage` | Tax percentage          |
| `category`    | Auto-detected category     |
| `vendor_tax_id` | VAT/tax registration ID |
| `invoice_number` | Invoice number          |
| `reference`   | Receipt/reference ID       |
| `vendor_address` | Vendor address          |
| `line_items`  | Itemized lines             |
| `payment_method` | CARD/CASH/ONLINE       |
| `card_last_4` | Last 4 digits of card      |
| `currency_code` | ISO 3-letter currency    |
| `exchange_rate` | Currency conversion rate |
| `net_amount`  | Net amount before tax      |
| `confidence_scores` | Field confidence map |
| `document_type` | INVOICE or RECEIPT       |
| `document_type_confidence` | Receipt/invoice confidence |

---

## Project structure

```
AiRecieptOCR/
├── src/
│   ├── api/             # Modular API package
│   │   ├── __init__.py  # App factory
│   │   ├── routes.py    # Route definitions
│   │   ├── controllers.py # Request handlers
│   │   ├── worker.py    # Background job queue
│   │   └── utils.py     # Image/Callback helpers
│   ├── api_server.py    # Flask API Entry Point
│   ├── gradio_ui.py     # Gradio UI
│   ├── pipeline.py      # Shared pipeline
│   ├── llm_normalize.py # Vision extraction
│   └── models/          # Model data
├── requirements.txt
├── .env.example
└── README.md
```

---

## Production (API)

For production, run the Flask app with Gunicorn:

```bash
gunicorn -w 1 -b 0.0.0.0:5050 --chdir src api_server:app
```

---

## License

Use and modify as you like. If you use this in a project, attribution is appreciated.
