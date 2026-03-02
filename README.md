# 🧾 Receipt OCR

> Extract structured data from receipt images using a **vision model** (Ollama API) — with a shared pipeline for both API and UI.

---

## Overview

**Receipt OCR** sends receipt images to an **Ollama vision model** and returns structured JSON:

- **Vision model** (OLLAMA_VISION_MODEL) — image → receipt fields in one step.

The same pipeline powers a **Flask REST API** and a **Gradio** web UI.

---

## ✨ Features

- **Dual interfaces**: REST API (`api.py`) and Gradio UI (`app.py`) using one pipeline.
- **Ollama vision**: [Ollama](https://ollama.ai) vision model (e.g. qwen3-vl, llava) for receipt extraction via API.
- **Structured output**: Fixed receipt schema: `store_name`, `shop_name`, `date`, `total_amount`, `tax_amount`, `gst_amount`, `sales_tax`, `received`, `payable`.

---

## Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Receipt     │ ──► │ Vision model     │ ──► │ Normalized      │
│ Image       │     │ (Ollama API)     │     │ receipt JSON    │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

---

## Prerequisites

- **Python** 3.10+
- **Ollama** (or compatible API): [Install Ollama](https://ollama.ai) and pull a vision model, e.g. `ollama pull qwen2-vl:7b` or use a cloud vision model via `OLLAMA_VISION_MODEL`

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
# Required: vision model for receipt extraction (e.g. qwen2-vl:7b, llava)
OLLAMA_VISION_MODEL=qwen2-vl:7b
```

**API behavior:**

- `API_MODE=async` (default) — `POST /api/process` returns **202** with `job_id`; processing runs in the background and results are sent to `CALLBACK_URL`.
- `API_MODE=sync` — `POST /api/process` blocks until done and returns **200** with receipt JSON (no callback).
- `INCLUDE_RAW` is ignored; responses contain only `receipt` (and `receipt_meta` if there was an error).
- `CALLBACK_URL` — URL to POST results to when a job completes (async mode only). Required for receiving results in async; see [Async API and callback](#async-api-and-callback) below.

---

## How to Use

### Option A — Gradio UI (easiest)

1. Set `OLLAMA_VISION_MODEL` in `.env` and ensure Ollama (or your API) is running with that vision model.
2. Run the app:

```bash
python app.py
```

3. Open the URL in your browser (e.g. http://127.0.0.1:7860), upload a receipt image, and click **Process**. You’ll see the receipt JSON from the vision model.

---

### Option B — Flask API

1. Start the API server:

```bash
python api.py
```

By default it runs at **http://0.0.0.0:5050**.

2. **Health check**

```bash
curl http://localhost:5050/health
```

3. **Process a receipt**

With `API_MODE=async` (default), the API returns immediately with a job ID; processing runs in the background and results are POSTed to `CALLBACK_URL`. With `API_MODE=sync`, the request blocks and the response is the receipt JSON directly.

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

**Example response (sync, 200 OK):** same receipt/raw structure as the callback success payload (see below).

---

#### Async API and callback

Set `CALLBACK_URL` in your `.env` (e.g. `CALLBACK_URL=https://your-server.com/receipt-callback`). When a job finishes, the API POSTs JSON to that URL.

**Success payload:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "receipt": {
    "store_name": "Coffee Shop",
    "shop_name": "Coffee Shop",
    "date": "2024-01-15",
    "total_amount": 12.50,
    "tax_amount": 1.25,
    "gst_amount": null,
    "sales_tax": null,
    "received": 15.00,
    "payable": 12.50
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
from pipeline import process_receipt_image

image = Image.open("receipt.jpg").convert("RGB")
result = process_receipt_image(image)

print(result["receipt"])           # Normalized receipt (RECEIPT_KEYS only)
print(result["receipt_meta"])      # None or {_error, _raw} if extraction failed
```

---

## Output schema

The merged **receipt** object uses these keys (values may be `null` if not found):

| Key           | Description                |
|---------------|----------------------------|
| `store_name`  | Store or business name     |
| `shop_name`   | Shop name                  |
| `date`        | Transaction date           |
| `total_amount`| Total amount               |
| `tax_amount`  | Tax amount                 |
| `gst_amount`  | GST amount                 |
| `sales_tax`   | Sales tax                  |
| `received`    | Amount received            |
| `payable`     | Amount payable             |

---

## Project structure

```
receiptOcr/
├── api.py           # Flask API (POST /api/process async → 202 + job_id; GET /health)
├── app.py           # Gradio UI
├── pipeline.py      # Shared pipeline: image → vision model → receipt JSON
├── llm_normalize.py # Vision extraction (Ollama API)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Production (API)

For production, run the Flask app with Gunicorn:

```bash
gunicorn -w 1 -b 0.0.0.0:5050 api:app
```

---

## License

Use and modify as you like. If you use this in a project, attribution is appreciated.
