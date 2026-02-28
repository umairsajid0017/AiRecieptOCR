# 🧾 Receipt OCR

> Extract structured data from receipt images using **LayoutLM**, **Donut**, and an **LLM** — with a shared pipeline for both API and UI.

---

## Overview

**Receipt OCR** runs a three-stage pipeline on receipt/document images:

1. **LayoutLM** (document question-answering) — answers fixed questions (store name, date, total, tax, etc.).
2. **Donut** (document understanding) — full document extraction into structured fields.
3. **LLM normalization** — merges both outputs into a single, consistent JSON schema (Ollama or Minimax).

The same pipeline powers a **Flask REST API** and a **Gradio** web UI.

---

## ✨ Features

- **Dual interfaces**: REST API (`api.py`) and Gradio UI (`app.py`) using one pipeline.
- **Flexible LLM backend**: Local [Ollama](https://ollama.ai) (default) or cloud [Minimax](https://www.minimax.io) for merging.
- **Structured output**: Fixed receipt schema: `store_name`, `shop_name`, `date`, `total_amount`, `tax_amount`, `gst_amount`, `sales_tax`, `received`, `payable`.
- **Optional raw outputs**: API can return LayoutLM Q&A and Donut extraction alongside the merged receipt.

---

## Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Receipt     │ ──► │ LayoutLM    │ ──► │ LLM (Ollama /    │ ──► │ Normalized      │
│ Image       │     │ (Q&A)       │     │ Minimax) merge   │     │ receipt JSON    │
└─────────────┘     └─────────────┘     └──────────────────┘     └─────────────────┘
       │                    │
       └────────────────────┼────────────────────┐
                            ▼                    ▼
                     ┌─────────────┐      Donut (full extraction)
                     │ Donut       │
                     └─────────────┘
```

---

## Prerequisites

- **Python** 3.10+
- **Ollama** (for local LLM merge): [Install Ollama](https://ollama.ai) and pull a model, e.g. `ollama pull llama3.2`
- **Optional**: CUDA-capable GPU for faster LayoutLM/Donut inference

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

### 4. Environment configuration

Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

**Minimal `.env` (Ollama, default):**

```env
# Optional: defaults to llama3.2 if not set
OLLAMA_MODEL=llama3.2
```

**Using Minimax instead of Ollama:**

```env
LLM_PROVIDER=minimax
MINIMAX_API_KEY=your_api_key_here
# Optional
MINIMAX_BASE_URL=https://api.minimax.io
MINIMAX_MODEL=MiniMax-M2.5
```

**API behavior:**

- `API_MODE=async` (default) — `POST /api/process` returns **202** with `job_id`; processing runs in the background and results are sent to `CALLBACK_URL`.
- `API_MODE=sync` — `POST /api/process` blocks until done and returns **200** with receipt JSON (no callback).
- `INCLUDE_RAW=true` (default) — callback payload (async) or response (sync) includes `raw.layoutlm` and `raw.donut`.
- `INCLUDE_RAW=false` — callback/response contains only `receipt` (and `receipt_meta` if there was an LLM error).
- `CALLBACK_URL` — URL to POST results to when a job completes (async mode only). Required for receiving results in async; see [Async API and callback](#async-api-and-callback) below.

---

## How to Use

### Option A — Gradio UI (easiest)

1. Start Ollama (if using local LLM):  
   `ollama serve` and `ollama pull llama3.2` (or your chosen model).
2. Run the app:

```bash
python app.py
```

3. Open the URL in your browser (e.g. http://127.0.0.1:7860), upload a receipt image, and click **Process**. You’ll see the merged receipt JSON and optional raw LayoutLM/Donut outputs.

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

**Optional custom questions (JSON array in form):**

```bash
curl -X POST http://localhost:5050/api/process \
  -F "image=@receipt.jpg" \
  -F 'questions=["What is the store name?","What is the total?"]'
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
  },
  "raw": {
    "layoutlm": [{"question": "...", "answer": "..."}],
    "donut": { ... }
  }
}
```

If `INCLUDE_RAW=false`, the payload omits `raw`. If the LLM step had issues, `receipt_meta` may be present with `_error` or `_raw`.

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
print(result["layoutlm_results"])  # LayoutLM Q&A list
print(result["donut_data"])        # Donut extraction dict
print(result["receipt_meta"])      # None or {_error, _raw} if LLM failed
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
├── pipeline.py      # Shared pipeline: LayoutLM → Donut → LLM → receipt JSON
├── llm_normalize.py # LLM merge (Ollama / Minimax)
├── models/
│   ├── layoutlm.py  # LayoutLM document Q&A
│   └── donut.py     # Donut document extraction
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

Use `-w 1` if your LayoutLM/Donut models are loaded in process and you want to avoid multiple heavy workers.

---

## License

Use and modify as you like. If you use this in a project, attribution is appreciated.
