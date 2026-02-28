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

- `INCLUDE_RAW=true` (default) — response includes `raw.layoutlm` and `raw.donut`.
- `INCLUDE_RAW=false` — response contains only `receipt` (and `receipt_meta` if there was an LLM error).

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

By default it runs at **http://0.0.0.0:5000**.

2. **Health check**

```bash
curl http://localhost:5000/health
```

3. **Process a receipt**

**Upload a file (multipart):**

```bash
curl -X POST http://localhost:5000/api/process -F "image=@/path/to/receipt.jpg"
```

**JSON with server file path:**

```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"image_path": "C:\\path\\to\\receipt.png"}'
```

**Optional custom questions (JSON array in form):**

```bash
curl -X POST http://localhost:5000/api/process \
  -F "image=@receipt.jpg" \
  -F 'questions=["What is the store name?","What is the total?"]'
```

**Example response:**

```json
{
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

If the LLM step fails, you may also see `receipt_meta` with `_error` or `_raw`.

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
├── api.py           # Flask API (POST /api/process, GET /health)
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
gunicorn -w 1 -b 0.0.0.0:5000 api:app
```

Use `-w 1` if your LayoutLM/Donut models are loaded in process and you want to avoid multiple heavy workers.

---

## License

Use and modify as you like. If you use this in a project, attribution is appreciated.
