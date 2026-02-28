"""
Gradio UI: same pipeline as API — image → LayoutLM (default questions) → Donut → LLM normalize → receipt JSON.
"""
import json
import os

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from models.layoutlm import LayoutLMQA
from models.donut import DonutExtractor
from llm_normalize import normalize_receipt, RECEIPT_KEYS

# Same default questions as API
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

layoutlm_model = LayoutLMQA()
donut_model = DonutExtractor()


def parse_layoutlm_answer(raw: str) -> str:
    """Extract answer text from LayoutLM output (strips ' (Confidence: 0.xx)')."""
    if not raw or not isinstance(raw, str):
        return raw or ""
    if "\n\n(Confidence:" in raw:
        raw = raw.split("\n\n(Confidence:")[0].strip()
    return raw.strip()


def ensure_receipt_schema(receipt: dict) -> dict:
    """Ensure receipt has exactly RECEIPT_KEYS for display."""
    out = {}
    for key in RECEIPT_KEYS:
        out[key] = receipt.get(key) if key in receipt else None
    return out


def run_pipeline(image):
    """Same flow as API: LayoutLM (all questions) → Donut → LLM normalize → receipt JSON."""
    if image is None:
        return "Please upload an image.", "", ""

    # 1. LayoutLM with default questions
    layoutlm_results = []
    for q in DEFAULT_QUESTIONS:
        try:
            raw = layoutlm_model.process(image, q)
            answer = parse_layoutlm_answer(raw)
            layoutlm_results.append({"question": q, "answer": answer})
        except Exception as e:
            layoutlm_results.append({"question": q, "answer": f"[Error: {e!s}]"})

    # 2. Donut
    try:
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

    # 3. LLM normalize (Ollama / Minimax)
    receipt = normalize_receipt(layoutlm_results, donut_data)
    receipt_clean = ensure_receipt_schema(receipt)

    # Build display strings
    receipt_str = json.dumps(receipt_clean, indent=2, ensure_ascii=False)
    if "_error" in receipt:
        receipt_str = receipt_str + "\n\n⚠️ " + receipt.get("_error", "")

    raw_layoutlm_str = json.dumps(layoutlm_results, indent=2, ensure_ascii=False)
    raw_donut_str = json.dumps(donut_data, indent=2, ensure_ascii=False) if isinstance(donut_data, dict) else str(donut_data)

    return receipt_str, raw_layoutlm_str, raw_donut_str


with gr.Blocks(title="Receipt/Document Analysis") as demo:
    gr.Markdown("# 🧾 Receipt/Document Analysis")
    gr.Markdown(
        "Upload a receipt image. Same pipeline as the API: **LayoutLM** (fixed questions) → **Donut** → **Ollama** merge → final receipt JSON."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Document/Receipt")
            process_btn = gr.Button("🚀 Process", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_receipt = gr.Textbox(
                label="Receipt (merged JSON)",
                lines=14,
                interactive=False,
            )
            with gr.Accordion("Raw outputs", open=False):
                output_layoutlm = gr.Textbox(
                    label="LayoutLM Q&A",
                    lines=8,
                    interactive=False,
                )
                output_donut = gr.Textbox(
                    label="Donut extraction",
                    lines=8,
                    interactive=False,
                )

    process_btn.click(
        fn=run_pipeline,
        inputs=[image_input],
        outputs=[output_receipt, output_layoutlm, output_donut],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
