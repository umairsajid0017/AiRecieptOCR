"""
Gradio UI for receipt processing. Uses shared pipeline.process_receipt_image().
"""
import json

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from pipeline import process_receipt_image


def run_ui(image):
    """Gradio handler: call shared pipeline and format outputs for UI."""
    if image is None:
        return "Please upload an image.", "", ""

    result = process_receipt_image(image)

    receipt_str = json.dumps(result["receipt"], indent=2, ensure_ascii=False)
    if result.get("receipt_meta") and "_error" in result["receipt_meta"]:
        receipt_str += "\n\n⚠️ " + result["receipt_meta"]["_error"]

    return receipt_str


with gr.Blocks(title="Receipt/Document Analysis") as demo:
    gr.Markdown("# 🧾 Receipt/Document Analysis")
    gr.Markdown(
        "Upload a receipt image. **Vision model** (Ollama API) → receipt JSON (same as API)."
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

    process_btn.click(
        fn=run_ui,
        inputs=[image_input],
        outputs=[output_receipt],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
