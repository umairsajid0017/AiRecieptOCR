import gradio as gr
from models.layoutlm import LayoutLMQA
from models.donut import DonutExtractor

# Initialize model wrappers (models themselves are loaded lazily on first use)
layoutlm_model = LayoutLMQA()
donut_model = DonutExtractor()


def run_all_models(image, question):
    """Run both models sequentially, yielding partial results as each completes."""
    if image is None:
        yield "", "Please upload an image."
        return

    result_1 = ""
    result_2 = ""

    # ── Model 1: LayoutLM (needs question) ──
    try:
        if not question or not question.strip():
            result_1 = "⚠️ Skipped — no question provided."
        else:
            result_1 = layoutlm_model.process(image, question)
    except Exception as e:
        result_1 = f"❌ Error: {str(e)}"
    yield result_1, result_2

    # ── Model 2: Donut Invoice/Receipt Extractor (no question needed) ──
    try:
        result_2 = donut_model.process(image)
    except Exception as e:
        result_2 = f"❌ Error: {str(e)}"
    yield result_1, result_2


with gr.Blocks(title="Receipt/Document Analysis") as demo:
    gr.Markdown("# 🧾 Receipt/Document Analysis")
    gr.Markdown(
        "Upload an image and optionally ask a question. "
        "LayoutLM (document QA) uses the question; Donut extracts invoice/receipt fields without one."
    )

    with gr.Row():
        # ── Left column: inputs ──
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Document/Receipt")
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g., What is the total amount?",
                info="Used by LayoutLM document QA. Donut extractor ignores this.",
            )
            process_btn = gr.Button("🚀 Process Both Models", variant="primary", size="lg")

        # ── Right column: two result panels ──
        with gr.Column(scale=1):
            output_layoutlm = gr.Textbox(
                label="① impira/layoutlm-document-qa",
                lines=6,
                interactive=False,
            )
            output_donut = gr.Textbox(
                label="② mychen76/invoice-and-receipts_donut_v1",
                lines=6,
                interactive=False,
            )

    process_btn.click(
        fn=run_all_models,
        inputs=[image_input, question_input],
        outputs=[output_layoutlm, output_donut],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
