import gradio as gr
from models.layoutlm import LayoutLMQA
from models.donut import DonutExtractor

# Initialize model wrappers (models themselves are loaded lazily on first use)
layoutlm_model = LayoutLMQA()
donut_model = DonutExtractor()

def process_document(image, model_choice, question):
    if image is None:
        return "Please upload an image."
    
    try:
        if model_choice == "impira/layoutlm-document-qa":
            return layoutlm_model.process(image, question)
        elif model_choice == "mychen76/invoice-and-receipts_donut_v1":
            return donut_model.process(image)
            
    except Exception as e:
        return f"Error processing document: {str(e)}\n\n(Check terminal for more details)"

def update_ui(model_choice):
    # Hide the question textbox if Donut is selected since it just extracts the whole document to JSON
    if model_choice == "impira/layoutlm-document-qa":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

with gr.Blocks(title="Receipt/Document Analysis") as demo:
    gr.Markdown("# Receipt/Document Analysis Multi-Model")
    gr.Markdown("Upload an image of a document or receipt. Choose a model to either ask targeted questions or extract the complete JSON structure.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Document/Receipt")
            model_dropdown = gr.Dropdown(
                choices=["impira/layoutlm-document-qa", "mychen76/invoice-and-receipts_donut_v1"],
                value="impira/layoutlm-document-qa",
                label="Select Model Pipeline"
            )
            question_input = gr.Textbox(
                label="Question (Only needed for LayoutLM model)", 
                placeholder="e.g., What is the total amount?"
            )
            process_btn = gr.Button("Process Document", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Result", lines=15)
            
    model_dropdown.change(fn=update_ui, inputs=model_dropdown, outputs=question_input)
    
    process_btn.click(
        fn=process_document,
        inputs=[image_input, model_dropdown, question_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
