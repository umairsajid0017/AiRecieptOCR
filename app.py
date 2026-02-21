import gradio as gr
from transformers import pipeline

# Initialize the document-question-answering pipeline
print("Loading model... (This might take a minute on the first run)")
pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
print("Model loaded successfully!")

def process_document(image, question):
    if image is None:
        return "Please upload an image."
    if not question or not question.strip():
        return "Please provide a question."
    
    try:
        # The pipeline accepts an image and a question
        result = pipe(image=image, question=question)
        
        # Extract the answer from the result
        if isinstance(result, list) and len(result) > 0:
            # Usually returns a list of dictionaries like [{'score': 0.99, 'answer': 'value', 'start': 1, 'end': 2}]
            answer = result[0].get('answer', str(result))
            score = result[0].get('score', 0)
            return f"{answer}\n\n(Confidence: {score:.2f})"
        elif isinstance(result, dict):
            answer = result.get('answer', str(result))
            score = result.get('score', 0)
            return f"{answer}\n\n(Confidence: {score:.2f})"
            
        return str(result)
    except Exception as e:
        return f"Error processing document: {str(e)}\n\nNote: If you see an error about Tesseract, please make sure Tesseract OCR is installed on your system and added to your PATH."

# Create the Gradio interface
demo = gr.Interface(
    fn=process_document,
    inputs=[
        gr.Image(type="pil", label="Upload Document/Receipt"),
        gr.Textbox(label="Question", placeholder="e.g., What is the total amount?")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Receipt/Document Question Answering",
    description="Upload an image of a document or receipt and ask a question about its contents using the `impira/layoutlm-document-qa` model.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
