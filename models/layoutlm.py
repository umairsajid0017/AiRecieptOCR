import time
from datetime import datetime
import torch
from transformers import pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LayoutLMQA:
    def __init__(self):
        self.model_name = "impira/layoutlm-document-qa"
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            print(f"Loading {self.model_name} on {DEVICE.upper()}... (This might take a while on the first run)")
            self.pipe = pipeline("document-question-answering", model=self.model_name, device=DEVICE)
            print(f"Model {self.model_name} loaded successfully on {DEVICE.upper()}!")

    def process(self, image, question):
        if not question or not question.strip():
            return "Please provide a question for the Question Answering model."
        
        self.load_model()
        
        start_time = time.time()
        start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"[LAYOUTLM] Inference START: {start_dt}")
        print(f"[LAYOUTLM] Device: {DEVICE.upper()}")
        print(f"[LAYOUTLM] Question: {question}")
        print(f"{'='*60}")
        
        result = self.pipe(image=image, question=question)
        
        end_time = time.time()
        end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"[LAYOUTLM] Inference END:   {end_dt}")
        print(f"[LAYOUTLM] Elapsed time:    {elapsed:.2f} seconds")
        print(f"{'='*60}\n")
        
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get('answer', str(result))
            score = result[0].get('score', 0)
            return f"{answer}\n\n(Confidence: {score:.2f})"
        elif isinstance(result, dict):
            answer = result.get('answer', str(result))
            score = result.get('score', 0)
            return f"{answer}\n\n(Confidence: {score:.2f})"
            
        return str(result)

