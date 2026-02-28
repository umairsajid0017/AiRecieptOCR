"""
Test Ollama normalization using the same code as the real API (llm_normalize.normalize_receipt).
Run with: python test_ollama.py
Make sure .env is loaded (OLLAMA_MODEL) and Ollama is running (e.g. on 11434).
"""
import json
import os

from dotenv import load_dotenv
load_dotenv()

from llm_normalize import normalize_receipt

# Fake inputs like what the API would pass after LayoutLM + Donut
FAKE_LAYOUTLM = [
    {"question": "What is the store or business name?", "answer": "ABC Store"},
    {"question": "What is the shop name?", "answer": "ABC Store"},
    {"question": "What is the date on the receipt?", "answer": "2024-01-15"},
    {"question": "What is the total amount?", "answer": "42.50"},
    {"question": "What is the tax amount?", "answer": "3.50"},
    {"question": "What is the GST amount?", "answer": "3.50"},
    {"question": "What is the sales tax?", "answer": "3.50"},
    {"question": "What is the amount received?", "answer": "50.00"},
    {"question": "What is the amount payable?", "answer": "42.50"},
]
FAKE_DONUT = {
    "header": {"vendor": "ABC Store", "date": "2024-01-15"},
    "summary": {"total": "42.50", "tax": "3.50"},
}

if __name__ == "__main__":
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    print(f"OLLAMA_MODEL = {model!r}")
    print("Calling normalize_receipt (same as API)...")
    print()
    result = normalize_receipt(FAKE_LAYOUTLM, FAKE_DONUT)
    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if "_error" in result:
        print("\n--> Ollama test failed (check model name and 'ollama list')")
    else:
        print("\n--> Ollama test OK")
