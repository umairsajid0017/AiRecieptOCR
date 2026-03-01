"""
Test vision-based receipt extraction (same code path as the API).
Run with: python test_ollama.py
Requires OLLAMA_VISION_MODEL in .env and an image path as first argument, or a small test image.
"""
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from llm_normalize import extract_receipt_from_image

def main():
    model = os.environ.get("OLLAMA_VISION_MODEL", "").strip()
    if not model:
        print("OLLAMA_VISION_MODEL is not set in .env. Set it to a vision model (e.g. qwen3-vl:8b, llava).")
        sys.exit(1)
    print(f"OLLAMA_VISION_MODEL = {model!r}")

    image_path = (sys.argv[1:2] or [""])[0].strip()
    if not image_path or not os.path.isfile(image_path):
        print("Usage: python test_ollama.py <path-to-receipt-image>")
        print("  Or set image path as first argument.")
        sys.exit(1)

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    print("Calling extract_receipt_from_image (vision API)...")
    result = extract_receipt_from_image(image)
    print()
    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if "_error" in result:
        print("\n--> Vision extraction failed (check model name and Ollama/API)")
        sys.exit(1)
    print("\n--> Vision extraction OK")

if __name__ == "__main__":
    main()
