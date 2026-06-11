"""
Test vision-based receipt extraction (same code path as the API).
Run with: python test_ai_vision.py <path-to-receipt-image>
Requires AI_TASK_RECEIPT_VISION_MODEL (or legacy OLLAMA_VISION_MODEL) in .env.
"""
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from llm_normalize import extract_receipt_from_image


def main():
    model = (
        os.environ.get("AI_TASK_RECEIPT_VISION_MODEL", "").strip()
        or os.environ.get("OLLAMA_VISION_MODEL", "").strip()
    )
    provider = os.environ.get("AI_TASK_RECEIPT_VISION_PROVIDER", "ollama").strip()
    if not model:
        print(
            "AI_TASK_RECEIPT_VISION_MODEL is not set in .env. "
            "Set it to a vision model (e.g. ministral-3:8b-cloud, llava, gemini-2.0-flash)."
        )
        sys.exit(1)
    print(f"AI_TASK_RECEIPT_VISION_PROVIDER = {provider!r}")
    print(f"AI_TASK_RECEIPT_VISION_MODEL = {model!r}")

    image_path = (sys.argv[1:2] or [""])[0].strip()
    if not image_path or not os.path.isfile(image_path):
        print("Usage: python test_ai_vision.py <path-to-receipt-image>")
        sys.exit(1)

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    print("Calling extract_receipt_from_image (vision API)...")
    result = extract_receipt_from_image(image)
    print()
    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if "_error" in result:
        print("\n--> Vision extraction failed (check provider, model, and credentials in .env)")
        sys.exit(1)
    print("\n--> Vision extraction OK")


if __name__ == "__main__":
    main()
