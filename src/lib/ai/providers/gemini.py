from __future__ import annotations

import requests

from urllib.parse import quote

from .. import config
from ..types import GenerateInput


def is_configured() -> bool:
    return bool(config.GEMINI_API_KEY.strip())


def generate(input: GenerateInput) -> str:
    base = config.GEMINI_BASE_URL.rstrip("/")
    parts: list = [{"text": input.prompt}]
    if input.images:
        for img in input.images:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img}})

    generation_config: dict = {"temperature": input.temperature}
    if input.json_mode:
        generation_config["responseMimeType"] = "application/json"

    resp = requests.post(
        f"{base}/models/{quote(input.model, safe='')}:generateContent",
        params={"key": config.GEMINI_API_KEY.strip()},
        json={"contents": [{"parts": parts}], "generationConfig": generation_config},
        timeout=input.timeout_ms / 1000,
    )
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    content_parts = content.get("parts") or []
    if not content_parts:
        return ""
    return (content_parts[0].get("text") or "").strip()


gemini_provider = {"name": "gemini", "is_configured": is_configured, "generate": generate}
