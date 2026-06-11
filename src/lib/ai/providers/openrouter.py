from __future__ import annotations

import requests

from .. import config
from ..types import GenerateInput
from .groq import _build_content


def is_configured() -> bool:
    return bool(config.OPENROUTER_API_KEY.strip())


def generate(input: GenerateInput) -> str:
    base = config.OPENROUTER_BASE_URL.rstrip("/")
    body: dict = {
        "model": input.model,
        "messages": [{"role": "user", "content": _build_content(input.prompt, input.images)}],
        "temperature": input.temperature,
    }
    if input.json_mode:
        body["response_format"] = {"type": "json_object"}

    resp = requests.post(
        f"{base}/chat/completions",
        json=body,
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY.strip()}",
            "Content-Type": "application/json",
        },
        timeout=input.timeout_ms / 1000,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


openrouter_provider = {
    "name": "openrouter",
    "is_configured": is_configured,
    "generate": generate,
}
