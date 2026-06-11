from __future__ import annotations

import requests

from .. import config
from ..types import GenerateInput


def is_configured() -> bool:
    return bool(config.GROQ_API_KEY.strip())


def _build_content(prompt: str, images: list[str] | None) -> str | list:
    if not images:
        return prompt
    parts: list = [{"type": "text", "text": prompt}]
    for img in images:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            }
        )
    return parts


def generate(input: GenerateInput) -> str:
    base = config.GROQ_BASE_URL.rstrip("/")
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
            "Authorization": f"Bearer {config.GROQ_API_KEY.strip()}",
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


groq_provider = {"name": "groq", "is_configured": is_configured, "generate": generate}
