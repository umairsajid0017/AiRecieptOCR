from __future__ import annotations

import requests

from .. import config
from ..types import GenerateInput


def is_configured() -> bool:
    return bool(config.OLLAMA_URL.strip())


def _chat_url() -> str:
    base = config.OLLAMA_URL.rstrip("/")
    if base.endswith("/api/generate"):
        return base[: -len("/api/generate")] + "/api/chat"
    if base.endswith("/api/chat"):
        return base
    return base + "/api/chat"


def generate(input: GenerateInput) -> str:
    headers = {"Content-Type": "application/json"}
    if config.OLLAMA_API_KEY.strip():
        headers["Authorization"] = f"Bearer {config.OLLAMA_API_KEY.strip()}"

    timeout = input.timeout_ms / 1000

    if input.images:
        body: dict = {
            "model": input.model,
            "messages": [
                {
                    "role": "user",
                    "content": input.prompt,
                    "images": input.images,
                }
            ],
            "stream": False,
            "options": {"temperature": input.temperature},
        }
        if input.json_mode:
            body["format"] = "json"
        resp = requests.post(_chat_url(), json=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message") or {}).get("content", "").strip()

    resp = requests.post(
        config.OLLAMA_URL,
        json={
            "model": input.model,
            "prompt": input.prompt,
            "stream": False,
            "options": {"temperature": input.temperature},
        },
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


ollama_provider = {"name": "ollama", "is_configured": is_configured, "generate": generate}
