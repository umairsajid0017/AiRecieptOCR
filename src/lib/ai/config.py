from __future__ import annotations

import os
import warnings

from dotenv import load_dotenv

from .types import PROVIDER_NAMES, ProviderName, TaskConfig

load_dotenv()


def _to_bool(value: str | None, fallback: bool) -> bool:
    if value is None:
        return fallback
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return fallback


def _to_int(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value is not None else fallback
    except ValueError:
        return fallback


def parse_provider(value: str | None, fallback: ProviderName = "ollama") -> ProviderName:
    normalized = (value or "").strip().lower()
    if normalized in PROVIDER_NAMES:
        return normalized  # type: ignore[return-value]
    return fallback


AI_ENABLED = _to_bool(os.getenv("AI_ENABLED"), True)
AI_TIMEOUT_MS = _to_int(os.getenv("AI_TIMEOUT_MS"), 20000)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)


def _receipt_vision_model() -> str:
    model = os.getenv("AI_TASK_RECEIPT_VISION_MODEL", "").strip()
    if model:
        return model
    legacy = os.getenv("OLLAMA_VISION_MODEL", "").strip()
    if legacy:
        warnings.warn(
            "OLLAMA_VISION_MODEL is deprecated; use AI_TASK_RECEIPT_VISION_MODEL",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy
    return ""


TASKS: dict[str, TaskConfig] = {
    "receipt_vision": TaskConfig(
        enabled=_to_bool(os.getenv("AI_TASK_RECEIPT_VISION_ENABLED"), True),
        provider=parse_provider(os.getenv("AI_TASK_RECEIPT_VISION_PROVIDER"), "ollama"),
        model=_receipt_vision_model(),
    ),
}
