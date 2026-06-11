from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ProviderName = Literal["ollama", "groq", "gemini", "openrouter"]

PROVIDER_NAMES: tuple[ProviderName, ...] = ("ollama", "groq", "gemini", "openrouter")


@dataclass
class GenerateInput:
    model: str
    prompt: str
    images: Optional[list[str]] = None
    json_mode: bool = False
    temperature: float = 0.0
    timeout_ms: int = 20000


@dataclass
class CompletionResult:
    text: str = ""
    provider: str = ""
    model: str = ""
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class TaskConfig:
    enabled: bool
    provider: ProviderName
    model: str
