from __future__ import annotations

import json
import re
from typing import Any, Optional

from . import config
from .providers import REGISTRY
from .types import CompletionResult, GenerateInput

__all__ = ["run_ai_task", "run_ai_task_json", "CompletionResult"]


def _empty_result(provider: str, model: str) -> CompletionResult:
    return CompletionResult(text="", provider=provider, model=model)


def extract_json_object(text: str) -> Optional[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?\n?", "", cleaned)
    cleaned = re.sub(r"\n?```", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return cleaned[start : end + 1]


def run_ai_task(
    task: str,
    prompt: str,
    *,
    images: list[str] | None = None,
    json_mode: bool = False,
    temperature: float = 0.0,
    timeout_ms: int | None = None,
) -> CompletionResult:
    task_cfg = config.TASKS.get(task)
    if not task_cfg:
        return CompletionResult(
            text="",
            provider="",
            model="",
            skipped=True,
            skip_reason=f"Unknown task: {task}",
        )

    provider_name = task_cfg.provider
    model = task_cfg.model

    if not config.AI_ENABLED or not task_cfg.enabled:
        return CompletionResult(
            **_empty_result(provider_name, model).__dict__,
            skipped=True,
            skip_reason="AI disabled for this task",
        )

    if not model:
        return CompletionResult(
            **_empty_result(provider_name, model).__dict__,
            skipped=True,
            skip_reason=f"No model configured for task {task}",
        )

    provider = REGISTRY.get(provider_name)
    if not provider:
        return CompletionResult(
            **_empty_result(provider_name, model).__dict__,
            skipped=True,
            skip_reason=f"Unknown provider: {provider_name}",
        )

    if not provider["is_configured"]():
        return CompletionResult(
            **_empty_result(provider_name, model).__dict__,
            skipped=True,
            skip_reason=f"{provider_name} not configured",
        )

    try:
        text = provider["generate"](
            GenerateInput(
                model=model,
                prompt=prompt,
                images=images,
                json_mode=json_mode,
                temperature=temperature,
                timeout_ms=timeout_ms or config.AI_TIMEOUT_MS,
            )
        )
        if not text:
            return CompletionResult(
                **_empty_result(provider_name, model).__dict__,
                skipped=True,
                skip_reason="empty AI response",
            )
        return CompletionResult(text=text, provider=provider_name, model=model)
    except Exception as exc:
        return CompletionResult(
            **_empty_result(provider_name, model).__dict__,
            skipped=True,
            skip_reason=str(exc),
        )


def run_ai_task_json(
    task: str,
    prompt: str,
    **kwargs: Any,
) -> tuple[dict | None, CompletionResult]:
    result = run_ai_task(task, prompt, json_mode=True, temperature=0.0, **kwargs)
    if result.skipped or not result.text:
        return None, result

    json_str = extract_json_object(result.text)
    if not json_str:
        return None, CompletionResult(
            text=result.text,
            provider=result.provider,
            model=result.model,
            skipped=True,
            skip_reason="no JSON object in AI response",
        )

    try:
        return json.loads(json_str), result
    except json.JSONDecodeError as exc:
        return None, CompletionResult(
            text=result.text,
            provider=result.provider,
            model=result.model,
            skipped=True,
            skip_reason=f"JSON parse failed: {exc}",
        )
