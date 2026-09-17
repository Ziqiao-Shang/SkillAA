"""OpenLux model client used by the teacher and student roles."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from openai import OpenAI

_optimizer_client: OpenAI | None = None
_target_client: OpenAI | None = None
_client_lock = threading.Lock()

_optimizer_endpoint = "https://api.openlux.ai/v1"
_target_endpoint = "https://api.openlux.ai/v1"
_optimizer_api_key = ""
_target_api_key = ""
_optimizer_deployment = "gpt-5.6-sol:nitro"
_target_deployment = "gpt-5.6-sol:nitro"
_reasoning_effort: str | None = "medium"
_provider_sort: str | None = None


class TokenTracker:
    """Thread-safe request and token counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, int]] = {}

    def record(self, stage: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            row = self._data.setdefault(
                stage,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            row["calls"] += 1
            row["prompt_tokens"] += int(prompt_tokens)
            row["completion_tokens"] += int(completion_tokens)

    def summary(self) -> dict[str, dict[str, int]]:
        with self._lock:
            output: dict[str, dict[str, int]] = {}
            total_calls = total_prompt = total_completion = 0
            for stage, row in sorted(self._data.items()):
                prompt = row["prompt_tokens"]
                completion = row["completion_tokens"]
                output[stage] = {
                    **row,
                    "total_tokens": prompt + completion,
                }
                total_calls += row["calls"]
                total_prompt += prompt
                total_completion += completion
            output["_total"] = {
                "calls": total_calls,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            }
            return output


tracker = TokenTracker()


def configure_openlux(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    optimizer_endpoint: str | None = None,
    optimizer_api_key: str | None = None,
    target_endpoint: str | None = None,
    target_api_key: str | None = None,
) -> None:
    """Configure role-specific OpenAI-compatible OpenLux clients."""
    global _optimizer_endpoint, _target_endpoint
    global _optimizer_api_key, _target_api_key
    global _optimizer_client, _target_client

    shared_endpoint = str(endpoint or os.environ.get("OPENLUX_BASE_URL") or _optimizer_endpoint)
    shared_key = str(
        api_key
        or os.environ.get("OPENLUX_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    _optimizer_endpoint = str(optimizer_endpoint or shared_endpoint).rstrip("/")
    _target_endpoint = str(target_endpoint or shared_endpoint).rstrip("/")
    _optimizer_api_key = str(optimizer_api_key or shared_key)
    _target_api_key = str(target_api_key or shared_key)
    with _client_lock:
        _optimizer_client = None
        _target_client = None


def _make_client(role: str) -> OpenAI:
    endpoint = _optimizer_endpoint if role == "optimizer" else _target_endpoint
    key = _optimizer_api_key if role == "optimizer" else _target_api_key
    if not key:
        raise RuntimeError("OPENLUX_API_KEY is not configured")
    return OpenAI(api_key=key, base_url=endpoint)


def get_optimizer_client() -> OpenAI:
    global _optimizer_client
    with _client_lock:
        if _optimizer_client is None:
            _optimizer_client = _make_client("optimizer")
        return _optimizer_client


def get_target_client() -> OpenAI:
    global _target_client
    with _client_lock:
        if _target_client is None:
            _target_client = _make_client("target")
        return _target_client


def set_optimizer_deployment(deployment: str) -> None:
    global _optimizer_deployment
    _optimizer_deployment = str(deployment)


def set_target_deployment(deployment: str) -> None:
    global _target_deployment
    _target_deployment = str(deployment)


def set_reasoning_effort(effort: str | None) -> None:
    global _reasoning_effort
    value = str(effort or "").strip()
    _reasoning_effort = value or None


def set_provider_sort(provider_sort: str | None) -> None:
    global _provider_sort
    _provider_sort = str(provider_sort).strip() if provider_sort else None


def get_target_backend() -> str:
    return "openai_chat"


def is_target_exec_backend() -> bool:
    return False


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(getattr(usage, "total_tokens", 0) or prompt + completion),
    }


def _chat_messages(
    *,
    client: OpenAI,
    deployment: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            kwargs: dict[str, Any] = {
                "model": deployment,
                "messages": messages,
                "max_tokens": int(max_completion_tokens),
            }
            effort = reasoning_effort or _reasoning_effort
            if effort:
                kwargs["reasoning_effort"] = effort
            if timeout is not None:
                kwargs["timeout"] = int(timeout)
            if _provider_sort:
                kwargs["extra_body"] = {"provider": {"sort": _provider_sort}}
            response = client.chat.completions.create(**kwargs)
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError(f"OpenLux returned no choices: {response!r}")
            text = choices[0].message.content or ""
            usage = _usage_dict(response)
            tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
            return text, usage
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < max(1, int(retries)):
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenLux call failed after {max(1, int(retries))} attempts: {last_error}")


def chat_optimizer(
    *,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return _chat_messages(
        client=get_optimizer_client(),
        deployment=_optimizer_deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_target(
    *,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "rollout",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return _chat_messages(
        client=get_target_client(),
        deployment=_target_deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_target_messages(
    *,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "rollout",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return _chat_messages(
        client=get_target_client(),
        deployment=_target_deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()
