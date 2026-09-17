"""OpenLux routing helpers for model suffixes or provider sorting."""

from __future__ import annotations

from typing import Any

OPENLUX_ROUTE_SUFFIXES: dict[str, str] = {
    "nitro": ":nitro",
    "speed": ":nitro",
    "floor": ":floor",
    "price": ":floor",
    "stable": ":stable",
    "success_rate": ":stable",
}
OPENLUX_PROVIDER_SORT: dict[str, str] = {
    "nitro": "speed",
    "speed": "speed",
    "floor": "price",
    "price": "price",
    "stable": "success_rate",
    "success_rate": "success_rate",
}
OPENLUX_KNOWN_SUFFIXES = frozenset(OPENLUX_ROUTE_SUFFIXES.values())
DEFAULT_OPENLUX_ROUTING = "stable"
DEFAULT_OPENLUX_ROUTING_STYLE = "suffix"  # suffix | provider_sort


def strip_openlux_routing_suffix(model: str) -> str:
    for suffix in OPENLUX_KNOWN_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def routing_to_suffix(routing: str | None) -> str | None:
    mode = (routing or "").strip().lower()
    if mode in {"", "none", "off", "false"}:
        return None
    suffix = OPENLUX_ROUTE_SUFFIXES.get(mode)
    if suffix is None:
        raise ValueError(f"unknown OpenLux routing mode: {routing!r}")
    return suffix


def routing_to_provider_sort(routing: str | None) -> str | None:
    mode = (routing or "").strip().lower()
    if mode in {"", "none", "off", "false"}:
        return None
    sort = OPENLUX_PROVIDER_SORT.get(mode)
    if sort is None:
        raise ValueError(f"unknown OpenLux routing mode: {routing!r}")
    return sort


def apply_suffix_routing(model: str, routing: str | None = None) -> str:
    raw = str(model).strip()
    if any(raw.endswith(s) for s in OPENLUX_KNOWN_SUFFIXES):
        return raw
    suffix = routing_to_suffix(routing if routing is not None else DEFAULT_OPENLUX_ROUTING)
    return raw if suffix is None else raw + suffix


def nitro_deployed_model(base_model: str, routing: str | None = None) -> str:
    """Return OpenLux API model string with ``:nitro`` (or other routing suffix)."""
    base = strip_openlux_routing_suffix(str(base_model).strip())
    deployed, _ = resolve_openlux_request(base, routing=routing or "nitro", style="suffix")
    return deployed


def nitro_catalog(models: tuple[str, ...] | list[str]) -> dict[str, str]:
    """Map base console ID → deployed ID, e.g. gpt5.5 → gpt5.5:nitro."""
    return {m: nitro_deployed_model(m) for m in models}


def resolve_openlux_request(
    base_model: str,
    *,
    routing: str | None = None,
    style: str | None = None,
) -> tuple[str, str | None]:
    """Return (model_for_api, provider_sort or None).

    style=suffix     → gpt-5.6-sol:nitro, provider_sort=None
    style=provider_sort → gpt-5.6-sol, provider_sort=speed
    """
    routing = routing if routing is not None else DEFAULT_OPENLUX_ROUTING
    style = (style or DEFAULT_OPENLUX_ROUTING_STYLE).strip().lower()
    base = strip_openlux_routing_suffix(base_model)

    if style in {"suffix", "model_suffix", "model"}:
        return apply_suffix_routing(base, routing), None

    if style in {"provider_sort", "provider", "sort"}:
        return base, routing_to_provider_sort(routing)

    print(f"  [model] warning: unknown openlux_routing_style={style!r}; using provider_sort")
    return base, routing_to_provider_sort(routing)


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    provider_sort: str | None = None,
    max_tokens: int = 16,
    stream: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build OpenAI SDK kwargs for OpenLux (provider.sort via extra_body)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if provider_sort:
        body = dict(extra or {})
        body["provider"] = {"sort": provider_sort}
        kwargs["extra_body"] = body
    elif extra:
        kwargs["extra_body"] = extra
    return kwargs


def install_openlux_provider_sort_hook(provider_sort: str | None) -> None:
    """Configure ``provider.sort`` for local OpenLux chat calls."""
    from graphopt.model import set_provider_sort

    set_provider_sort(provider_sort)
