"""Shared deterministic task-type signatures for grouped benchmark splits."""

from __future__ import annotations

import re
from typing import Any


def primary_livemath_type(row: dict[str, Any]) -> str:
    values = [str(value).strip() for value in (row.get("theorem_type") or [])]
    return next((value for value in values if value), "unknown")


def static_split_type(benchmark: str, row: dict[str, Any]) -> str:
    """Fine type shared by the materializer and runtime preflight."""
    if benchmark == "livemathematicianbench":
        return primary_livemath_type(row)
    raise ValueError(f"unsupported static grouped benchmark: {benchmark}")
