"""Lossless exact-value references for model-facing JSON payloads.

Only byte-for-byte equal canonical JSON subtrees are interned.  The original
artifacts are never changed, and every encoded package is decoded and compared
with its source before it may be sent to a model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


REF_KEY = "__graphopt_exact_ref__"
SCHEMA = "graphopt-exact-evidence-references"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        yield value
        for child in value:
            yield from _walk(child)
    elif isinstance(value, str):
        yield value


def encode_exact_references(value: Any, *, min_chars: int = 256) -> dict[str, Any]:
    """List repeated exact values once and replace every occurrence by a ref."""
    canonical_values: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    for item in _walk(value):
        canonical = _canonical(item)
        if len(canonical) < min_chars:
            continue
        counts[canonical] += 1
        canonical_values.setdefault(canonical, item)

    # Prefer complete repeated objects.  A repeated child already contained in
    # one selected parent is not separately catalogued; this is what makes a
    # six-times repeated evidence object appear exactly once, not once per field.
    selected: list[str] = []
    covered_descendants: set[str] = set()
    for canonical in sorted(
        (text for text, count in counts.items() if count > 1),
        key=lambda text: (-len(text), text),
    ):
        if canonical in covered_descendants:
            continue
        selected.append(canonical)
        parent = canonical_values[canonical]
        for child in _walk(parent):
            child_canonical = _canonical(child)
            if child_canonical != canonical:
                covered_descendants.add(child_canonical)

    ids = {canonical: f"sha256:{_digest(canonical)}" for canonical in selected}
    catalog = {ids[canonical]: canonical_values[canonical] for canonical in selected}

    def encode(item: Any) -> Any:
        if isinstance(item, (dict, list, str)):
            canonical = _canonical(item)
            if canonical in ids:
                return {REF_KEY: ids[canonical]}
        if isinstance(item, dict):
            return {str(key): encode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [encode(child) for child in item]
        return item

    package = {
        "schema_version": SCHEMA,
        "reference_rule": (
            f"An object containing only {REF_KEY} means the exact JSON value "
            "stored under that ID in evidence_catalog. Resolve it verbatim."
        ),
        "evidence_catalog": catalog,
        "payload": encode(value),
    }
    if _canonical(decode_exact_references(package)) != _canonical(value):
        raise ValueError("exact-reference round-trip validation failed")
    return package


def decode_exact_references(package: dict[str, Any]) -> Any:
    """Reconstruct a package produced by :func:`encode_exact_references`."""
    if package.get("schema_version") != SCHEMA:
        raise ValueError("unknown exact-reference schema")
    catalog = package.get("evidence_catalog")
    if not isinstance(catalog, dict):
        raise ValueError("evidence_catalog must be an object")

    def decode(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {REF_KEY}:
            ref = str(item[REF_KEY])
            if ref not in catalog:
                raise ValueError(f"unknown exact evidence reference: {ref}")
            expected = ref.removeprefix("sha256:")
            canonical = _canonical(catalog[ref])
            if _digest(canonical) != expected:
                raise ValueError(f"exact evidence hash mismatch: {ref}")
            return catalog[ref]
        if isinstance(item, dict):
            return {str(key): decode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        return item

    return decode(package.get("payload"))


def exact_reference_json(value: Any, *, indent: int | None = 2) -> str:
    """Encode, validate, and render a lossless model-facing JSON package."""
    package = encode_exact_references(value)
    kwargs = {"indent": indent} if indent is not None else {"separators": (",", ":")}
    return json.dumps(package, ensure_ascii=False, **kwargs)
