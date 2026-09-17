"""Synthesize a narrow child rule for a cited-but-not-executed parent skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphopt.debug.artifacts import ArtifactStore, save_llm_call, save_template_call
from graphopt.evolution.exact_refs import exact_reference_json
from graphopt.types import SkillGraph

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _normalize_avoid(value: Any) -> list[str]:
    """Normalize the harmless single-string variant emitted by teachers.

    Graph nodes store ``avoid`` as a list, but a teacher can serialize one
    sentence as a JSON string. That representation-only difference must not
    discard an otherwise complete child. Other value types remain invalid.
    """
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(
            "execution child avoid must be a JSON array of strings "
            '(for one item use {"avoid": ["..."]})'
        )
    return [item.strip() for item in value if item.strip()]


def _fallback_fields(node_id: str, reminders: str) -> dict[str, Any]:
    lines = [line.strip(" -\t") for line in reminders.splitlines() if line.strip()]
    procedure = lines[0] if lines else "Pause before acting and execute the cited parent rule."
    return {
        "title": f"Execution check for {node_id}",
        "when_to_use": f"When {node_id} has been retrieved and the next environment action is about to be chosen.",
        "how_to_use": procedure,
        "avoid": ["Citing the parent rule without applying its required state check to the next action."],
    }


def synthesize_execution_child(
    graph: SkillGraph,
    parent_node: str,
    child_node: str,
    reminders: str,
    support: int,
    source_case_ids: list[str],
    *,
    lapse_evidence: list[dict[str, Any]] | None = None,
    failure_examples: list[dict[str, Any]] | None = None,
    success_examples: list[dict[str, Any]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Create one executable child while leaving the correct parent untouched.

    This is intentionally one teacher call, not synthesis plus a second semantic
    audit. The shared mapped-validation small Gate is the behavioral acceptance test.
    """
    parent = graph.nodes.get(parent_node)
    if parent is None:
        return None
    fallback = _fallback_fields(parent_node, reminders)
    evidence = {
        "parent_node": parent.to_dict(),
        "child_node_id": child_node,
        "distinct_failure_support": support,
        "source_case_ids": list(source_case_ids),
        "execution_lapse_reminders": reminders,
        "semantic_cluster_evidence": list(lapse_evidence or []),
        "source_bad_cases": list(failure_examples or []),
        "protected_right_cases": list(success_examples or []),
    }
    if mode != "teacher" or chat_fn is None:
        if store is not None:
            save_template_call(
                store,
                child_node,
                stage="execution_child_synthesis",
                inputs=evidence,
                outputs=fallback,
            )
        return fallback

    system = (PROMPTS / "execution_child_synthesis.md").read_text(encoding="utf-8")
    user = exact_reference_json(evidence)
    from graphopt.json_utils import extract_json

    def parse(response: str) -> dict[str, Any] | None:
        obj = extract_json(response)
        required = {"title", "when_to_use", "how_to_use", "avoid"}
        if not isinstance(obj, dict) or set(obj) != required:
            raise ValueError(f"execution child JSON must contain exactly {sorted(required)}")
        if all(obj[key] is None for key in required):
            return None
        for key in ("title", "when_to_use", "how_to_use"):
            if not isinstance(obj[key], str) or not obj[key].strip():
                raise ValueError(f"execution child {key} must be a non-empty string")
        avoid = _normalize_avoid(obj["avoid"])
        return {
            "title": obj["title"].strip(),
            "when_to_use": obj["when_to_use"].strip(),
            "how_to_use": obj["how_to_use"].strip(),
            "avoid": avoid,
        }

    current_user = user
    for attempt in range(1, 3):
        response = ""
        usage: Any = None
        try:
            response, usage = chat_fn(
                system=system,
                user=current_user,
                max_completion_tokens=4096,
                retries=2,
                stage="execution_child_synthesis",
            )
            result = parse(response)
            if store is not None:
                save_llm_call(
                    store,
                    child_node,
                    stage="execution_child_synthesis",
                    system=system,
                    user=current_user,
                    response=response,
                    usage=usage,
                    parsed=result,
                )
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if store is not None:
                save_llm_call(
                    store,
                    child_node,
                    stage="execution_child_synthesis",
                    system=system,
                    user=current_user,
                    response=response or None,
                    usage=usage,
                    error=f"attempt {attempt}/2: {exc}",
                )
            if attempt == 1:
                current_user = (
                    user
                    + "\n\n## Format correction\nThe previous answer was invalid: "
                    + error
                    + "\nReturn only the five-key JSON object. title, meaning, "
                    "when_to_use, and how_to_use must be non-empty strings. "
                    'avoid must be a JSON array of strings, for example "avoid": '
                    '["Do not repeat a known-bad search action."]. Use null for all '
                    "four values only if the evidence cannot support a narrow "
                    "execution detail. Do not change supported semantics merely to "
                    "repair the JSON format.\n\n## Invalid previous answer\n"
                    + response
                )
    return None
