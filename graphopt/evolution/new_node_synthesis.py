"""Synthesize one complete node only after a novel proposal reaches support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphopt.debug.artifacts import ArtifactStore, save_llm_call, save_template_call
from graphopt.evolution.exact_refs import exact_reference_json
from graphopt.evolution.execution_child import _normalize_avoid
from graphopt.types import SkillGraph

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def synthesize_new_node(
    node_id: str,
    merged_opinion: str,
    support: int,
    source_case_ids: list[str],
    *,
    graph: SkillGraph | None = None,
    failure_examples: list[dict[str, Any]] | None = None,
    success_examples: list[dict[str, Any]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Return title plus trigger/action/avoid; two invalid answers abort."""
    parts: dict[str, str] = {}
    for section in merged_opinion.split("||"):
        key, separator, value = section.strip().partition(":")
        if separator:
            parts[key.strip().upper()] = value.strip()
    how_to_use = parts.get("DO") or merged_opinion.strip()
    title_text = how_to_use.rstrip(". ").casefold()
    title = title_text[:72] or node_id
    fallback = {
        "title": title,
        "when_to_use": parts.get("WHEN")
        or "When the recurring failure pattern in the supporting cases appears.",
        "how_to_use": how_to_use,
        "avoid": [parts["AVOID"]] if parts.get("AVOID") else [
            "Ignoring this recurring failure pattern."
        ],
    }
    if mode != "teacher" or chat_fn is None:
        if store is not None:
            save_template_call(
                store,
                node_id,
                stage="new_node_synthesis",
                inputs={
                    "node_id": node_id,
                    "merged_opinion": merged_opinion,
                    "support": support,
                    "source_case_ids": source_case_ids,
                },
                outputs=fallback,
            )
        return fallback

    system = (PROMPTS / "new_node_synthesis.md").read_text(encoding="utf-8")
    evidence = {
        "node_id": node_id,
        "support": support,
        "source_case_ids": source_case_ids,
        "merged_novel_opinion": merged_opinion,
        "source_bad_cases": list(failure_examples or []),
        "same_task_successes": list(success_examples or []),
        "existing_graph": graph.to_dict() if graph is not None else None,
    }
    user = exact_reference_json(evidence)
    from graphopt.json_utils import extract_json

    def parse(response: str) -> dict[str, Any]:
        obj = extract_json(response)
        required = {"title", "when_to_use", "how_to_use", "avoid"}
        if not isinstance(obj, dict) or set(obj) != required:
            raise ValueError(f"new node JSON must contain exactly {sorted(required)}")
        for key in ("title", "when_to_use", "how_to_use"):
            if not isinstance(obj[key], str) or not obj[key].strip():
                raise ValueError(f"new node {key} must be a non-empty string")
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
                max_completion_tokens=8192,
                retries=3,
                stage="new_node_synthesis",
            )
            result = parse(response)
            if graph is not None:
                audit_system = (PROMPTS / "new_node_audit.md").read_text(encoding="utf-8")
                audit_user = exact_reference_json(
                    {**evidence, "candidate_node": result}
                )
                audit_response, audit_usage = chat_fn(
                    system=audit_system, user=audit_user,
                    max_completion_tokens=4096, retries=3, stage="new_node_audit",
                )
                audit = extract_json(audit_response)
                if (
                    not isinstance(audit, dict)
                    or set(audit) != {"accept", "reason"}
                    or not isinstance(audit["accept"], bool)
                    or not isinstance(audit["reason"], str)
                    or not audit["reason"].strip()
                ):
                    raise ValueError(
                        "new node audit must contain exactly boolean accept and non-empty reason"
                    )
                if store is not None:
                    save_llm_call(
                        store, node_id, stage="new_node_audit", system=audit_system,
                        user=audit_user, response=audit_response, usage=audit_usage,
                        parsed=audit,
                    )
                if not audit["accept"]:
                    return None
            if store is not None:
                save_llm_call(
                    store,
                    node_id,
                    stage="new_node_synthesis",
                    system=system,
                    user=current_user,
                    response=response,
                    usage=usage,
                    parsed=result,
                )
            return result
        except Exception as exc:
            error = str(exc)
            if store is not None:
                save_llm_call(
                    store,
                    node_id,
                    stage="new_node_synthesis",
                    system=system,
                    user=current_user,
                    response=response or None,
                    usage=usage,
                    error=f"attempt {attempt}/2: {error}",
                )
            if attempt == 1:
                current_user = (
                    user
                    + "\n\n## Format Correction\nYour previous answer was invalid: "
                    + error
                    + "\nReturn the complete strict JSON object again with exactly title, "
                    + "when_to_use, how_to_use, and avoid. Do not include an ID, "
                    + "extra keys, or Markdown. avoid must be a JSON array of strings, "
                    + 'for example "avoid": ["Unsupported behavior to avoid."].'
                    + "\n\n## Invalid Previous Answer\n"
                    + str(response)
                )
    return None
