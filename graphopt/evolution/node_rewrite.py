"""Apply merged node revisions with LLM (preserve valid old semantics)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from graphopt.debug.artifacts import ArtifactStore, save_llm_call, save_template_call
from graphopt.evolution.exact_refs import exact_reference_json
from graphopt.types import SkillGraph

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _normalized_text(value: str) -> str:
    """Normalize an instruction for conservative idempotency checks."""
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _append_preserving(base: str, addition: str, *, label: str) -> str:
    """Append one learned clause without allowing a rewrite to erase *base*."""
    original = (base or "").strip()
    extra = (addition or "").strip()
    if not extra:
        return original
    normalized_base = _normalized_text(original)
    normalized_extra = _normalized_text(extra)
    if not normalized_extra or normalized_extra in normalized_base:
        return original
    suffix = f"{label}: {extra}"
    return f"{original} {suffix}" if original else suffix


_ACTION_WORDS = {
    "choose", "continue", "open", "take", "put", "place", "go", "move",
    "search", "inspect", "examine", "verify", "record", "skip", "switch",
    "return", "stop", "clean", "heat", "cool", "use", "update", "check",
    "select", "confirm", "reject", "maintain", "track", "prefer", "avoid", "retry",
}
_CONDITION_WORDS = {
    "when", "if", "after", "before", "while", "once", "unless", "until",
}
_STOP_WORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "then", "to", "with",
}


def _addition_quality_error(node: Any, addition: str) -> str:
    """Cheap pre-apply filter for redundant or non-executable additions."""
    tokens = set(_normalized_text(addition).split())
    original = " ".join((
        str(node.meaning), str(node.when_to_use), str(node.how_to_use),
        " ".join(str(x) for x in node.avoid),
    ))
    original_tokens = set(_normalized_text(original).split())
    novel = {token for token in tokens - original_tokens if token not in _STOP_WORDS}
    if len(novel) < 2:
        return "addition has no substantive new semantics"
    if not tokens.intersection(_CONDITION_WORDS):
        return "addition lacks an observable condition"
    if not tokens.intersection(_ACTION_WORDS):
        return "addition lacks a concrete next action or state update"
    return ""


def rewrite_node(
    graph: SkillGraph,
    node_id: str,
    merged_revision: str,
    *,
    positive_examples: list[dict[str, Any]] | None = None,
    failure_examples: list[dict[str, Any]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Return revised node body fields; node identity/title stay immutable."""
    node = graph.nodes.get(node_id)
    if node is None:
        out = {"how_to_use": merged_revision}
        if store is not None:
            save_template_call(
                store,
                node_id,
                stage="node_rewrite",
                inputs={"node_id": node_id, "merged_revision": merged_revision, "missing_node": True},
                outputs=out,
            )
        return out

    if mode == "teacher" and chat_fn is not None:
        system = (PROMPTS / "node_rewrite.md").read_text(encoding="utf-8")
        positives = list(positive_examples or [])
        failures = list(failure_examples or [])
        user = (
            f"## Original Node {node_id}\n"
            f"title: {node.title}\n"
            f"meaning: {node.meaning}\n"
            f"when_to_use: {node.when_to_use}\n"
            f"how_to_use: {node.how_to_use}\n"
            f"avoid: {node.avoid}\n\n"
            f"## Merged Revision Proposal\n{merged_revision}\n\n"
            f"## Successful Uses of This Node ({len(positives)} cases)\n"
            "These are a bounded, locally clean, task-relevant sample where "
            "the node was attributed as correct. Preserve compatibility with "
            "every case. Each record contains the task and complete action path.\n"
            + exact_reference_json(positives)
            + "\n\n## Bad Cases This Patch Must Help ("
            + str(len(failures))
            + " cases)\nInfer one reusable missing condition/action that would change "
            "these failures without changing the protected successful paths.\n"
            + exact_reference_json(failures)
            + "\n"
        )
        from graphopt.json_utils import extract_json

        def parse_response(response: str) -> str | None:
            obj = extract_json(response)
            if not isinstance(obj, dict) or set(obj) != {"addition"}:
                raise ValueError("node experience update must contain exactly addition")
            value = obj["addition"]
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError("node experience addition must be a string or null")
            value = value.strip()
            if len(value) > 600:
                raise ValueError("node experience addition exceeds 600 characters")
            return value

        current_user = user
        for attempt in range(1, 3):
            resp = ""
            token_usage: Any = None
            try:
                resp, token_usage = chat_fn(
                    system=system,
                    user=current_user,
                    max_completion_tokens=1024,
                    retries=2,
                    stage="node_rewrite",
                )
                addition = parse_response(resp)
                if addition is None:
                    if store is not None:
                        save_llm_call(
                            store, node_id, stage="node_rewrite", system=system,
                            user=current_user, response=resp, usage=token_usage,
                            parsed={"addition": None, "decision": "no_safe_addition"},
                        )
                    return None
                quality_error = _addition_quality_error(node, addition)
                if quality_error:
                    raise ValueError(quality_error)
                out = {
                    "meaning": node.meaning,
                    "when_to_use": node.when_to_use,
                    "how_to_use": _append_preserving(
                        node.how_to_use, addition, label="Learned extension"
                    ),
                    "avoid": list(node.avoid),
                }
                if store is not None:
                    save_llm_call(
                        store,
                        node_id,
                        stage="node_rewrite",
                        system=system,
                        user=current_user,
                        response=resp,
                        usage=token_usage,
                        parsed=out,
                    )
                return out
            except Exception as exc:
                error = str(exc)
                if store is not None:
                    save_llm_call(
                        store,
                        node_id,
                        stage="node_rewrite",
                        system=system,
                        user=current_user,
                        response=resp or None,
                        usage=token_usage,
                        error=f"attempt {attempt}/2: {error}",
                    )
                if attempt == 1:
                    current_user = (
                        user
                        + "\n\n## Format Correction\nYour previous answer was invalid: "
                        + error
                        + "\nReturn strict JSON again with exactly the key "
                        + "addition. Do not rewrite the original node or include extra "
                        + "keys or Markdown.\n\n## Invalid Previous Answer\n"
                        + str(resp)
                    )
        # Invalid raw proposals are not safe learned experience. Keep the node
        # intact and wait for future evidence instead of force-appending them.
        return None

    out = {
        "meaning": node.meaning,
        "when_to_use": node.when_to_use,
        "how_to_use": _append_preserving(
            node.how_to_use, merged_revision, label="Learned extension"
        ),
        "avoid": list(node.avoid),
    }
    if store is not None:
        save_template_call(
            store,
            node_id,
            stage="node_rewrite",
            inputs={
                "node_id": node_id,
                "merged_revision": merged_revision,
                "original_how_to_use": node.how_to_use,
            },
            outputs=out,
        )
    return out


def rewrite_toxic_node(
    graph: SkillGraph,
    node_id: str,
    toxic_text: str,
    merged_revision: str,
    *,
    positive_examples: list[dict[str, Any]] | None = None,
    failure_examples: list[dict[str, Any]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Replace one exact harmful span while preserving every other byte."""
    node = graph.nodes.get(node_id)
    toxic = (toxic_text or "").strip()
    if node is None or not toxic:
        return None
    fields: dict[str, Any] = {
        "meaning": node.meaning,
        "when_to_use": node.when_to_use,
        "how_to_use": node.how_to_use,
        "avoid": list(node.avoid),
    }
    locations = [key for key in ("meaning", "when_to_use", "how_to_use") if toxic in fields[key]]
    locations += [f"avoid:{i}" for i, value in enumerate(fields["avoid"]) if toxic in value]
    if len(locations) != 1:
        return None
    replacement = merged_revision.strip()
    if mode == "teacher" and chat_fn is not None:
        system = (PROMPTS / "toxic_node_rewrite.md").read_text(encoding="utf-8")
        user = exact_reference_json({
            "node_id": node_id,
            "toxic_text": toxic,
            "replacement_proposals": merged_revision,
            "bad_cases": list(failure_examples or []),
            "protected_successes": list(positive_examples or []),
        })
        from graphopt.json_utils import extract_json
        for attempt in range(1, 3):
            response = ""
            usage: Any = None
            try:
                response, usage = chat_fn(
                    system=system, user=user, max_completion_tokens=4096,
                    retries=2, stage="toxic_node_rewrite",
                )
                obj = extract_json(response)
                if not isinstance(obj, dict) or set(obj) != {"replacement"}:
                    raise ValueError("toxic rewrite must contain exactly replacement")
                replacement = obj["replacement"]
                if replacement is None:
                    return None
                if not isinstance(replacement, str) or not replacement.strip() or len(replacement) > 600:
                    raise ValueError("replacement must be a string <=600 characters or null")
                replacement = replacement.strip()
                if store is not None:
                    save_llm_call(store, node_id, stage="toxic_node_rewrite", system=system,
                                  user=user, response=response, usage=usage,
                                  parsed={"replacement": replacement})
                break
            except Exception as exc:
                if store is not None:
                    save_llm_call(store, node_id, stage="toxic_node_rewrite", system=system,
                                  user=user, response=response or None, usage=usage,
                                  error=f"attempt {attempt}/2: {exc}")
                if attempt == 2:
                    return None
    if not replacement or replacement == toxic:
        return None
    location = locations[0]
    if location.startswith("avoid:"):
        index = int(location.split(":", 1)[1])
        fields["avoid"][index] = fields["avoid"][index].replace(toxic, replacement, 1)
    else:
        fields[location] = fields[location].replace(toxic, replacement, 1)
    return fields


def rewrite_node_when_to_use(
    graph: SkillGraph,
    node_id: str,
    merged_revision: str,
    *,
    positive_examples: list[dict[str, Any]] | None = None,
    failure_examples: list[dict[str, Any]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Clarify only a correct node's retrieval trigger after repeated misses."""
    node = graph.nodes.get(node_id)
    if node is None:
        return None
    system = (PROMPTS / "retrieval_rewrite.md").read_text(encoding="utf-8")
    positives = list(positive_examples or [])
    failures = list(failure_examples or [])
    user = (
        f"## Original Node {node_id}\n"
        f"title: {node.title}\n"
        f"meaning: {node.meaning}\n"
        f"when_to_use: {node.when_to_use}\n"
        f"how_to_use: {node.how_to_use}\n"
        f"avoid: {node.avoid}\n\n"
        f"## Merged Retrieval-Trigger Opinions\n{merged_revision}\n\n"
        f"## Successful Uses of This Node ({len(positives)} cases)\n"
        "These are a bounded, locally clean, task-relevant sample where "
        "the original trigger retrieved the node correctly. The revised trigger "
        "must continue to cover every case.\n"
        + exact_reference_json(positives)
        + "\n\n## Retrieval-Miss Cases ("
        + str(len(failures))
        + " cases)\nUse only cues observable immediately before the missed retrieval.\n"
        + exact_reference_json(failures)
        + "\n"
    )

    if mode == "teacher" and chat_fn is not None:
        from graphopt.json_utils import extract_json

        def parse_response(response: str) -> str | None:
            obj = extract_json(response)
            if not isinstance(obj, dict) or set(obj) != {"additional_when"}:
                raise ValueError("retrieval update must contain exactly additional_when")
            value = obj["additional_when"]
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError("additional_when must be a non-empty string or null")
            return value.strip()

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
                    stage="retrieval_rewrite",
                )
                additional_when = parse_response(response)
                if additional_when is None:
                    return None
                out = {"when_to_use": _append_preserving(
                    node.when_to_use, additional_when, label="Also retrieve when"
                )}
                if store is not None:
                    save_llm_call(
                        store,
                        node_id,
                        stage="retrieval_rewrite",
                        system=system,
                        user=current_user,
                        response=response,
                        usage=usage,
                        parsed=out,
                    )
                return out
            except Exception as exc:
                error = str(exc)
                if store is not None:
                    save_llm_call(
                        store,
                        node_id,
                        stage="retrieval_rewrite",
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
                        + "\nReturn strict JSON with exactly the key "
                        + "additional_when. Do not rewrite the original trigger or "
                        + "output any other field; use null to abstain.\n\n"
                        + "## Invalid Previous Answer\n"
                        + str(response)
                    )
        return None
    trigger = merged_revision.strip()
    original = (node.when_to_use or "").strip()
    value = _append_preserving(original, trigger, label="Also retrieve when")
    out = {"when_to_use": value}
    if store is not None:
        save_template_call(
            store,
            node_id,
            stage="retrieval_rewrite",
            inputs={
                "node_id": node_id,
                "original_when_to_use": original,
                "merged_revision": merged_revision,
            },
            outputs=out,
        )
    return out


def reinforce_node_execution(
    graph: SkillGraph,
    node_id: str,
    reinforcement: str,
    *,
    positive_examples: list[dict[str, Any]] | None = None,
    failure_examples: list[dict[str, Any]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Lightly emphasize execution wording without changing node semantics."""
    node = graph.nodes.get(node_id)
    if node is None:
        return None
    positives = list(positive_examples or [])
    failures = list(failure_examples or [])
    if mode == "teacher" and chat_fn is not None:
        system = (PROMPTS / "execution_reinforcement.md").read_text(encoding="utf-8")
        user = (
            f"## Original Node {node_id}\nmeaning: {node.meaning}\n"
            f"when_to_use: {node.when_to_use}\nhow_to_use: {node.how_to_use}\n"
            f"avoid: {node.avoid}\n\n## Repeated Execution Reminder\n{reinforcement}\n\n"
            f"## Successful Uses ({len(positives)} cases)\n"
            + exact_reference_json(positives)
            + f"\n\n## Repeated Execution-Lapse Cases ({len(failures)} cases)\n"
            + exact_reference_json(failures)
        )
        from graphopt.json_utils import extract_json

        for attempt in range(1, 3):
            response = ""
            usage: Any = None
            try:
                response, usage = chat_fn(
                    system=system, user=user, max_completion_tokens=4096,
                    retries=2, stage="execution_reinforcement",
                )
                obj = extract_json(response)
                if not isinstance(obj, dict) or set(obj) != {"emphasis"}:
                    raise ValueError("execution reinforcement must contain exactly emphasis")
                value = obj["emphasis"]
                if value is None:
                    return None
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("execution emphasis must be a non-empty string or null")
                out = {"how_to_use": _append_preserving(
                    node.how_to_use, value, label="Execution emphasis"
                )}
                if store is not None:
                    save_llm_call(
                        store, node_id, stage="execution_reinforcement",
                        system=system, user=user, response=response,
                        usage=usage, parsed=out,
                    )
                return out
            except Exception as exc:
                if store is not None:
                    save_llm_call(
                        store, node_id, stage="execution_reinforcement",
                        system=system, user=user, response=response or None,
                        usage=usage, error=f"attempt {attempt}/2: {exc}",
                    )
        return None
    base = (node.how_to_use or "").rstrip()
    reminder = reinforcement.strip()
    out = {"how_to_use": _append_preserving(
        base, reminder, label="Execution emphasis"
    )}
    if store is not None:
        save_template_call(
            store, node_id, stage="execution_reinforcement",
            inputs={"node_id": node_id, "reinforcement": reinforcement,
                    "positive_examples": positives}, outputs=out,
        )
    return out


def audit_combined_node_update(
    graph: SkillGraph,
    node_id: str,
    candidate_fields: dict[str, Any],
    component_updates: list[dict[str, Any]],
    *,
    positive_examples: list[dict[str, Any]] | None = None,
    failure_examples: list[dict[str, Any]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Remove only exact new conflicting/redundant spans from the final node."""
    node = graph.nodes.get(node_id)
    if node is None:
        return None
    if mode != "teacher" or chat_fn is None:
        return dict(candidate_fields)
    system = (PROMPTS / "node_combination_audit.md").read_text(encoding="utf-8")
    user = exact_reference_json({
            "node_id": node_id,
            "original_node": node.to_dict(),
            "candidate_node": {
                "title": node.title,
                "meaning": candidate_fields.get("meaning", node.meaning),
                "when_to_use": candidate_fields.get("when_to_use", node.when_to_use),
                "how_to_use": candidate_fields.get("how_to_use", node.how_to_use),
                "avoid": candidate_fields.get("avoid", list(node.avoid)),
            },
            "component_updates": component_updates,
            "source_bad_cases": list(failure_examples or []),
            "protected_successes": list(positive_examples or []),
        })
    from graphopt.json_utils import extract_json

    current_user = user
    for attempt in range(1, 3):
        response = ""
        usage: Any = None
        try:
            response, usage = chat_fn(
                system=system, user=current_user, max_completion_tokens=4096,
                retries=3, stage="node_combination_audit",
            )
            obj = extract_json(response)
            if (
                not isinstance(obj, dict)
                or set(obj) != {"remove_spans", "unresolved_conflict", "reason"}
                or not isinstance(obj["remove_spans"], list)
                or any(not isinstance(item, dict) for item in obj["remove_spans"])
                or not isinstance(obj["unresolved_conflict"], bool)
                or not isinstance(obj["reason"], str)
                or not obj["reason"].strip()
            ):
                raise ValueError(
                    "combination audit requires remove_spans, unresolved_conflict, and reason"
                )
            if obj["unresolved_conflict"]:
                if obj["remove_spans"]:
                    raise ValueError("unresolved audit cannot also request partial removals")
                if store is not None:
                    save_llm_call(
                        store, node_id, stage="node_combination_audit", system=system,
                        user=current_user, response=response, usage=usage, parsed=obj,
                    )
                return None
            cleaned = dict(candidate_fields)
            original_by_field = {
                "when_to_use": node.when_to_use,
                "how_to_use": node.how_to_use,
            }
            for index, removal in enumerate(obj["remove_spans"]):
                if set(removal) != {"field", "text", "reason"}:
                    raise ValueError(
                        f"remove_spans[{index}] must contain field, text, and reason"
                    )
                field = removal["field"]
                text = removal["text"]
                reason = removal["reason"]
                if field not in original_by_field:
                    raise ValueError("only additive when_to_use/how_to_use spans may be removed")
                if not isinstance(text, str) or not text.strip() or not isinstance(reason, str) or not reason.strip():
                    raise ValueError("removal text and reason must be non-empty strings")
                allowed_labels = (
                    "Learned extension:", "Also retrieve when:", "Execution emphasis:"
                )
                if not text.startswith(allowed_labels):
                    raise ValueError(
                        "removal must be one complete labeled additive span"
                    )
                value = str(cleaned.get(field) or "")
                if value.count(text) != 1:
                    raise ValueError("removal text must occur exactly once in the candidate field")
                original = original_by_field[field]
                if text in original or not value.startswith(original):
                    raise ValueError("combination audit cannot delete original node semantics")
                if value.find(text) < len(original):
                    raise ValueError("removal span overlaps original node semantics")
                cleaned[field] = " ".join(value.replace(text, "", 1).split())
            if store is not None:
                save_llm_call(
                    store, node_id, stage="node_combination_audit", system=system,
                    user=current_user, response=response, usage=usage, parsed=obj,
                )
            return cleaned
        except Exception as exc:
            if store is not None:
                save_llm_call(
                    store, node_id, stage="node_combination_audit", system=system,
                    user=current_user, response=response or None, usage=usage,
                    error=f"attempt {attempt}/2: {exc}",
                )
            if attempt == 1:
                current_user = (
                    user
                    + "\n\n## Format Correction\nReturn strict JSON with exactly "
                    + "remove_spans, unresolved_conflict, and reason. Each removal must "
                    + "quote one exact newly added span. Reject on uncertainty.\n\n"
                    + "## Invalid Previous Answer\n"
                    + str(response)
                )
    return None
