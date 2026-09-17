"""Cheap, deterministic evidence selection for experience synthesis.

These helpers do not judge whether an episode is globally good.  They select a
small set of *locally useful* contrasts for one node, so a lucky successful
episode cannot silently become positive evidence for the exact behaviour that
was wrong inside that episode.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


SEARCH_CONTROL_NODES = {"G02", "G03", "G08", "H01", "H02"}
TASK_NODE_TYPES = {
    "T01": "pick_and_place",
    "T02": "pick_two_obj_and_place",
    "T03": "look_at_obj_in_light",
    "T04": "pick_clean_then_place_in_recep",
    "T05": "pick_heat_then_place_in_recep",
    "T06": "pick_cool_then_place_in_recep",
}


def extract_actions(trajectory: str) -> list[str]:
    """Extract the executable action sequence from a formatted trajectory."""
    actions: list[str] = []
    for line in str(trajectory or "").splitlines():
        marker = "Action:" if "Action:" in line else "Next action:"
        if marker not in line:
            continue
        action = line.split(marker, 1)[1].split(
            " | Environment feedback:", 1
        )[0].strip()
        if action:
            actions.append(action)
    return actions


def _go_target(action: str) -> str:
    text = str(action or "").strip().casefold()
    return text[6:].strip() if text.startswith("go to ") else ""


def first_pre_take_location_revisit(actions: list[str]) -> dict[str, Any] | None:
    """Return the first exact search-location revisit before acquiring an item."""
    seen: dict[str, int] = {}
    for index, action in enumerate(actions):
        normalized = str(action or "").strip().casefold()
        if normalized.startswith("take "):
            break
        target = _go_target(normalized)
        if not target:
            continue
        if target in seen:
            return {
                "action_index": index,
                "action": action,
                "first_action_index": seen[target],
                "kind": "pre_take_location_revisit",
            }
        seen[target] = index
    return None


def extract_action_feedback_events(trajectory: str) -> list[dict[str, Any]]:
    """Parse the harness-authored action/feedback pairs with stable indices."""
    events: list[dict[str, Any]] = []
    for line in str(trajectory or "").splitlines():
        if "Action:" not in line and "Next action:" not in line:
            continue
        action_marker = "Action:" if "Action:" in line else "Next action:"
        tail = line.split(action_marker, 1)[1]
        if " | Environment feedback:" in tail:
            action, feedback = tail.split(" | Environment feedback:", 1)
        else:
            action, feedback = tail, ""
        action = str(action).strip()
        feedback = str(feedback).strip()
        if action:
            events.append(
                {
                    "action_index": len(events),
                    "action": action,
                    "feedback": feedback,
                }
            )
    return events


def diagnose_no_progress_loops(
    trajectory: str, *, evidence_cap: int = 6
) -> dict[str, Any]:
    """Return deterministic no-progress evidence without judging graph semantics."""
    events = extract_action_feedback_events(trajectory)
    normalized_actions = [
        " ".join(str(event["action"]).casefold().split()) for event in events
    ]
    pair_occurrences: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        key = (
            normalized_actions[index],
            " ".join(str(event["feedback"]).casefold().split()),
        )
        pair_occurrences[key].append(index)

    repeated_pairs: list[dict[str, Any]] = []
    for (action, feedback), indices in pair_occurrences.items():
        if len(indices) < 2:
            continue
        repeated_pairs.append(
            {
                "action": action,
                "feedback": feedback[:240],
                "action_indices": indices[:12],
                "count": len(indices),
            }
        )
    repeated_pairs.sort(key=lambda item: (-int(item["count"]), item["action"]))

    cycles: list[dict[str, Any]] = []
    seen_cycles: set[tuple[str, ...]] = set()
    for period in (2, 3, 4):
        for start in range(0, max(0, len(normalized_actions) - (2 * period) + 1)):
            pattern = tuple(normalized_actions[start : start + period])
            if not pattern or pattern in seen_cycles:
                continue
            repeats = 1
            cursor = start + period
            while normalized_actions[cursor : cursor + period] == list(pattern):
                repeats += 1
                cursor += period
            if repeats >= 2:
                seen_cycles.add(pattern)
                cycles.append(
                    {
                        "start_action_index": start,
                        "period": period,
                        "repeats": repeats,
                        "actions": list(pattern),
                    }
                )
    cycles.sort(key=lambda item: (-int(item["repeats"]), int(item["period"])))

    revisit = first_pre_take_location_revisit(
        [str(event["action"]) for event in events]
    )
    return {
        "action_count": len(events),
        "repeated_action_feedback": repeated_pairs[: max(0, int(evidence_cap))],
        "repeated_action_cycles": cycles[: max(0, int(evidence_cap))],
        "pre_take_location_revisit": revisit,
        "interpretation": (
            "Deterministic candidate evidence only; the teacher must decide "
            "whether it caused the failure and whether graph knowledge was missing."
        ),
    }


def positive_is_locally_clean(node_id: str, example: dict[str, Any]) -> bool:
    """Accept a successful example only when it carries usable local evidence."""
    del node_id
    return bool(
        str(example.get("case_id") or "").strip()
        and (
            str(example.get("evidence_excerpt") or "").strip()
            or str(example.get("task_description") or "").strip()
        )
    )


def compact_failure_example(example: dict[str, Any], *, radius: int = 4) -> dict[str, Any]:
    """Keep task context plus the action window around the first obvious failure."""
    compact = {
        "case_id": str(example.get("case_id") or example.get("id") or ""),
        "task_type": str(example.get("task_type") or ""),
        "task_description": str(example.get("task_description") or ""),
        "fail_reason": str(example.get("fail_reason") or ""),
        "student_step_limit_failure": bool(
            example.get("student_step_limit_failure")
        ),
    }
    for key in (
        "question", "instruction", "response", "predicted_answer",
        "predicted_label", "predicted_text", "training_reference",
        "reference_answer", "gold_answer",
    ):
        value = example.get(key)
        if value is not None and value != "":
            compact[key] = value
    graph_refs = example.get("graph_refs")
    if isinstance(graph_refs, dict):
        compact["graph_refs"] = dict(graph_refs)
    trace_evidence = example.get("trace_evidence")
    if isinstance(trace_evidence, dict):
        compact["trace_evidence"] = dict(trace_evidence)
    semantic_trace = str(example.get("semantic_reasoning_trace") or "").strip()
    if semantic_trace:
        compact["semantic_reasoning_trace"] = semantic_trace
    compact["semantic_reasoning_trace_status"] = str(
        example.get("semantic_reasoning_trace_status") or "not_requested"
    )
    compact["semantic_reasoning_trace_required"] = bool(
        example.get("semantic_reasoning_trace_required")
    )
    reference = example.get("training_reference_plan")
    if isinstance(reference, dict):
        compact["training_reference_plan"] = dict(reference)
    trajectory = str(example.get("trajectory") or "")
    if compact["student_step_limit_failure"] or trajectory:
        compact["loop_diagnostics"] = diagnose_no_progress_loops(trajectory)
    actions = [str(x) for x in (example.get("action_path") or []) if str(x).strip()]
    if not actions:
        actions = extract_actions(str(example.get("trajectory") or ""))
    focus_step = example.get("focus_step")
    if (
        isinstance(focus_step, int)
        and not isinstance(focus_step, bool)
        and 0 <= focus_step < len(actions)
    ):
        event = {
            "action_index": focus_step,
            "action": actions[focus_step],
            "kind": "teacher_grounded_first_wrong_step",
        }
    else:
        event = first_pre_take_location_revisit(actions)
    if event is None:
        start = max(0, len(actions) - (2 * radius + 1))
        end = len(actions)
    else:
        center = int(event["action_index"])
        start = max(0, center - radius)
        end = min(len(actions), center + radius + 1)
    compact["action_path"] = actions[start:end]
    compact["action_window_start"] = start
    compact["heuristic_event"] = event
    return compact


def _round_robin_by_task(
    examples: list[dict[str, Any]], cap: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[str(example.get("task_type") or "unknown")].append(example)
    for values in groups.values():
        values.sort(key=lambda x: (int(x.get("n_turns") or 10**9), str(x.get("case_id") or "")))
    chosen: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys and len(chosen) < cap:
        next_keys: list[str] = []
        for key in keys:
            values = groups[key]
            if values and len(chosen) < cap:
                chosen.append(values.pop(0))
            if values:
                next_keys.append(key)
        keys = next_keys
    return chosen


def select_positive_examples(
    node_id: str,
    examples: list[dict[str, Any]],
    failure_examples: list[dict[str, Any]],
    *,
    cap: int = 12,
) -> list[dict[str, Any]]:
    """Select bounded, clean and task-relevant rightcase counter-evidence."""
    cap = max(0, int(cap))
    if cap == 0:
        return []
    failure_types = {
        str(x.get("task_type") or "") for x in failure_examples
        if str(x.get("task_type") or "")
    }
    deduped: dict[str, dict[str, Any]] = {}
    for raw in examples:
        example = dict(raw)
        case_id = str(example.get("case_id") or "")
        if case_id and positive_is_locally_clean(node_id, example):
            deduped.setdefault(case_id, example)
    values = list(deduped.values())
    matched = [x for x in values if str(x.get("task_type") or "") in failure_types]
    other = [x for x in values if x not in matched]
    selected = _round_robin_by_task(matched, cap)
    if len(selected) < cap:
        selected.extend(_round_robin_by_task(other, cap - len(selected)))
    return selected


def select_failure_examples(
    examples: list[dict[str, Any]], *, cap: int = 8
) -> list[dict[str, Any]]:
    """Bound badcase context while retaining task diversity and action evidence."""
    cap = max(0, int(cap))
    deduped: dict[str, dict[str, Any]] = {}
    for raw in examples:
        compact = compact_failure_example(dict(raw))
        case_id = str(compact.get("case_id") or "")
        if case_id:
            deduped.setdefault(case_id, compact)
    values = list(deduped.values())
    for item in values:
        item["n_turns"] = len(item.get("action_path") or [])
    return _round_robin_by_task(values, cap)
