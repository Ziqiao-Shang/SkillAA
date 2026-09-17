"""Compatibility helpers for lossless GraphOpt Meta evidence transfer.

The active entry points at the bottom return defensive copies of complete Gate
records. Questions, responses, trajectories, references, before/after states,
teacher explanations, and edit evidence are never summarized, truncated,
removed, or rewritten. The downstream model transport may list an exactly repeated JSON value once and refer to it by a validated SHA-256 ID; decoding reconstructs these complete records. Legacy private helpers remain checkpoint-only.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any


META_LEDGER_SCHEMA = "graphopt-meta-decision-ledger-v1"
META_LEDGER_MAX_CHARS = 500_000
PRIOR_GATE_LEDGER_MAX_CHARS = 350_000

_OMIT = object()
_RAW_EVIDENCE_KEYS = frozenset({
    "before",
    "after",
    "paired_cases",
    "changed_pairs",
    "evidence_items",
    "question",
    "task_description",
    "response",
    "trajectory",
    "training_reference",
    "conversation",
    "target_system_prompt",
    "target_user_prompt",
    "system",
    "user",
    "attempts",
    "original_task",
    "student_output",
})

_STATE_FIELDS = (
    "id",
    "hard",
    "soft",
    "task_type",
    "predicted_answer",
    "predicted_label",
    "correct_answer",
    "correct_label",
    "fail_reason",
    "semantic_reasoning_trace_status",
    "exclude_from_metrics",
    "metric_exclusion_reason",
)

_ROOT_FIELDS = (
    "step",
    "epoch",
    "action",
    "accepted",
    "big_gate_policy",
    "train_gate_passed",
    "full_candidate_gate_passed",
    "update_protocol",
    "update_boundary",
    "n_edits",
    "n_edits_kept",
    "n_edits_rolled_back",
    "kept_edit_indices",
    "rolled_back_edit_indices",
    "n_evidence_cases",
    "n_small_gates",
    "n_small_gates_accepted",
    "pending_bad_case_prompt_set",
    "reference_train_reused",
    "reference_train_source",
    "test_used_for_decision",
)

_AUDIT_FIELDS = (
    "n_eligible",
    "reference_hard_successes",
    "candidate_hard_successes",
    "hard_net_case_gain",
    "n_improved",
    "n_regressed",
    "improved_case_ids",
    "regressed_case_ids",
    "accept_if",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _semantic_value(value: Any, *, key: str = "") -> Any:
    """Copy semantic content verbatim and omit only named raw evidence bodies."""
    if key in _RAW_EVIDENCE_KEYS:
        return _OMIT
    if isinstance(value, dict):
        rendered: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child = _semantic_value(child_value, key=str(child_key))
            if child is not _OMIT:
                rendered[str(child_key)] = child
        return rendered
    if isinstance(value, list):
        rendered_list: list[Any] = []
        for child_value in value:
            child = _semantic_value(child_value)
            if child is not _OMIT:
                rendered_list.append(child)
        return rendered_list
    return value


def _evidence_reference(value: Any) -> dict[str, Any]:
    items = list(value) if isinstance(value, list) else []
    case_ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or item.get("id") or "")
        if case_id and case_id not in case_ids:
            case_ids.append(case_id)
    return {
        "item_count": len(items),
        "case_ids": case_ids,
        "raw_evidence_sha256": _sha256(value),
        "raw_evidence_location": "unchanged Gate artifacts under outputs",
    }


def compact_edit(edit: Any) -> dict[str, Any]:
    """Preserve every edit semantic field; replace only raw evidence items."""
    if not isinstance(edit, dict):
        return {}
    result = {
        str(key): value
        for key, value in edit.items()
        if str(key) != "evidence_items"
    }
    if "evidence_items" in edit:
        result["evidence_reference"] = _evidence_reference(edit.get("evidence_items"))
    semantic_without_raw = {
        str(key): value
        for key, value in edit.items()
        if str(key) != "evidence_items"
    }
    result["semantic_sha256"] = _sha256(semantic_without_raw)
    return result


def _compact_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        key: value[key]
        for key in _STATE_FIELDS
        if value.get(key) not in (None, "", [], {})
    }
    refs = value.get("graph_refs")
    if isinstance(refs, dict):
        result["graph_usage"] = {
            key: refs[key]
            for key in ("status", "used_nodes", "used_edges", "nodes", "edges")
            if refs.get(key) not in (None, "", [], {})
        }
    return result


def _compact_pair(pair: Any) -> dict[str, Any]:
    if not isinstance(pair, dict):
        return {}
    result = {
        key: pair[key]
        for key in (
            "case_id",
            "id",
            "category",
            "transition",
            "hard_before",
            "hard_after",
            "soft_before",
            "soft_after",
        )
        if pair.get(key) not in (None, "", [], {})
    }
    # Stable/persistent rows are represented by ID, category, and transition.
    # Changed/source rows additionally retain a readable answer/usage card.
    category = str(pair.get("category") or pair.get("transition") or "")
    if category in {"effective", "harmful", "unresolved_source", "0->1", "1->0"}:
        before = _compact_state(pair.get("before"))
        after = _compact_state(pair.get("after"))
        if before:
            result["before_state"] = before
        if after:
            result["after_state"] = after
    result["raw_pair_sha256"] = _sha256(pair)
    return result


def _compact_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        key: value[key]
        for key in _AUDIT_FIELDS
        if value.get(key) not in (None, "", [], {})
    }
    transitions = value.get("transitions")
    if isinstance(transitions, dict):
        result["improved_case_ids"] = list(transitions.get("0->1") or [])
        result["regressed_case_ids"] = list(transitions.get("1->0") or [])
        result["persistent_fail_count"] = len(transitions.get("0->0") or [])
        result["stable_success_count"] = len(transitions.get("1->1") or [])
    return result


def _compact_patch(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "reasoning": value.get("reasoning") or "",
        "edits": [compact_edit(edit) for edit in value.get("edits") or []],
    }


def _compact_local_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, child in value.items():
        if key == "paired_cases":
            result["paired_case_evidence_reference"] = {
                **_evidence_reference(child),
                "raw_pair_sha256": _sha256(child),
            }
        elif key == "semantic_explanation":
            rendered = _compact_semantic_explanation(child)
            if rendered is not _OMIT:
                result[key] = rendered
        elif key == "per_edit_explanations":
            # joint_patch owns edit semantics; local_gate owns group-wide IDs.
            result[key] = [
                _compact_per_edit_explanation(row)
                for row in child or []
                if isinstance(row, dict)
            ]
        elif key not in _RAW_EVIDENCE_KEYS:
            rendered = _semantic_value(child, key=key)
            if rendered is not _OMIT:
                result[key] = rendered
    return result


_REPEATED_LOCAL_CASE_ID_KEYS = frozenset({
    "effective_case_ids",
    "harmful_case_ids",
    "ineffective_case_ids",
    "unresolved_source_case_ids",
    "source_case_ids",
    "persistent_failure_case_ids",
    "protected_success_case_ids",
    "affected_case_ids",
})


def _compact_per_edit_explanation(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        str(key): child
        for key, child in value.items()
        if key not in {"edit", "semantic_explanation"}
        and key not in _RAW_EVIDENCE_KEYS
        and key not in _REPEATED_LOCAL_CASE_ID_KEYS
    }
    for key in _REPEATED_LOCAL_CASE_ID_KEYS:
        case_ids = value.get(key)
        if isinstance(case_ids, list):
            result[key.replace("_case_ids", "_case_count")] = len(case_ids)
    return result


def _compact_semantic_explanation(value: Any) -> Any:
    """Keep causal prose while removing copies of canonical edits and case lists."""
    if isinstance(value, dict):
        rendered: dict[str, Any] = {}
        for key, child in value.items():
            if key == "edit" or key in _REPEATED_LOCAL_CASE_ID_KEYS:
                continue
            compact = _compact_semantic_explanation(child)
            if compact is not _OMIT:
                rendered[str(key)] = compact
        return rendered
    if isinstance(value, list):
        return [_compact_semantic_explanation(child) for child in value]
    return _semantic_value(value)


def _compact_small_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    scalar_keys = (
        "epoch",
        "joint_candidate_index",
        "attempt",
        "accepted",
        "action",
        "patch_sha256",
        "n_edits",
        "n_evidence_cases",
        "train_size",
        "validation_size",
        "train_case_ids",
        "validation_case_ids",
        "precise_refinement_triggered",
    )
    result = {
        key: value[key]
        for key in scalar_keys
        if value.get(key) not in (None, "", [], {})
    }
    if isinstance(value.get("affected_scope"), dict):
        result["affected_scope"] = _semantic_value(value["affected_scope"])
    result["joint_patch"] = _compact_patch(value.get("joint_patch"))
    result["local_gate"] = _compact_local_gate(value.get("local_gate"))
    return result


def _compact_attribution(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "atomic_group_id",
        "edit_indices",
        "touch_keys",
        "direct_source_train_case_ids",
        "script_keep",
        "script_decision",
        "usage_complete",
        "scope_coverage_complete",
    ):
        if value.get(key) not in (None, "", [], {}):
            result[key] = value[key]
    result["train_outcome"] = _compact_audit(value.get("train"))
    result["validation_outcome"] = _compact_audit(value.get("validation"))
    result["edits"] = []
    for row in value.get("edits") or []:
        if not isinstance(row, dict):
            continue
        item = {
            key: row[key]
            for key in (
                "edit_index", "joint_group_id", "decision", "reason",
                "effective_case_ids", "ineffective_case_ids",
                "individual_isolation", "why_not_isolated",
            )
            if row.get(key) not in (None, "", [], {})
        }
        # Bind final attribution to the canonical Local-Gate edit without
        # repeating the same rule text and evidence.
        item["edit_semantic_sha256"] = _sha256(
            _edit_semantics_without_evidence(row.get("edit"))
        )
        result["edits"].append(item)
    return result


def _compact_record(record: dict[str, Any], *, environment: str) -> dict[str, Any]:
    result = {
        key: record[key]
        for key in _ROOT_FIELDS
        if record.get(key) not in (None, "", [], {})
    }
    result.update({
        "schema_version": META_LEDGER_SCHEMA,
        "environment": environment,
        "meta_history_view": "human_readable_semantic_decision_ledger",
        "reading_guide": {
            "purpose": "Summarize measured graph-edit lessons for the next epoch.",
            "semantic_contract": (
                "Rule text, edit reasons, Gate decisions, transition case IDs, and "
                "teacher semantic explanations are copied verbatim."
            ),
            "evidence_contract": (
                "Repeated full questions, responses, trajectories, and before/after "
                "objects remain unchanged on disk and are represented here by readable "
                "case evidence plus SHA-256 references."
            ),
            "decision_rule": "Use measured outcomes; do not infer facts from hashes.",
        },
        "raw_gate_artifact": {
            "path": "gate_history.jsonl",
            "epoch": record.get("epoch"),
            "record_sha256": _sha256(record),
        },
        "full_train_candidate_outcome": _compact_audit(
            record.get("candidate_epoch_train_audit")
        ),
        "committed_full_train_outcome": _compact_audit(
            record.get("final_epoch_train_audit")
        ),
        "update_pool_diagnostic": _compact_audit(
            record.get("update_pool_diagnostic_audit")
        ),
        "joint_local_gates": [
            _compact_small_gate(gate) for gate in record.get("small_gates") or []
        ],
        "final_edit_attribution": [
            _compact_attribution(row) for row in record.get("edit_attribution") or []
        ],
    })
    if record.get("bad_case_summary") not in (None, "", [], {}):
        result["big_gate_failure_summary"] = _semantic_value(
            record.get("bad_case_summary"), key="bad_case_summary"
        )
    return result


def _edit_semantics_without_evidence(edit: Any) -> dict[str, Any]:
    if not isinstance(edit, dict):
        return {}
    return {str(key): value for key, value in edit.items() if key != "evidence_items"}


def validate_meta_decision_ledger(
    raw_history: list[dict[str, Any]],
    compact_history: list[dict[str, Any]],
) -> None:
    """Fail closed if a decision, edit semantic, or transition ID changed."""
    if len(raw_history) != len(compact_history):
        raise ValueError("Meta ledger changed the Gate record count")
    for raw, compact in zip(raw_history, compact_history):
        if compact.get("schema_version") != META_LEDGER_SCHEMA:
            raise ValueError("Meta ledger schema mismatch")
        raw_ref = compact.get("raw_gate_artifact") or {}
        if raw_ref.get("record_sha256") != _sha256(raw):
            raise ValueError("Meta ledger raw Gate hash mismatch")
        for key in ("epoch", "action", "accepted"):
            if raw.get(key) != compact.get(key):
                raise ValueError(f"Meta ledger changed root decision field: {key}")

        raw_gates = list(raw.get("small_gates") or [])
        compact_gates = list(compact.get("joint_local_gates") or [])
        if len(raw_gates) != len(compact_gates):
            raise ValueError("Meta ledger changed Local-Gate count")
        for raw_gate, compact_gate in zip(raw_gates, compact_gates):
            for key in ("joint_candidate_index", "attempt", "accepted", "action"):
                if raw_gate.get(key) != compact_gate.get(key):
                    raise ValueError(f"Meta ledger changed Local-Gate field: {key}")
            raw_local = raw_gate.get("local_gate") or {}
            compact_local = compact_gate.get("local_gate") or {}
            for key in (
                "accepted",
                "effective_case_ids",
                "harmful_case_ids",
                "ineffective_case_ids",
                "unresolved_source_case_ids",
                "source_case_ids",
                "decision_reason",
            ):
                if raw_local.get(key) != compact_local.get(key):
                    raise ValueError(f"Meta ledger changed Local-Gate semantics: {key}")
            expected_explanation = _compact_semantic_explanation(
                raw_local.get("semantic_explanation")
            )
            if expected_explanation is _OMIT:
                expected_explanation = None
            if compact_local.get("semantic_explanation") != expected_explanation:
                raise ValueError("Meta ledger changed teacher semantic explanation")
            raw_edits = (raw_gate.get("joint_patch") or {}).get("edits") or []
            compact_edits = (compact_gate.get("joint_patch") or {}).get("edits") or []
            if len(raw_edits) != len(compact_edits):
                raise ValueError("Meta ledger changed joint edit count")
            for raw_edit, compact_edit_value in zip(raw_edits, compact_edits):
                expected = _edit_semantics_without_evidence(raw_edit)
                actual = {
                    key: value
                    for key, value in compact_edit_value.items()
                    if key not in {"evidence_reference", "semantic_sha256"}
                }
                if actual != expected or compact_edit_value.get("semantic_sha256") != _sha256(expected):
                    raise ValueError("Meta ledger changed joint edit semantics")

        raw_attr = list(raw.get("edit_attribution") or [])
        compact_attr = list(compact.get("final_edit_attribution") or [])
        if len(raw_attr) != len(compact_attr):
            raise ValueError("Meta ledger changed final attribution count")
        for raw_row, compact_row in zip(raw_attr, compact_attr):
            if list(raw_row.get("edit_indices") or []) != list(
                compact_row.get("edit_indices") or []
            ):
                raise ValueError("Meta ledger changed attribution edit indices")
            raw_attr_edits = list(raw_row.get("edits") or [])
            compact_attr_edits = list(compact_row.get("edits") or [])
            if len(raw_attr_edits) != len(compact_attr_edits):
                raise ValueError("Meta ledger changed attribution edit count")
            for raw_edit_row, compact_edit_row in zip(raw_attr_edits, compact_attr_edits):
                expected_hash = _sha256(
                    _edit_semantics_without_evidence(raw_edit_row.get("edit"))
                )
                if compact_edit_row.get("edit_semantic_sha256") != expected_hash:
                    raise ValueError("Meta ledger changed attribution edit binding")
            for raw_key, compact_key in (
                ("train", "train_outcome"),
                ("validation", "validation_outcome"),
            ):
                raw_transitions = (raw_row.get(raw_key) or {}).get("transitions") or {}
                compact_outcome = compact_row.get(compact_key) or {}
                if (
                    list(raw_transitions.get("0->1") or [])
                    != list(compact_outcome.get("improved_case_ids") or [])
                    or list(raw_transitions.get("1->0") or [])
                    != list(compact_outcome.get("regressed_case_ids") or [])
                    or len(raw_transitions.get("0->0") or [])
                    != int(compact_outcome.get("persistent_fail_count") or 0)
                    or len(raw_transitions.get("1->1") or [])
                    != int(compact_outcome.get("stable_success_count") or 0)
                ):
                    raise ValueError("Meta ledger changed transition semantics")

    rendered_chars = len(_canonical_json(compact_history))
    if rendered_chars > META_LEDGER_MAX_CHARS:
        raise ValueError(
            f"Meta semantic ledger exceeds safe prompt budget: "
            f"{rendered_chars} > {META_LEDGER_MAX_CHARS} chars"
        )


def validate_full_meta_gate_history(
    raw_history: list[dict[str, Any]],
    prepared_history: list[dict[str, Any]],
) -> None:
    """Fail closed unless every model-facing field equals the raw evidence."""
    if _canonical_json(prepared_history) != _canonical_json(raw_history):
        raise ValueError("model-facing Meta Gate history differs from raw evidence")


def compact_meta_decision_ledger(
    history: list[dict[str, Any]], *, environment: str = ""
) -> list[dict[str, Any]]:
    """Build and validate a bounded Gate decision ledger without raw rollout bodies."""
    prepared = [
        _compact_record(record, environment=environment)
        for record in history
    ]
    validate_meta_decision_ledger(history, prepared)
    return prepared


def compact_meta_gate_history(
    history: list[dict[str, Any]], *, environment: str = ""
) -> list[dict[str, Any]]:
    """Return complete Gate evidence without compression or field removal.

    The legacy function name is retained for compatibility. ``environment``
    is audit-only; model-facing content is an exact defensive copy.
    """
    del environment
    prepared = copy.deepcopy(history)
    validate_full_meta_gate_history(history, prepared)
    return prepared


def compact_prior_gate_experiences(
    experiences: list[dict[str, Any]], *, environment: str = ""
) -> dict[str, Any]:
    """Wrap complete prior Gate records without compressing their evidence."""
    records = copy.deepcopy(experiences)
    ledger = {
        "schema_version": "graphopt-prior-gate-full-evidence",
        "environment": environment,
        "evidence_policy": (
            "complete verbatim Gate evidence; no summarization, truncation, "
            "hash replacement, or evidence-field removal"
        ),
        "epochs": records,
    }
    if _canonical_json(records) != _canonical_json(experiences):
        raise ValueError("model-facing prior Gate experience differs from raw evidence")
    return ledger

    """Build the same readable semantics-first view for repeated case prompts."""
    records: list[dict[str, Any]] = []
    for experience in experiences:
        local_records: list[dict[str, Any]] = []
        for local in experience.get("local_gate_experiences") or []:
            local_records.append({
                key: (
                    _compact_patch(value)
                    if key == "joint_patch"
                    else _semantic_value(value, key=key)
                )
                for key, value in local.items()
                if key not in _RAW_EVIDENCE_KEYS
            })
        records.append({
            "epoch": experience.get("epoch"),
            "accepted": bool(experience.get("accepted")),
            "action": experience.get("action"),
            "big_gate_policy": experience.get("big_gate_policy"),
            "candidate_train_outcome": _compact_audit(
                experience.get("candidate_train_audit")
            ),
            "update_pool_diagnostic": _compact_audit(
                experience.get("update_pool_diagnostic_audit")
            ),
            "joint_local_gates": local_records,
            "raw_experience_sha256": _sha256(experience),
        })
    ledger = {
        "schema_version": "graphopt-prior-gate-decision-ledger-v1",
        "environment": environment,
        "reading_guide": (
            "Measured Gate lessons only. Rule-edit semantics and decision reasons are "
            "verbatim; repeated rollout bodies remain in outputs."
        ),
        "epochs": records,
    }
    validate_prior_gate_experience_ledger(experiences, ledger)
    return ledger


def validate_prior_gate_experience_ledger(
    experiences: list[dict[str, Any]], ledger: dict[str, Any]
) -> None:
    records = list(ledger.get("epochs") or [])
    if len(experiences) != len(records):
        raise ValueError("Prior Gate ledger changed the epoch count")
    for raw, compact in zip(experiences, records):
        for key in ("epoch", "action"):
            if raw.get(key) != compact.get(key):
                raise ValueError(f"Prior Gate ledger changed field: {key}")
        if bool(raw.get("accepted")) != compact.get("accepted"):
            raise ValueError("Prior Gate ledger changed acceptance")
        if compact.get("raw_experience_sha256") != _sha256(raw):
            raise ValueError("Prior Gate ledger raw experience hash mismatch")
        raw_locals = list(raw.get("local_gate_experiences") or [])
        compact_locals = list(compact.get("joint_local_gates") or [])
        if len(raw_locals) != len(compact_locals):
            raise ValueError("Prior Gate ledger changed Local-Gate count")
        for raw_local, compact_local in zip(raw_locals, compact_locals):
            for key in ("accepted", "action", "decision_reason"):
                if raw_local.get(key) != compact_local.get(key):
                    raise ValueError(f"Prior Gate ledger changed Local-Gate field: {key}")
            raw_patch = raw_local.get("joint_patch") or {}
            compact_patch = compact_local.get("joint_patch") or {}
            raw_edits = list(raw_patch.get("edits") or [])
            compact_edits = list(compact_patch.get("edits") or [])
            if len(raw_edits) != len(compact_edits):
                raise ValueError("Prior Gate ledger changed edit count")
            for raw_edit, compact_edit_value in zip(raw_edits, compact_edits):
                expected = _edit_semantics_without_evidence(raw_edit)
                actual = {
                    key: value for key, value in compact_edit_value.items()
                    if key not in {"evidence_reference", "semantic_sha256"}
                }
                if actual != expected:
                    raise ValueError("Prior Gate ledger changed edit semantics")
            expected_explanation = _semantic_value(
                raw_local.get("semantic_explanation"), key="semantic_explanation"
            )
            if expected_explanation is _OMIT:
                expected_explanation = None
            if compact_local.get("semantic_explanation") != expected_explanation:
                raise ValueError("Prior Gate ledger changed semantic explanation")
    rendered_chars = len(_canonical_json(ledger))
    if rendered_chars > PRIOR_GATE_LEDGER_MAX_CHARS:
        raise ValueError(
            f"Prior Gate semantic ledger exceeds safe repeated-prompt budget: "
            f"{rendered_chars} > {PRIOR_GATE_LEDGER_MAX_CHARS} chars"
        )


def summarize_longitudinal_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Readable lossless view of already-small longitudinal pair records."""
    categories = Counter(str(pair.get("category") or "unknown") for pair in pairs)
    return {
        "counts": dict(sorted(categories.items())),
        "pairs": list(pairs),
    }
