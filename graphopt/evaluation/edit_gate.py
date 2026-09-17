"""Shared, measurable per-atomic-edit audit for small and big Gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphopt.evaluation.graph_usage import extract_graph_refs, simulate_edit_touch_keys
from graphopt.optimizer.skill import atomic_edit_groups
from graphopt.types import GraphEdit, SkillGraph


def _transition(before: dict[str, Any], after: dict[str, Any]) -> str:
    old = float(before.get("hard") or 0.0)
    new = float(after.get("hard") or 0.0)
    if old < 1.0 <= new:
        return "0->1"
    if new < 1.0 <= old:
        return "1->0"
    return "1->1" if old >= 1.0 else "0->0"


def _summary(case_ids, before, after) -> dict[str, Any]:
    transitions = {key: [] for key in ("0->1", "1->0", "0->0", "1->1")}
    changed_pairs = []
    for case_id in sorted(case_ids):
        kind = _transition(before[case_id], after[case_id])
        transitions[kind].append(case_id)
        if kind in {"0->1", "1->0"}:
            changed_pairs.append({
                "case_id": case_id, "transition": kind,
                "before": before[case_id], "after": after[case_id],
            })
    n_improved = len(transitions["0->1"])
    n_regressed = len(transitions["1->0"])
    return {
        "case_ids": sorted(case_ids), "transitions": transitions,
        "n_improved": n_improved, "n_regressed": n_regressed,
        "net": n_improved - n_regressed, "changed_pairs": changed_pairs,
    }


def _touches(row: dict[str, Any], touch_keys: set[str]) -> bool:
    nodes, edges = extract_graph_refs(row)
    for key in touch_keys:
        if key.startswith("node:") and key[5:] in nodes:
            return True
        if key.startswith("edge_id:") and key[8:] in edges:
            return True
    return False


def _unknown_usage_case_ids(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]],
) -> set[str]:
    """Return static cases whose usage sidecar is missing or malformed.

    Unknown usage is not an automatic rollback. The caller expands every
    edit scope to the affected fixed group so its real hard transition is
    still counted conservatively.
    """
    unknown = set()
    for case_id in before:
        for row in (before[case_id], after[case_id]):
            refs = row.get("graph_refs")
            if isinstance(refs, dict):
                status = str(refs.get("status") or "")
                if status.startswith(("missing_", "invalid_")):
                    unknown.add(case_id)
                    break
    return unknown


def build_atomic_edit_audits(
    *, grouped_batch: Any, base_graph: SkillGraph, edits: list[GraphEdit],
    reference_val: list[dict[str, Any]], candidate_val: list[dict[str, Any]],
    reference_train: list[dict[str, Any]], candidate_train: list[dict[str, Any]],
    allow_validation_tie: bool = False,
    require_nonnegative_combined_on_validation_gain: bool = False,
) -> list[dict[str, Any]]:
    """Measure every atomic edit on its reported-usage and fixed-group scope.

    Direct source IDs locate a fixed group as a provenance fallback; they do
    not themselves cast a Gate vote. Ungrouped schedules use the whole update
    batch as the protection group.
    """
    def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        ids = [str(row.get("id")) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"per-edit audit requires unique {label} case IDs")
        return dict(zip(ids, rows))

    val_before = _index(reference_val, "reference validation")
    val_after = _index(candidate_val, "candidate validation")
    train_before = _index(reference_train, "reference train")
    train_after = _index(candidate_train, "candidate train")
    if set(val_before) != set(val_after) or set(train_before) != set(train_after):
        raise ValueError("per-edit audit requires identical before/after case IDs")

    touch_sets = simulate_edit_touch_keys(base_graph, edits)
    unknown_val = _unknown_usage_case_ids(val_before, val_after)
    unknown_train = _unknown_usage_case_ids(train_before, train_after)
    complete = not unknown_val and not unknown_train
    audits = []
    for group_id, edit_indices in atomic_edit_groups(edits):
        touch_keys = set().union(*(touch_sets[index] for index in edit_indices))
        used_val = {case_id for case_id in val_before if (
            _touches(val_before[case_id], touch_keys)
            or _touches(val_after[case_id], touch_keys)
        )}
        used_train = {case_id for case_id in train_before if (
            _touches(train_before[case_id], touch_keys)
            or _touches(train_after[case_id], touch_keys)
        )}
        direct_sources = {
            str(case_id) for index in edit_indices
            for case_id in edits[index].source_case_ids if str(case_id)
        }
        trace_source_ids = {
            str(evidence.get("case_id") or "")
            for index in edit_indices
            for evidence in edits[index].evidence_items
            if isinstance(evidence, dict)
            and isinstance(evidence.get("trace_evidence"), dict)
            and bool((evidence.get("trace_evidence") or {}).get("verified"))
            and str(evidence.get("case_id") or "")
        } & direct_sources
        related_val = set(used_val)
        related_train = set(used_train)
        related_group_indices = []
        if grouped_batch is not None:
            for position, fixed_group in enumerate(grouped_batch.groups):
                group_train = set(map(str, fixed_group.train_ids))
                group_val = set(map(str, fixed_group.val_ids))
                # Missing usage means "possibly used any edit". Include
                # its fixed group for every edit and let real hard outcomes vote.
                if ((used_train | direct_sources | unknown_train) & group_train
                    or (used_val | unknown_val) & group_val):
                    related_group_indices.append(position)
                    related_train.update(group_train)
                    related_val.update(group_val)
        elif used_val or used_train or direct_sources or unknown_val or unknown_train:
            related_train.update(train_before)
            related_val.update(unknown_val)

        related_val.intersection_update(val_before)
        related_train.intersection_update(train_before)
        validation = _summary(related_val, val_before, val_after)
        train = _summary(related_train, train_before, train_after)
        trace_source_train = trace_source_ids & set(train_before)
        trace_source_unobserved = trace_source_ids - set(train_before)
        trace_source_summary = _summary(
            trace_source_train, train_before, train_after
        )
        trace_source_observed_failures = {
            case_id for case_id in trace_source_train
            if float(train_before[case_id].get("hard") or 0.0) < 1.0
        }
        protected_validation_regressions = sorted(
            case_id for case_id in related_val
            if float(val_before[case_id].get("hard") or 0.0) >= 1.0
            and _transition(val_before[case_id], val_after[case_id]) == "1->0"
        )
        protected_train_regressions = sorted(
            case_id for case_id in related_train - direct_sources
            if float(train_before[case_id].get("hard") or 0.0) >= 1.0
            and _transition(train_before[case_id], train_after[case_id]) == "1->0"
        )
        trace_source_repaired = bool(
            not trace_source_ids
            or (
                trace_source_train
                and (
                    trace_source_summary["n_improved"] > 0
                    if trace_source_observed_failures
                    else validation["net"] > 0 or train["net"] > 0
                )
            )
            or (not train_before and validation["net"] > 0)
        )
        # Individual 1->0 transitions are exposed to the teacher veto but
        # cannot be a deterministic hard stop: repeated runs of the unchanged
        # graph can flip individual cases. Net paired outcomes remain the script
        # authority, while the verified source repair is still mandatory.
        trace_guard_passed = bool(
            not trace_source_ids or trace_source_repaired
        )
        has_scope = bool(related_val or related_train)
        combined_protected = bool(
            not require_nonnegative_combined_on_validation_gain
            or not train_before
            or validation["net"] + train["net"] >= 0
        )
        coverage_complete = bool(
            unknown_val <= related_val and unknown_train <= related_train
        )
        script_keep = bool(
            coverage_complete and has_scope and trace_guard_passed and (
                (validation["net"] > 0 and combined_protected)
                or (
                    allow_validation_tie and validation["net"] == 0
                    and combined_protected
                )
                or (validation["net"] == 0 and train["net"] > 0)
            )
        )
        if not coverage_complete:
            script_decision = "ROLLBACK_UNCOVERED_UNKNOWN_USAGE"
        elif trace_source_unobserved:
            script_decision = "ROLLBACK_TRACE_SOURCE_UNOBSERVED"
        elif not trace_guard_passed:
            script_decision = "ROLLBACK_TRACE_SOURCE_NOT_REPAIRED"
        elif not has_scope:
            script_decision = "ROLLBACK_NO_REPORTED_USAGE_OR_GROUP"
        elif validation["net"] >= 0 and not combined_protected:
            script_decision = "ROLLBACK_COMBINED_NET_REGRESSION"
        elif validation["net"] > 0:
            script_decision = "KEEP_VALIDATION_NET_GAIN"
        elif allow_validation_tie and validation["net"] == 0:
            script_decision = "KEEP_VALIDATION_HARD_TIE"
        elif validation["net"] == 0 and train["net"] > 0:
            script_decision = "KEEP_VALIDATION_TIE_TRAIN_NET_GAIN"
        else:
            script_decision = "ROLLBACK_NET_NOT_POSITIVE"
        audits.append({
            "schema_version": "graphopt-atomic-edit-audit-v1",
            "atomic_group_id": str(group_id),
            "edit_indices": list(edit_indices),
            "edits": [edits[index].to_dict() for index in edit_indices],
            "touch_keys": sorted(touch_keys),
            "direct_source_train_case_ids": sorted(direct_sources),
            "trace_verified_source_case_ids": sorted(trace_source_ids),
            "trace_source_unobserved_case_ids": sorted(trace_source_unobserved),
            "trace_source_observation_complete": not trace_source_unobserved,
            "trace_source_train_audit": trace_source_summary,
            "trace_source_observed_failure_case_ids": sorted(
                trace_source_observed_failures
            ),
            "trace_source_repaired": trace_source_repaired,
            "protected_validation_regression_case_ids": (
                protected_validation_regressions
            ),
            "protected_train_regression_case_ids": protected_train_regressions,
            "trace_guard_passed": trace_guard_passed,
            "used_validation_case_ids": sorted(used_val),
            "used_train_case_ids": sorted(used_train),
            "related_group_indices": related_group_indices,
            "related_validation_case_ids": sorted(related_val),
            "related_train_case_ids": sorted(related_train),
            "validation": validation, "train": train,
            "unknown_validation_case_ids": sorted(unknown_val),
            "unknown_train_case_ids": sorted(unknown_train),
            "usage_complete": complete,
            "scope_coverage_complete": coverage_complete,
            "script_keep": script_keep,
            "script_decision": script_decision,
        })
    return audits


def review_atomic_edits(
    audits: list[dict[str, Any]], *, chat_fn: Any,
    prompt_path: Path, stage: str,
) -> tuple[set[str], dict[str, Any]]:
    """Default eligible edits to KEEP; accept only grounded teacher vetoes."""
    eligible = {row["atomic_group_id"] for row in audits if row["script_keep"]}
    artifact = {
        "schema_version": "graphopt-per-edit-audit-artifact-v1",
        "decision_policy": "script_eligible_default_keep_high_confidence_grounded_veto_only",
        "invalid_twice_policy": "fall_back_to_deterministic_script_decision",
        "audits": audits, "reviews": [],
    }
    if chat_fn is None:
        artifact.update({
            "status": "script_only_no_teacher",
            "kept_group_ids": sorted(eligible),
        })
        return eligible, artifact

    from graphopt.json_utils import extract_json
    system = prompt_path.read_text(encoding="utf-8")
    kept = set()
    required = {
        "schema_version", "atomic_group_id", "decision", "veto_basis",
        "confidence", "reason", "supporting_case_ids",
    }
    for row in audits:
        payload = json.dumps({
            "schema_version": "graphopt-per-edit-teacher-audit-input-v1",
            "atomic_edit_audit": row,
        }, ensure_ascii=False, indent=2)
        user = payload
        verdict = None
        attempts = []
        valid_case_ids = set(row["related_validation_case_ids"]) | set(row["related_train_case_ids"])
        changed_case_ids = {
            pair["case_id"] for split in (row["validation"], row["train"])
            for pair in split["changed_pairs"]
        }
        regressed_case_ids = {
            pair["case_id"] for split in (row["validation"], row["train"])
            for pair in split["changed_pairs"] if pair["transition"] == "1->0"
        }
        for attempt in range(1, 3):
            response = ""
            usage = None
            try:
                response, usage = chat_fn(
                    system=system, user=user, max_completion_tokens=2048,
                    retries=3, stage=f"{stage}_{row['atomic_group_id']}",
                )
                obj = extract_json(response)
                if not isinstance(obj, dict) or set(obj) != required:
                    raise ValueError("output must contain exactly the required keys")
                if obj["schema_version"] != "graphopt-per-edit-teacher-audit-v2" or obj["atomic_group_id"] != row["atomic_group_id"]:
                    raise ValueError("schema_version or atomic_group_id mismatch")
                if obj["decision"] not in {"KEEP", "ROLLBACK"}:
                    raise ValueError("decision must be KEEP or ROLLBACK")
                if obj["veto_basis"] not in {
                    "NONE", "SCRIPT_INELIGIBLE", "DIRECT_CAUSAL_REGRESSION",
                    "EXPLICIT_SEMANTIC_CONTRADICTION",
                }:
                    raise ValueError("invalid veto_basis")
                if obj["confidence"] not in {"NOT_APPLICABLE", "HIGH"}:
                    raise ValueError("confidence must be NOT_APPLICABLE or HIGH")
                if not isinstance(obj["reason"], str) or not obj["reason"].strip():
                    raise ValueError("reason must be a non-empty string")
                supporting = obj["supporting_case_ids"]
                if (not isinstance(supporting, list)
                    or any(not isinstance(case_id, str) for case_id in supporting)
                    or len(supporting) != len(set(supporting))
                    or not set(supporting) <= valid_case_ids):
                    raise ValueError("supporting_case_ids must be unique related IDs")
                if changed_case_ids and not supporting:
                    raise ValueError("changed evidence requires at least one supporting case")
                if supporting and not set(supporting) <= changed_case_ids:
                    raise ValueError("supporting_case_ids must identify 0->1 or 1->0 cases")
                if obj["decision"] == "KEEP" and not row["script_keep"]:
                    raise ValueError("teacher cannot rescue a script-ineligible edit")
                if obj["decision"] == "KEEP" and (
                    obj["veto_basis"] != "NONE"
                    or obj["confidence"] != "NOT_APPLICABLE"
                ):
                    raise ValueError("KEEP requires veto_basis=NONE and confidence=NOT_APPLICABLE")
                if obj["decision"] == "ROLLBACK" and not row["script_keep"] and (
                    obj["veto_basis"] != "SCRIPT_INELIGIBLE"
                    or obj["confidence"] != "NOT_APPLICABLE"
                ):
                    raise ValueError(
                        "script-ineligible ROLLBACK requires "
                        "veto_basis=SCRIPT_INELIGIBLE and confidence=NOT_APPLICABLE"
                    )
                if obj["decision"] == "ROLLBACK" and row["script_keep"]:
                    if obj["confidence"] != "HIGH" or obj["veto_basis"] not in {
                        "DIRECT_CAUSAL_REGRESSION",
                        "EXPLICIT_SEMANTIC_CONTRADICTION",
                    }:
                        raise ValueError("eligible edit veto requires a high-confidence evidence basis")
                    if (
                        obj["veto_basis"] == "DIRECT_CAUSAL_REGRESSION"
                        and not set(supporting).intersection(regressed_case_ids)
                    ):
                        raise ValueError("causal-regression veto must cite at least one 1->0 case")
                verdict = obj
                attempts.append({"attempt": attempt, "response": response, "usage": usage, "error": None})
                break
            except Exception as exc:
                attempts.append({"attempt": attempt, "response": response or None, "usage": usage, "error": str(exc)})
                user = (
                    payload
                    + "\n\nYour previous output was invalid: "
                    + str(exc)
                    + "\nReturn the complete strict JSON object again."
                )
        if verdict is None:
            fallback_keep = bool(row["script_keep"])
            verdict = {
                "schema_version": "graphopt-per-edit-teacher-audit-v2",
                "atomic_group_id": row["atomic_group_id"],
                "decision": "KEEP" if fallback_keep else "ROLLBACK",
                "veto_basis": "NONE" if fallback_keep else "SCRIPT_INELIGIBLE",
                "confidence": "NOT_APPLICABLE",
                "reason": "invalid twice; fall back to the deterministic script decision",
                "supporting_case_ids": sorted(changed_case_ids)[:1],
            }
        artifact["reviews"].append({
            "atomic_group_id": row["atomic_group_id"],
            "attempts": attempts, "verdict": verdict,
        })
        if verdict["decision"] == "KEEP" and row["script_keep"]:
            kept.add(row["atomic_group_id"])
    artifact.update({"status": "complete", "kept_group_ids": sorted(kept)})
    return kept, artifact
