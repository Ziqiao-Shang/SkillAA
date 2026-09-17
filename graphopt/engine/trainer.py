"""GraphOpt trainer: SkillAA loop with SkillGraph G_t instead of skill document S_t.

Only the trainable state + edit space change. Rollout / splits / gate / meta /
optimizer model are configured by this package.
"""

from __future__ import annotations

import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib
import json
import os
import statistics
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from graphopt.debug.artifacts import (
    StepRecorder,
    archive_epoch_artifacts,
    save_json,
    save_rollout_artifact,
    write_epoch_versions_doc,
)
from graphopt.evaluation.gate import (
    GateDecision, decide, mark_step_limit_exclusions, normalize_results, score_results,
)
from graphopt.evaluation.task_metrics import canonical_task_type, compute_task_metrics
from graphopt.evaluation.edit_gate import build_atomic_edit_audits, review_atomic_edits
from graphopt.evolution import EvolutionCache, EvolutionConfig, run_evolution_step, save_skill_json
from graphopt.evolution.ablation import run_ablation_evolution
from graphopt.evolution.experience_quality import (
    extract_actions, positive_is_locally_clean,
)
from graphopt.evolution.types import EvolutionResult, GraphEditPlan
from graphopt.evolution.case_analyzer import badcase_analysis_protocol
from graphopt.engine.rollout_readout import (
    attach_graph_usage,
    attach_semantic_reasoning_traces,
    attach_trajectories,
)
from graphopt.evolution.trace_attribution import attach_trace_evidence
from graphopt.engine.grouped_schedule import (
    GroupedBatch,
    TrainValidationGroup,
    build_grouped_schedule,
    resolve_group_batch_sizes,
    validate_group_batch_size,
)
from graphopt.gradient.reflect import format_graph
from graphopt.optimizer.meta_skill import format_meta, update_meta
from graphopt.optimizer.semantic_convert import graph_to_prompt, prompt_node_ids
from graphopt.optimizer.skill import apply_patch, atomic_edit_groups, load_graph
from graphopt.runtime import resolve_steps
from graphopt.types import GraphEdit, GraphPatch, SkillGraph, normalize_edge_type

GATE_REPEATS = 1
TERMINAL_LOCAL_GATE_EXPLANATION_STATUSES = frozenset({
    "complete",
    "deterministic_only_no_teacher",
    "invalid_twice_deterministic_fallback",
})
LEGACY_G0_ONLY_PROTOCOL_SEMANTICS_VERSION = (
    "case-complete-800-update-train-validation-big-gate-3x-epoch-test-final-v1"
)
UPDATE_PROTOCOL_SEMANTICS_VERSION = (
    "case-complete-update-pool-no-validation-big-gate-3x-epoch-test-final-v2"
)
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def graph_sha256(graph: SkillGraph) -> str:
    """Stable hash for graph identity and component-result reuse."""
    payload = json.dumps(
        graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_patch_sha256(patch: GraphPatch) -> str:
    """Stable Local-Gate identity, invariant to opaque group-id relabeling."""
    record = GraphPatch.from_dict(copy.deepcopy(patch.to_dict())).to_dict()
    canonical_groups: dict[str, str] = {}
    for edit in record.get("edits") or []:
        group_id = str(edit.get("group_id") or "")
        if not group_id:
            edit["group_id"] = None
            continue
        if group_id not in canonical_groups:
            canonical_groups[group_id] = f"group-{len(canonical_groups) + 1}"
        edit["group_id"] = canonical_groups[group_id]
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legacy_graph_patch_sha256(patch: GraphPatch) -> str:
    payload = json.dumps(
        patch.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reusable_local_gate_candidate_dir(
    update_dir: Path,
    *,
    candidate_index: int,
    patch: GraphPatch,
) -> Path:
    """Find a completed legacy directory for the same semantic joint patch."""
    digest = graph_patch_sha256(patch)
    root = update_dir / "local_gates"
    preferred = root / f"joint_{candidate_index:03d}_{digest[:12]}"
    for candidate in sorted(root.glob(f"joint_{candidate_index:03d}_*")):
        for tested_path in sorted(candidate.glob("attempt_*/tested_patch.json")):
            if not (
                tested_path.with_name("local_gate.json").is_file()
                and tested_path.with_name("reference_reuse.json").is_file()
            ):
                continue
            try:
                tested = GraphPatch.from_dict(json.loads(
                    tested_path.read_text(encoding="utf-8")
                ))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if graph_patch_sha256(tested) == digest:
                return candidate
    return preferred




def _load_reusable_epoch_local_gate_attempt(
    attempt_dir: Path,
    *,
    patch: GraphPatch,
    patch_digest: str,
    base_graph: SkillGraph,
    affected: dict[str, Any],
) -> dict[str, Any] | None:
    """Reuse a completed exact Local Gate attempt, never a partial or stale one."""
    gate_path = attempt_dir / "local_gate.json"
    tested_path = attempt_dir / "tested_patch.json"
    reference_path = attempt_dir / "reference_reuse.json"
    if not all(path.is_file() for path in (gate_path, tested_path, reference_path)):
        return None
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        tested_record = json.loads(tested_path.read_text(encoding="utf-8"))
        tested = GraphPatch.from_dict(tested_record)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    train_ids = list(map(str, affected.get("train_case_ids") or []))
    val_ids = list(map(str, affected.get("validation_case_ids") or []))
    stored_patch_digest = str(reference.get("patch_sha256") or "")
    tested_record_digest = hashlib.sha256(json.dumps(
        tested_record, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    recorded_rows = [
        row for row in (gate.get("per_edit_explanations") or [])
        if isinstance(row, dict) and isinstance(row.get("edit"), dict)
    ]
    recorded_rows.sort(key=lambda row: int(row.get("edit_index", -1)))
    recorded_patch_digest = ""
    if (
        len(recorded_rows) == len(tested.edits)
        and [int(row.get("edit_index", -1)) for row in recorded_rows]
        == list(range(len(tested.edits)))
    ):
        recorded_patch_digest = hashlib.sha256(json.dumps({
            "reasoning": str(tested_record.get("reasoning") or ""),
            "edits": [row["edit"] for row in recorded_rows],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )).hexdigest()
    if (
        graph_patch_sha256(tested) != patch_digest
        or graph_patch_sha256(patch) != patch_digest
        or stored_patch_digest not in {
            patch_digest, tested_record_digest, recorded_patch_digest,
            _legacy_graph_patch_sha256(tested),
        }
        or str(reference.get("base_graph_sha256") or "") != graph_sha256(base_graph)
        or list(map(str, reference.get("train_case_ids") or [])) != train_ids
        or list(map(str, reference.get("validation_case_ids") or [])) != val_ids
        or list(map(str, gate.get("affected_case_ids") or [])) != [*train_ids, *val_ids]
        or not isinstance(gate.get("accepted"), bool)
    ):
        return None
    return gate


def stage_results_signature(results: list[dict[str, Any]]) -> str:
    payload = [
        {
            "id": str(row.get("id") or ""),
            "hard": row.get("hard"),
            "soft": row.get("soft"),
            "response": row.get("response"),
            "predicted_answer": row.get("predicted_answer"),
        }
        for row in results
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _mean_gate_repeat_results(
    repeat_results: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Normalize one aligned Gate run at case level for paired attribution."""
    if len(repeat_results) != GATE_REPEATS:
        raise ValueError(
            f"Gate requires exactly {GATE_REPEATS} repeats, got {len(repeat_results)}"
        )
    reference_ids = [str(row.get("id")) for row in repeat_results[0]]
    by_repeat: list[dict[str, dict[str, Any]]] = []
    for index, rows in enumerate(repeat_results, start=1):
        ids = [str(row.get("id")) for row in rows]
        if ids != reference_ids:
            raise ValueError(
                f"Gate repeat {index} case IDs/order differ from repeat 1"
            )
        by_repeat.append({str(row.get("id")): row for row in rows})

    averaged: list[dict[str, Any]] = []
    for case_id in reference_ids:
        rows = [mapping[case_id] for mapping in by_repeat]
        item = dict(rows[0])
        hard_values = [float(row.get("hard") or 0.0) for row in rows]
        soft_values = [float(row.get("soft") or 0.0) for row in rows]
        item["hard"] = statistics.fmean(hard_values)
        item["soft"] = statistics.fmean(soft_values)
        item["gate_repeat_count"] = GATE_REPEATS
        item["hard_repeat_values"] = hard_values
        item["soft_repeat_values"] = soft_values
        item["representative_repeat"] = 1
        if all(value >= 1.0 for value in hard_values):
            item["same_sample_outcome"] = "all_correct"
        elif all(value <= 0.0 for value in hard_values):
            item["same_sample_outcome"] = "all_wrong"
        else:
            item["same_sample_outcome"] = "mixed"
        averaged.append(item)
    return averaged

def _batch_size_for_logging(batch: Any, fallback: int) -> int:
    """Return a BatchSpec/list size without assuming the object implements len."""
    if batch is None:
        return int(fallback)
    declared = getattr(batch, "batch_size", None)
    if declared is not None:
        return int(declared)
    try:
        return len(batch)
    except TypeError:
        payload = getattr(batch, "payload", None)
        if payload is not None:
            return len(payload)
        return int(fallback)


def _train_shard_sizes(train_size: int, num_epochs: int) -> list[int]:
    """Put the remainder in the last shards (3553/4 -> 888,888,888,889)."""
    if train_size <= 0 or num_epochs <= 0:
        return []
    base, remainder = divmod(train_size, num_epochs)
    return [base] * (num_epochs - remainder) + [base + 1] * remainder


def plan_disjoint_train_shard(
    dataloader,
    *,
    epoch: int,
    num_epochs: int,
    batch_size: int,
    seed: int,
    train_size: int = 0,
) -> list[Any]:
    """Return epoch ``epoch`` from one fixed, no-replacement train permutation."""
    items = list(getattr(dataloader, "train_items", []) or [])
    target_size = int(train_size or len(items))
    if target_size > len(items):
        raise ValueError(
            f"configured train_size={target_size} exceeds loaded train cases={len(items)}"
        )
    if not 1 <= epoch <= num_epochs:
        raise ValueError(f"epoch must be in [1, {num_epochs}], got {epoch}")
    if num_epochs > target_size:
        raise ValueError(
            f"num_epochs={num_epochs} exceeds train_size={target_size}; empty shards are forbidden"
        )

    # Keep the disjoint schedule aligned with the repository's original
    # epoch-1 full-pass sampler (which used ``seed + epoch * 1000``). This
    # preserves deterministic case identity across the full-pass -> sharded
    # migration while still assigning each case to exactly one epoch.
    rng = random.Random(seed + 1000)
    rng.shuffle(items)
    items = items[:target_size]
    sizes = _train_shard_sizes(target_size, num_epochs)
    start = sum(sizes[: epoch - 1])
    shard = items[start: start + sizes[epoch - 1]]
    metadata_fn = getattr(dataloader, "_metadata_for_items", None)
    from graphopt.runtime_envs.data import BatchSpec

    batches: list[Any] = []
    for batch_index in range(0, len(shard), batch_size):
        batch_items = shard[batch_index: batch_index + batch_size]
        metadata = (
            dict(metadata_fn(batch_items, "train", "train"))
            if callable(metadata_fn)
            else {}
        )
        metadata.update(
            {
                "shard_epoch": epoch,
                "shard_count": num_epochs,
                "shard_size": len(shard),
                "shard_start": start,
            }
        )
        batches.append(
            BatchSpec(
                phase="train",
                split="train",
                seed=seed + epoch * 1000 + len(batches) + 1,
                batch_size=len(batch_items),
                payload=batch_items,
                metadata=metadata,
            )
        )
    return batches


def _batch_case_ids(batch: Any) -> list[str] | None:
    """Extract deterministic case IDs from BatchSpec or list-like batches."""
    if batch is None:
        return None
    metadata = getattr(batch, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("result_ids"):
        return [str(case_id) for case_id in metadata["result_ids"]]
    payload = getattr(batch, "payload", None)
    if payload is None and isinstance(batch, list):
        payload = batch
    if isinstance(payload, list):
        ids = [str(row.get("id")) for row in payload if isinstance(row, dict) and row.get("id")]
        return ids or None
    return None

def build_exact_split_batch(
    dataloader: Any,
    split: str,
    case_ids: list[str],
    *,
    seed: int,
) -> Any:
    """Build one deterministic BatchSpec containing exactly ``case_ids``."""
    if dataloader is None:
        raise ValueError("an exact batch requires a dataloader")
    requested = list(dict.fromkeys(str(case_id) for case_id in case_ids))
    canonical = "valid_seen" if split in {"val", "valid", "valid_seen"} else split
    if canonical == "train":
        items = list(getattr(dataloader, "train_items", []) or [])
        phase = "train"
    else:
        items = list(dataloader.get_split_items(canonical) or [])
        phase = "eval"
    by_id = {
        str(item.get("id")): item
        for item in items
        if isinstance(item, dict) and item.get("id") is not None
    }
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(f"exact {canonical} batch cases are missing: {missing}")
    payload = [by_id[case_id] for case_id in requested]
    # Environment-owned materialization (notably LiveMath choice shuffling)
    # is part of the evaluation protocol.  Exact-ID batches must preserve it;
    # bypassing this hook made small Gates see different question variants
    # from the benchmark's ordinary train/eval batches.
    materialize_fn = getattr(dataloader, "_materialize_batch", None)
    if callable(materialize_fn):
        payload = list(materialize_fn(payload, int(seed)))
    metadata_fn = getattr(dataloader, "_metadata_for_items", None)
    metadata = (
        dict(metadata_fn(payload, canonical, phase))
        if callable(metadata_fn) else {}
    )
    metadata["result_ids"] = requested
    metadata["exact_case_batch"] = True
    from graphopt.runtime_envs.data import BatchSpec

    return BatchSpec(
        phase=phase,
        split=canonical,
        seed=int(seed),
        batch_size=len(payload),
        payload=payload,
        metadata=metadata,
    )


def drop_orphan_weight_refreshes(
    patch: GraphPatch,
) -> tuple[GraphPatch, int]:
    """Drop statistical weight refreshes when no semantic edit survived.

    A weight-only patch cannot stand in for learned node or structural-edge
    knowledge. Keeping refreshes alongside a real semantic edit is allowed.
    """
    weight_refreshes = [
        edit
        for edit in patch.edits
        if edit.op == "add_edge"
        and str(edit.reasoning or "").startswith("weight refresh ")
    ]
    if not weight_refreshes or len(weight_refreshes) != len(patch.edits):
        return patch, 0
    return GraphPatch(reasoning=patch.reasoning, edits=[]), len(weight_refreshes)

def drop_protected_root_updates(
    patch: GraphPatch, protected_node_ids: set[str] | frozenset[str],
) -> tuple[GraphPatch, list[str]]:
    """Prevent case-specific text growth on environment-owned global roots."""
    protected = set(map(str, protected_node_ids))
    dropped = [
        str(edit.node_id)
        for edit in patch.edits
        if edit.op == "update_node" and str(edit.node_id) in protected
    ]
    if not dropped:
        return patch, []
    edits = [
        edit for edit in patch.edits
        if not (edit.op == "update_node" and str(edit.node_id) in protected)
    ]
    return GraphPatch(reasoning=patch.reasoning, edits=edits), dropped


def limit_patch_to_atomic_groups(
    patch: GraphPatch, max_groups: int,
) -> tuple[GraphPatch, GraphPatch, list[str]]:
    """Select support-ranked atomic candidates for independent small-Gate A/B."""
    groups = atomic_edit_groups(patch.edits)
    limit = max(1, int(max_groups))
    if len(groups) <= limit:
        return patch, GraphPatch(reasoning=patch.reasoning, edits=[]), []

    def rank(item: tuple[str, list[int]]) -> tuple[int, int, int, int]:
        _, indices = item
        edits = [patch.edits[index] for index in indices]
        semantic = any(
            not (edit.op == "add_edge" and str(edit.reasoning or "").startswith("weight refresh "))
            for edit in edits
        )
        source_support = len({
            str(case_id) for edit in edits for case_id in edit.source_case_ids
            if str(case_id)
        })
        node_semantics = any(edit.op in {"add_node", "update_node"} for edit in edits)
        return (-int(semantic), -source_support, -int(node_semantics), indices[0])

    chosen_groups = sorted(groups, key=rank)[:limit]
    chosen_indices = {index for _, indices in chosen_groups for index in indices}
    chosen_ids = {group_id for group_id, _ in chosen_groups}
    deferred_ids = [group_id for group_id, _ in groups if group_id not in chosen_ids]
    selected = GraphPatch(
        reasoning=patch.reasoning,
        edits=[edit for index, edit in enumerate(patch.edits) if index in chosen_indices],
    )
    deferred = GraphPatch(
        reasoning=patch.reasoning,
        edits=[edit for index, edit in enumerate(patch.edits) if index not in chosen_indices],
    )
    return selected, deferred, deferred_ids


def group_epoch_joint_patch(
    patch: GraphPatch,
) -> tuple[GraphPatch, list[dict[str, Any]]]:
    """Join edits that share evidence or touch any common graph node.

    The joint component, not one text field or edge, is the atomic Local-Gate and materialization
    unit. An update of Q020 and any other node/edge edit incident on Q020 must
    therefore be synthesized, tested, accepted, and rolled back together; a
    later group is never allowed to overwrite an earlier meaning of Q020.
    Pure statistical weight refreshes have no source case and are not part of
    the epoch semantic candidate stream.
    """
    edits = [copy.deepcopy(edit) for edit in patch.edits if not (
        edit.op == "add_edge"
        and str(edit.reasoning or "").startswith("weight refresh ")
    )]
    parent = list(range(len(edits)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    # Preserve explicit/new-node dependency groups first.
    for _, indices in atomic_edit_groups(edits):
        for index in indices[1:]:
            union(indices[0], index)
    first_by_source: dict[str, int] = {}
    first_by_incident_node: dict[str, int] = {}
    for index, edit in enumerate(edits):
        incident_nodes = {
            str(value)
            for value in (edit.node_id, edit.src, edit.dst)
            if str(value or "")
        }
        for node_id in incident_nodes:
            if node_id in first_by_incident_node:
                union(first_by_incident_node[node_id], index)
            else:
                first_by_incident_node[node_id] = index
        for case_id in dict.fromkeys(map(str, edit.source_case_ids)):
            if not case_id:
                continue
            if case_id in first_by_source:
                union(first_by_source[case_id], index)
            else:
                first_by_source[case_id] = index

    components: dict[int, list[int]] = {}
    for index in range(len(edits)):
        components.setdefault(find(index), []).append(index)
    manifest: list[dict[str, Any]] = []
    for indices in components.values():
        source_ids = sorted({
            str(case_id) for index in indices
            for case_id in edits[index].source_case_ids if str(case_id)
        })
        identity = json.dumps({
            "sources": source_ids,
            "edits": [
                {
                    key: value for key, value in edits[index].to_dict().items()
                    if key != "group_id"
                }
                for index in indices
            ],
        }, ensure_ascii=False, sort_keys=True)
        group_id = "epoch-joint-" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]
        for index in indices:
            edits[index].group_id = group_id
        manifest.append({
            "group_id": group_id,
            "edit_indices": list(indices),
            "source_case_ids": source_ids,
            "n_edits": len(indices),
            "ops": [edits[index].op for index in indices],
            "targets": sorted({
                value for index in indices
                for value in (
                    edits[index].node_id, edits[index].src, edits[index].dst
                ) if value
            }),
            "incident_node_ids": sorted({
                str(value) for index in indices
                for value in (
                    edits[index].node_id, edits[index].src, edits[index].dst
                ) if str(value or "")
            }),
            "semantic_conflict_policy": (
                "same-node meanings are precomposed with protected successes; "
                "all incident structural edits are one atomic Local-Gate unit"
            ),
        })
    return GraphPatch(reasoning=patch.reasoning, edits=edits), manifest


def select_epoch_affected_cases(
    *,
    grouped_batch: GroupedBatch,
    base_graph: SkillGraph,
    patch: GraphPatch,
    reference_train: list[dict[str, Any]],
    reference_val: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the baseline-usage and fixed-group scope touched by one joint patch."""
    from graphopt.evaluation.graph_usage import (
        extract_graph_refs, simulate_edit_touch_keys,
    )

    touch_sets = simulate_edit_touch_keys(base_graph, patch.edits)
    touch_keys = set().union(*touch_sets) if touch_sets else set()
    added_nodes = {
        str(edit.node_id) for edit in patch.edits
        if edit.op == "add_node" and str(edit.node_id or "")
    }
    specialist_only = bool(added_nodes) and all(
        edit.op == "add_node"
        or (
            edit.op == "add_edge"
            and bool({str(edit.src or ""), str(edit.dst or "")}.intersection(added_nodes))
        )
        for edit in patch.edits
    )
    if specialist_only:
        # A narrow specialist and its activation edge do not rewrite the parent
        # node. Local Gate therefore tests direct evidence plus its fixed sibling
        # groups. Treating the parent endpoint as a broad update expands every
        # specialist to the full epoch; the complete valid_seen Big Gate remains
        # the global guard against accidental broad activation.
        touch_keys = {f"node:{node_id}" for node_id in added_nodes}
    direct_sources = {
        str(case_id) for edit in patch.edits
        for case_id in edit.source_case_ids if str(case_id)
    }

    def classify(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
        used: set[str] = set()
        unknown: set[str] = set()
        for row in rows:
            case_id = str(row.get("id") or "")
            refs = row.get("graph_refs")
            status = str(refs.get("status") or "") if isinstance(refs, dict) else "missing"
            if not case_id:
                continue
            if status.startswith(("missing_", "invalid_")) or status == "missing":
                unknown.add(case_id)
                continue
            nodes, edges = extract_graph_refs(row)
            if any(
                (key.startswith("node:") and key[5:] in nodes)
                or (key.startswith("edge_id:") and key[8:] in edges)
                for key in touch_keys
            ):
                used.add(case_id)
        return used, unknown

    used_train, unknown_train = classify(reference_train)
    used_val, unknown_val = classify(reference_val)
    affected_train = set(used_train) | set(direct_sources) | set(unknown_train)
    affected_val = set(used_val) | set(unknown_val)
    related_groups: list[int] = []
    for position, group in enumerate(grouped_batch.groups):
        train_ids = set(map(str, group.train_ids))
        val_ids = set(map(str, group.val_ids))
        if (affected_train & train_ids) or (affected_val & val_ids):
            related_groups.append(position)
            affected_train.update(train_ids)
            affected_val.update(val_ids)

    train_order = [str(row.get("id")) for row in reference_train]
    val_order = [str(row.get("id")) for row in reference_val]
    return {
        "scope_policy": (
            "specialist_sources_and_fixed_sibling_groups"
            if specialist_only
            else "direct_graph_usage_and_fixed_sibling_groups"
        ),
        "touch_keys": sorted(touch_keys),
        "direct_source_case_ids": sorted(direct_sources),
        "used_train_case_ids": sorted(used_train),
        "used_validation_case_ids": sorted(used_val),
        "unknown_train_case_ids": sorted(unknown_train),
        "unknown_validation_case_ids": sorted(unknown_val),
        "related_group_indices": related_groups,
        "train_case_ids": [case_id for case_id in train_order if case_id in affected_train],
        "validation_case_ids": [case_id for case_id in val_order if case_id in affected_val],
    }



def _build_comparison_pairs(
    prev_results: list[dict[str, Any]],
    curr_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prev = {str(r.get("id")): r for r in prev_results if r.get("id") is not None}
    curr = {str(r.get("id")): r for r in curr_results if r.get("id") is not None}
    if len(prev) != len(prev_results) or len(curr) != len(curr_results):
        raise ValueError("meta comparison contains missing or duplicate case ids")
    if set(prev) != set(curr):
        raise ValueError("meta comparison requires identical validation case ids")
    pairs = []
    for rid in sorted(prev):
        ph = float(prev[rid].get("hard") or 0)
        ch = float(curr[rid].get("hard") or 0)
        if ph < 1e-9 and ch >= 1e-9:
            cat = "improved"
        elif ph >= 1e-9 and ch < 1e-9:
            cat = "regressed"
        elif ph < 1e-9 and ch < 1e-9:
            cat = "persistent_fail"
        else:
            cat = "stable_success"
        pairs.append(
            {
                "id": rid,
                "category": cat,
                "prev_hard": ph,
                "curr_hard": ch,
                "task_type": curr[rid].get("task_type") or prev[rid].get("task_type"),
            }
        )
    return pairs


def _require_same_case_ids(
    reference_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
) -> None:
    """Reject every Gate comparison whose paired decision pools differ."""
    reference_ids = [str(row.get("id")) for row in reference_results]
    candidate_ids = [str(row.get("id")) for row in candidate_results]
    if len(reference_ids) != len(set(reference_ids)) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Gate results contain duplicate case ids")
    if set(reference_ids) != set(candidate_ids):
        missing = sorted(set(reference_ids) - set(candidate_ids))
        extra = sorted(set(candidate_ids) - set(reference_ids))
        raise ValueError(
            "Gate requires identical decision-pool case ids; "
            f"missing_in_candidate={missing}, extra_in_candidate={extra}"
        )


def _hard_transition_summary(
    reference_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Count one-run hard successes and paired 0->1 / 1->0 transitions."""
    _require_same_case_ids(reference_results, candidate_results)
    reference = {
        str(row.get("id")): row for row in reference_results
        if not row.get("exclude_from_metrics")
    }
    candidate = {
        str(row.get("id")): row for row in candidate_results
        if not row.get("exclude_from_metrics")
    }
    eligible = sorted(set(reference) & set(candidate))
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    for case_id in eligible:
        before = float(reference[case_id].get("hard") or 0.0)
        after = float(candidate[case_id].get("hard") or 0.0)
        if after > before + eps:
            improved.append(case_id)
        elif after < before - eps:
            regressed.append(case_id)
        else:
            unchanged.append(case_id)
    before_successes = sum(
        float(reference[case_id].get("hard") or 0.0) for case_id in eligible
    )
    after_successes = sum(
        float(candidate[case_id].get("hard") or 0.0) for case_id in eligible
    )
    return {
        "n_eligible": len(eligible),
        "reference_hard_successes": before_successes,
        "candidate_hard_successes": after_successes,
        "hard_success_delta": after_successes - before_successes,
        "improved_case_ids": improved,
        "regressed_case_ids": regressed,
        "unchanged_case_ids": unchanged,
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "hard_net_case_gain": len(improved) - len(regressed),
    }


def _combined_big_gate_audit(
    validation_audit: dict[str, Any], train_audit: dict[str, Any],
) -> dict[str, Any]:
    """Combine disjoint full-train and full-valid_seen hard transitions."""
    validation_n = int(validation_audit.get("n_eligible") or 0)
    train_n = int(train_audit.get("n_eligible") or 0)
    validation_net = int(validation_audit.get("hard_net_case_gain") or 0)
    train_net = int(train_audit.get("hard_net_case_gain") or 0)
    return {
        "schema_version": "graphopt-combined-big-gate-audit-v1",
        "n_eligible": validation_n + train_n,
        "validation_n": validation_n,
        "train_n": train_n,
        "reference_hard_successes": (
            float(validation_audit.get("reference_hard_successes") or 0.0)
            + float(train_audit.get("reference_hard_successes") or 0.0)
        ),
        "candidate_hard_successes": (
            float(validation_audit.get("candidate_hard_successes") or 0.0)
            + float(train_audit.get("candidate_hard_successes") or 0.0)
        ),
        "validation_hard_net_case_gain": validation_net,
        "train_hard_net_case_gain": train_net,
        "hard_net_case_gain": validation_net + train_net,
        "n_improved": (
            int(validation_audit.get("n_improved") or 0)
            + int(train_audit.get("n_improved") or 0)
        ),
        "n_regressed": (
            int(validation_audit.get("n_regressed") or 0)
            + int(train_audit.get("n_regressed") or 0)
        ),
        "accept_if": "validation_hard_net_case_gain + train_hard_net_case_gain > 0",
    }


def _compact_transition_audit(audit: dict[str, Any] | None) -> dict[str, Any]:
    audit = dict(audit or {})
    return {
        key: audit.get(key)
        for key in (
            "n_eligible", "reference_hard_successes",
            "candidate_hard_successes", "hard_net_case_gain",
            "n_improved", "n_regressed", "improved_case_ids",
            "regressed_case_ids",
        )
        if key in audit
    }


def _rolled_back_modification_lessons(
    patch: GraphPatch, record: dict[str, Any],
) -> list[dict[str, Any]]:
    """List only the concrete edits rejected by the complete Big Gate."""
    no_validation = bool(record.get("candidate_update_pool_evaluated"))
    train_audit = _compact_transition_audit(
        record.get("candidate_epoch_train_audit")
    )
    validation_audit = _compact_transition_audit(
        record.get("candidate_update_validation_audit")
    )
    why_failed = {
        "gate_action": str(record.get("action") or ""),
        "causal_scope": (
            "the complete candidate failed as a whole; this record does not "
            "prove that this individual edit alone caused the regression"
        ),
    }
    if no_validation:
        why_failed["update_pool_outcome"] = _compact_transition_audit(
            record.get("candidate_update_pool_audit")
        )
    else:
        why_failed["train_outcome"] = train_audit
        why_failed["validation_diagnostic"] = validation_audit
    lessons: list[dict[str, Any]] = []
    for edit in patch.edits:
        semantic = edit.to_dict()
        semantic.pop("evidence_items", None)
        lessons.append({
            "edit": semantic,
            "lookup": {
                "node_ids": sorted({
                    str(value) for value in (edit.node_id, edit.src, edit.dst)
                    if str(value or "")
                }),
                "edge": {
                    "source": str(edit.src or ""),
                    "target": str(edit.dst or ""),
                    "type": str(edit.edge_type or ""),
                } if edit.src or edit.dst else None,
            },
            "why_failed": copy.deepcopy(why_failed),
        })
    return lessons


def _is_reusable_big_gate_rollback_experience(record: dict[str, Any]) -> bool:
    """Carry lessons only from a candidate that Big Gate actually rolled back."""
    action = str(record.get("action") or "")
    return (
        not bool(record.get("accepted"))
        and action.startswith("reject_complete_candidate")
        and int(record.get("n_edits") or 0) > 0
    )


def _compact_gate_experience(record: dict[str, Any]) -> dict[str, Any]:
    """Retain measured edit lessons without unseen diagnostics or unchanged IDs."""
    local_experiences: list[dict[str, Any]] = []
    for local_record in record.get("small_gates") or []:
        local_gate = dict(local_record.get("local_gate") or {})
        local_experiences.append({
            "joint_candidate_index": local_record.get("joint_candidate_index"),
            "attempt": local_record.get("attempt"),
            "accepted": bool(local_record.get("accepted")),
            "action": local_record.get("action"),
            "joint_patch": local_record.get("joint_patch"),
            "effective_case_ids": list(local_gate.get("effective_case_ids") or []),
            "ineffective_case_ids": list(local_gate.get("ineffective_case_ids") or []),
            "n_effective": int(local_gate.get("n_effective") or 0),
            "n_ineffective": int(local_gate.get("n_ineffective") or 0),
            "decision_reason": local_gate.get("decision_reason"),
            "semantic_explanation": local_gate.get("semantic_explanation"),
            "precise_refinement_triggered": bool(
                local_record.get("precise_refinement_triggered")
            ),
        })
    return {
        "schema_version": "graphopt-cumulative-gate-experience-v1",
        "epoch": int(record.get("epoch") or record.get("step") or 0),
        "action": record.get("action"),
        "accepted": bool(record.get("accepted")),
        "big_gate_policy": record.get("big_gate_policy"),
        "candidate_update_pool_audit": _compact_transition_audit(
            record.get("candidate_update_pool_audit")
        ),
        "candidate_train_audit": (
            {} if record.get("candidate_update_pool_evaluated")
            else _compact_transition_audit(
                record.get("candidate_epoch_train_audit")
            )
        ),
        "candidate_update_validation_audit": (
            {} if record.get("candidate_update_pool_evaluated")
            else _compact_transition_audit(
                record.get("candidate_update_validation_audit")
            )
        ),
        "update_pool_diagnostic_audit": dict(
            record.get("update_pool_diagnostic_audit") or {}
        ),
        "local_gate_experiences": local_experiences,
    }


def _big_gate_accepts_complete_candidate(
    validation_audit: dict[str, Any],
    train_audit: dict[str, Any],
    *,
    allow_validation_tie: bool,
    use_train_tiebreak: bool,
) -> bool:
    """Accept a complete candidate on strict paired decision-pool net gain."""
    validation_net = int(validation_audit.get("hard_net_case_gain") or 0)
    validation_observed = int(validation_audit.get("n_eligible") or 0) > 0
    train_net = int(train_audit.get("hard_net_case_gain") or 0)
    train_observed = int(train_audit.get("n_eligible") or 0) > 0
    del allow_validation_tie
    if not use_train_tiebreak:
        return bool(validation_observed and validation_net > 0)
    return bool(
        validation_observed
        and train_observed
        and validation_net + train_net > 0
    )


def _hard_transition_rows(
    reference_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return paired cases whose one-run hard outcome changed.

    This is evidence compaction only: the rows are supplied to the bounded
    recovery planner. They never cast per-edit votes and are never treated as
    independent graph-reference attribution.
    """
    audit = _hard_transition_summary(reference_results, candidate_results)
    transition_ids = set(audit["improved_case_ids"]) | set(
        audit["regressed_case_ids"]
    )
    return (
        [row for row in reference_results if str(row.get("id")) in transition_ids],
        [row for row in candidate_results if str(row.get("id")) in transition_ids],
    )


def _gate_failure_evidence(
    patch: GraphPatch,
    reference_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    audit: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Build the complete paired evidence for every behavior-changing case."""
    reference = {str(row.get("id")): row for row in reference_results}
    candidate = {str(row.get("id")): row for row in candidate_results}
    changed_ids = list(dict.fromkeys(
        [str(x) for x in audit.get("improved_case_ids") or []]
        + [str(x) for x in audit.get("regressed_case_ids") or []]
    ))
    failed_patch = patch.to_dict()
    for edit in failed_patch.get("edits") or []:
        edit.pop("evidence_items", None)
    return {
        "schema_version": "graphopt-gate-failure-evidence-v3",
        "stage": stage,
        "gate_rule": "validation_net_positive_or_validation_tie_with_train_net_positive",
        "failed_patch": failed_patch,
        "hard_transition_audit": audit,
        "paired_changed_cases": [
            {
                "case_id": case_id,
                "before": reference[case_id],
                "after": candidate[case_id],
            }
            for case_id in changed_ids
            if case_id in reference and case_id in candidate
        ],
    }


def _format_small_gate_rejection(
    patch: GraphPatch,
    reference_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    audit: dict[str, Any],
    *,
    decision: str = "",
    atomic_edit_audit: dict[str, Any] | None = None,
) -> str:
    """Give the next teacher one-shot, fully paired behavioral evidence."""
    evidence = _gate_failure_evidence(
        patch, reference_results, candidate_results, audit,
        stage="rejected_candidate",
    )
    evidence["rejection_decision"] = str(decision)
    if atomic_edit_audit is not None:
        compact_keys = (
            "atomic_group_id", "touch_keys",
            "direct_source_train_case_ids",
            "trace_verified_source_case_ids",
            "trace_source_unobserved_case_ids",
            "trace_source_observation_complete",
            "trace_source_train_audit",
            "trace_source_observed_failure_case_ids",
            "trace_source_repaired",
            "protected_validation_regression_case_ids",
            "protected_train_regression_case_ids",
            "related_validation_case_ids", "related_train_case_ids",
            "validation", "train", "script_keep", "script_decision",
        )
        evidence["atomic_edit_audit"] = {
            key: atomic_edit_audit.get(key) for key in compact_keys
        }
    return (
        "## Previous Gate rejection (one-shot evidence)\n"
        "The environment rejected the complete patch below. Do not repeat it verbatim. "
        "Use the paired before/after cases to propose a narrower reusable rule. "
        "This evidence is diagnostic only and cannot approve any edit.\n\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
    )

def _group_hard_audit(
    grouped_batch: GroupedBatch,
    reference_val: list[dict[str, Any]],
    candidate_val: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose each fixed group's mapped validation vote behind a small Gate."""
    before_val = {str(row.get("id")): row for row in reference_val}
    after_val = {str(row.get("id")): row for row in candidate_val}
    audits: list[dict[str, Any]] = []
    for index, group in enumerate(grouped_batch.groups, start=1):
        before_rows = [
            dict(before_val[case_id], id=case_id)
            for case_id in group.val_ids if case_id in before_val
        ]
        after_rows = [
            dict(after_val[case_id], id=case_id)
            for case_id in group.val_ids if case_id in after_val
        ]
        if not before_rows and not after_rows:
            continue
        summary = _hard_transition_summary(before_rows, after_rows)
        audits.append({
            "group_index": index,
            "group_type": group.group_type,
            "train_case_ids": list(group.train_ids),
            "validation_case_ids": list(group.val_ids),
            "expected_validation_size": len(group.val_ids),
            **summary,
        })
    return audits


def _attach_same_group_rightcases(
    grouped_batch: GroupedBatch,
    train_results: list[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    """Attach successful update cases to failures in the same fixed group.

    This is batch-local generation context only. It is not persisted, scored by
    a curator, attributed across groups, or counted as an independent Gate.
    """
    for row in train_results:
        row.pop("same_group_rightcases", None)
        row.pop("same_group_id", None)
    if not enabled:
        return []
    by_id = {str(row.get("id")): row for row in train_results}
    manifest: list[dict[str, Any]] = []
    fields = (
        "id", "environment", "task_type", "task_description", "hard", "soft",
        "response", "predicted_answer", "predicted_label", "predicted_text",
        "trajectory", "n_turns", "semantic_reasoning_trace",
        "semantic_reasoning_trace_status", "semantic_reasoning_trace_required",
    )
    for group_index, group in enumerate(grouped_batch.groups, start=1):
        rows = [by_id[case_id] for case_id in group.train_ids if case_id in by_id]
        successes = [
            row for row in rows
            if not row.get("exclude_from_evolution")
            and float(row.get("hard") or 0.0) >= 1.0 - 1e-12
        ]
        failures = [
            row for row in rows
            if not row.get("exclude_from_evolution")
            and float(row.get("hard") or 0.0) < 1.0 - 1e-12
        ]
        if not failures:
            continue
        peers = [
            {key: row.get(key) for key in fields if row.get(key) is not None}
            for row in successes
        ]
        for row in failures:
            row["same_group_id"] = group_index
            row["same_group_rightcases"] = copy.deepcopy(peers)
        manifest.append({
            "group_index": group_index,
            "group_type": group.group_type,
            "failure_case_ids": [str(row.get("id")) for row in failures],
            "rightcase_ids": [str(row.get("id")) for row in successes],
        })
    return manifest


def _related_group_train_ids(
    grouped_batch: GroupedBatch,
    edits: list[GraphEdit],
) -> list[str]:
    """Expand direct edit sources to every update case in their fixed groups.

    ``GraphEdit.source_case_ids`` remains exact proposal provenance. Gate
    protection is broader: if any source belongs to a fixed group, all train
    siblings in that group must be rerun under the candidate graph.
    """
    direct_ids = {
        str(case_id)
        for edit in edits
        for case_id in edit.source_case_ids
        if str(case_id)
    }
    related: list[str] = []
    for group in grouped_batch.groups:
        if direct_ids.intersection(str(case_id) for case_id in group.train_ids):
            related.extend(str(case_id) for case_id in group.train_ids)
    return list(dict.fromkeys(related))



def _mapped_small_gate_attribution(
    grouped_batch: GroupedBatch,
    base_graph: SkillGraph,
    candidate_graph: SkillGraph,
    edits: list[GraphEdit],
    reference_val: list[dict[str, Any]],
    candidate_val: list[dict[str, Any]],
    reference_train: list[dict[str, Any]] | None = None,
    candidate_train: list[dict[str, Any]] | None = None,
    retry_round: int = 0,
    chat_fn: Any = None,
    require_nonnegative_combined_on_validation_gain: bool = False,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    """Independently audit every atomic edit, then return the teacher-vetted subset."""
    del candidate_graph, retry_round
    audits = build_atomic_edit_audits(
        grouped_batch=grouped_batch, base_graph=base_graph, edits=edits,
        reference_val=reference_val, candidate_val=candidate_val,
        reference_train=list(reference_train or []),
        candidate_train=list(candidate_train or []),
        require_nonnegative_combined_on_validation_gain=require_nonnegative_combined_on_validation_gain,
    )
    kept_group_ids, teacher_artifact = review_atomic_edits(
        audits, chat_fn=chat_fn, prompt_path=PROMPTS / "per_edit_gate_audit.md",
        stage="small_gate_per_edit_audit",
    )
    kept = {
        index for row in audits if row["atomic_group_id"] in kept_group_ids
        for index in row["edit_indices"]
    }
    review_by_group = {
        row["atomic_group_id"]: row
        for row in teacher_artifact.get("reviews") or []
    }
    rows: list[dict[str, Any]] = []
    single_atomic_ab = len(audits) == 1
    by_index = {
        index: audit for audit in audits for index in audit["edit_indices"]
    }
    for index, edit in enumerate(edits):
        audit = by_index[index]
        rows.append({
            "edit_index": index,
            "atomic_group_id": audit["atomic_group_id"],
            "decision_scope": (
                "single_atomic_group_ab_test" if single_atomic_ab
                else "independent_atomic_edit_usage_scope"
            ),
            "individual_isolation": single_atomic_ab,
            "environment_usage_statistics": audit,
            "teacher_review": review_by_group.get(audit["atomic_group_id"]),
            "teacher_audit_status": teacher_artifact.get("status"),
            "keep": index in kept,
            "request_reoptimization": False,
            "attribution_decision": (
                "keep_script_and_teacher" if index in kept
                else "rollback_script_or_teacher_veto"
            ),
            "attribution_basis": (
                "single_atomic_candidate_rollout_plus_fixed_group_expansion"
                if single_atomic_ab else "reported_graph_usage_plus_fixed_group_expansion"
            ),
            "edit": edit.to_dict(),
        })
    return rows, kept, set()

def _mean_result_scores(
    results: list[dict[str, Any]], *, mixed_weight: float
) -> tuple[float, float]:
    """Reconstruct hard/soft means for a reused previous-step result file."""
    results = [row for row in results if not row.get("exclude_from_metrics")]
    if not results:
        return 0.0, 0.0
    hard = sum(float(row.get("hard") or 0) for row in results) / len(results)
    soft = sum(
        float(row.get("hard") or 0)
        if row.get("soft") is None
        else float(row["soft"])
        for row in results
    ) / len(results)
    del mixed_weight
    return hard, soft


def _range_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"values": [], "min": 0.0, "max": 0.0, "mean": 0.0, "half_range": 0.0}
    low = min(values)
    high = max(values)
    return {
        "values": values,
        "min": low,
        "max": high,
        "mean": statistics.fmean(values),
        "half_range": (high - low) / 2.0,
    }


def format_final_test_display(summary: dict[str, Any]) -> str:
    """Format final-test percentages with conventional half-up rounding."""
    def percent(key: str) -> str:
        if key == "hard_half_range" and {"hard_min", "hard_max"}.issubset(summary):
            value = (
                Decimal(str(summary["hard_max"]))
                - Decimal(str(summary["hard_min"]))
            ) * Decimal("50")
        else:
            value = Decimal(str(summary[key])) * Decimal("100")
        return format(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), ".1f")

    return (
        f"min={percent('hard_min')}; "
        f"max={percent('hard_max')}; "
        f"mean={percent('hard_mean')} ± "
        f"{percent('hard_half_range')}"
    )


def _rollout_context_payload(
    *, split: str, seed: int, skill: str, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Return the complete identity of a rollout that is safe to reuse."""
    return {
        "schema_version": "graphopt-rollout-context-v8",
        "environment": str(cfg.get("env_name") or cfg.get("env") or "searchqa"),
        "split": split,
        "seed": int(seed),
        "skill_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
        "student_model_base": str(cfg.get("student_model_base") or ""),
        "student_model_deployed": str(
            cfg.get("target_model") or cfg.get("llm_model") or ""
        ),
        "target_backend": str(cfg.get("target_backend") or ""),
        "environment_mode": str(cfg.get("mode") or ""),
        "max_turns": int(cfg.get("max_turns") or 0),
        "max_completion_tokens": int(cfg.get("max_completion_tokens") or 0),
        "student_rollout_protocol": str(
            cfg.get("student_rollout_protocol") or "standard"
        ),
        "student_rollout_prompt_sha256": str(
            cfg.get("student_rollout_prompt_sha256") or ""
        ),
        "exec_timeout": int(cfg.get("exec_timeout") or 0),
        "workers": int(cfg.get("workers") or 0),
        "openlux_routing": str(cfg.get("openlux_routing") or ""),
        "student_openlux_routing": str(
            cfg.get("student_openlux_routing") or cfg.get("openlux_routing") or ""
        ),
        "openlux_routing_style": str(cfg.get("openlux_routing_style") or ""),
        "response_soft_char_limit": int(cfg.get("rollout_response_soft_char_limit") or 16384),
        "response_hard_char_limit": int(cfg.get("rollout_response_hard_char_limit") or 16384),
        "response_guard_enabled": bool(cfg.get("rollout_response_guard_enabled", False)),
        "max_steps": int(cfg.get("max_steps") or 100),
        "persistent_episode_ledger_enabled": bool(
            cfg.get("persistent_episode_ledger_enabled", False)
        ),
        "persistent_episode_ledger_max_locations": int(
            cfg.get("persistent_episode_ledger_max_locations") or 64
        ),
        "persistent_episode_ledger_observation_chars": int(
            cfg.get("persistent_episode_ledger_observation_chars") or 240
        ),
        "exclude_step_limit_failures": bool(
            cfg.get("exclude_step_limit_failures", False)
        ),
        "step_limit_exclusion_policy": "step_limit_is_scored_failure_v2",
        "eval_overlength_retry_unbounded": bool(
            cfg.get("eval_overlength_retry_unbounded", False)
        ),
        "eval_overlength_retry_max_completion_tokens": "backend_default",
    }


def _prepare_rollout_context(
    rollout_dir: Path, *, split: str, seed: int, skill: str, cfg: dict[str, Any]
) -> None:
    """Reuse checkpoints only when split, seed, prompt, and output policy match."""
    context = _rollout_context_payload(split=split, seed=seed, skill=skill, cfg=cfg)
    context_path = rollout_dir / "rollout_context.json"
    previous = None
    if context_path.is_file():
        try:
            previous = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    has_legacy_payload = previous is None and any(rollout_dir.iterdir())
    if has_legacy_payload or (previous is not None and previous != context):
        reason = "missing_context" if has_legacy_payload else "context_mismatch"
        shutil.rmtree(rollout_dir)
        rollout_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[graphopt resume] invalidated stale rollout cache: {rollout_dir} "
            f"reason={reason}",
            flush=True,
        )
    save_json(context_path, context)


def aggregate_final_test_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeats with minimum, maximum, and mean plus half-range.

    Paper-facing outputs expose min, max, and mean(scores) with
    (max(scores)-min(scores))/2 as the spread.
    """
    if not runs:
        raise ValueError("final-test aggregation requires at least one run")
    counts = {int(run.get("n") or 0) for run in runs}
    if len(counts) != 1:
        raise ValueError(f"final-test repeats have different case counts: {sorted(counts)}")

    hard = _range_summary([float(run.get("hard") or 0.0) for run in runs])
    soft = _range_summary([float(run.get("soft") or 0.0) for run in runs])
    gate = _range_summary([float(run.get("gate") or 0.0) for run in runs])
    for summary in (hard, soft, gate):
        values = summary["values"]
        summary["mean"] = statistics.fmean(values)
        summary["std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    raw_metric_maps = [
        (run.get("raw_task_metrics") or run.get("task_metrics") or {})
        for run in runs
    ]
    metric_key_sets = [set(metric_map.keys()) for metric_map in raw_metric_maps]
    if any(keys != metric_key_sets[0] for keys in metric_key_sets[1:]):
        repeat_keys = {
            int(run.get("repeat") or index + 1): sorted(metric_key_sets[index])
            for index, run in enumerate(runs)
        }
        raise ValueError(
            "final-test repeats disagree on task metric buckets: "
            f"{repeat_keys}"
        )
    metric_keys = sorted(metric_key_sets[0])
    metrics: dict[str, dict[str, Any]] = {}
    for key in metric_keys:
        items = [(run.get("task_metrics") or {}).get(key) or {} for run in runs]
        raw_items = [metric_map.get(key) or {} for metric_map in raw_metric_maps]
        raw_count_values = [int(item.get("n") or 0) for item in raw_items]
        raw_metric_counts = set(raw_count_values)
        if len(raw_metric_counts) != 1:
            repeat_counts = {
                int(run.get("repeat") or index + 1): raw_count_values[index]
                for index, run in enumerate(runs)
            }
            raise ValueError(
                f"final-test repeats disagree on raw {key!r} case count: "
                f"{repeat_counts}"
            )
        effective_counts = [int(item.get("n") or 0) for item in items]
        hard_metric = _range_summary(
            [float(item.get("hard_percent") or 0.0) for item in items]
        )
        soft_metric = _range_summary(
            [float(item.get("soft_percent") or 0.0) for item in items]
        )
        for summary in (hard_metric, soft_metric):
            values = summary["values"]
            summary["mean"] = statistics.fmean(values)
            summary["std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        metrics[key] = {
            "label": str(items[0].get("label") or key),
            "n": next(iter(raw_metric_counts)),
            "effective_n_values": effective_counts,
            "hard": hard_metric["mean"] / 100.0,
            "soft": soft_metric["mean"] / 100.0,
            "hard_percent": hard_metric["mean"],
            "soft_percent": soft_metric["mean"],
            "hard_mean_percent": hard_metric["mean"],
            "hard_std_percent": hard_metric["std"],
            "hard_min_percent": hard_metric["min"],
            "hard_max_percent": hard_metric["max"],
            "hard_half_range_percent": hard_metric["half_range"],
            "hard_repeat_values_percent": hard_metric["values"],
            "soft_min_percent": soft_metric["min"],
            "soft_max_percent": soft_metric["max"],
            "soft_half_range_percent": soft_metric["half_range"],
            "soft_repeat_values_percent": soft_metric["values"],
            "soft_mean_percent": soft_metric["mean"],
            "soft_std_percent": soft_metric["std"],
        }

    return {
        "aggregation": "min_max_mean_plus_half_range",
        "spread_name": "half_range",
        "spread_formula": "(max(scores) - min(scores)) / 2",
        "spread_is_standard_deviation": False,
        "test_repeats": len(runs),
        "n": next(iter(counts)),
        "effective_n_values": [int(run.get("effective_n", run.get("n") or 0)) for run in runs],
        "excluded_n_values": [int(run.get("excluded_n") or 0) for run in runs],
        "excluded_case_ids_by_repeat": [list(run.get("excluded_case_ids") or []) for run in runs],
        "hard": hard["mean"],
        "soft": soft["mean"],
        "gate": gate["mean"],
        "hard_mean": hard["mean"],
        "soft_mean": soft["mean"],
        "gate_mean": gate["mean"],
        "hard_std": hard["std"],
        "soft_std": soft["std"],
        "gate_std": gate["std"],
        "hard_min": hard["min"],
        "hard_max": hard["max"],
        "hard_half_range": hard["half_range"],
        "hard_repeat_values": hard["values"],
        "soft_min": soft["min"],
        "soft_max": soft["max"],
        "soft_half_range": soft["half_range"],
        "soft_repeat_values": soft["values"],
        "gate_min": gate["min"],
        "gate_max": gate["max"],
        "gate_half_range": gate["half_range"],
        "gate_repeat_values": gate["values"],
        "task_metrics": metrics,
    }


def _snapshot_case_analyzer_cache(step_dir: Path, backup_dir: Path) -> int:
    """Snapshot only exact-input teacher case artifacts from an incomplete step."""
    index_path = step_dir / "artifact_index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    artifacts = index.get("artifacts") or {}
    safe = {
        key: list(entries)
        for key, entries in artifacts.items()
        if isinstance(key, str)
        and key.startswith("llm/case_analyze/")
        and isinstance(entries, list)
    }
    if not safe:
        return 0

    temporary = backup_dir.with_name(backup_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    copied = 0
    filtered: dict[str, list[dict[str, Any]]] = {}
    for key, entries in safe.items():
        kept: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("file"):
                continue
            relative = Path(str(entry["file"]))
            source = step_dir / relative
            if not source.is_file():
                continue
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            kept.append(dict(entry))
            copied += 1
        if kept:
            filtered[key] = kept
    if not filtered:
        shutil.rmtree(temporary)
        return 0
    save_json(
        temporary / "artifact_index.json",
        {
            "schema_version": "graphopt-incomplete-analyzer-cache-v1",
            "artifacts": filtered,
            "source_step": step_dir.name,
        },
    )
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    temporary.replace(backup_dir)
    return copied


def _restore_case_analyzer_cache(backup_dir: Path, step_dir: Path) -> int:
    """Restore a safe Analyzer-only snapshot after unsafe step state is cleared."""
    index_path = backup_dir / "artifact_index.json"
    if not index_path.is_file():
        return 0
    if step_dir.exists():
        shutil.rmtree(step_dir)
    shutil.copytree(backup_dir, step_dir)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return sum(
        len(entries)
        for entries in (payload.get("artifacts") or {}).values()
        if isinstance(entries, list)
    )


def _recover_committed_state(out: Path) -> bool:
    """Recover a Gate-complete epoch written by the pre-checkpoint code.

    Older runs wrote graph/Gate/meta first and delayed ``trainer_state.json``
    until three diagnostic test passes had finished.  Those artifacts are a
    complete commit record and must not be discarded merely because a test was
    interrupted.
    """
    if (out / "trainer_state.json").is_file():
        return False
    history: list[dict[str, Any]] = []
    epoch = 1
    while True:
        step_dir = out / "steps" / f"step_{epoch:04d}"
        required = [
            step_dir / "gate.json",
            step_dir / "evolution_cache.json",
            out / "graphs" / f"graph_step{epoch:04d}.json",
            out / f"meta_epoch_{epoch:02d}.txt",
        ]
        if not all(path.is_file() for path in required):
            break
        try:
            gate = json.loads(required[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        if int(gate.get("epoch") or 0) != epoch or "accepted" not in gate:
            break
        history.append(gate)
        epoch += 1
    completed = len(history)
    if completed == 0:
        return False

    last = history[-1]
    step_dir = out / "steps" / f"step_{completed:04d}"
    action = str(last.get("action") or "")
    results_candidates = [step_dir / "gate_committed_results.json"]
    if bool(last.get("accepted")):
        if "partial" in action:
            results_candidates.append(
                out / "rollouts" / f"step_{completed:04d}_val_g2_r1" / "results.json"
            )
        results_candidates.append(
            out / "rollouts" / f"step_{completed:04d}_val_r1" / "results.json"
        )
    else:
        results_candidates.append(step_dir / "gate_reference_results.json")
    results_path = next((path for path in results_candidates if path.is_file()), None)
    if results_path is None:
        return False
    try:
        last_results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(last_results, list) or not last_results:
        return False

    baseline_path = out / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {}
    best_score = float(baseline.get("gate") or 0.0)
    best_step = 0
    for record in history:
        value = float(record.get("val_score") or 0.0)
        if record.get("selection_policy") == "unconditional_terminal_graph":
            best_score = value
            best_step = int(record.get("step") or record.get("epoch") or 0)
        elif bool(record.get("accepted")) and value >= best_score - 1e-12:
            best_score = max(best_score, value)
            best_step = int(record.get("step") or record.get("epoch") or 0)
    payload = {
        "schema_version": "graphopt-trainer-state-v1-recovered",
        "completed_epochs": completed,
        "graph_path": f"graphs/graph_step{completed:04d}.json",
        "best_graph_path": "best_graph.json",
        "evolution_cache_path": f"steps/step_{completed:04d}/evolution_cache.json",
        "current_score": float(last.get("val_score", last.get("current_score", 0.0))),
        "best_score": best_score,
        "best_step": best_step,
        "history": history,
        "epoch_archives": [],
        "last_selection_results": last_results,
        "meta": (out / f"meta_epoch_{completed:02d}.txt").read_text(encoding="utf-8"),
        "recovered_from_committed_artifacts": True,
    }
    temporary = out / "trainer_state.json.recovering"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(out / "trainer_state.json")
    print(
        f"[graphopt resume] recovered committed epoch={completed} from graph/Gate/meta artifacts; "
        "pending tests will resume separately",
        flush=True,
    )
    return True


def diff_skill_graphs(base: SkillGraph, candidate: SkillGraph) -> GraphPatch:
    """Collapse sequential batch edits into one exact epoch-level patch.

    Every connected component of newly added nodes and all edges incident to
    that component share one atomic group. This prevents a selective rollback
    from keeping a dependent edge while removing the node it references.
    """
    edits: list[GraphEdit] = []
    base_nodes = set(base.nodes)
    candidate_nodes = set(candidate.nodes)
    added_nodes = candidate_nodes - base_nodes
    removed_nodes = base_nodes - candidate_nodes

    parent: dict[str, str] = {node_id: node_id for node_id in added_nodes}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for edge in candidate.edges:
        if edge.src in added_nodes and edge.dst in added_nodes:
            union(edge.src, edge.dst)
    components: dict[str, list[str]] = {}
    for node_id in sorted(added_nodes):
        components.setdefault(find(node_id), []).append(node_id)
    component_group = {
        node_id: "new_component:" + "+".join(members)
        for members in components.values() for node_id in members
    }

    for node_id in sorted(removed_nodes):
        edits.append(GraphEdit(op="delete_node", node_id=node_id, group_id=f"node:{node_id}"))
    for node_id in sorted(added_nodes):
        node = candidate.nodes[node_id]
        parent_node = next((
            edge.src for edge in candidate.edges
            if edge.dst == node_id and normalize_edge_type(edge.type) == "enhance"
        ), "")
        edits.append(GraphEdit(
            op="add_node", node_id=node_id, name=node.title,
            meaning=node.meaning, when_to_use=node.when_to_use,
            how_to_use=node.how_to_use, avoid=list(node.avoid),
            category=node.category, parent_node=parent_node,
            group_id=component_group[node_id], source_type="epoch_batch_diff",
        ))
    for node_id in sorted(base_nodes & candidate_nodes):
        before = base.nodes[node_id]
        after = candidate.nodes[node_id]
        before_fields = (
            before.meaning, before.when_to_use, before.how_to_use,
            tuple(before.avoid), before.category,
        )
        after_fields = (
            after.meaning, after.when_to_use, after.how_to_use,
            tuple(after.avoid), after.category,
        )
        if before_fields != after_fields:
            edits.append(GraphEdit(
                op="update_node", node_id=node_id,
                meaning=after.meaning, when_to_use=after.when_to_use,
                how_to_use=after.how_to_use, avoid=list(after.avoid),
                category=after.category, group_id=f"node:{node_id}",
                source_type="epoch_batch_diff",
            ))

    def edge_map(graph: SkillGraph) -> dict[tuple[str, str, str], Any]:
        return {
            (edge.src, edge.dst, normalize_edge_type(edge.type)): edge
            for edge in graph.edges
        }

    before_edges = edge_map(base)
    after_edges = edge_map(candidate)
    for key in sorted(before_edges.keys() - after_edges.keys()):
        src, dst, typ = key
        if src in removed_nodes or dst in removed_nodes:
            continue
        edits.append(GraphEdit(
            op="delete_edge", src=src, dst=dst, edge_type=typ,
            group_id=f"edge:{src}:{dst}", source_type="epoch_batch_diff",
        ))
    for key in sorted(after_edges):
        after = after_edges[key]
        before = before_edges.get(key)
        changed = before is None or (
            float(before.w), before.rationale, before.active
        ) != (float(after.w), after.rationale, after.active)
        if changed:
            src, dst, typ = key
            incident_new = src if src in added_nodes else dst if dst in added_nodes else ""
            edits.append(GraphEdit(
                op="add_edge", src=src, dst=dst, edge_type=typ,
                w=float(after.w), rationale=after.rationale,
                group_id=(component_group[incident_new] if incident_new else f"edge:{src}:{dst}"),
                source_type="epoch_batch_diff",
            ))
    return GraphPatch(reasoning="collapsed accepted small-G batch edits", edits=edits)


def _edit_provenance_keys(edit: GraphEdit) -> set[str]:
    """Return stable graph-element keys used to carry batch provenance forward."""
    keys: set[str] = set()
    if edit.node_id:
        keys.add(f"node:{edit.node_id}")
    if edit.src and edit.dst:
        keys.add(
            f"edge:{edit.src}:{edit.dst}:{normalize_edge_type(edit.edge_type)}"
        )
    return keys


def _bounded_local_gate_text(value: Any, limit: int) -> Any:
    """Bound teacher-only evidence text without changing measured artifacts."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated {len(value) - limit} chars]"


def _local_gate_case_view(row: dict[str, Any], *, detailed: bool) -> dict[str, Any]:
    """Build a bounded outcome/trace view of one rollout for Local-Gate reasoning.

    Raw rollout rows can recursively carry all same-group successes. Passing those
    blobs for every affected case made broad-node gates exceed the teacher context.
    The measured rows remain untouched on disk; shared question/gold fields are
    stored once by the pair view instead of duplicated before/after.
    """
    view: dict[str, Any] = {}
    for key in (
        "predicted_answer", "hard", "graph_usage", "graph_refs",
    ):
        value = row.get(key)
        if value not in (None, "", [], {}):
            view[key] = value
    if detailed:
        status = row.get("semantic_reasoning_trace_status")
        if status not in (None, ""):
            view["semantic_reasoning_trace_status"] = status
        trace = row.get("semantic_reasoning_trace") or row.get("reasoning_trace")
        if trace not in (None, ""):
            view["semantic_reasoning_trace"] = _bounded_local_gate_text(trace, 5000)
        for key in ("evaluator_feedback", "fail_reason"):
            value = row.get(key)
            if value not in (None, ""):
                view[key] = _bounded_local_gate_text(value, 1600)
        if "semantic_reasoning_trace" not in view:
            response = row.get("response")
            if isinstance(response, str) and response.strip():
                view["response_tail"] = _bounded_local_gate_text(response[-5000:], 5000)
    return view


def _local_gate_pair_view(row: dict[str, Any], *, detailed: bool) -> dict[str, Any]:
    before = row.get("before") if isinstance(row.get("before"), dict) else {}
    after = row.get("after") if isinstance(row.get("after"), dict) else {}
    question = before.get("question") or before.get("task_description")
    if question in (None, ""):
        question = after.get("question") or after.get("task_description")
    gold = (
        before.get("gold_answers") or before.get("reference_answer")
        or after.get("gold_answers") or after.get("reference_answer")
    )
    pair = {
        "case_id": str(row.get("case_id") or ""),
        "category": str(row.get("category") or ""),
        "before": _local_gate_case_view(before, detailed=detailed),
        "after": _local_gate_case_view(after, detailed=detailed),
    }
    if question not in (None, ""):
        pair["question"] = _bounded_local_gate_text(question, 800)
    if gold not in (None, "", [], {}):
        pair["gold_answers"] = gold
    return pair


def _build_local_gate_teacher_evidence(
    gate_evidence: dict[str, Any],
    *,
    max_payload_chars: int = 600_000,
) -> dict[str, Any]:
    """Represent every A/B case compactly and decisive cases in trace detail."""
    pairs = [
        row for row in (gate_evidence.get("paired_cases") or [])
        if isinstance(row, dict)
    ]
    summaries = [
        _local_gate_pair_view(row, detailed=False) for row in pairs
    ]

    # Every voting case gets priority for full trace evidence. Protected successes
    # are all present above and a deterministic spread is also expanded below.
    decisive = [
        row for row in pairs
        if str(row.get("category") or "") in {
            "effective", "harmful", "unresolved_source",
        }
    ]
    protected = [
        row for row in pairs
        if str(row.get("category") or "") == "protected_success"
    ]
    if protected:
        stride = max(1, len(protected) // 16)
        protected = protected[::stride][:16]
    detail_candidates = [*decisive, *protected]
    detailed_pairs: list[dict[str, Any]] = []
    base = {
        "coverage": {
            "all_paired_case_count": len(pairs),
            "all_case_summaries_complete": True,
            "detailed_pair_count": 0,
            "detail_policy": (
                "all effective/harmful/unresolved-source cases first, then a "
                "deterministic spread of protected successes within the payload budget"
            ),
        },
        "all_case_summaries": summaries,
        "detailed_trace_pairs": detailed_pairs,
    }
    for row in detail_candidates:
        detail = _local_gate_pair_view(row, detailed=True)
        detailed_pairs.append(detail)
        base["coverage"]["detailed_pair_count"] = len(detailed_pairs)
        if len(json.dumps(base, ensure_ascii=False, separators=(",", ":"))) > max_payload_chars:
            detailed_pairs.pop()
            base["coverage"]["detailed_pair_count"] = len(detailed_pairs)
            base["coverage"]["detail_budget_exhausted"] = True
            break
    return base


def attach_epoch_patch_provenance(
    patch: GraphPatch,
    small_gate_records: list[dict[str, Any]],
) -> GraphPatch:
    """Restore accepted batch/group provenance lost by a G0/G1 diff.

    The exact epoch diff is intentionally reconstructed from graph state, but
    graph state does not contain optimizer-only source_case_ids. Match its
    final node/edge edits back to accepted small-G edits. If an updater edit
    omitted direct case provenance (notably weight refreshes), its accepted
    batch is the conservative provenance fallback, so the big Gate can still
    expand the edit to real fixed update groups.
    """
    sources_by_key: dict[str, set[str]] = {}
    for record in small_gate_records:
        if not bool(record.get("accepted")):
            continue
        batch_sources = {
            str(case_id) for case_id in (record.get("train_case_ids") or [])
            if str(case_id)
        }
        for row in record.get("edit_attribution") or []:
            if not bool(row.get("keep")) or not isinstance(row.get("edit"), dict):
                continue
            accepted_edit = GraphEdit.from_dict(row["edit"])
            direct_sources = {
                str(case_id) for case_id in accepted_edit.source_case_ids
                if str(case_id)
            }
            provenance = direct_sources or batch_sources
            for key in _edit_provenance_keys(accepted_edit):
                sources_by_key.setdefault(key, set()).update(provenance)

    for edit in patch.edits:
        restored = set(edit.source_case_ids)
        for key in _edit_provenance_keys(edit):
            restored.update(sources_by_key.get(key, set()))
        edit.source_case_ids = sorted(restored)
    return patch



def explain_epoch_local_gate(
    patch: GraphPatch,
    gate_evidence: dict[str, Any],
    *,
    chat_fn: Any,
) -> dict[str, Any]:
    """Explain each member of a joint edit from complete decisive A/B traces."""
    if chat_fn is None:
        return {
            "status": "deterministic_only_no_teacher",
            "joint_summary": str(gate_evidence.get("decision_reason") or ""),
            "edits": copy.deepcopy(gate_evidence.get("per_edit_explanations") or []),
        }
    teacher_evidence = _build_local_gate_teacher_evidence(gate_evidence)
    payload = {
        "joint_patch": patch.to_dict(),
        "decision": {
            key: gate_evidence.get(key) for key in (
                "accepted", "n_effective", "n_ineffective",
                "effective_case_ids", "ineffective_case_ids",
                "decision_reason",
            )
        },
        "measured_before_after_evidence": teacher_evidence,
    }
    system = """Explain a measured local joint graph edit. Read every compact paired
case summary and use the expanded decisive/protected traces for causal detail. Use only
the supplied questions, reasoning traces, graph_usage, answers, and hard outcomes. Return
strict JSON with keys joint_summary and edits. edits must contain one object per edit:
{"edit_index":0,"effective_mechanism":"...","ineffective_mechanism":"...","observable_boundary":"...","evidence_case_ids":["..."]}.
Do not claim that one member was independently effective because shared-source node/edge
edits were intervened on jointly. Distinguish observed facts from hypotheses."""
    from graphopt.json_utils import extract_json

    valid_ids = set(map(str, gate_evidence.get("affected_case_ids") or []))
    expected_indices = list(range(len(patch.edits)))
    attempts: list[dict[str, Any]] = []
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for attempt in range(1, 3):
        response = ""
        usage: Any = None
        try:
            response, usage = chat_fn(
                system=system, user=user, max_completion_tokens=4096,
                retries=2, stage="epoch_local_gate_explanation",
            )
            obj = extract_json(response)
            if not isinstance(obj, dict) or set(obj) != {"joint_summary", "edits"}:
                raise ValueError("explanation must contain joint_summary and edits")
            if not isinstance(obj["joint_summary"], str) or not obj["joint_summary"].strip():
                raise ValueError("joint_summary must be non-empty")
            edits = obj["edits"]
            if not isinstance(edits, list) or sorted(
                int(row.get("edit_index", -1)) for row in edits
                if isinstance(row, dict)
            ) != expected_indices:
                raise ValueError("explanation must cover every edit index exactly once")
            required = {
                "edit_index", "effective_mechanism", "ineffective_mechanism",
                "observable_boundary", "evidence_case_ids",
            }
            for row in edits:
                if not isinstance(row, dict) or set(row) != required:
                    raise ValueError("each explanation has the exact five fields")
                if any(
                    not isinstance(row[key], str) or not row[key].strip()
                    for key in (
                        "effective_mechanism", "ineffective_mechanism",
                        "observable_boundary",
                    )
                ):
                    raise ValueError("explanation text fields must be non-empty")
                citations = row["evidence_case_ids"]
                if (
                    not isinstance(citations, list)
                    or any(not isinstance(case_id, str) for case_id in citations)
                    or len(citations) != len(set(citations))
                    or not set(citations) <= valid_ids
                ):
                    raise ValueError("explanation citations must be unique affected IDs")
            attempts.append({"attempt": attempt, "usage": usage})
            return {"status": "complete", **obj, "attempts": attempts}
        except Exception as exc:
            attempts.append({
                "attempt": attempt, "response": response or None,
                "usage": usage, "error": str(exc),
            })
            user = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n\nPrevious response was invalid: " + str(exc)
                + "\nReturn the complete strict JSON again."
            )
    return {
        "status": "invalid_twice_deterministic_fallback",
        "joint_summary": str(gate_evidence.get("decision_reason") or ""),
        "edits": copy.deepcopy(gate_evidence.get("per_edit_explanations") or []),
        "attempts": attempts,
    }


def summarize_epoch_local_gate(
    patch: GraphPatch,
    *,
    reference_train: list[dict[str, Any]],
    candidate_train: list[dict[str, Any]],
    reference_val: list[dict[str, Any]],
    candidate_val: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count strict local votes and retain only effective > ineffective."""
    _require_same_case_ids(reference_train, candidate_train)
    _require_same_case_ids(reference_val, candidate_val)
    before = {
        str(row.get("id")): row
        for row in [*reference_train, *reference_val]
    }
    after = {
        str(row.get("id")): row
        for row in [*candidate_train, *candidate_val]
    }
    if set(before) != set(after):
        raise ValueError("local Gate train/validation union must be paired")
    improved: list[str] = []
    regressed: list[str] = []
    persistent_fail: list[str] = []
    stable_success: list[str] = []
    for case_id in before:
        old = float(before[case_id].get("hard") or 0.0)
        new = float(after[case_id].get("hard") or 0.0)
        if old < 1.0 <= new:
            improved.append(case_id)
        elif new < 1.0 <= old:
            regressed.append(case_id)
        elif old < 1.0 and new < 1.0:
            persistent_fail.append(case_id)
        else:
            stable_success.append(case_id)
    source_ids = {
        str(case_id) for edit in patch.edits
        for case_id in edit.source_case_ids if str(case_id)
    }
    unresolved_sources = sorted(source_ids.intersection(persistent_fail))
    # A persistent failure is attributable to this candidate only when it is
    # direct source evidence for one of the edits. Other affected 0→0 cases are
    # valuable diagnostics, but counting all of them as negative votes makes a
    # broad-scope node mathematically impossible to improve.
    ineffective = sorted(set(regressed) | set(unresolved_sources))
    accepted = len(improved) > len(ineffective)
    pairs = [{
        "case_id": case_id,
        "category": (
            "effective" if case_id in improved
            else "harmful" if case_id in regressed
            else "unresolved_source" if case_id in unresolved_sources
            else "persistent_failure" if case_id in persistent_fail
            else "protected_success"
        ),
        "before": before[case_id],
        "after": after[case_id],
    } for case_id in before]
    decision_reason = (
        f"KEEP: effective={len(improved)} > ineffective={len(ineffective)}"
        if accepted else
        f"REVISE_OR_DROP: effective={len(improved)} <= ineffective={len(ineffective)}"
    )
    per_edit = [{
        "edit_index": index,
        "edit": edit.to_dict(),
        "joint_group_id": str(edit.group_id or ""),
        "decision": "KEEP_WITH_JOINT_GROUP" if accepted else "REVISE_WITH_JOINT_GROUP",
        "effective_case_ids": sorted(improved),
        "ineffective_case_ids": ineffective,
        "reason": decision_reason,
        "individual_isolation": False,
        "why_not_isolated": (
            "edits sharing source cases or graph dependencies are one local intervention"
        ),
    } for index, edit in enumerate(patch.edits)]
    return {
        "schema_version": "graphopt-epoch-local-gate-v2",
        "policy": "effective_greater_than_harmful_plus_unresolved_sources",
        "accepted": accepted,
        "n_effective": len(improved),
        "n_ineffective": len(ineffective),
        "effective_case_ids": sorted(improved),
        "harmful_case_ids": sorted(regressed),
        "unresolved_source_case_ids": unresolved_sources,
        "ineffective_case_ids": ineffective,
        "persistent_failure_case_ids": sorted(persistent_fail),
        "protected_success_case_ids": sorted(stable_success),
        "affected_case_ids": list(before),
        "source_case_ids": sorted(source_ids),
        "decision_reason": decision_reason,
        "paired_cases": pairs,
        "per_edit_explanations": per_edit,
    }


def _local_refinement_units(patch: GraphPatch) -> list[dict[str, Any]]:
    """Bind each semantic node edit to its dependent new-node activation edges."""
    node_indices = [
        index for index, edit in enumerate(patch.edits)
        if edit.op in {"update_node", "add_node"}
    ]
    new_node_owner = {
        str(patch.edits[index].node_id): index for index in node_indices
        if patch.edits[index].op == "add_node"
        and str(patch.edits[index].node_id or "")
    }
    members: dict[int, list[int]] = {index: [index] for index in node_indices}
    standalone: list[int] = []
    for index, edit in enumerate(patch.edits):
        if index in members:
            continue
        owners = sorted({
            new_node_owner[node_id]
            for node_id in (str(edit.src or ""), str(edit.dst or ""))
            if node_id in new_node_owner
        })
        if owners:
            members[owners[0]].append(index)
        else:
            standalone.append(index)
    units: list[dict[str, Any]] = []
    for primary_index in [*node_indices, *standalone]:
        member_indices = members.get(primary_index, [primary_index])
        units.append({
            "unit_index": len(units),
            "kind": (
                "semantic_node_with_dependencies"
                if primary_index in members else "standalone_structural_edit"
            ),
            "primary_patch_edit_index": primary_index,
            "member_patch_edit_indices": member_indices,
            "primary_edit": patch.edits[primary_index].to_dict(),
            "dependent_edits": [
                patch.edits[index].to_dict()
                for index in member_indices if index != primary_index
            ],
        })
    return units


def refine_rejected_joint_patch(
    *,
    base_graph: SkillGraph,
    patch: GraphPatch,
    gate_evidence: dict[str, Any],
    chat_fn: Any,
) -> tuple[GraphPatch, dict[str, Any]]:
    """Regenerate a rejected node/edge component from all paired Gate evidence."""
    empty = GraphPatch(reasoning="no precise local-Gate refinement", edits=[])
    if chat_fn is None:
        return empty, {"status": "no_teacher", "attempts": []}
    editable = [
        edit for edit in patch.edits if edit.op in {"update_node", "add_node"}
    ]
    editable_ids = list(dict.fromkeys(
        str(edit.node_id) for edit in editable if str(edit.node_id)
    ))
    resolution_units = _local_refinement_units(patch)
    incident_ids = sorted({
        str(value) for edit in patch.edits
        for value in (edit.node_id, edit.src, edit.dst)
        if str(value or "")
    })

    ineffective_ids = set(map(str, gate_evidence.get("ineffective_case_ids") or []))
    valid_ids = set(map(str, gate_evidence.get("affected_case_ids") or []))
    teacher_evidence = _build_local_gate_teacher_evidence(gate_evidence)
    compact_gate = {
        key: copy.deepcopy(gate_evidence.get(key)) for key in (
            "accepted", "policy", "n_effective", "n_ineffective",
            "effective_case_ids", "harmful_case_ids",
            "unresolved_source_case_ids", "ineffective_case_ids",
            "protected_success_case_ids", "source_case_ids", "decision_reason",
        )
    }
    compact_gate["measured_before_after_evidence"] = teacher_evidence
    payload = {
        "schema_version": "graphopt-local-gate-refinement-input-v3",
        "current_subgraph": {
            "nodes": {
                node_id: base_graph.nodes[node_id].to_dict()
                for node_id in incident_ids if node_id in base_graph.nodes
            },
            "incident_edges": [
                edge.to_dict() for edge in base_graph.edges
                if edge.src in incident_ids or edge.dst in incident_ids
            ],
        },
        "rejected_joint_patch": patch.to_dict(),
        "resolution_units": resolution_units,
        "local_gate_evidence": compact_gate,
        "required_node_ids": editable_ids,
    }
    system = """You refine one rejected local joint graph component. Read every compact
before/after case summary, including every protected success, and use the expanded trace
pairs for detailed causal evidence. Resolve all node and edge edits together. Diagnose why
effective cases improved and why harmful or unresolved source cases failed.
Use an observable conditional branch when cases require different actions. Preserve every
initialized node meaning verbatim. You may keep or drop each supplied structural edit, but
must not invent edits, node IDs, endpoints, facts, or relation types. ABSTAIN if the paired
evidence has no reliable observable separator.

Return strict JSON with exactly decision, diagnosis, evidence_case_ids, resolved_units.
decision is REVISE or ABSTAIN. diagnosis has exactly why_effective, why_ineffective,
observable_separator. resolved_units contains exactly one row per supplied resolution unit
with unit_index, keep, when_to_use, how_to_use, avoid, rationale. A semantic-node unit
already includes its dependent activation edges: decide it once and do not emit separate
rows for those edges. For a kept semantic-node unit, return its complete final
when_to_use/how_to_use/avoid and retain the original initialized text verbatim. For a
standalone structural unit or a dropped unit, those three fields are null. REVISE must cite
an ineffective case and must actually change or drop at least one unit. Conditions must be
observable before the decision; labels, reference answers, and case IDs are forbidden as
conditions."""
    from graphopt.json_utils import extract_json

    attempts: list[dict[str, Any]] = []
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for attempt in range(1, 4):
        response = ""
        usage: Any = None
        try:
            response, usage = chat_fn(
                system=system, user=user, max_completion_tokens=4096,
                retries=2, stage="epoch_local_gate_precise_refinement",
            )
            obj = extract_json(response)
            required = {
                "decision", "diagnosis", "evidence_case_ids", "resolved_units",
            }
            if not isinstance(obj, dict) or set(obj) != required:
                raise ValueError("refinement must contain exactly the required keys")
            if obj["decision"] not in {"REVISE", "ABSTAIN"}:
                raise ValueError("decision must be REVISE or ABSTAIN")
            diagnosis = obj["diagnosis"]
            if not isinstance(diagnosis, dict) or set(diagnosis) != {
                "why_effective", "why_ineffective", "observable_separator",
            } or any(
                not isinstance(diagnosis[key], str) or not diagnosis[key].strip()
                for key in diagnosis
            ):
                raise ValueError("diagnosis must contain three non-empty strings")
            citations = obj["evidence_case_ids"]
            if (
                not isinstance(citations, list)
                or any(not isinstance(case_id, str) for case_id in citations)
                or len(citations) != len(set(citations))
                or not set(citations) <= valid_ids
            ):
                raise ValueError("evidence_case_ids must be unique affected case IDs")
            if obj["decision"] == "ABSTAIN":
                attempts.append({"attempt": attempt, "parsed": obj, "usage": usage})
                return empty, {
                    "status": "teacher_abstained", "attempts": attempts,
                    "diagnosis": diagnosis,
                }
            if ineffective_ids and not set(citations).intersection(ineffective_ids):
                raise ValueError("REVISE must cite an ineffective or harmful case")
            rows = obj["resolved_units"]
            row_fields = {
                "unit_index", "keep", "when_to_use", "how_to_use", "avoid",
                "rationale",
            }
            expected_indices = set(range(len(resolution_units)))
            if (
                not isinstance(rows, list) or len(rows) != len(resolution_units)
                or any(not isinstance(row, dict) or set(row) != row_fields for row in rows)
                or {int(row.get("unit_index", -1)) for row in rows} != expected_indices
            ):
                raise ValueError(
                    "resolved_units must cover every resolution unit exactly once"
                )
            by_index = {int(row["unit_index"]): row for row in rows}
            refined_by_index: dict[int, GraphEdit] = {}
            changed = False
            kept_new_nodes: set[str] = set()
            for unit in resolution_units:
                unit_index = int(unit["unit_index"])
                row = by_index[unit_index]
                primary_index = int(unit["primary_patch_edit_index"])
                original = patch.edits[primary_index]
                if not isinstance(row["keep"], bool):
                    raise ValueError("keep must be boolean")
                if not isinstance(row["rationale"], str) or not row["rationale"].strip():
                    raise ValueError("every unit resolution needs a rationale")
                node_fields = (row["when_to_use"], row["how_to_use"], row["avoid"])
                if not row["keep"]:
                    if any(value is not None for value in node_fields):
                        raise ValueError("dropped units must use null node fields")
                    changed = True
                    continue
                primary = copy.deepcopy(original)
                if primary.op in {"update_node", "add_node"}:
                    when, how, avoid = node_fields
                    if (
                        not isinstance(when, str) or not when.strip()
                        or not isinstance(how, str) or not how.strip()
                        or not isinstance(avoid, list)
                        or any(not isinstance(value, str) for value in avoid)
                    ):
                        raise ValueError("kept node edits need complete when/how/avoid")
                    if primary.op == "update_node":
                        node = base_graph.nodes.get(str(primary.node_id))
                        if node is None:
                            raise ValueError("refined update targets a missing node")
                        if node.when_to_use.strip() and node.when_to_use.strip() not in when:
                            raise ValueError("refinement removed initialized when_to_use")
                        if node.how_to_use.strip() and node.how_to_use.strip() not in how:
                            raise ValueError("refinement removed initialized how_to_use")
                        if not set(node.avoid).issubset(set(avoid)):
                            raise ValueError("refinement removed initialized avoid semantics")
                    else:
                        kept_new_nodes.add(str(primary.node_id))
                    changed = changed or (
                        primary.when_to_use != when.strip()
                        or primary.how_to_use != how.strip()
                        or primary.avoid != list(avoid)
                    )
                    primary.when_to_use = when.strip()
                    primary.how_to_use = how.strip()
                    primary.avoid = list(avoid)
                elif any(value is not None for value in node_fields):
                    raise ValueError("structural units must use null node fields")
                for member_index in unit["member_patch_edit_indices"]:
                    edit = (
                        primary if int(member_index) == primary_index
                        else copy.deepcopy(patch.edits[int(member_index)])
                    )
                    edit.reasoning = (
                        str(edit.reasoning or "").rstrip()
                        + " | local_gate_resolution: " + row["rationale"].strip()
                    ).strip(" |")
                    refined_by_index[int(member_index)] = edit
            if not changed:
                raise ValueError("REVISE must change or drop at least one unit")
            refined_edits = [
                refined_by_index[index] for index in sorted(refined_by_index)
            ]
            available_nodes = set(base_graph.nodes) | kept_new_nodes
            for edit in refined_edits:
                if edit.op == "add_edge" and (
                    edit.src not in available_nodes or edit.dst not in available_nodes
                ):
                    raise ValueError("kept edge points to a dropped new node")
            refined = GraphPatch(
                reasoning="precise retry after rejected epoch local Gate",
                edits=refined_edits,
            )
            trial = base_graph.copy()
            report = apply_patch(trial, refined)
            if report.get("n_failed"):
                raise ValueError("refined joint patch does not materialize atomically")
            attempts.append({"attempt": attempt, "parsed": obj, "usage": usage})
            return refined, {
                "status": "revised", "attempts": attempts,
                "diagnosis": diagnosis, "evidence_case_ids": citations,
            }
        except Exception as exc:
            attempts.append({
                "attempt": attempt, "response": response or None,
                "usage": usage, "error": str(exc),
            })
            user = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n\nPrevious output was invalid: " + str(exc)
                + "\nReturn the complete strict JSON again with a narrower "
                "observable separator and every required resolution unit."
            )
    return empty, {"status": "invalid_three_times", "attempts": attempts}


class Trainer:
    def __init__(
        self,
        *,
        init_graph: str | Path,
        out_root: str | Path,
        cfg: dict[str, Any] | None = None,
        adapter=None,
        rollout_fn: Callable[[str, str, dict], list[dict[str, Any]]] | None = None,
        chat_fn=None,
        reflect_mode: str = "template",
    ):
        self.cfg = dict(cfg or {})
        self.out = Path(out_root)
        self.out.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.dataloader = adapter.get_dataloader() if adapter is not None else None
        self.initial_graph_path = Path(init_graph).expanduser().resolve()
        self.initial_graph_file_sha256 = hashlib.sha256(
            self.initial_graph_path.read_bytes()
        ).hexdigest()
        initial_lock_path = self.out / "initial_graph_lock.json"
        if initial_lock_path.is_file():
            previous_lock = json.loads(initial_lock_path.read_text(encoding="utf-8"))
            previous_hash = str(previous_lock.get("sha256") or "")
            if previous_hash and previous_hash != self.initial_graph_file_sha256:
                raise RuntimeError(
                    "initial graph changed since this run was created; "
                    "GraphOpt refuses to rewrite or silently replace G0"
                )
        save_json(initial_lock_path, {
            "schema_version": "graphopt-initial-graph-lock-v1",
            "path": str(self.initial_graph_path),
            "sha256": self.initial_graph_file_sha256,
            "policy": "read_only_input_never_mutated",
        })
        self.graph = load_graph(self.initial_graph_path)
        self.best = self.graph.copy()
        self.score = -1.0
        self.best_score = -1.0
        self.best_step = -1
        self.history: list[dict[str, Any]] = []
        self.epoch_archives: list[str] = []
        self.meta = ""
        self._last_sel_results: list[dict[str, Any]] = []
        self._selection_excluded_case_ids: set[str] | None = None
        self.resume_completed_epochs = 0
        self.resume_requested = bool(self.cfg.get("resume", False))
        self._restored_trainer_state = False
        self._resume_cache_path: Path | None = None
        self._pending_meta_epoch = 0
        self._pending_meta_prev_results: list[dict[str, Any]] = []
        self._pending_meta_prev_graph_text = ""
        meta_path = self.out / "meta.txt"
        if meta_path.exists():
            self.meta = meta_path.read_text(encoding="utf-8")
        if self.resume_requested:
            _recover_committed_state(self.out)
        if self.resume_requested and (self.out / "trainer_state.json").is_file():
            state = json.loads(
                (self.out / "trainer_state.json").read_text(encoding="utf-8")
            )
            graph_path = self.out / str(state["graph_path"])
            best_graph_path = self.out / str(
                state.get("best_graph_path") or "best_graph.json"
            )
            self.graph = load_graph(graph_path)
            self.best = load_graph(best_graph_path)
            self.score = float(state["current_score"])
            self.best_score = float(state["best_score"])
            self.best_step = int(state["best_step"])
            self.history = list(state.get("history") or [])
            self.epoch_archives = list(state.get("epoch_archives") or [])
            self._last_sel_results = list(state.get("last_selection_results") or [])
            stored_exclusions = state.get("selection_excluded_case_ids")
            if stored_exclusions is not None:
                self._selection_excluded_case_ids = {
                    str(case_id) for case_id in stored_exclusions
                }
            elif self._last_sel_results:
                self._selection_excluded_case_ids = {
                    str(row.get("id"))
                    for row in self._last_sel_results
                    if row.get("exclude_from_metrics")
                }
            self.resume_completed_epochs = int(state.get("completed_epochs") or 0)
            pending_meta = state.get("pending_meta")
            if pending_meta is not None:
                if not isinstance(pending_meta, dict):
                    raise RuntimeError("trainer_state pending_meta must be an object or null")
                pending_epoch = int(pending_meta.get("epoch") or 0)
                previous_results = pending_meta.get("previous_selection_results")
                previous_graph_text = pending_meta.get("previous_graph_text")
                if (
                    pending_epoch != self.resume_completed_epochs
                    or not isinstance(previous_results, list)
                    or not isinstance(previous_graph_text, str)
                ):
                    raise RuntimeError(
                        "trainer_state pending_meta does not match its completed epoch"
                    )
                self._pending_meta_epoch = pending_epoch
                self._pending_meta_prev_results = list(previous_results)
                self._pending_meta_prev_graph_text = previous_graph_text
            cache_relative = str(state.get("evolution_cache_path") or "").strip()
            if not cache_relative:
                raise RuntimeError(
                    "trainer_state.json is missing evolution_cache_path; "
                    "safe resume requires an epoch-boundary cache snapshot"
                )
            self._resume_cache_path = self.out / cache_relative
            if not self._resume_cache_path.is_file():
                raise RuntimeError(
                    f"trainer state evolution cache is missing: {self._resume_cache_path}"
                )
            self.meta = str(state.get("meta") or self.meta)
            self._restored_trainer_state = True
            print(
                f"[graphopt resume] restored epoch={self.resume_completed_epochs} "
                f"graph={graph_path} current_score={self.score:.4f} "
                f"best_score={self.best_score:.4f}",
                flush=True,
            )
        if self.resume_requested:
            self._prepare_resume_workspace(self.resume_completed_epochs)
        self.rollout_fn = rollout_fn  # dry-run only; real path uses adapter
        self.chat_fn = chat_fn
        self.reflect_mode = reflect_mode
        self.evolution_cfg = EvolutionConfig.from_cfg(self.cfg)
        cache_path = self.out / "evolution_cache.json"
        cache_source = self._resume_cache_path if self._restored_trainer_state else (
            None if self.resume_requested else cache_path
        )
        self.evolution_cache = EvolutionCache.load(cache_source) or EvolutionCache.from_graph(
            self.graph, source_graph=str(init_graph)
        )
        self.evolution_cache.reconcile_graph(self.graph, source_graph=str(init_graph))
        if self.cfg.get("delete_used_threshold") is not None:
            self.evolution_cfg.delete_used_threshold = int(self.cfg["delete_used_threshold"])
        self.max_node = self.cfg.get("max_node_edits")
        self.max_edge = self.cfg.get("max_edge_edits")
        if self.max_node is not None:
            self.max_node = int(self.max_node)
        if self.max_edge is not None:
            self.max_edge = int(self.max_edge)
        # SkillAA paper-faithful evaluation selects graphs by hard task
        # success. Soft remains an audited diagnostic unless explicitly chosen.
        self.gate_metric = str(self.cfg.get("gate_metric") or "hard").strip().lower()
        if self.gate_metric not in {"hard", "soft", "mixed"}:
            raise ValueError("gate_metric must be one of: hard, soft, mixed")
        mixed_raw = self.cfg.get("gate_mixed_weight")
        self.mixed_weight = float(0.8 if mixed_raw is None else mixed_raw)
        if not 0.0 <= self.mixed_weight <= 1.0:
            raise ValueError("gate_mixed_weight must be in [0, 1]")
        legacy_use_gate = bool(self.cfg.get("use_gate", True))
        self.use_small_gate = bool(
            self.cfg.get("use_small_gate", legacy_use_gate)
        )
        self.use_big_gate = bool(
            self.cfg.get("use_big_gate", legacy_use_gate)
        )
        self.selective_gate = bool(self.cfg.get("selective_gate", True))
        self.ablation_mode = str(self.cfg.get("ablation_mode") or "custom").strip().lower()
        # SkillAA is single-component. Tuple fields are: updater, online
        # small Gate, epoch complete-candidate big Gate, per-edit small-Gate selection,
        # same-group rightcase context, train tie-break, teacher veto.
        # The formal full method is g_full and is reported once.
        ablation_modes = {
            "no_rightcase_context": ("evidence", True, True, True, False, True, True),
            "no_small_gate": ("evidence", False, True, True, True, True, True),
            "no_big_gate": ("evidence", True, False, True, True, True, True),
            "g_full": ("evidence", True, True, True, True, True, True),
        }
        if self.ablation_mode != "custom" and self.ablation_mode not in ablation_modes:
            raise ValueError(
                "ablation_mode must be one of: " + ", ".join(ablation_modes)
            )
        if self.ablation_mode == "custom":
            self.update_strategy = "evidence"
            self.use_train_tiebreak = bool(self.cfg.get("use_train_tiebreak", True))
            self.use_teacher_veto = bool(self.cfg.get("use_teacher_veto", True))
        else:
            (
                self.update_strategy,
                self.use_small_gate,
                self.use_big_gate,
                self.selective_gate,
                self.evolution_cfg.use_positive_context,
                self.use_train_tiebreak,
                self.use_teacher_veto,
            ) = ablation_modes[self.ablation_mode]
        # Local Gates compare affected update cases. The active formal Big Gate
        # evaluates the complete candidate once on the complete update pool and
        # accepts only a strict positive hard-success net gain. A separate train
        # tiebreak exists only for the legacy split protocol.
        no_validation_protocol = bool(
            self.cfg.get("no_validation_split", False)
        )
        self.use_big_gate_train_tiebreak = bool(
            self.cfg.get(
                "use_big_gate_train_tiebreak", not no_validation_protocol
            )
        )
        if no_validation_protocol and self.use_big_gate_train_tiebreak:
            raise ValueError(
                "no-validation protocol cannot enable a separate train tiebreak"
            )
        # Legacy non-grouped code has one Gate switch; it corresponds to the
        # epoch-level big Gate.  The grouped pipeline uses the two explicit
        # switches above.
        self.use_gate = self.use_big_gate
        self.environment = str(
            self.cfg.get("env_name") or self.cfg.get("env") or "searchqa"
        ).strip().lower()
        if self.environment not in {"searchqa", "docvqa", "livemathematicianbench"}:
            raise ValueError(f"unsupported update/rollback pipeline: {self.environment}")
        self.immutable_prompt_file = str(
            self.cfg.get("immutable_prompt_file") or "immutable_prompt.md"
        ).strip()
        self.semantic_reasoning_trace_required = bool(
            self.cfg.get("semantic_reasoning_trace", False)
        )
        trace_prompt_selected = (
            self.immutable_prompt_file == "immutable_prompt_reasoning.md"
        )
        if self.semantic_reasoning_trace_required != trace_prompt_selected:
            raise ValueError(
                f"{self.environment} semantic_reasoning_trace and "
                "immutable_prompt_file must select the reasoning protocol together"
            )
        self.pipeline_policy = importlib.import_module(
            f"graphopt.envs.{self.environment}.pipeline"
        )
        self.allow_execution_children = bool(
            getattr(self.pipeline_policy, "ALLOW_EXECUTION_CHILDREN", True)
        )
        self.max_atomic_groups_per_small_gate = max(
            1, int(getattr(self.pipeline_policy, "MAX_ATOMIC_GROUPS_PER_SMALL_GATE", 1))
        )
        self.require_nonnegative_combined_on_validation_gain = bool(
            getattr(
                self.pipeline_policy,
                "REQUIRE_NONNEGATIVE_COMBINED_ON_VALIDATION_GAIN",
                False,
            )
        )
        self.protected_root_node_ids = frozenset(
            map(str, getattr(self.pipeline_policy, "PROTECTED_ROOT_NODE_IDS", ()))
        )
        # Small Gates remain strict. Epoch big Gates may explicitly retain a
        # complete validation hard-score tie; configure this per dataset.
        self.big_gate_allow_hard_tie = bool(
            self.cfg.get("big_gate_allow_hard_tie", False)
        )
        self.max_big_gate_passes = int(self.cfg.get("max_big_gate_passes") or 1)
        if self.max_big_gate_passes != 1:
            raise ValueError(
                "max_big_gate_passes must be 1; big Gate partial rollback is disabled"
            )
        self.experiment_mode = str(
            self.cfg.get("experiment_mode") or "graphopt"
        ).strip().lower()
        if self.experiment_mode not in {
            "graphopt", "graphopt_best", "no_skill", "initial_skill", "skillaa_md"
        }:
            raise ValueError(
                "experiment_mode must be one of: graphopt, graphopt_best, "
                "no_skill, initial_skill, skillaa_md"
            )
        self.test_only = bool(self.cfg.get("test_only", False))
        if self.test_only and self.experiment_mode == "graphopt":
            raise ValueError("test_only is valid only for frozen evaluation modes")
        if self.experiment_mode == "graphopt_best" and not self.test_only:
            raise ValueError("graphopt_best requires test_only=true")
        requested_protocol = str(
            os.environ.get("GRAPHOPT_UPDATE_PROTOCOL")
            or self.cfg.get("update_protocol")
            or "legacy"
        ).strip().lower()
        if requested_protocol == "case_complete":
            requested_protocol = "case_complete_v1"
        if requested_protocol not in {"legacy", "case_complete_v1"}:
            raise ValueError("update_protocol must be legacy or case_complete")
        # Baseline modes keep the exact historical rendering. GraphOpt uses
        # the reasoning graph-execution renderer without adding a probe layer.
        self.update_protocol = (
            requested_protocol
            if self.experiment_mode in {"graphopt", "graphopt_best"}
            else "legacy"
        )
        self.badcase_analysis_protocol = (
            badcase_analysis_protocol(self.update_protocol)
            if self.experiment_mode == "graphopt"
            else "not_applicable"
        )
        self.renderer_protocol = (
            "case_complete_v1" if self.update_protocol == "case_complete_v1" else "legacy"
        )
        # Complete-case synthesis retains cross-batch evidence until epoch merge.
        if self.update_protocol == "case_complete_v1":
            self.evolution_cfg.persist_proposal_pool = True
        protocol_path = self.out / "update_protocol.json"
        if self.resume_requested and protocol_path.is_file():
            stored_protocol_record = json.loads(
                protocol_path.read_text(encoding="utf-8")
            )
            stored_protocol = str(
                stored_protocol_record.get("update_protocol") or "legacy"
            )
            if stored_protocol != self.update_protocol:
                raise RuntimeError(
                    "resume update_protocol mismatch: "
                    f"run={stored_protocol} requested={self.update_protocol}"
                )
            stored_analysis_protocol = str(
                stored_protocol_record.get("badcase_analysis_protocol") or ""
            )
            if (
                self.experiment_mode == "graphopt"
                and stored_analysis_protocol != self.badcase_analysis_protocol
            ):
                raise RuntimeError(
                    "resume badcase analysis protocol mismatch: "
                    f"run={stored_analysis_protocol or chr(60) + chr(109) + chr(105) + chr(115) + chr(115) + chr(105) + chr(110) + chr(103) + chr(62)} "
                    f"requested={self.badcase_analysis_protocol}"
                )
            if self.update_protocol == "case_complete_v1":
                stored_semantics = str(
                    stored_protocol_record.get(
                        "update_protocol_semantics_version"
                    ) or ""
                )
                legacy_g0_only_migration = bool(
                    stored_semantics == LEGACY_G0_ONLY_PROTOCOL_SEMANTICS_VERSION
                    and no_validation_protocol
                    and not (self.out / "trainer_state.json").is_file()
                    and (self.out / "baseline.json").is_file()
                    and not (
                        self.out / "epochs" / "epoch_01" / "epoch_update"
                        / "joint_proposed_patch.json"
                    ).is_file()
                    and not (
                        self.out / "steps" / "step_0001" / "gate.json"
                    ).is_file()
                )
                if (
                    stored_semantics != UPDATE_PROTOCOL_SEMANTICS_VERSION
                    and not legacy_g0_only_migration
                ):
                    raise RuntimeError(
                        "resume update protocol semantics mismatch; final protocol "
                        "requires a fresh run: "
                        f"run={stored_semantics or chr(60) + chr(109) + chr(105) + chr(115) + chr(115) + chr(105) + chr(110) + chr(103) + chr(62)} "
                        f"requested={UPDATE_PROTOCOL_SEMANTICS_VERSION}"
                    )
                if legacy_g0_only_migration:
                    print(
                        "[graphopt resume] upgraded a verified G0-only run to "
                        "the no-validation update-pool protocol; no epoch edit "
                        "or Gate artifact was reused",
                        flush=True,
                    )
        self.skill_markdown_path: Path | None = None
        self.skill_markdown = ""
        if self.experiment_mode == "skillaa_md":
            raw_path = str(self.cfg.get("skill_markdown_path") or "").strip()
            environment = str(
                self.cfg.get("env_name") or self.cfg.get("env") or "searchqa"
            ).strip().lower()
            path = (
                Path(raw_path).expanduser().resolve()
                if raw_path
                else Path(__file__).resolve().parents[1]
                / "envs" / environment / "skills" / "gpt5.5_skill.md"
            )
            if not path.is_file():
                raise FileNotFoundError(f"skillaa_md file not found: {path}")
            self.skill_markdown_path = path
            self.skill_markdown = path.read_text(encoding="utf-8")
            if not self.skill_markdown.strip():
                raise ValueError(f"skillaa_md file is empty: {path}")
        self.accumulation = max(1, int(self.cfg.get("accumulation") or 1))
        self.grouped_batch_gate = bool(self.cfg.get("grouped_batch_gate", False))
        self.batch_size = max(1, int(self.cfg.get("batch_size") or 48))
        self.batch_size_scope = str(
            self.cfg.get("batch_size_scope") or "train"
        ).strip().lower()
        self.train_batch_size = self.batch_size
        self.validation_batch_size = 0
        if (
            self.experiment_mode == "graphopt"
            and self.grouped_batch_gate
            and not bool(self.cfg.get("single_full_pool_input", False))
        ):
            (
                self.train_batch_size,
                self.validation_batch_size,
                _,
            ) = resolve_group_batch_sizes(
                self.batch_size,
                train_per_group=int(self.cfg.get("train_per_validation") or 3),
                scope=self.batch_size_scope,
            )
        if (
            self.experiment_mode == "graphopt"
            and self.update_protocol == "case_complete_v1"
            and not self.grouped_batch_gate
        ):
            raise ValueError(
                "case_complete_v1 requires grouped_batch_gate=true so support is "
                "counted by independent fixed update groups"
            )
        self.minibatch_size = max(1, int(self.cfg.get("minibatch_size") or 8))
        self.analyst_workers = max(1, int(self.cfg.get("analyst_workers") or 16))
        self.seed = int(self.cfg.get("seed") or 42)
        self.ablation_random_sample_rate = float(
            self.cfg.get("ablation_random_sample_rate") or 0.2
        )
        if not 0.0 < self.ablation_random_sample_rate <= 1.0:
            raise ValueError("ablation_random_sample_rate must be in (0, 1]")
        self.final_test_repeats = max(
            1, int(self.cfg.get("final_test_repeats") or 1)
        )
        self.run_epoch_test = bool(
            self.cfg.get(
                "run_epoch_test",
                self.cfg.get(
                    "eval_test", self.update_protocol == "case_complete_v1"
                ),
            )
        )
        self.epoch_test_repeats = max(
            1, int(self.cfg.get("epoch_test_repeats") or 3)
        )
        if (
            self.experiment_mode == "graphopt"
            and self.update_protocol == "case_complete_v1"
            and self.run_epoch_test
            and self.epoch_test_repeats != 3
        ):
            raise ValueError(
                "an enabled full GraphOpt epoch test requires exactly 3 repeats"
            )
        self.run_component_ablation_tests = bool(
            self.cfg.get("run_component_ablation_tests", False)
        )
        if self.run_component_ablation_tests and not self.grouped_batch_gate:
            raise ValueError(
                "run_component_ablation_tests requires grouped_batch_gate=true"
            )
        if self.run_component_ablation_tests and self.update_protocol != "case_complete_v1":
            raise ValueError(
                "run_component_ablation_tests is defined only for case_complete_v1"
            )
        self.test_repeat_seed_mode = str(
            self.cfg.get("test_repeat_seed_mode") or "offset"
        ).strip().lower()
        if self.test_repeat_seed_mode not in {"fixed", "offset"}:
            raise ValueError("test_repeat_seed_mode must be 'fixed' or 'offset'")
        self._epoch_start_graph_text = ""
        if not self._restored_trainer_state:
            save_skill_json(self.graph, str(self.out / "graphs" / "graph_step0000.json"))
            save_skill_json(self.graph, str(self.out / "best_graph.json"))
        save_json(self.out / "update_protocol.json", {
            "schema_version": "graphopt-update-protocol-v1",
            "update_protocol": self.update_protocol,
            "badcase_analysis_protocol": self.badcase_analysis_protocol,
            "renderer_protocol": self.renderer_protocol,
            "persist_proposal_pool": self.evolution_cfg.persist_proposal_pool,
            "update_protocol_semantics_version": (
                UPDATE_PROTOCOL_SEMANTICS_VERSION
                if self.update_protocol == "case_complete_v1" else None
            ),
            "candidate_policy": "all_epoch_joint_groups",
            "compatibility": "final protocol only; older GraphOpt checkpoints are rejected",
            "initial_graph_sha256": self.initial_graph_file_sha256,
        })
        (self.out / "config.json").write_text(
            json.dumps({k: v for k, v in self.cfg.items() if "api_key" not in k.lower()}, indent=2),
            encoding="utf-8",
        )

    def _prepare_resume_workspace(self, completed_epoch: int) -> None:
        """Clear artifacts beyond the resumable frontier without losing progress.

        The next incomplete step and its candidate-validation episodes are kept.
        Their artifact/cache input hashes and rollout context invalidate them if
        the reconstructed graph differs.  Deleting them eagerly caused every
        manual resume to restart the same 130+ completed validation episodes.
        """
        incomplete_step = completed_epoch + 1
        step_dir = self.out / "steps" / f"step_{incomplete_step:04d}"

        patterns = (
            (self.out / "graphs", "graph_step*.json", r"graph_step(\d+)", completed_epoch),
            (self.out / "steps", "step_*", r"step_(\d+)", incomplete_step),
            (self.out / "epoch_archives", "epoch_v*", r"epoch_v(\d+)", completed_epoch),
        )
        removed: list[str] = []
        for root, glob_pattern, number_pattern, retain_through in patterns:
            if not root.is_dir():
                continue
            for item in root.glob(glob_pattern):
                match = re.search(number_pattern, item.name)
                if not match or int(match.group(1)) <= retain_through:
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                removed.append(str(item.relative_to(self.out)))

        rollouts = self.out / "rollouts"
        if rollouts.is_dir():
            for item in rollouts.glob("step_*_val*"):
                match = re.match(r"step_(\d+)_val", item.name)
                if match and int(match.group(1)) > incomplete_step:
                    shutil.rmtree(item)
                    removed.append(str(item.relative_to(self.out)))

        for pattern in ("meta_epoch_*.txt", "comparison_pairs_epoch_*.json"):
            for item in self.out.glob(pattern):
                match = re.search(r"epoch_(\d+)", item.name)
                if match and int(match.group(1)) > completed_epoch:
                    item.unlink()
                    removed.append(str(item.relative_to(self.out)))

        if removed:
            print(
                f"[graphopt resume] cleared {len(removed)} graph-dependent "
                f"artifact(s) newer than epoch {completed_epoch}; "
                "completed train/final-test rollout caches were retained",
                flush=True,
            )

    def _save_trainer_state(self, completed_epoch: int) -> None:
        """Atomically persist graph, cache, and control state at an epoch boundary."""
        cache_dir = self.out / "resume_state"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"evolution_cache_epoch_{completed_epoch:04d}.json"
        cache_temporary = cache_path.with_suffix(".json.tmp")
        cache_temporary.write_text(
            json.dumps(self.evolution_cache.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cache_temporary.replace(cache_path)
        payload = {
            "schema_version": "graphopt-trainer-state-v1",
            "update_protocol": self.update_protocol,
            "badcase_analysis_protocol": self.badcase_analysis_protocol,
            "renderer_protocol": self.renderer_protocol,
            "persist_proposal_pool": self.evolution_cfg.persist_proposal_pool,
            "update_protocol_semantics_version": (
                UPDATE_PROTOCOL_SEMANTICS_VERSION
                if self.update_protocol == "case_complete_v1" else None
            ),
            "candidate_policy": "all_epoch_joint_groups",
            "initial_graph_sha256": self.initial_graph_file_sha256,
            "completed_epochs": int(completed_epoch),
            "graph_path": f"graphs/graph_step{completed_epoch:04d}.json",
            "best_graph_path": "best_graph.json",
            "evolution_cache_path": str(cache_path.relative_to(self.out)),
            "current_score": self.score,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "history": self.history,
            "epoch_archives": self.epoch_archives,
            "last_selection_results": self._last_sel_results,
            "selection_excluded_case_ids": sorted(
                self._selection_excluded_case_ids or set()
            ),
            "meta": self.meta,
            "pending_meta": (
                {
                    "epoch": self._pending_meta_epoch,
                    "previous_selection_results": self._pending_meta_prev_results,
                    "previous_graph_text": self._pending_meta_prev_graph_text,
                }
                if self._pending_meta_epoch
                else None
            ),
        }
        path = self.out / "trainer_state.json"
        temporary = self.out / "trainer_state.json.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        print(
            f"[graphopt checkpoint] saved trainer_state epoch={completed_epoch} "
            f"path={path}",
            flush=True,
        )

    def _checkpoint_gate_before_meta(
        self,
        epoch: int,
        *,
        previous_selection_results: list[dict[str, Any]],
        previous_graph_text: str,
    ) -> None:
        """Commit a Gate-complete epoch before the advisory Meta stage."""
        # The formal epoch boundary is the Gate, not advisory Meta/test. Keep
        # the in-memory resume cursor synchronized with the checkpoint payload
        # so a process resumed at epoch N can commit epoch N+1 in the same run.
        self.resume_completed_epochs = int(epoch)
        self._pending_meta_epoch = int(epoch)
        self._pending_meta_prev_results = list(previous_selection_results)
        self._pending_meta_prev_graph_text = str(previous_graph_text)
        self._save_trainer_state(epoch)

    def _finish_pending_meta(self) -> None:
        """Finish Meta for a Gate-complete checkpoint without replaying its Gate."""
        epoch = int(self._pending_meta_epoch)
        if epoch <= 0:
            return
        if epoch != self.resume_completed_epochs and self.resume_completed_epochs > 0:
            raise RuntimeError(
                "pending Meta epoch does not match the resumable Gate checkpoint"
            )
        pairs = _build_comparison_pairs(
            self._pending_meta_prev_results, self._last_sel_results
        )
        use_teacher = self.reflect_mode == "teacher" and self.chat_fn is not None
        meta_dir = self.out / "meta"
        meta_dir.mkdir(exist_ok=True)
        self.meta = update_meta(
            [h for h in self.history if h.get("epoch") == epoch],
            current=self.meta,
            chat_fn=self.chat_fn if use_teacher else None,
            mode="teacher" if use_teacher else "template",
            comparison_pairs=pairs,
            prev_graph_text=self._pending_meta_prev_graph_text,
            curr_graph_text=format_graph(self.graph),
            artifact_dir=meta_dir,
            epoch=epoch,
        )
        (self.out / "meta.txt").write_text(self.meta, encoding="utf-8")
        (self.out / f"meta_epoch_{epoch:02d}.txt").write_text(
            self.meta, encoding="utf-8"
        )
        (self.out / f"comparison_pairs_epoch_{epoch:02d}.json").write_text(
            json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._pending_meta_epoch = 0
        self._pending_meta_prev_results = []
        self._pending_meta_prev_graph_text = ""
        with open(self.out / "gate_history.jsonl", "w", encoding="utf-8") as handle:
            for record in self.history:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._save_trainer_state(epoch)
        print(
            f"[graphopt checkpoint] completed pending Meta epoch={epoch} without Gate replay",
            flush=True,
        )

    def _finish_pending_epoch_tests(self) -> None:
        """Complete any missing three-repeat diagnostic test after resume."""
        if self.resume_completed_epochs <= 0:
            return
        changed = False
        for epoch in (
            range(1, self.resume_completed_epochs + 1)
            if self.run_epoch_test else []
        ):
            record = next((
                item for item in self.history
                if int(item.get("epoch") or item.get("step") or 0) == epoch
            ), None)
            if record is None:
                raise RuntimeError(f"committed epoch {epoch} is missing Gate history")
            graph_path = self.out / "graphs" / f"graph_step{epoch:04d}.json"
            if not graph_path.is_file():
                raise RuntimeError(
                    f"committed epoch {epoch} is missing graph checkpoint: {graph_path}"
                )
            committed_graph = load_graph(graph_path)
            aggregate_path = (
                self.out / "epoch_tests" / f"epoch_{epoch:02d}" / "epoch_test.json"
            )
            if aggregate_path.is_file():
                epoch_test = json.loads(aggregate_path.read_text(encoding="utf-8"))
                if epoch_test.get("graph_sha256") != graph_sha256(committed_graph):
                    raise RuntimeError(
                        f"epoch {epoch} diagnostic test graph identity mismatch"
                    )
                if int(epoch_test.get("test_repeats") or 0) != 3:
                    raise RuntimeError(
                        f"epoch {epoch} diagnostic test must contain exactly 3 repeats"
                    )
            else:
                print(
                    f"[graphopt resume] completing pending three-repeat test for epoch={epoch}",
                    flush=True,
                )
                epoch_test, _ = self._run_epoch_test_repeats(epoch, committed_graph)
            epoch_record = self._epoch_test_record(epoch, epoch_test)
            if record.get("epoch_test") != epoch_record:
                record["epoch_test"] = epoch_record
                changed = True

        missing_archives: list[tuple[int, Path, dict[str, Any] | None]] = []
        for epoch in range(1, self.resume_completed_epochs + 1):
            archive_dir = self.out / "epoch_archives" / f"epoch_v{epoch}"
            record = next((
                item for item in self.history
                if int(item.get("epoch") or item.get("step") or 0) == epoch
            ), None)
            archive_text = str(archive_dir)
            if archive_text not in self.epoch_archives:
                self.epoch_archives.append(archive_text)
                changed = True
            if record is not None and record.get("epoch_archive") != archive_text:
                record["epoch_archive"] = archive_text
                changed = True
            if not archive_dir.is_dir():
                missing_archives.append((epoch, archive_dir, record))

        if changed or missing_archives:
            with open(self.out / "gate_history.jsonl", "w", encoding="utf-8") as handle:
                for record in self.history:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            write_epoch_versions_doc(self.out, self.history)
            for epoch, archive_dir, record in missing_archives:
                print(
                    f"[graphopt resume] materializing pending epoch archive "
                    f"epoch={epoch} path={archive_dir}",
                    flush=True,
                )
                archive_epoch_artifacts(
                    self.out, epoch=epoch, step=epoch, decision=record
                )
            self._save_trainer_state(self.resume_completed_epochs)

    def _skill_for(self, graph: SkillGraph, *, env: Any = None, results=None):
        """Render the current JSON graph directly into the Agent prompt."""
        del env, results
        if self.experiment_mode == "no_skill":
            return "", []
        if self.experiment_mode == "skillaa_md":
            return self.skill_markdown, []
        environment = str(self.cfg.get("env_name") or self.cfg.get("env") or "searchqa")
        return (
            graph_to_prompt(
                graph, environment=environment, protocol=self.renderer_protocol,
                immutable_prompt_file=self.immutable_prompt_file,
            ),
            prompt_node_ids(graph),
        )

    def _run_evaluation_only(self) -> dict[str, Any]:
        """Evaluate a baseline without training, teacher calls, mutation, or Gate."""
        mode = self.experiment_mode
        if mode not in {"graphopt_best", "no_skill", "initial_skill", "skillaa_md"}:
            raise RuntimeError(f"evaluation-only mode is invalid: {mode!r}")

        selection_baseline = None
        if mode == "initial_skill" and not self.test_only:
            # Produce the one exact G0 valid_seen rollout that SkillAA methods
            # may reuse fail-closed. It is an initialization measurement, not
            # training and not a Gate decision.
            self._baseline()
            selection_baseline = json.loads(
                (self.out / "baseline.json").read_text(encoding="utf-8")
            )
        print(f"[graphopt] EVALUATION ONLY — mode={mode}, split=valid_unseen")
        final, skill = self._run_final_test_repeats(self.graph)
        final["experiment_mode"] = mode
        save_json(self.out / "final_test.json", final)
        (self.out / "best_skill.md").write_text(skill, encoding="utf-8")
        summary = {
            **final,
            "out_root": str(self.out),
            "update_protocol": self.update_protocol,
            "badcase_analysis_protocol": self.badcase_analysis_protocol,
            "renderer_protocol": self.renderer_protocol,
            "initial_graph_sha256": self.initial_graph_file_sha256,
            "teacher_profile": self.cfg.get("teacher_profile"),
            "teacher_model_base": self.cfg.get("teacher_model_base"),
            "student_model_base": self.cfg.get("student_model_base"),
            "best_graph": str(self.out / "best_graph.json"),
            "best_skill_md": str(self.out / "best_skill.md"),
            "teacher_calls": 0,
            "training_rollouts": 0,
            "test_only": self.test_only,
            "evaluation_scope": (
                "valid_unseen_only" if self.test_only
                else "valid_seen_initialization_plus_valid_unseen"
                if mode == "initial_skill"
                else "valid_unseen_only"
            ),
            "validation_rollouts": 0 if self.test_only else (
                int(selection_baseline.get("n") or 0) if selection_baseline else 0
            ),
            "selection_baseline": selection_baseline,
            "selection_rollout_cases": (
                int(selection_baseline.get("n") or 0) if selection_baseline else 0
            ),
            "selection_baseline_role": (
                "frozen_g0_reuse_source_not_a_gate" if selection_baseline else "not_run"
            ),
            "updates_per_epoch": 0,
            "candidate_gates_per_epoch": 0,
            "num_epochs": 0,
            "update_boundary": "none",
            "use_gate": False,
            "selective_gate": False,
            "run_epoch_test": False,
            "epoch_test_repeats": 0,
            "epoch_test_role": "not_applicable_evaluation_only",
            "skill_markdown_path": str(self.skill_markdown_path) if self.skill_markdown_path else None,
            "skill_markdown_sha256": hashlib.sha256(self.skill_markdown.encode("utf-8")).hexdigest() if self.skill_markdown_path else None,
            "final_test_repeats": self.final_test_repeats,
            "final_test_aggregation": (
                "single_terminal_graph_evaluation"
                if self.final_test_repeats == 1
                else "min_max_mean_plus_half_range"
            ),
        }
        save_json(self.out / "summary.json", summary)
        return summary

    def _run_final_test_repeats(
        self, graph: SkillGraph
    ) -> tuple[dict[str, Any], str]:
        if self.experiment_mode == "graphopt" and self.run_epoch_test:
            expected_hash = graph_sha256(graph)
            for prior in reversed(self.history):
                prior_test = prior.get("epoch_test")
                if not isinstance(prior_test, dict):
                    continue
                source_path = Path(str(prior_test.get("file") or ""))
                if not source_path.is_absolute():
                    source_path = self.out / source_path
                if not source_path.is_file():
                    continue
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                if (
                    payload.get("graph_sha256") != expected_hash
                    or int(payload.get("test_repeats") or 0) != 3
                ):
                    continue
                final = copy.deepcopy(payload)
                final_runs: list[dict[str, Any]] = []
                for repeat, source_record in enumerate(
                    payload.get("repeat_results") or [], start=1
                ):
                    source_file = Path(str(source_record.get("file") or ""))
                    if not source_file.is_absolute():
                        source_file = self.out / source_file
                    if not source_file.is_file():
                        raise FileNotFoundError(
                            f"epoch-test reuse source is missing: {source_file}"
                        )
                    raw_run = json.loads(source_file.read_text(encoding="utf-8"))
                    destination = self.out / f"final_test_v{repeat}.json"
                    raw_run.update({
                        "reused_from_epoch_test": True,
                        "epoch_test_source": str(source_file),
                    })
                    save_json(destination, raw_run)
                    final_runs.append({
                        **dict(source_record),
                        "file": destination.name,
                        "epoch_test_source": str(source_file),
                    })
                if len(final_runs) != 3:
                    raise RuntimeError("final epoch test must contain exactly 3 repeats")
                final.update({
                    "role": "final_epoch_committed_graph_diagnostic",
                    "used_for_gate": False,
                    "used_for_graph_selection": False,
                    "graph_sha256": expected_hash,
                    "repeat_results": final_runs,
                })
                save_json(self.out / "final_test.json", final)
                return final, self._skill_for(graph)[0]
            raise RuntimeError(
                "final GraphOpt graph has no exact three-repeat epoch test to reuse"
            )

        runs: list[dict[str, Any]] = []
        skill = ""
        test_seed_offset = int(self.cfg.get("test_seed_offset") or 50_000)
        for repeat in range(1, self.final_test_repeats + 1):
            tag = f"final_test_v{repeat}"
            repeat_offset = 0 if self.test_repeat_seed_mode == "fixed" else repeat - 1
            results, hard, soft, gate, skill = self._rollout(
                graph,
                "valid_unseen",
                tag,
                seed_offset=repeat_offset,
            )
            record = {
                "repeat": repeat,
                "seed": self.seed + test_seed_offset + repeat_offset,
                "seed_mode": self.test_repeat_seed_mode,
                "tag": tag,
                "hard": hard,
                "soft": soft,
                "gate": gate,
                "n": len(results),
                **self._metric_audit(results),
                "task_metrics": compute_task_metrics(results, environment=str(self.cfg.get("env_name") or self.cfg.get("env") or "searchqa")),
                "raw_task_metrics": compute_task_metrics(results, environment=str(self.cfg.get("env_name") or self.cfg.get("env") or "searchqa"), include_excluded=True),
            }
            save_json(self.out / f"final_test_v{repeat}.json", record)
            runs.append(record)
        final = aggregate_final_test_runs(runs)
        final.update({
            "role": "final_committed_graph_evaluation_only",
            "used_for_gate": False,
            "used_for_graph_selection": False,
            "graph_sha256": graph_sha256(graph),
        })
        final["repeat_results"] = [
            {
                "repeat": run["repeat"],
                "seed": run["seed"],
                "tag": run["tag"],
                "file": f"final_test_v{run['repeat']}.json",
                "hard": run["hard"],
                "soft": run["soft"],
                "gate": run["gate"],
                "n": run["n"],
            }
            for run in runs
        ]
        save_json(self.out / "final_test.json", final)
        return final, skill

    def _run_epoch_test_repeats(
        self, epoch: int, graph: SkillGraph
    ) -> tuple[dict[str, Any], str]:
        """Evaluate the committed graph three times; never use test for selection."""
        output_dir = self.out / "epoch_tests" / f"epoch_{epoch:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, Any]] = []
        skill = ""
        test_seed_offset = int(self.cfg.get("test_seed_offset") or 50_000)
        role = "committed_graph_epoch_diagnostic_only"
        for repeat in range(1, self.epoch_test_repeats + 1):
            tag = f"epoch_{epoch:02d}_test_v{repeat}"
            repeat_offset = 0 if self.test_repeat_seed_mode == "fixed" else repeat - 1
            results, hard, soft, gate, skill = self._rollout(
                graph, "valid_unseen", tag, seed_offset=repeat_offset
            )
            record = {
                "epoch": epoch,
                "repeat": repeat,
                "seed": self.seed + test_seed_offset + repeat_offset,
                "seed_mode": self.test_repeat_seed_mode,
                "tag": tag,
                "role": role,
                "hard": hard,
                "soft": soft,
                "gate": gate,
                "n": len(results),
                **self._metric_audit(results),
                "task_metrics": compute_task_metrics(
                    results, environment=str(
                        self.cfg.get("env_name") or self.cfg.get("env") or "searchqa"
                    )
                ),
                "raw_task_metrics": compute_task_metrics(
                    results, environment=str(
                        self.cfg.get("env_name") or self.cfg.get("env") or "searchqa"
                    ), include_excluded=True
                ),
            }
            save_json(output_dir / f"epoch_test_v{repeat}.json", record)
            runs.append(record)
        aggregate = aggregate_final_test_runs(runs)
        relative_dir = output_dir.relative_to(self.out)
        aggregate.update({
            "epoch": epoch,
            "graph_sha256": graph_sha256(graph),
            "role": role,
            "used_for_gate": False,
            "used_for_graph_selection": False,
            "selection_independent": True,
            "repeat_results": [
                {
                    "repeat": run["repeat"],
                    "seed": run["seed"],
                    "tag": run["tag"],
                    "file": str(relative_dir / f"epoch_test_v{run['repeat']}.json"),
                    "rollout_dir": str(Path("rollouts") / run["tag"]),
                    "hard": run["hard"],
                    "soft": run["soft"],
                    "gate": run["gate"],
                    "n": run["n"],
                }
                for run in runs
            ],
        })
        aggregate["display"] = format_final_test_display(aggregate)
        save_json(output_dir / "epoch_test.json", aggregate)
        print(
            f"[graphopt] EPOCH TEST epoch={epoch} "
            f"repeats={self.epoch_test_repeats} hard_min={aggregate['hard_min']:.4f} "
            f"hard_max={aggregate['hard_max']:.4f} "
            f"hard_mean={aggregate['hard_mean']:.4f}±{aggregate['hard_half_range']:.4f} "
            f"seed_mode={self.test_repeat_seed_mode}",
            flush=True,
        )
        return aggregate, skill

    def _epoch_test_record(
        self, epoch: int, aggregate: dict[str, Any]
    ) -> dict[str, Any]:
        """Store the canonical compact pointer and interval for one epoch."""
        return {
            "status": "evaluated_committed_graph",
            "file": str(Path("epoch_tests") / f"epoch_{epoch:02d}" / "epoch_test.json"),
            "graph_sha256": aggregate.get("graph_sha256"),
            "hard_min": aggregate.get("hard_min"),
            "hard_max": aggregate.get("hard_max"),
            "hard_mean": aggregate.get("hard_mean"),
            "hard_std": aggregate.get("hard_std"),
            "hard_half_range": aggregate.get("hard_half_range"),
            "display": aggregate.get("display"),
            "test_repeats": aggregate.get("test_repeats"),
            "used_for_gate": False,
            "used_for_graph_selection": False,
            "rollout_performed": True,
        }

    def _run_component_ablation_test(
        self,
        *,
        stage: str,
        graph: SkillGraph,
        graph_source: str,
    ) -> tuple[dict[str, Any], str]:
        """Evaluate one fixed-stream component graph once, without leakage."""
        allowed = {
            "g0",
            "raw_proposal_replay",
            "local_gate_replay",
        }
        if stage not in allowed:
            raise ValueError(f"unsupported component ablation point: {stage}")
        output_dir = self.out / "component_ablation" / "tests" / stage
        output_dir.mkdir(parents=True, exist_ok=True)
        aggregate_path = output_dir / "test.json"
        expected_hash = graph_sha256(graph)
        if aggregate_path.is_file():
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if str(aggregate.get("graph_sha256") or "") != expected_hash:
                raise RuntimeError(
                    "component-test graph mismatch on resume: "
                    f"stage={stage}"
                )
            return aggregate, self._skill_for(graph)[0]

        stage_order = (
            "g0", "raw_proposal_replay",
            "local_gate_replay",
        )
        for prior_stage in stage_order:
            if prior_stage == stage:
                break
            prior_path = (
                self.out / "component_ablation" / "tests"
                / prior_stage / "test.json"
            )
            if not prior_path.is_file():
                continue
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if str(prior.get("graph_sha256") or "") != expected_hash:
                continue
            aggregate = copy.deepcopy(prior)
            aggregate.update({
                "stage": stage,
                "graph_source": graph_source,
                "reused_from_stage": prior_stage,
                "paired_test_policy": (
                    "same_valid_unseen_ids_and_seed_once_per_unique_graph_hash"
                ),
            })
            source_run = (
                self.out / str(
                    (prior.get("repeat_results") or [{}])[0].get("file") or ""
                )
            )
            if source_run.is_file():
                reused_run = json.loads(source_run.read_text(encoding="utf-8"))
                reused_run.update({
                    "reused_from_stage": prior_stage,
                    "reused_for_stage": stage,
                })
                save_json(output_dir / "test_v1.json", reused_run)
                aggregate["repeat_results"] = [{
                    **dict((prior.get("repeat_results") or [{}])[0]),
                    "file": str(
                        Path("component_ablation") / "tests" / stage
                        / "test_v1.json"
                    ),
                    "reused_from_stage": prior_stage,
                }]
            save_json(aggregate_path, aggregate)
            save_skill_json(graph, str(output_dir / "graph.json"))
            print(
                f"[graphopt] COMPONENT TEST stage={stage} reused={prior_stage} "
                f"hard={aggregate['hard']:.4f} graph={expected_hash[:12]} "
                "role=diagnostic_only",
                flush=True,
            )
            return aggregate, self._skill_for(graph)[0]

        tag = f"component_ablation_{stage}_v1"
        results, hard, soft, gate, skill = self._rollout(
            graph, "valid_unseen", tag, seed_offset=0
        )
        run = {
            "repeat": 1,
            "seed": self.seed + int(self.cfg.get("test_seed_offset") or 50_000),
            "seed_mode": "paired_fixed_stage_seed",
            "tag": tag,
            "hard": hard,
            "soft": soft,
            "gate": gate,
            "n": len(results),
            **self._metric_audit(results),
            "task_metrics": compute_task_metrics(
                results,
                environment=str(
                    self.cfg.get("env_name")
                    or self.cfg.get("env")
                    or "searchqa"
                ),
            ),
            "raw_task_metrics": compute_task_metrics(
                results,
                environment=str(
                    self.cfg.get("env_name")
                    or self.cfg.get("env")
                    or "searchqa"
                ),
                include_excluded=True,
            ),
        }
        save_json(output_dir / "test_v1.json", run)
        aggregate = aggregate_final_test_runs([run])
        aggregate.update({
            "schema_version": "graphopt-component-ablation-test-v1",
            "stage": stage,
            "graph_sha256": expected_hash,
            "graph_source": graph_source,
            "role": "diagnostic_only_not_used_by_gate_or_optimizer",
            "used_for_gate": False,
            "used_for_graph_selection": False,
            "paired_test_policy": "same_valid_unseen_ids_and_seed_once_per_unique_graph_hash",
            "repeat_results": [{
                "repeat": 1,
                "seed": run["seed"],
                "tag": tag,
                "file": str(
                    Path("component_ablation") / "tests" / stage / "test_v1.json"
                ),
                "hard": hard,
                "soft": soft,
                "gate": gate,
                "n": len(results),
            }],
        })
        save_json(aggregate_path, aggregate)
        save_skill_json(graph, str(output_dir / "graph.json"))
        print(
            f"[graphopt] COMPONENT TEST stage={stage} repeats=1 "
            f"hard={hard:.4f} graph={expected_hash[:12]} role=diagnostic_only",
            flush=True,
        )
        return aggregate, skill

    def _component_batch_dirs(self) -> list[Path]:
        paths: list[Path] = []
        for epoch_dir in sorted((self.out / "epochs").glob("epoch_[0-9][0-9]")):
            epoch_update = epoch_dir / "epoch_update"
            if epoch_update.is_dir():
                paths.append(epoch_update)
            else:
                paths.extend(sorted(
                    (epoch_dir / "batch_updates").glob("batch_[0-9][0-9][0-9][0-9]")
                ))
        return paths

    @staticmethod
    def _load_patch(path: Path) -> GraphPatch:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"patch artifact is not an object: {path}")
        return GraphPatch.from_dict(payload)

    def _local_gate_replay_patch(self, batch_dir: Path) -> GraphPatch:
        epoch_patch = batch_dir / "local_gate_accepted_patch.json"
        if epoch_patch.is_file():
            return self._load_patch(epoch_patch)
        gate_path = batch_dir / "small_gate.json"
        if not gate_path.is_file():
            raise FileNotFoundError(f"small-Gate artifact is missing: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not bool(gate.get("accepted", False)):
            return GraphPatch(reasoning="small Gate rejected this batch", edits=[])
        effective_path = batch_dir / "effective_patch.json"
        if not effective_path.is_file():
            raise FileNotFoundError(
                f"accepted small Gate has no effective patch: {effective_path}"
            )
        effective = self._load_patch(effective_path)
        raw_kept = gate.get("kept_edit_indices")
        if not isinstance(raw_kept, list):
            return effective
        kept = {int(index) for index in raw_kept}
        return GraphPatch(
            reasoning=effective.reasoning,
            edits=[
                edit for index, edit in enumerate(effective.edits)
                if index in kept
            ],
        )

    def _build_component_replay_graphs(
        self,
    ) -> tuple[dict[str, SkillGraph], dict[str, Any]]:
        """Replay one GraphOpt run's proposal stream under cumulative safeguards.

        These are conditional counterfactuals: every shadow sees proposals made
        by the real GraphOpt run. No shadow can influence later proposals,
        Gate decisions, the committed graph, or model calls.
        """
        root = self.out / "component_ablation"
        graph_dir = root / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        g0_path = self.out / "graphs" / "graph_step0000.json"
        if not g0_path.is_file():
            raise FileNotFoundError(f"G0 graph is missing: {g0_path}")
        g0 = load_graph(g0_path)
        graphs = {
            "raw_proposal_replay": g0.copy(),
            "local_gate_replay": g0.copy(),
        }
        patch_names = {
            "raw_proposal_replay": "component_raw_patch.json",
        }
        audits: dict[str, dict[str, Any]] = {
            stage: {
                "stage": stage,
                "atomic_groups_attempted": 0,
                "atomic_groups_applied": 0,
                "atomic_groups_failed": 0,
                "edits_attempted": 0,
                "edits_applied": 0,
                "edits_failed": 0,
                "batches": [],
            }
            for stage in graphs
        }
        batch_dirs = self._component_batch_dirs()
        if not batch_dirs:
            raise RuntimeError(
                "component replay requires at least one completed epoch update"
            )
        for batch_dir in batch_dirs:
            for stage, graph in graphs.items():
                if stage == "local_gate_replay":
                    patch = self._local_gate_replay_patch(batch_dir)
                    patch_file = "local_gate_accepted_patch.json"
                else:
                    patch_path = batch_dir / patch_names[stage]
                    if not patch_path.is_file():
                        raise FileNotFoundError(
                            f"component replay patch is missing: {patch_path}"
                        )
                    patch = self._load_patch(patch_path)
                    patch_file = patch_path.name
                report = apply_patch(graph, patch)
                group_reports = list(report.get("group_reports") or [])
                applied_groups = sum(
                    bool(item.get("applied")) for item in group_reports
                )
                audit = audits[stage]
                audit["atomic_groups_attempted"] += len(group_reports)
                audit["atomic_groups_applied"] += applied_groups
                audit["atomic_groups_failed"] += len(group_reports) - applied_groups
                audit["edits_attempted"] += int(report.get("n_edits") or 0)
                audit["edits_applied"] += int(report.get("n_applied") or 0)
                audit["edits_failed"] += int(report.get("n_failed") or 0)
                audit["batches"].append({
                    "batch_dir": str(batch_dir.relative_to(self.out)),
                    "patch_source": patch_file,
                    "n_edits": int(report.get("n_edits") or 0),
                    "n_applied": int(report.get("n_applied") or 0),
                    "n_failed": int(report.get("n_failed") or 0),
                    "warnings": list(report.get("warnings") or []),
                })
        graph_sources: dict[str, str] = {}
        for stage, graph in graphs.items():
            path = graph_dir / f"{stage}.json"
            save_skill_json(graph, str(path))
            graph_sources[stage] = str(path.relative_to(self.out))
            audits[stage]["graph_sha256"] = graph_sha256(graph)
        manifest = {
            "schema_version": "graphopt-fixed-proposal-component-replay-v2",
            "design": "conditional_fixed_proposal_stream_replay",
            "proposal_stream_source": "the_single_graphopt_training_run",
            "shadow_feedback_into_training": False,
            "test_feedback_into_training": False,
            "raw_stage_semantics": (
                "apply every structurally sanitized proposal group before Local Gate"
            ),
            "local_gate_stage_semantics": (
                "apply only joint edits accepted by affected-scope Local Gate, bypassing Big Gate"
            ),
            "big_gate_stage_semantics": (
                "reuse repeat 1 of the real GraphOpt final-best evaluation"
            ),
            "n_batches": len(batch_dirs),
            "g0_graph_sha256": graph_sha256(g0),
            "graph_sources": graph_sources,
            "replay_audits": audits,
        }
        save_json(root / "replay_manifest.json", manifest)
        return graphs, manifest

    def _write_component_ablation_summary(
        self, replay_manifest: dict[str, Any]
    ) -> None:
        root = self.out / "component_ablation"
        order = {
            "g0": 0,
            "raw_proposal_replay": 1,
            "local_gate_replay": 2,
            "graphopt_final": 3,
        }
        records: list[dict[str, Any]] = []
        for path in sorted((root / "tests").glob("*/test.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append({
                "stage": payload.get("stage"),
                "graph_sha256": payload.get("graph_sha256"),
                "graph_source": payload.get("graph_source"),
                "hard": payload.get("hard_mean"),
                "soft": payload.get("soft_mean"),
                "gate": payload.get("gate_mean"),
                "n": payload.get("n"),
                "file": str(path.relative_to(self.out)),
                "source": "one_pass_component_test",
            })
        final_path = self.out / "final_test_v1.json"
        best_path = self.out / "best_graph.json"
        if not final_path.is_file() or not best_path.is_file():
            raise FileNotFoundError(
                "component summary requires final_test_v1.json and best_graph.json"
            )
        final = json.loads(final_path.read_text(encoding="utf-8"))
        best = load_graph(best_path)
        records.append({
            "stage": "graphopt_final",
            "graph_sha256": graph_sha256(best),
            "graph_source": str(best_path.relative_to(self.out)),
            "hard": final.get("hard"),
            "soft": final.get("soft"),
            "gate": final.get("gate"),
            "n": final.get("n"),
            "file": str(final_path.relative_to(self.out)),
            "source": "reused_final_test_repeat_1",
        })
        records.sort(key=lambda row: order.get(str(row.get("stage") or ""), 99))
        save_json(root / "summary.json", {
            "schema_version": "graphopt-component-ablation-summary-v2",
            "role": "diagnostic_only_not_used_by_gate_or_optimizer",
            "design": "conditional_fixed_proposal_stream_replay",
            "paired_test_policy": (
                "same_valid_unseen_ids_and_seed_once_per_unique_graph_hash"
            ),
            "component_audit": (
                "G0 to raw proposals to Local-Gate accepted graph to real Big-Gate final"
            ),
            "update_protocol": self.update_protocol,
            "replay_manifest": "component_ablation/replay_manifest.json",
            "replay_audits": replay_manifest.get("replay_audits") or {},
            "records": records,
        })

    def _run_component_ablation_suite(self) -> None:
        if not self.run_component_ablation_tests:
            return
        graphs, manifest = self._build_component_replay_graphs()
        g0_path = self.out / "graphs" / "graph_step0000.json"
        self._run_component_ablation_test(
            stage="g0",
            graph=load_graph(g0_path),
            graph_source=str(g0_path.relative_to(self.out)),
        )
        for stage in (
            "raw_proposal_replay",
            "local_gate_replay",
        ):
            source = str(manifest["graph_sources"][stage])
            self._run_component_ablation_test(
                stage=stage, graph=graphs[stage], graph_source=source
            )
        self._write_component_ablation_summary(manifest)

    def _rollout_adapter(self, skill: str, split: str, out_dir: str, *, batch=None, seed: int = 0):
        assert self.adapter is not None
        no_validation = bool(self.cfg.get("no_validation_split", False))
        split_map = {
            "selection": "train" if no_validation else "valid_seen",
            "val": "train" if no_validation else "valid_seen",
            "valid_seen": "train" if no_validation else "valid_seen",
            "test": "valid_unseen",
        }
        aw_split = split_map.get(split, split)
        if batch is not None:
            env = self.adapter.build_env_from_batch(batch, out_root=str(self.out))
        elif aw_split == "train":
            train_batch_size = self.train_batch_size
            if no_validation and split in {"selection", "val", "valid_seen"}:
                train_batch_size = len(
                    list(getattr(self.dataloader, "train_items", []) or [])
                )
                if train_batch_size <= 0:
                    raise RuntimeError(
                        "complete update-pool Gate requested an empty train split"
                    )
            env = self.adapter.build_train_env(
                batch_size=train_batch_size, seed=seed, out_root=str(self.out)
            )
        else:
            env_num = int(self.cfg.get("sel_env_num") or self.cfg.get("test_env_num") or 0)
            if aw_split == "valid_unseen":
                env_num = int(self.cfg.get("test_env_num") or env_num)
            env = self.adapter.build_eval_env(
                env_num=env_num, split=aw_split, seed=seed, out_root=str(self.out)
            )
        return env, self.adapter.rollout(env, skill, out_dir)

    def _rollout(
        self,
        graph: SkillGraph,
        split: str,
        tag: str,
        *,
        batch=None,
        seed_offset: int = 0,
        skill_override: str | None = None,
        prompt_nodes_override: list[str] | None = None,
    ) -> tuple[list[dict], float, float, float, str]:
        d = self.out / "rollouts" / tag
        d.mkdir(parents=True, exist_ok=True)
        seed = self.seed + seed_offset
        # Every graph compared by a gate must see exactly the same sampled cases.
        # Training may vary per step; validation/test sampling is fixed by split.
        no_validation = bool(self.cfg.get("no_validation_split", False))
        split_map = {
            "selection": "train" if no_validation else "valid_seen",
            "val": "train" if no_validation else "valid_seen",
            "valid_seen": "train" if no_validation else "valid_seen",
            "test": "valid_unseen",
        }
        canonical_split = split_map.get(split, split)
        if canonical_split == "valid_seen":
            seed = self.seed + int(self.cfg.get("selection_seed_offset") or 0)
        elif canonical_split == "valid_unseen":
            seed = (
                self.seed
                + int(self.cfg.get("test_seed_offset") or 50_000)
                + seed_offset
            )

        if self.adapter is not None:
            # Build env first so retrieval can see batch task types / items
            aw_split = split_map.get(split, split)
            if batch is not None:
                env = self.adapter.build_env_from_batch(batch, out_root=str(self.out))
            elif aw_split == "train":
                train_batch_size = self.train_batch_size
                if no_validation and split in {"selection", "val", "valid_seen"}:
                    train_batch_size = len(
                        list(getattr(self.dataloader, "train_items", []) or [])
                    )
                    if train_batch_size <= 0:
                        raise RuntimeError(
                            "complete update-pool Gate requested an empty train split"
                        )
                env = self.adapter.build_train_env(
                    batch_size=train_batch_size, seed=seed, out_root=str(self.out)
                )
            else:
                env_num = int(self.cfg.get("sel_env_num") or 0)
                if aw_split == "valid_unseen":
                    env_num = int(self.cfg.get("test_env_num") or 0)
                env = self.adapter.build_eval_env(
                    env_num=env_num, split=aw_split, seed=seed, out_root=str(self.out)
                )
            if skill_override is None:
                skill, order = self._skill_for(graph, env=env)
            else:
                skill = str(skill_override)
                order = list(prompt_nodes_override or prompt_node_ids(graph))
            _prepare_rollout_context(
                d, split=canonical_split, seed=seed, skill=skill, cfg=self.cfg
            )
            (d / "skill_prompt.md").write_text(skill, encoding="utf-8")
            (d / "prompt_nodes.json").write_text(
                json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            results = self.adapter.rollout(env, skill, str(d))
            environment = str(
                self.cfg.get("env_name") or self.cfg.get("env") or "searchqa"
            ).strip().lower()
            item_by_id = (
                {
                    str(item.get("id")): item
                    for item in env
                    if isinstance(item, dict) and item.get("id") is not None
                }
                if isinstance(env, list)
                else {}
            )
            for row in results:
                row.setdefault("environment", environment)
                row["task_type"] = canonical_task_type(
                    row, environment=environment
                )
                row.setdefault(
                    "task_description",
                    row.get("question") or row.get("instruction") or "",
                )
                if canonical_split != "train":
                    continue
                item = item_by_id.get(str(row.get("id")))
                if not item:
                    continue
                reference: dict[str, Any] = {
                    "scope": "teacher_diagnosis_only_not_student_input",
                    "environment": environment,
                }
                if environment == "docvqa":
                    reference.update({
                        "gold_answers": (
                            item.get("answers")
                            if self.cfg.get("teacher_use_reference", True)
                            else []
                        ),
                    })
                elif environment == "searchqa":
                    reference.update({
                        "gold_answers": (
                            item.get("answers")
                            if self.cfg.get("teacher_use_reference", True)
                            else []
                        ),
                    })
                elif environment == "livemathematicianbench":
                    reference.update({
                        "correct_choice": item.get("correct_choice"),
                        "theorem": item.get("theorem") if self.cfg.get("teacher_use_reference", True) else "",
                        "sketch": item.get("sketch") if self.cfg.get("teacher_use_reference", True) else "",
                    })
                if any(value for key, value in reference.items() if key not in {"scope", "environment"}):
                    row["training_reference"] = reference
        else:
            if skill_override is None:
                skill, order = self._skill_for(graph)
            else:
                skill = str(skill_override)
                order = list(prompt_nodes_override or prompt_node_ids(graph))
            _prepare_rollout_context(
                d, split=canonical_split, seed=seed, skill=skill, cfg=self.cfg
            )
            (d / "skill_prompt.md").write_text(skill, encoding="utf-8")
            (d / "prompt_nodes.json").write_text(
                json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self.rollout_fn is None:
                raise RuntimeError("No adapter and no rollout_fn (use --dry_run or --config)")
            results = self.rollout_fn(
                skill, canonical_split,
                {**self.cfg, "_rollout_out": str(d), "seed": seed},
            )

        results = normalize_results(results)
        filter_enabled = bool(self.cfg.get("exclude_step_limit_failures", False)) and canonical_split in {
            "valid_seen", "valid_unseen"
        }
        metric_audit = mark_step_limit_exclusions(results, enabled=filter_enabled)
        if canonical_split == "valid_seen" and self._selection_excluded_case_ids is not None:
            for row in results:
                case_id = str(row.get("id") or "")
                if case_id in self._selection_excluded_case_ids:
                    row["exclude_from_metrics"] = True
                    row["metric_exclusion_reason"] = "G0_FIXED_EXCLUSION"
                elif row.get("exclude_from_metrics"):
                    # The G0 eligibility mask fixes the denominator. A
                    # candidate-only API/environment exclusion may reject an
                    # update, but can never improve it by shrinking the
                    # candidate denominator.
                    original_reason = str(
                        row.get("metric_exclusion_reason") or "UNKNOWN_EXCLUSION"
                    )
                    row["exclude_from_metrics"] = False
                    row["hard"] = 0.0
                    row["soft"] = 0.0
                    row["candidate_only_exclusion_reason"] = original_reason
                    row["metric_exclusion_reason"] = (
                        "CANDIDATE_ONLY_EXCLUSION_COUNTED_AS_FAILURE"
                    )
            metric_audit = self._metric_audit(results)
            metric_audit["metric_policy"] = "fixed_g0_eligibility_mask"
        if not any(not row.get("exclude_from_metrics") for row in results):
            raise ValueError(f"rollout {tag} has no metric-eligible cases")
        attach_trajectories(d, results)
        rollout_env = str(self.cfg.get("env_name") or self.cfg.get("env") or "")
        if rollout_env in {"searchqa", "docvqa", "livemathematicianbench"}:
            attach_semantic_reasoning_traces(
                d, results, required=self.semantic_reasoning_trace_required
            )
        if rollout_env in {"searchqa", "docvqa", "livemathematicianbench"}:
            attach_graph_usage(
                d, results, valid_nodes=set(graph.nodes),
                valid_edges={edge.id for edge in graph.edges if edge.id},
            )
            attach_trace_evidence(graph, results)
        (d / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        save_json(
            d / "rollout_progress.json",
            {
                "schema_version": "graphopt-rollout-progress-v2",
                "scope": "complete_rollout",
                "status": "complete",
                "episodes": len(results),
                "completed": len(results),
                "pending": 0,
                "completed_case_ids": [result.get("id") for result in results],
            },
        )
        hard, soft, gate = score_results(
            results, metric=self.gate_metric, mixed_weight=self.mixed_weight, skill_content=skill
        )
        raw_results = [
            {k: v for k, v in row.items() if k not in {"exclude_from_metrics", "metric_exclusion_reason"}}
            for row in results
        ]
        raw_hard, raw_soft, raw_gate = score_results(
            raw_results, metric=self.gate_metric, mixed_weight=self.mixed_weight, skill_content=skill
        )
        metric_audit.update({
            "filtered_hard": hard, "filtered_soft": soft, "filtered_gate": gate,
            "official_unfiltered_hard": raw_hard,
            "official_unfiltered_soft": raw_soft,
            "official_unfiltered_gate": raw_gate,
        })
        save_json(d / "metric_filter.json", metric_audit)
        save_rollout_artifact(
            d,
            stage=f"rollout_{tag}",
            inputs={
                "tag": tag,
                "split": split,
                "seed": seed,
                "batch_size": _batch_size_for_logging(batch, self.train_batch_size),
            },
            outputs={"n": len(results), **metric_audit, "hard": hard, "soft": soft, "gate": gate, "case_ids": [r.get("id") for r in results]},
        )
        save_json(
            d / "rollout_summary.json",
            {"tag": tag, "split": split, "seed": seed, "n": len(results), **metric_audit, "hard": hard, "soft": soft, "gate": gate},
        )
        return results, hard, soft, gate, skill

    @staticmethod
    def _metric_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
        excluded = [r for r in results if r.get("exclude_from_metrics")]
        return {
            "metric_policy": "fixed_g0_eligibility_mask",
            "raw_n": len(results),
            "effective_n": len(results) - len(excluded),
            "excluded_n": len(excluded),
            "excluded_case_ids": [str(r.get("id")) for r in excluded],
        }

    def _run_gate_repeats(
        self,
        graph: SkillGraph,
        tag_prefix: str,
        *,
        batch: Any = None,
    ) -> tuple[list[dict[str, Any]], float, float, float, str, list[str]]:
        """Run exactly one complete Gate pass for one graph version."""
        runs: list[list[dict[str, Any]]] = []
        hard_values: list[float] = []
        soft_values: list[float] = []
        gate_values: list[float] = []
        tags: list[str] = []
        skill = ""
        for repeat in range(1, GATE_REPEATS + 1):
            tag = f"{tag_prefix}_r{repeat}"
            tags.append(tag)
            results, hard, soft, gate, skill = self._rollout(
                graph,
                "valid_seen",
                tag,
                batch=batch,
                seed_offset=repeat - 1,
            )
            runs.append(results)
            hard_values.append(hard)
            soft_values.append(soft)
            gate_values.append(gate)
        averaged = _mean_gate_repeat_results(runs)
        return (
            averaged,
            statistics.fmean(hard_values),
            statistics.fmean(soft_values),
            statistics.fmean(gate_values),
            skill,
            tags,
        )

    def _restore_incomplete_baseline(self) -> bool:
        """Reuse a complete G0 selection rollout when no epoch was committed yet."""
        if not self.resume_requested or self._restored_trainer_state:
            return False
        baseline_path = self.out / "baseline.json"
        results_path = self.out / "rollouts" / "baseline_selection_r1" / "results.json"
        context_path = self.out / "rollouts" / "baseline_selection_r1" / "rollout_context.json"
        if not all(path.is_file() for path in (baseline_path, results_path, context_path)):
            return False
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            results = json.loads(results_path.read_text(encoding="utf-8"))
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        selection_split = (
            "train" if bool(self.cfg.get("no_validation_split", False))
            else "valid_seen"
        )
        if selection_split == "train":
            expected_case_ids = [
                str(item.get("id"))
                for item in list(
                    getattr(self.dataloader, "train_items", []) or []
                )
                if isinstance(item, dict) and item.get("id") is not None
            ]
        else:
            get_split_items = getattr(
                self.dataloader, "get_split_items", None
            )
            expected_case_ids = [
                str(item.get("id"))
                for item in (
                    list(get_split_items("valid_seen") or [])
                    if callable(get_split_items) else []
                )
                if isinstance(item, dict) and item.get("id") is not None
            ]
        stored_case_ids = [str(row.get("id") or "") for row in results]
        if (
            not isinstance(results, list)
            or not results
            or len(results) != int(baseline.get("n") or -1)
            or str(context.get("split") or "") != selection_split
            or int(context.get("seed") or -1) != self.seed
            or len({str(row.get("id") or "") for row in results}) != len(results)
            or (
                bool(expected_case_ids)
                and (
                    len(expected_case_ids) != len(set(expected_case_ids))
                    or len(stored_case_ids) != len(expected_case_ids)
                    or set(stored_case_ids) != set(expected_case_ids)
                )
            )
        ):
            raise RuntimeError("stored baseline artifacts failed identity/coverage validation")
        self.score = float(baseline.get("gate") or 0.0)
        self.best_score = self.score
        self.best_step = 0
        self._last_sel_results = list(results)
        self._selection_excluded_case_ids = {
            str(row.get("id")) for row in results if row.get("exclude_from_metrics")
        }
        print(
            f"[graphopt stage resume] reused complete baseline {selection_split} "
            f"({len(results)} cases, gate={self.score:.4f})",
            flush=True,
        )
        return True

    def _baseline(self) -> None:
        print("[graphopt] BASELINE — initial SkillGraph on complete update pool")
        frozen = self._load_frozen_g0()
        if frozen is None:
            res, hard, soft, gate, skill, tags = self._run_gate_repeats(
                self.graph, "baseline_selection"
            )
            baseline_source = None
        else:
            res, hard, soft, gate, skill, tags, baseline_source = frozen
        self.score = gate
        self.best_score = gate
        self.best_step = 0
        self._last_sel_results = res
        self._selection_excluded_case_ids = {
            str(row.get("id"))
            for row in res
            if row.get("exclude_from_metrics")
        }
        (self.out / "baseline.json").write_text(
            json.dumps({
                "hard": hard,
                "soft": soft,
                "gate": gate,
                "n": len(res),
                "gate_repeats": GATE_REPEATS,
                "rollout_tags": tags,
                "frozen_g0_reused": baseline_source is not None,
                "frozen_g0_source": baseline_source,
            }, indent=2),
            encoding="utf-8",
        )
        (self.out / "best_skill.md").write_text(skill, encoding="utf-8")
        print(
            f"[graphopt] baseline gate[{self.gate_metric}]={gate:.4f} "
            f"hard={hard:.4f} soft={soft:.4f} repeats={GATE_REPEATS}"
        )

    def _load_frozen_g0(
        self,
    ) -> tuple[
        list[dict[str, Any]], float, float, float, str, list[str], str
    ] | None:
        """Load an exact frozen-G0 rollout instead of paying to regenerate it.

        Reuse is deliberately fail-closed. The validation IDs and order, seed,
        rendered initial prompt, deployed student model, routing, backend, and
        response policy must all match the current run. A configured frozen
        root that contains no exact match is an error, never a silent fallback.
        """
        raw_root = str(self.cfg.get("frozen_g0_rollout_dir") or "").strip()
        if not raw_root:
            return None
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"frozen_g0_rollout_dir does not exist: {root}")

        skill, order = self._skill_for(self.graph)
        no_validation = bool(self.cfg.get("no_validation_split", False))
        selection_split = "train" if no_validation else "valid_seen"
        seed = (
            self.seed
            if no_validation
            else self.seed + int(self.cfg.get("selection_seed_offset") or 0)
        )
        expected_context = _rollout_context_payload(
            split=selection_split, seed=seed, skill=skill, cfg=self.cfg
        )
        expected_ids = [
            str(item.get("id"))
            for item in (
                (
                    list(getattr(self.dataloader, "train_items", []) or [])
                    if no_validation
                    else self.dataloader.get_split_items("valid_seen")
                )
                if self.dataloader is not None
                else []
            )
        ]
        if expected_ids and len(expected_ids) != len(set(expected_ids)):
            raise ValueError(
                f"current {selection_split} split contains duplicate case ids"
            )

        candidates = [root]
        candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))
        mismatch_notes: list[str] = []
        selected: tuple[Path, list[dict[str, Any]]] | None = None
        for candidate in candidates:
            context_path = candidate / "rollout_context.json"
            results_path = candidate / "results.json"
            if not context_path.is_file() or not results_path.is_file():
                continue
            try:
                context = json.loads(context_path.read_text(encoding="utf-8"))
                results = json.loads(results_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                mismatch_notes.append(f"{candidate.name}: unreadable ({exc})")
                continue
            differing = [
                key
                for key, value in expected_context.items()
                if context.get(key) != value
            ]
            actual_ids = (
                [str(row.get("id")) for row in results]
                if isinstance(results, list)
                else []
            )
            if differing:
                mismatch_notes.append(
                    f"{candidate.name}: context mismatch {','.join(differing)}"
                )
                continue
            if expected_ids and actual_ids != expected_ids:
                mismatch_notes.append(
                    f"{candidate.name}: validation IDs/order mismatch "
                    f"expected={len(expected_ids)} actual={len(actual_ids)}"
                )
                continue
            if len(actual_ids) != len(set(actual_ids)) or not actual_ids:
                mismatch_notes.append(f"{candidate.name}: empty or duplicate result IDs")
                continue
            selected = candidate, results
            break

        if selected is None:
            detail = (
                "; ".join(mismatch_notes[:8])
                or "no results.json + rollout_context.json"
            )
            raise RuntimeError(
                f"no exact frozen G0 matches the current run under {root}: {detail}"
            )

        source, raw_results = selected
        results = normalize_results(raw_results)
        filter_enabled = bool(self.cfg.get("exclude_step_limit_failures", False))
        metric_audit = mark_step_limit_exclusions(results, enabled=filter_enabled)
        if not any(not row.get("exclude_from_metrics") for row in results):
            raise ValueError(f"frozen G0 at {source} has no metric-eligible cases")
        hard, soft, gate = score_results(
            results,
            metric=self.gate_metric,
            mixed_weight=self.mixed_weight,
            skill_content=skill,
        )

        tag = "baseline_selection_r1"
        destination = self.out / "rollouts" / tag
        destination.mkdir(parents=True, exist_ok=True)
        save_json(destination / "rollout_context.json", expected_context)
        save_json(destination / "results.json", results)
        save_json(destination / "prompt_nodes.json", order)
        (destination / "skill_prompt.md").write_text(skill, encoding="utf-8")
        save_json(
            destination / "rollout_progress.json",
            {
                "schema_version": "graphopt-rollout-progress-v2",
                "scope": "complete_rollout",
                "status": "complete",
                "episodes": len(results),
                "completed": len(results),
                "pending": 0,
                "completed_case_ids": [row.get("id") for row in results],
                "reused_frozen_g0": True,
            },
        )
        raw_rows = [
            {
                key: value
                for key, value in row.items()
                if key not in {"exclude_from_metrics", "metric_exclusion_reason"}
            }
            for row in results
        ]
        raw_hard, raw_soft, raw_gate = score_results(
            raw_rows,
            metric=self.gate_metric,
            mixed_weight=self.mixed_weight,
            skill_content=skill,
        )
        metric_audit.update(
            {
                "filtered_hard": hard,
                "filtered_soft": soft,
                "filtered_gate": gate,
                "official_unfiltered_hard": raw_hard,
                "official_unfiltered_soft": raw_soft,
                "official_unfiltered_gate": raw_gate,
                "frozen_g0_source": str(source),
            }
        )
        save_json(destination / "metric_filter.json", metric_audit)
        save_json(
            destination / "rollout_summary.json",
            {
                "tag": tag,
                "split": selection_split,
                "seed": seed,
                "n": len(results),
                **metric_audit,
                "hard": hard,
                "soft": soft,
                "gate": gate,
                "reused_frozen_g0": True,
            },
        )
        save_json(
            self.out / "frozen_g0_reuse.json",
            {
                "schema_version": "graphopt-frozen-g0-reuse-v1",
                "source": str(source),
                "destination": str(destination),
                "case_count": len(results),
                "case_ids_exact": True,
                "context_exact": True,
                "hard": hard,
                "soft": soft,
                "gate": gate,
            },
        )
        print(
            f"[graphopt] reused frozen G0 source={source} cases={len(results)} "
            f"gate[{self.gate_metric}]={gate:.4f}; no baseline API rollout",
            flush=True,
        )
        return results, hard, soft, gate, skill, [tag], str(source)

    def _run_grouped_big_gate(
        self,
        *,
        epoch: int,
        base_graph: SkillGraph,
        base_results: list[dict[str, Any]],
        base_score: float,
        small_gate_records: list[dict[str, Any]],
        grouped_batches: list[GroupedBatch] | None = None,
    ) -> dict[str, Any]:
        """Commit the complete epoch candidate, or restore the epoch-start graph."""
        no_validation = bool(self.cfg.get("no_validation_split", False))
        if no_validation and self.use_big_gate_train_tiebreak:
            raise ValueError(
                "no-validation protocol cannot enable a separate train tiebreak"
            )
        step_dir = self.out / "steps" / f"step_{epoch:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        candidate = self.graph.copy()
        patch = attach_epoch_patch_provenance(
            diff_skill_graphs(base_graph, candidate), small_gate_records
        )
        save_json(step_dir / "effective_patch.json", patch.to_dict())
        save_json(step_dir / "small_gate_records.json", small_gate_records)
        save_json(step_dir / "gate_reference_results.json", base_results)
        local_train_case_ids = {
            str(case_id) for row in small_gate_records
            for case_id in (row.get("train_case_ids") or []) if str(case_id)
        }
        local_evidence_case_ids = {
            str(case_id) for row in small_gate_records
            for edit in ((row.get("joint_patch") or {}).get("edits") or [])
            for case_id in (edit.get("source_case_ids") or []) if str(case_id)
        }
        train_count = len(local_train_case_ids)
        evidence_count = len(local_evidence_case_ids)

        if not patch.edits:
            committed_train: list[dict[str, Any]] = (
                copy.deepcopy(base_results) if no_validation else []
            )
            committed_train_source = (
                "complete_update_pool_base_results" if no_validation else "disabled"
            )
            if self.use_big_gate_train_tiebreak:
                source = (
                    self.out / "steps" / f"step_{epoch - 1:04d}"
                    / "gate_committed_train_results.json"
                    if epoch > 1
                    else self.out / "epochs" / "epoch_01" / "epoch_update"
                    / "epoch_train_results.json"
                )
                if not source.is_file():
                    raise RuntimeError(
                        "cannot carry the combined Big Gate reference through a "
                        f"no-edit epoch; missing={source}"
                    )
                try:
                    committed_train = list(json.loads(source.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError, TypeError) as exc:
                    raise RuntimeError(
                        f"invalid committed train reference for no-edit epoch: {exc}"
                    ) from exc
                expected_train_ids = list(dict.fromkeys(
                    str(case_id)
                    for batch in (grouped_batches or [])
                    for case_id in batch.train_ids
                ))
                if expected_train_ids:
                    _require_same_case_ids(
                        [{"id": case_id} for case_id in expected_train_ids],
                        committed_train,
                    )
                committed_train_source = str(source.relative_to(self.out))
            self.graph, self.score = base_graph, base_score
            self._last_sel_results = list(base_results)
            record = {
                "step": epoch, "epoch": epoch,
                "action": "skip_no_small_gate_edits", "accepted": False,
                "current_score": base_score, "val_score": base_score,
                "candidate_val_score": base_score, "n_edits": 0,
                "n_edits_kept": 0, "n_edits_rolled_back": 0,
                "small_gates": small_gate_records,
                "n_small_gates": len(small_gate_records),
                "n_small_gates_accepted": sum(bool(row.get("accepted")) for row in small_gate_records),
                "n_rollout_cases": train_count,
                "n_reference_train_cases": len(committed_train),
                "n_reference_train_rollout_cases": 0,
                "reference_train_source": committed_train_source,
                "reference_train_reused": bool(committed_train),
                "n_evidence_cases": evidence_count,
                "n_excluded_cases": train_count - evidence_count,
                "gate_passes": 0,
            }
            self.history.append(record)
            gate_experience = self._prepare_gate_experience(record)
            self.evolution_cache.record_gate_experience(gate_experience)
            self.evolution_cache.save(self.out / "evolution_cache.json")
            save_json(step_dir / "gate_experience.json", gate_experience)
            save_json(step_dir / "gate.json", record)
            save_json(step_dir / "gate_committed_results.json", base_results)
            save_json(step_dir / "gate_committed_train_results.json", committed_train)
            save_skill_json(self.graph, str(self.out / "graphs" / f"graph_step{epoch:04d}.json"))
            return record

        rebuilt = base_graph.copy()
        rebuild_report = apply_patch(rebuilt, patch)
        reconstruction_delta = diff_skill_graphs(rebuilt, candidate)
        if rebuild_report.get("n_failed") or reconstruction_delta.edits:
            raise RuntimeError("epoch patch does not reconstruct the cumulative candidate")
        save_skill_json(candidate, str(step_dir / "candidate_graph.json"))

        if not self.use_big_gate:
            # This is a diagnostic full-validation rollout, not a Gate vote.
            # The ablation commits the terminal graph unconditionally, but its
            # graph, score and result cache must still describe the same model.
            (
                diagnostic_results, diagnostic_hard, diagnostic_soft,
                diagnostic_score, final_skill, diagnostic_tags,
            ) = self._run_gate_repeats(
                candidate,
                f"step_{epoch:04d}_update_pool"
                if no_validation else f"step_{epoch:04d}_val",
            )
            _require_same_case_ids(base_results, diagnostic_results)
            save_json(step_dir / "gate_candidate_results.json", diagnostic_results)
            diagnostic_audit = _hard_transition_summary(base_results, diagnostic_results)
            self.graph = candidate
            self.score = diagnostic_score
            self._last_sel_results = list(diagnostic_results)
            self.best = candidate.copy()
            self.best_score = diagnostic_score
            self.best_step = epoch
            save_skill_json(self.best, str(self.out / "best_graph.json"))
            (self.out / "best_skill.md").write_text(final_skill, encoding="utf-8")
            self.evolution_cache.ensure_graph_keys(self.graph)
            self.evolution_cache.save(self.out / "evolution_cache.json")
            save_json(step_dir / "gate_committed_results.json", diagnostic_results)
            record = {
                "step": epoch, "epoch": epoch,
                "action": "accept_big_gate_disabled", "accepted": True,
                "current_score": diagnostic_score, "val_score": diagnostic_score,
                "val_hard": diagnostic_hard, "val_soft": diagnostic_soft,
                "candidate_val_score": diagnostic_score,
                "diagnostic_hard_audit": diagnostic_audit,
                "n_edits": len(patch.edits),
                "n_edits_kept": len(patch.edits), "n_edits_rolled_back": 0,
                "kept_edit_indices": list(range(len(patch.edits))),
                "rolled_back_edit_indices": [], "gate_passes": 1,
                "rollout_tags": diagnostic_tags, "small_gates": small_gate_records,
                "big_gate_policy": "disabled_by_ablation_diagnostic_not_used_for_decision",
                "used_for_decision": False,
                "selection_policy": "unconditional_terminal_graph",
            }
            self.history.append(record)
            gate_experience = self._prepare_gate_experience(record)
            self.evolution_cache.record_gate_experience(gate_experience)
            self.evolution_cache.save(self.out / "evolution_cache.json")
            save_json(step_dir / "gate_experience.json", gate_experience)
            save_json(step_dir / "gate.json", record)
            save_json(step_dir / "gate_attribution.json", {"edit_attribution": []})
            save_skill_json(self.graph, str(self.out / "graphs" / f"graph_step{epoch:04d}.json"))
            return record

        grouped_batches = list(grouped_batches or [])
        combined_grouped = GroupedBatch(
            train_ids=list(dict.fromkeys(
                str(case_id) for batch in grouped_batches for case_id in batch.train_ids
            )),
            val_ids=(
                [] if no_validation
                else [str(row.get("id")) for row in base_results]
            ),
            groups=[group for batch in grouped_batches for group in batch.groups],
            tail=any(batch.tail for batch in grouped_batches),
        )
        if not combined_grouped.train_ids:
            combined_grouped.train_ids = list(dict.fromkeys(
                str(case_id)
                for row in small_gate_records
                for case_id in (row.get("train_case_ids") or [])
            ))

        canonical_train_seed_offset = 900001
        reference_train_source = "disabled"
        reference_train_reused = False

        def evaluate_epoch_train(
            graph: SkillGraph, suffix: str, *, case_ids: list[str] | None = None,
        ) -> list[dict[str, Any]]:
            gate_ids = list(case_ids or combined_grouped.train_ids)
            if not gate_ids or not self.use_big_gate_train_tiebreak:
                return []
            batch = build_exact_split_batch(
                self.dataloader, "train", gate_ids,
                seed=self.seed + canonical_train_seed_offset,
            )
            rows, _, _, _, _ = self._rollout(
                graph, "train", f"epoch_{epoch:02d}_big_g_{suffix}_train",
                batch=batch, seed_offset=canonical_train_seed_offset,
            )
            wanted = set(gate_ids)
            rows = [row for row in rows if str(row.get("id")) in wanted]
            _require_same_case_ids(
                [{"id": case_id} for case_id in gate_ids], rows
            )
            return rows

        def previous_committed_train() -> list[dict[str, Any]] | None:
            if epoch <= 1 or not self.use_big_gate_train_tiebreak:
                return None
            previous_path = (
                self.out / "steps" / f"step_{epoch - 1:04d}"
                / "gate_committed_train_results.json"
            )
            if not previous_path.is_file():
                raise RuntimeError(
                    "previous epoch committed train Gate results are required; "
                    f"missing={previous_path}"
                )
            try:
                rows = json.loads(previous_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"invalid previous committed train Gate results: {exc}"
                ) from exc
            _require_same_case_ids(
                [{"id": case_id} for case_id in combined_grouped.train_ids], rows
            )
            return list(rows)

        # Big Gate evaluates one complete candidate graph on its full decision
        # pool. In the active no-validation protocol this is exactly the update
        # pool and is executed once; legacy split mode may add a train tiebreak.
        # Local-Gate fragments are never spliced into this decision.
        results, hard, soft, score, skill, rollout_tags = self._run_gate_repeats(
            candidate, f"step_{epoch:04d}_val"
        )
        _require_same_case_ids(base_results, results)
        save_json(step_dir / "gate_candidate_results.json", results)
        candidate_hard_audit = _hard_transition_summary(base_results, results)
        primary_gate_evaluated = True
        primary_diagnostic_path = step_dir / (
            "gate_update_pool_diagnostic.json"
            if no_validation else "gate_validation_diagnostic.json"
        )
        save_json(
            primary_diagnostic_path,
            {
                "schema_version": (
                    "graphopt-big-gate-update-pool-diagnostic-v1"
                    if no_validation
                    else "graphopt-big-gate-validation-diagnostic-v1"
                ),
                "status": "evaluated_and_used_for_decision",
                "split": "train" if no_validation else "valid_seen",
                "reference_case_count": len(base_results),
                "candidate_rollout_case_count": len(results),
                "decision_authority": (
                    "complete_update_pool_hard_net_case_gain_only"
                    if no_validation
                    else "full_train_plus_validation_hard_net_case_gain"
                    if self.use_big_gate_train_tiebreak
                    else "full_validation_hard_net_case_gain_only"
                ),
            },
        )
        reference_train = previous_committed_train()
        if reference_train is None:
            reference_train = evaluate_epoch_train(base_graph, "reference")
            reference_train_source = "current_epoch_reference_rollout_or_exact_cache"
        else:
            reference_train_reused = True
            reference_train_source = (
                f"steps/step_{epoch - 1:04d}/gate_committed_train_results.json"
            )
        candidate_train = evaluate_epoch_train(
            candidate, "candidate",
            case_ids=[str(row.get("id")) for row in reference_train],
        )
        if no_validation:
            reference_train = copy.deepcopy(base_results)
            candidate_train = copy.deepcopy(results)
            reference_train_source = "complete_update_pool_base_results"
            reference_train_reused = True
        if self.use_big_gate_train_tiebreak:
            _require_same_case_ids(reference_train, candidate_train)
        save_json(step_dir / "gate_reference_train_results.json", reference_train)
        save_json(step_dir / "gate_candidate_train_results.json", candidate_train)
        candidate_train_audit = (
            _hard_transition_summary(reference_train, candidate_train)
            if reference_train or candidate_train
            else {"hard_net_case_gain": 0, "n_eligible": 0}
        )
        combined_big_gate_audit = _combined_big_gate_audit(
            candidate_hard_audit, candidate_train_audit
        )
        if no_validation:
            combined_big_gate_audit = {
                **candidate_hard_audit,
                "schema_version": "graphopt-update-pool-big-gate-audit-v1",
                "update_pool_n": int(candidate_hard_audit.get("n_eligible") or 0),
                "validation_n": 0,
                "train_n": 0,
                "update_pool_hard_net_case_gain": int(
                    candidate_hard_audit.get("hard_net_case_gain") or 0
                ),
                "validation_hard_net_case_gain": 0,
                "train_hard_net_case_gain": 0,
                "accept_if": "update_pool_hard_net_case_gain > 0",
            }
        expected_primary_cases = len(base_results)
        expected_train_cases = (
            len(combined_grouped.train_ids)
            if self.use_big_gate_train_tiebreak else 0
        )
        expected_big_gate_cases = expected_primary_cases + expected_train_cases
        if int(combined_big_gate_audit["n_eligible"]) != expected_big_gate_cases:
            raise RuntimeError(
                "Big Gate must cover the complete decision pool: "
                f"expected={expected_big_gate_cases} "
                f"actual={combined_big_gate_audit['n_eligible']}"
            )
        combined_gate_passed = self.pipeline_policy.accept_complete_candidate(
            candidate_hard_audit,
            candidate_train_audit,
            allow_validation_tie=self.big_gate_allow_hard_tie,
            use_train_tiebreak=self.use_big_gate_train_tiebreak,
        )
        primary_gate_passed = (
            int(candidate_hard_audit["hard_net_case_gain"]) > 0
            and int(candidate_hard_audit["n_eligible"])
            == expected_primary_cases
        )
        update_pool_gate_passed = primary_gate_passed if no_validation else False
        validation_gate_passed = primary_gate_passed if not no_validation else False
        train_gate_passed = (
            int(candidate_train_audit["hard_net_case_gain"]) > 0
            and int(candidate_train_audit["n_eligible"])
            == expected_train_cases
        ) if self.use_big_gate_train_tiebreak else False
        strict_combined_gain_passed = (
            int(candidate_hard_audit["n_eligible"]) == expected_primary_cases
            and (
                not self.use_big_gate_train_tiebreak
                or int(candidate_train_audit["n_eligible"]) == expected_train_cases
            )
            and int(combined_big_gate_audit["hard_net_case_gain"]) > 0
        )
        if combined_gate_passed != strict_combined_gain_passed:
            raise RuntimeError(
                "dataset Big Gate policy diverged from the strict complete "
                "decision-pool net gain"
            )
        full_candidate_gate_passed = combined_gate_passed
        audits = build_atomic_edit_audits(
            grouped_batch=combined_grouped,
            base_graph=base_graph,
            edits=patch.edits,
            reference_val=[] if no_validation else base_results,
            candidate_val=[] if no_validation else results,
            reference_train=base_results if no_validation else reference_train,
            candidate_train=results if no_validation else candidate_train,
            allow_validation_tie=self.big_gate_allow_hard_tie,
            require_nonnegative_combined_on_validation_gain=(
                self.require_nonnegative_combined_on_validation_gain
            ),
        )
        teacher_artifact = {
            "schema_version": "graphopt-per-edit-audit-artifact-v1",
            "status": "diagnostic_only_complete_candidate_big_gate",
            "audits": audits,
            "reviews": [],
            "kept_group_ids": (
                [str(audit["atomic_group_id"]) for audit in audits]
                if full_candidate_gate_passed else []
            ),
            "decision_authority": (
                "complete_update_pool_hard_net_case_gain"
                if no_validation else "full_train_plus_validation_hard_net_case_gain"
            ),
            "teacher_used_for_decision": False,
            "partial_rescue_supported": False,
        }
        kept_indices = (
            list(range(len(patch.edits))) if full_candidate_gate_passed else []
        )
        save_json(step_dir / "per_edit_gate_audits.json", audits)
        save_json(step_dir / "per_edit_teacher_audit.json", teacher_artifact)

        all_indices = set(range(len(patch.edits)))
        rolled_indices = sorted(all_indices - set(kept_indices))
        gate_passes = 1
        accepted = bool(full_candidate_gate_passed)
        final_graph: SkillGraph | None = candidate if accepted else None
        final_results = list(results if accepted else base_results)
        if accepted:
            final_hard, final_soft, final_score, final_skill = (
                hard, soft, score, skill
            )
            final_train = list(results if no_validation else candidate_train)
        else:
            final_hard, final_soft = _mean_result_scores(
                base_results, mixed_weight=self.mixed_weight
            )
            final_score = base_score
            final_skill = self._skill_for(base_graph)[0]
            final_train = list(base_results if no_validation else reference_train)
        related_train_ids = set(map(str, combined_grouped.train_ids))
        reference_related_train = list(
            base_results if no_validation else reference_train
        )
        final_related_train = list(final_train)
        final_train_audit = (
            _hard_transition_summary(reference_related_train, final_related_train)
            if reference_related_train or final_related_train
            else {"hard_net_case_gain": 0, "n_eligible": 0,
                  "n_improved": 0, "n_regressed": 0,
                  "improved_case_ids": [], "regressed_case_ids": []}
        )
        evaluated_final_hard_audit = _hard_transition_summary(
            base_results, final_results
        )

        if accepted:
            self.graph = final_graph
            self.score = final_score
            self._last_sel_results = list(final_results)
            self.evolution_cache.reconcile_graph(
                self.graph, source_graph="grouped_big_gate_final_graph"
            )
            self.best_score = final_score
            self.best_step = epoch
            self.best = final_graph.copy()
            save_skill_json(self.best, str(self.out / "best_graph.json"))
            (self.out / "best_skill.md").write_text(final_skill, encoding="utf-8")
        else:
            self.graph, self.score = base_graph, base_score
            self._last_sel_results = list(base_results)
            kept_indices = []
            rolled_indices = list(range(len(patch.edits)))
            final_score = base_score
            final_hard, final_soft = _mean_result_scores(
                base_results, mixed_weight=self.mixed_weight
            )
            self.evolution_cache.reconcile_graph(
                self.graph, source_graph="grouped_big_gate_epoch_rollback"
            )

        bad_case_summary = None
        if rolled_indices:
            failed_patch = GraphPatch(
                reasoning="rolled_back_complete_epoch_candidate",
                edits=[patch.edits[index] for index in rolled_indices],
            )
            validation_failure = _gate_failure_evidence(
                failed_patch, base_results, results, candidate_hard_audit,
                stage="big_gate_validation_component",
            )
            train_failure = _gate_failure_evidence(
                failed_patch, reference_train, candidate_train,
                candidate_train_audit,
                stage="big_gate_train_component",
            )
            validation_failure["gate_rule"] = "component_evidence_only"
            train_failure["gate_rule"] = "component_evidence_only"
            bad_case_summary = {
                "schema_version": "graphopt-combined-big-gate-failure-evidence-v1",
                "stage": "big_gate_complete_candidate_rollback",
                "gate_rule": (
                    "validation_hard_net_case_gain + "
                    "train_hard_net_case_gain > 0"
                ),
                "combined_hard_transition_audit": combined_big_gate_audit,
                "validation_component": validation_failure,
                "train_component": train_failure,
            }
            feedback = (
                "## Previous Big Gate rejection (one-shot evidence)\n"
                "The environment rejected the complete patch below on the "
                "full train+validation vote. Do not repeat it verbatim. Use "
                "both paired components to propose a narrower reusable rule. "
                "This evidence is diagnostic only and cannot approve an edit.\n\n"
                + json.dumps(bad_case_summary, ensure_ascii=False, indent=2)
            )
            if no_validation:
                update_pool_failure = _gate_failure_evidence(
                    failed_patch, base_results, results, candidate_hard_audit,
                    stage="big_gate_update_pool",
                )
                update_pool_failure["gate_rule"] = (
                    "complete_update_pool_hard_net_case_gain > 0"
                )
                bad_case_summary = {
                    "schema_version": "graphopt-update-pool-big-gate-failure-evidence-v1",
                    "stage": "big_gate_complete_candidate_rollback",
                    "gate_rule": "complete_update_pool_hard_net_case_gain > 0",
                    "update_pool_hard_transition_audit": combined_big_gate_audit,
                    "update_pool_component": update_pool_failure,
                }
                feedback = (
                    "## Previous Big Gate rejection (one-shot evidence)\n"
                    "The environment rejected the complete patch below on the "
                    "full update-pool vote. Do not repeat it verbatim. Propose a "
                    "narrower reusable rule from this paired evidence. This "
                    "evidence is diagnostic only and cannot approve an edit.\n\n"
                    + json.dumps(bad_case_summary, ensure_ascii=False, indent=2)
                )
            self.evolution_cache.set_pending_bad_case_prompt(feedback)
            save_json(step_dir / "bad_case_summary.json", bad_case_summary)
            (step_dir / "bad_case_prompt.txt").write_text(feedback, encoding="utf-8")
        self.evolution_cache.save(self.out / "evolution_cache.json")

        action = (
            "accept_complete_candidate_update_pool_big_gate"
            if no_validation and combined_gate_passed
            else "reject_complete_candidate_update_pool_big_gate"
            if no_validation
            else "accept_complete_candidate_train_validation_big_gate"
            if combined_gate_passed
            else "reject_complete_candidate_train_validation_big_gate"
        )
        attribution = {
            "schema_version": "graphopt-complete-big-gate-attribution-v1",
            "scope": (
                "complete_update_pool_decision"
                if no_validation
                else "full_train_plus_validation_joint_decision"
                if self.use_big_gate_train_tiebreak
                else "full_validation_decision"
            ),
            "environment_decides": True,
            "decision_granularity": "complete_candidate_only",
            "teacher_can_veto_only": False,
            "full_candidate_gate_passed": full_candidate_gate_passed,
            "combined_gate_passed": combined_gate_passed,
            "update_pool_gate_passed": update_pool_gate_passed,
            "validation_gate_passed": validation_gate_passed,
            "train_gate_passed": train_gate_passed,
            "test_used_for_decision": False,
            "update_protocol": self.update_protocol,
            "rollback_granularity": "complete_epoch_candidate_only",
            "partial_rescue_attempted": False,
            "teacher_used_for_decision": False,
            "audits": audits,
            "teacher_audit": teacher_artifact,
            "kept_edit_indices": kept_indices if accepted else [],
            "rolled_back_edit_indices": rolled_indices,
            "candidate_update_pool_audit": candidate_hard_audit if no_validation else None,
            "candidate_update_pool_evaluated": no_validation and primary_gate_evaluated,
            "candidate_validation_audit": (
                None if no_validation else candidate_hard_audit
            ),
            "candidate_validation_evaluated": (
                not no_validation and primary_gate_evaluated
            ),
            "candidate_train_audit": candidate_train_audit,
            "update_pool_diagnostic_audit": combined_big_gate_audit,
            "reference_train_source": reference_train_source,
            "reference_train_reused": reference_train_reused,
            "committed_update_pool_audit": (
                evaluated_final_hard_audit if no_validation else None
            ),
            "committed_validation_audit": (
                None if no_validation else evaluated_final_hard_audit
            ),
            "committed_train_audit": final_train_audit,
        }
        save_json(step_dir / "gate_attribution.json", attribution)
        record = {
            "step": epoch, "epoch": epoch, "action": action,
            "accepted": accepted, "current_score": final_score,
            "candidate_val_score": score, "val_score": final_score,
            "val_hard": final_hard, "val_soft": final_soft,
            "big_gate_policy": (
                "complete_candidate_full_update_pool_strict_net_gain"
                if no_validation
                else "complete_candidate_full_train_plus_validation_strict_net_gain"
            ),
            "big_gate_train_tiebreak": self.use_big_gate_train_tiebreak,
            "big_gate_allow_hard_tie": self.big_gate_allow_hard_tie,
            "full_candidate_gate_passed": full_candidate_gate_passed,
            "combined_gate_passed": combined_gate_passed,
            "update_pool_gate_passed": update_pool_gate_passed,
            "validation_gate_passed": validation_gate_passed,
            "train_gate_passed": train_gate_passed,
            "test_used_for_decision": False,
            "epoch_test": None,
            "update_protocol": self.update_protocol,
            "partial_rescue_attempted": False,
            "candidate_update_pool_audit": (
                candidate_hard_audit if no_validation else None
            ),
            "candidate_update_validation_audit": candidate_hard_audit,
            "candidate_update_pool_evaluated": no_validation and primary_gate_evaluated,
            "candidate_validation_evaluated": (
                not no_validation and primary_gate_evaluated
            ),
            "candidate_epoch_train_audit": candidate_train_audit,
            "update_pool_diagnostic_audit": combined_big_gate_audit,
            "reference_train_source": reference_train_source,
            "reference_train_reused": reference_train_reused,
            "evaluated_final_hard_audit": evaluated_final_hard_audit,
            "final_epoch_train_audit": final_train_audit,
            "epoch_related_train_case_ids": sorted(related_train_ids),
            "n_edits": len(patch.edits),
            "n_edits_kept": len(kept_indices) if accepted else 0,
            "n_edits_rolled_back": len(rolled_indices),
            "kept_edit_indices": kept_indices if accepted else [],
            "rolled_back_edit_indices": rolled_indices,
            "gate_passes": gate_passes,
            "max_gate_passes": 1,
            "rollout_tags": rollout_tags,
            "small_gates": small_gate_records,
            "n_small_gates": len(small_gate_records),
            "n_small_gates_accepted": sum(bool(row.get("accepted")) for row in small_gate_records),
            "n_rollout_cases": train_count,
            "n_reference_train_cases": len(reference_train),
            "n_reference_train_rollout_cases": (
                0 if reference_train_reused else len(reference_train)
            ),
            "n_evidence_cases": evidence_count,
            "n_excluded_cases": train_count - evidence_count,
            "edit_attribution": audits,
            "bad_case_summary": bad_case_summary,
            "pending_bad_case_prompt_set": bool(bad_case_summary),
            "update_boundary": (
                "epoch_synthesis_then_joint_local_gates_then_complete_update_pool_big_gate"
                if no_validation
                else "epoch_synthesis_then_joint_affected_scope_local_gates_then_complete_candidate_big_gate"
            ),
        }
        if not accepted:
            record["rolled_back_modifications"] = _rolled_back_modification_lessons(
                patch, record
            )
        self.history.append(record)
        gate_experience = self._prepare_gate_experience(record)
        self.evolution_cache.record_gate_experience(gate_experience)
        self.evolution_cache.save(self.out / "evolution_cache.json")
        save_json(step_dir / "gate_experience.json", gate_experience)
        save_json(step_dir / "gate.json", record)
        save_json(step_dir / "gate_committed_results.json", self._last_sel_results)
        save_json(step_dir / "gate_committed_train_results.json", final_train)
        save_skill_json(self.graph, str(self.out / "graphs" / f"graph_step{epoch:04d}.json"))
        return record

    def _prepare_gate_experience(self, record: dict[str, Any]) -> dict[str, Any]:
        """Carry only failed edits from an actual complete Big-Gate rollback."""
        epoch = int(record.get("epoch") or record.get("step") or 0)
        if not _is_reusable_big_gate_rollback_experience(record):
            return {
                "schema_version": "graphopt-gate-experience",
                "epoch": epoch,
                "action": record.get("action"),
                "accepted": bool(record.get("accepted")),
                "carry_to_next_epoch": False,
                "reason": "only complete-candidate Big-Gate rollbacks are reusable",
            }
        failed = copy.deepcopy(record.get("rolled_back_modifications") or [])
        return {
            "schema_version": "graphopt-rollback-lessons",
            "epoch": epoch,
            "action": record.get("action"),
            "accepted": False,
            "carry_to_next_epoch": True,
            "failed_modifications": failed,
            "usage_instruction": (
                "Use a lesson only when a later proposal touches the same node "
                "or edge; avoid repeating the failed edit for the recorded reason."
            ),
        }

    def _load_grouped_epoch_collection(
        self,
        *,
        epoch: int,
        update_dir: Path,
        combined: GroupedBatch,
        base_graph: SkillGraph,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Load a complete frozen-graph epoch collection or refuse stale reuse."""
        if not self.resume_requested:
            return None
        results_path = update_dir / "epoch_train_results.json"
        manifest_path = update_dir / "collection_manifest.json"
        if not results_path.is_file() and not manifest_path.is_file():
            return None
        if not results_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("partial epoch collection cannot be reused safely")
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid epoch collection artifact: {exc}") from exc
        records = list(manifest.get("records") or [])
        expected_hash = graph_sha256(base_graph)
        scheduled_ids = [
            str(case_id)
            for record in records
            for case_id in (record.get("train_case_ids") or [])
        ]
        result_ids = [str(row.get("id") or "") for row in results]
        expected_ids = [str(case_id) for case_id in combined.train_ids]
        valid = (
            manifest.get("schema_version") == "graphopt-epoch-collection-v1"
            and manifest.get("policy") == "all_batches_share_one_frozen_epoch_start_graph"
            and bool(records)
            and all(str(record.get("graph_sha256") or "") == expected_hash for record in records)
            and scheduled_ids == expected_ids
            and set(result_ids) == set(expected_ids)
            and len(result_ids) == len(expected_ids)
            and len(set(result_ids)) == len(result_ids)
            and all(str(row.get("response") or "").strip() for row in results)
        )
        if not valid:
            raise RuntimeError(
                f"epoch {epoch} collection identity/coverage changed; refusing stale stage reuse"
            )
        save_json(update_dir / "collection_resume_manifest.json", {
            "schema_version": "graphopt-collection-resume-v1",
            "epoch": epoch,
            "graph_sha256": expected_hash,
            "n_batches_reused": len(records),
            "n_cases_reused": len(results),
            "reused": True,
        })
        print(
            f"[graphopt stage resume] reused epoch {epoch} frozen collection "
            f"({len(records)} batches, {len(results)} cases)",
            flush=True,
        )
        return list(results), records

    def _load_post_joint_epoch_state(
        self,
        *,
        epoch: int,
        update_dir: Path,
        base_graph: SkillGraph,
        results: list[dict[str, Any]],
    ) -> tuple[GraphPatch, GraphEditPlan, EvolutionCache] | None:
        """Resume after a complete joint synthesis without regenerating it."""
        if not self.resume_requested:
            return None
        required = {
            "checkpoint": update_dir / "pre_joint_checkpoint.json",
            "joint": update_dir / "joint_semantic_synthesis.json",
            "proposed": update_dir / "proposed_patch.json",
            "plan": update_dir / "graph_edit_plan.json",
            "cache": update_dir / "evolution_cache.json",
        }
        if not all(path.is_file() for path in required.values()):
            return None
        try:
            checkpoint = json.loads(required["checkpoint"].read_text(encoding="utf-8"))
            joint = json.loads(required["joint"].read_text(encoding="utf-8"))
            proposed = GraphPatch.from_dict(
                json.loads(required["proposed"].read_text(encoding="utf-8"))
            )
            plan = GraphEditPlan.from_dict(
                json.loads(required["plan"].read_text(encoding="utf-8"))
            )
            cache = EvolutionCache.load(required["cache"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid post-joint checkpoint: {exc}") from exc
        valid = (
            str(checkpoint.get("graph_sha256") or "") == graph_sha256(base_graph)
            and str(checkpoint.get("results_signature") or "")
            == stage_results_signature(results)
            and int(checkpoint.get("step") or -1) == epoch * 10000
            and joint.get("status") == "complete"
            and int(joint.get("n_output_edits") or -1) == len(proposed.edits)
            and bool(proposed.edits)
            and cache is not None
            and int(cache.updated_step) == epoch * 10000
        )
        if not valid:
            raise RuntimeError("post-joint checkpoint identity or coverage changed")
        save_json(update_dir / "post_joint_resume_manifest.json", {
            "schema_version": "graphopt-post-joint-resume-v1",
            "epoch": epoch,
            "graph_sha256": graph_sha256(base_graph),
            "results_signature": stage_results_signature(results),
            "n_cases_reused": len(results),
            "n_joint_edits_reused": len(proposed.edits),
            "reused_through": "joint_semantic_synthesis",
            "recomputed_from": "local_gate",
        })
        print(
            f"[graphopt stage resume] reused completed joint synthesis "
            f"({len(proposed.edits)} edits); restarting at Local Gate",
            flush=True,
        )
        return proposed, plan, cache

    def run_grouped_epoch(
        self,
        *,
        epoch: int,
        grouped_batches: list[GroupedBatch],
    ) -> dict[str, Any]:
        """Synthesize once per epoch, locally Gate joint edits, then Big Gate."""
        base_graph = self.graph.copy()
        base_results = list(self._last_sel_results)
        base_score = self.score
        if not base_results:
            raise RuntimeError("grouped epoch requires the frozen full-valid_seen G0")

        no_validation = bool(self.cfg.get("no_validation_split", False))
        scheduled_val_ids = list(dict.fromkeys(
            str(case_id) for batch in grouped_batches
            for case_id in batch.val_ids
        ))
        if no_validation and scheduled_val_ids:
            raise RuntimeError(
                "no-validation protocol received non-empty validation IDs"
            )
        combined = GroupedBatch(
            train_ids=list(dict.fromkeys(
                str(case_id) for batch in grouped_batches
                for case_id in batch.train_ids
            )),
            val_ids=(
                [] if no_validation
                else [str(row.get("id")) for row in base_results]
            ),
            groups=[group for batch in grouped_batches for group in batch.groups],
            tail=any(batch.tail for batch in grouped_batches),
        )
        epoch_dir = self.out / "epochs" / f"epoch_{epoch:02d}"
        update_dir = epoch_dir / "epoch_update"
        update_dir.mkdir(parents=True, exist_ok=True)
        save_json(update_dir / "group_mapping.json", combined.to_dict())

        for prior_record in self.history:
            prior_epoch = int(
                prior_record.get("epoch") or prior_record.get("step") or 0
            )
            if 0 < prior_epoch < epoch:
                self.evolution_cache.record_gate_experience(
                    self._prepare_gate_experience(prior_record)
                )
        save_json(
            update_dir / "prior_gate_experiences.json",
            {
                "schema_version": "graphopt-prior-gate-experiences-v1",
                "current_epoch": epoch,
                "experiences": self.evolution_cache.gate_experiences,
                "valid_unseen_included": False,
            },
        )
        self.evolution_cache.save(self.out / "evolution_cache.json")

        # Every collection unit is rolled out against the same frozen G_t. On resume,
        # a complete collection is reused only after exact graph/ID validation.
        reused_collection = None if no_validation else self._load_grouped_epoch_collection(
            epoch=epoch, update_dir=update_dir, combined=combined,
            base_graph=base_graph,
        )
        if no_validation:
            _require_same_case_ids(
                [{"id": case_id} for case_id in combined.train_ids], base_results
            )
            # Keep the measured baseline rows immutable as the Big-Gate reference,
            # while restoring the fixed update-group positive context consumed by
            # the evolution stage. The old validation protocol attached this while
            # collecting train batches; full-pool reuse must do the same explicitly.
            all_train_results = copy.deepcopy(base_results)
            same_group_rightcases = _attach_same_group_rightcases(
                combined,
                all_train_results,
                enabled=self.evolution_cfg.use_positive_context,
            )
            collection_records = [{
                "collection_index": 1,
                "update_case_ids": list(map(str, combined.train_ids)),
                "same_group_rightcases": same_group_rightcases,
                "graph_sha256": graph_sha256(base_graph),
                "graph_mutated": False,
                "source": "complete_update_pool_baseline",
            }]
        elif reused_collection is not None:
            all_train_results, collection_records = reused_collection
        else:
            all_train_results = []
            collection_records = []
            for batch_index, grouped_batch in enumerate(grouped_batches, start=1):
                token = epoch * 10000 + batch_index
                train_batch = build_exact_split_batch(
                    self.dataloader, "train", grouped_batch.train_ids,
                    seed=self.seed + token,
                )
                rows, hard, soft, gate, _ = self._rollout(
                    base_graph, "train",
                    f"epoch_{epoch:02d}_collect_train_b{batch_index:04d}",
                    batch=train_batch, seed_offset=token,
                )
                rightcases = _attach_same_group_rightcases(
                    grouped_batch, rows,
                    enabled=self.evolution_cfg.use_positive_context,
                )
                all_train_results.extend(rows)
                record = {
                    "batch_index": batch_index,
                    "train_case_ids": list(map(str, grouped_batch.train_ids)),
                    "validation_case_ids": list(map(str, grouped_batch.val_ids)),
                    "train_hard": hard, "train_soft": soft, "train_gate": gate,
                    "same_group_rightcases": rightcases,
                    "graph_sha256": graph_sha256(base_graph),
                    "graph_mutated": False,
                }
                collection_records.append(record)
        if len({str(row.get("id")) for row in all_train_results}) != len(all_train_results):
            raise RuntimeError("epoch collection returned duplicate train case IDs")
        if set(map(str, combined.train_ids)) != {
            str(row.get("id")) for row in all_train_results
        }:
            raise RuntimeError("epoch collection did not cover the exact train schedule")
        save_json(update_dir / "collection_manifest.json", {
            "schema_version": "graphopt-epoch-collection-v1",
            "policy": "all_batches_share_one_frozen_epoch_start_graph",
            "records": collection_records,
        })
        save_json(update_dir / "epoch_train_results.json", all_train_results)

        all_update_results = (
            list(all_train_results)
            if no_validation else [*all_train_results, *base_results]
        )
        update_ids = [str(row.get("id") or "") for row in all_update_results]
        expected_update_ids = [
            *map(str, combined.train_ids), *map(str, combined.val_ids)
        ]
        if (
            len(update_ids) != len(expected_update_ids)
            or len(update_ids) != len(set(update_ids))
            or set(update_ids) != set(expected_update_ids)
        ):
            raise RuntimeError(
                "epoch update pool must cover the exact scheduled IDs without "
                "duplicates"
            )
        save_json(update_dir / "epoch_update_results.json", all_update_results)
        save_json(update_dir / "update_pool_manifest.json", {
            "schema_version": "graphopt-update-pool-final-v1",
            "policy": "complete_update_pool",
            "graph_sha256": graph_sha256(base_graph),
            "train_size": len(all_train_results),
            "validation_size": 0,
            "update_pool_size": len(all_update_results),
            "test_included": False,
        })

        eligible_epoch_results = [
            row for row in all_update_results
            if not row.get("exclude_from_evolution")
        ]
        evidence = [
            row for row in eligible_epoch_results
            if float(row.get("hard") or 0.0) < 1.0 - 1e-12
        ]
        epoch_summary: dict[str, Any] = {
            "epoch": epoch,
            "train_case_ids": list(map(str, combined.train_ids)),
            "validation_case_ids": list(map(str, combined.val_ids)),
            "train_size": len(combined.train_ids),
            "validation_size": 0,
            "update_pool_size": len(all_update_results),
            "update_pool_policy": "complete_update_pool_no_validation_split",
            "test_included_in_update": False,
            "n_evidence_cases": len(evidence),
            "update_boundary": (
                "complete_update_pool_then_joint_local_gates_then_update_pool_big_gate"
                if no_validation
                else "complete_train_validation_pool_then_joint_local_gates_then_big_gate"
            ),
        }
        if not evidence:
            self.graph = base_graph
            epoch_summary.update({
                "action": "skip_no_eligible_epoch_evidence",
                "accepted": False, "n_edits": 0, "edit_attribution": [],
            })
            save_json(update_dir / "epoch_local_gate_summary.json", epoch_summary)
            return self._run_grouped_big_gate(
                epoch=epoch, base_graph=base_graph, base_results=base_results,
                base_score=base_score, small_gate_records=[epoch_summary],
                grouped_batches=grouped_batches,
            )

        recorder = StepRecorder(update_dir)
        mode = self.reflect_mode if self.chat_fn else "template"
        epoch_cfg = copy.copy(self.evolution_cfg)
        if not self.allow_execution_children:
            epoch_cfg.max_execution_child_candidates_per_epoch = 0
        resume_stage = ""
        allow_legacy_v18_reconstruction = False
        if (
            reused_collection is not None
            and (update_dir / "patch_before_dedupe.json").is_file()
        ):
            resume_stage = "joint_semantic_synthesis"
        post_joint_reuse_allowed = True
        post_joint_state = (
            self._load_post_joint_epoch_state(
                epoch=epoch, update_dir=update_dir, base_graph=base_graph,
                results=eligible_epoch_results,
            )
            if (
                self.update_strategy == "evidence"
                and reused_collection is not None
                and post_joint_reuse_allowed
            )
            else None
        )
        if post_joint_state is not None:
            post_patch, post_plan, post_cache = post_joint_state
            evo = EvolutionResult(
                patch_edits=list(post_patch.edits), case_analyses=[],
                graph_statistics={}, merged_proposals={}, edit_plan=post_plan,
                reasoning=post_patch.reasoning, cache=post_cache,
            )
        elif self.update_strategy == "evidence":
            evo = run_evolution_step(
                base_graph, eligible_epoch_results, cfg=epoch_cfg,
                cache=self.evolution_cache,
                chat_fn=self.chat_fn, mode=mode,
                meta_context=format_meta(self.meta), step_dir=update_dir,
                step=epoch * 10000, recorder=recorder,
                update_protocol=self.update_protocol,
                resume_from_stage=resume_stage,
                allow_legacy_v18_reconstruction=allow_legacy_v18_reconstruction,
            )
        else:
            evo = run_ablation_evolution(
                base_graph, evidence, strategy=self.update_strategy,
                cache=self.evolution_cache, chat_fn=self.chat_fn, mode=mode,
                meta_context=format_meta(self.meta), step=epoch * 10000,
                random_seed=self.seed + epoch * 10000,
                random_sample_rate=self.ablation_random_sample_rate,
                recorder=recorder,
            )
        self.evolution_cache = evo.cache or self.evolution_cache
        proposed = GraphPatch(reasoning=evo.reasoning, edits=list(evo.patch_edits))
        save_json(update_dir / "proposed_patch.json", proposed.to_dict())
        proposed, protected = drop_protected_root_updates(
            proposed, self.protected_root_node_ids
        )
        proposed, orphan_weights = drop_orphan_weight_refreshes(proposed)
        proposed, joint_manifest = group_epoch_joint_patch(proposed)
        save_json(update_dir / "joint_group_manifest.json", {
            "schema_version": "graphopt-epoch-joint-edit-groups-v2",
            "policy": "shared_source_or_incident_node_transitive_closure",
            "pure_weight_refreshes_removed": True,
            "protected_root_updates_removed": protected,
            "orphan_weight_refreshes_removed": orphan_weights,
            "groups": joint_manifest,
        })
        save_json(update_dir / "joint_proposed_patch.json", proposed.to_dict())

        ranked: list[dict[str, Any]] = []
        groups = atomic_edit_groups(proposed.edits)
        for position, (group_id, indices) in enumerate(groups):
            ranked.append({
                "group_id": str(group_id), "indices": list(indices),
                "patch": GraphPatch(
                    reasoning=proposed.reasoning,
                    edits=[proposed.edits[index] for index in indices],
                ),
                "audit": {"position": position},
            })

        frozen_train_by_id = {
            str(row.get("id") or ""): row for row in all_train_results
        }
        frozen_val_by_id = {
            str(row.get("id") or ""): row
            for row in ([] if no_validation else base_results)
        }
        local_reference_val = [] if no_validation else base_results
        local_records: list[dict[str, Any]] = []
        locally_accepted: list[tuple[GraphPatch, dict[str, Any]]] = []
        for candidate_index, item in enumerate(ranked, start=1):
            current_patch = item["patch"]
            candidate_dir = _reusable_local_gate_candidate_dir(
                update_dir, candidate_index=candidate_index, patch=current_patch,
            )
            candidate_dir.mkdir(parents=True, exist_ok=True)
            final_record: dict[str, Any] | None = None
            for attempt in range(2):
                patch_digest = graph_patch_sha256(current_patch)
                token = epoch * 100000 + candidate_index * 10 + attempt
                affected = select_epoch_affected_cases(
                    grouped_batch=combined, base_graph=base_graph,
                    patch=current_patch, reference_train=all_train_results,
                    reference_val=local_reference_val,
                )
                scope_payload = {
                    "train_case_ids": list(affected["train_case_ids"]),
                    "validation_case_ids": list(affected["validation_case_ids"]),
                }
                scope_digest = hashlib.sha256(json.dumps(
                    scope_payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                plain_attempt_dir = candidate_dir / f"attempt_{attempt + 1}"
                attempt_dir = plain_attempt_dir
                old_scope_path = plain_attempt_dir / "affected_scope.json"
                if old_scope_path.is_file():
                    try:
                        old_scope = json.loads(old_scope_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        old_scope = {}
                    old_identity = {
                        "train_case_ids": list(old_scope.get("train_case_ids") or []),
                        "validation_case_ids": list(
                            old_scope.get("validation_case_ids") or []
                        ),
                    }
                    if old_identity != scope_payload:
                        attempt_dir = candidate_dir / (
                            f"attempt_{attempt + 1}_s{scope_digest[:12]}"
                        )
                attempt_dir.mkdir(parents=True, exist_ok=True)
                save_json(attempt_dir / "affected_scope.json", affected)
                candidate_graph = base_graph.copy()
                apply_report = apply_patch(candidate_graph, current_patch)
                save_json(attempt_dir / "apply_report.json", apply_report)
                reusable_local = _load_reusable_epoch_local_gate_attempt(
                    attempt_dir, patch=current_patch, patch_digest=patch_digest,
                    base_graph=base_graph, affected=affected,
                )
                if reusable_local is not None:
                    local = reusable_local
                    print(
                        "[graphopt stage resume] reused completed Local Gate "
                        f"joint={candidate_index} attempt={attempt + 1} "
                        f"patch={patch_digest[:12]}",
                        flush=True,
                    )
                elif apply_report.get("n_failed"):
                    local = {
                        "accepted": False, "materialization_failed": True,
                        "n_effective": 0, "n_ineffective": 1,
                        "effective_case_ids": [],
                        "ineffective_case_ids": [],
                        "affected_case_ids": [],
                        "decision_reason": "joint patch failed atomic materialization",
                        "per_edit_explanations": [],
                    }
                else:
                    train_ids = list(affected["train_case_ids"])
                    val_ids = list(affected["validation_case_ids"])
                    reference_train: list[dict[str, Any]] = []
                    candidate_train: list[dict[str, Any]] = []
                    reference_val: list[dict[str, Any]] = []
                    candidate_val: list[dict[str, Any]] = []
                    if train_ids:
                        train_batch = build_exact_split_batch(
                            self.dataloader, "train", train_ids,
                            seed=self.seed + token,
                        )
                        reference_train = [frozen_train_by_id[case_id] for case_id in train_ids]
                        candidate_train, _, _, _, _ = self._rollout(
                            candidate_graph, "train",
                            f"epoch_{epoch:02d}_local_j{candidate_index:03d}_a{attempt + 1}_p{patch_digest[:12]}_s{scope_digest[:12]}_post_train",
                            batch=train_batch, seed_offset=token,
                        )
                    if val_ids:
                        val_batch = build_exact_split_batch(
                            self.dataloader, "valid_seen", val_ids,
                            seed=self.seed + token + 500000,
                        )
                        reference_val = [frozen_val_by_id[case_id] for case_id in val_ids]
                        candidate_val, _, _, _, _, _ = self._run_gate_repeats(
                            candidate_graph,
                            f"epoch_{epoch:02d}_local_j{candidate_index:03d}_a{attempt + 1}_p{patch_digest[:12]}_s{scope_digest[:12]}_post_val",
                            batch=val_batch,
                        )
                    save_json(attempt_dir / "reference_reuse.json", {
                        "schema_version": "graphopt-local-gate-reference-reuse-v1",
                        "policy": "reuse_exact_frozen_epoch_start_observations",
                        "train_case_ids": train_ids,
                        "validation_case_ids": val_ids,
                        "base_graph_sha256": graph_sha256(base_graph),
                        "patch_sha256": patch_digest,
                        "candidate_only_reexecuted": True,
                    })
                    local = summarize_epoch_local_gate(
                        current_patch, reference_train=reference_train,
                        candidate_train=candidate_train,
                        reference_val=reference_val, candidate_val=candidate_val,
                    )
                explanation_status = str(
                    (local.get("semantic_explanation") or {}).get("status") or ""
                )
                if (
                    reusable_local is None
                    or explanation_status not in TERMINAL_LOCAL_GATE_EXPLANATION_STATUSES
                ):
                    semantic_explanation = explain_epoch_local_gate(
                        current_patch, local, chat_fn=self.chat_fn
                    )
                    local["semantic_explanation"] = semantic_explanation
                    explanation_by_index = {
                        int(row.get("edit_index")): row
                        for row in (semantic_explanation.get("edits") or [])
                        if isinstance(row, dict) and row.get("edit_index") is not None
                    }
                    for row in local.get("per_edit_explanations") or []:
                        row["semantic_explanation"] = copy.deepcopy(
                            explanation_by_index.get(
                                int(row.get("edit_index") or 0)
                            )
                        )
                    save_json(attempt_dir / "local_gate.json", local)
                    if reusable_local is None:
                        save_json(
                            attempt_dir / "tested_patch.json",
                            current_patch.to_dict(),
                        )
                    if reusable_local is not None:
                        print(
                            "[graphopt stage resume] regenerated incomplete Local Gate "
                            f"explanation joint={candidate_index} "
                            f"attempt={attempt + 1}",
                            flush=True,
                        )

                final_record = {
                    "epoch": epoch, "joint_candidate_index": candidate_index,
                    "attempt": attempt + 1,
                    "action": (
                        "accept_epoch_local_joint_gate"
                        if local["accepted"] else "reject_epoch_local_joint_gate"
                    ),
                    "accepted": bool(local["accepted"]),
                    "train_case_ids": list(affected.get("train_case_ids") or []),
                    "validation_case_ids": list(affected.get("validation_case_ids") or []),
                    "train_size": len(affected.get("train_case_ids") or []),
                    "validation_size": len(affected.get("validation_case_ids") or []),
                    "n_evidence_cases": len({
                        str(case_id) for edit in current_patch.edits
                        for case_id in edit.source_case_ids if str(case_id)
                    }),
                    "n_edits": len(current_patch.edits),
                    "patch_sha256": patch_digest,
                    "joint_patch": current_patch.to_dict(),
                    "affected_scope": affected,
                    "local_gate": local,
                    "edit_attribution": [
                        {**row, "keep": bool(local["accepted"])}
                        for row in local.get("per_edit_explanations") or []
                    ],
                    "update_protocol": self.update_protocol,
                }
                if local["accepted"]:
                    locally_accepted.append((current_patch, final_record))
                    break

                if (
                    attempt == 0
                    and not bool(local.get("accepted"))
                    and not bool(local.get("materialization_failed"))
                    and bool(local.get("ineffective_case_ids"))
                ):
                    refinement_path = attempt_dir / "precise_refinement.json"
                    refined_path = attempt_dir / "refined_patch.json"
                    refinement_reused = False
                    if (
                        reusable_local is not None
                        and refinement_path.is_file()
                        and refined_path.is_file()
                    ):
                        try:
                            refinement = json.loads(
                                refinement_path.read_text(encoding="utf-8")
                            )
                            refined = GraphPatch.from_dict(json.loads(
                                refined_path.read_text(encoding="utf-8")
                            ))
                            refinement_reused = True
                        except (OSError, ValueError, TypeError, json.JSONDecodeError):
                            refinement_reused = False
                    if not refinement_reused:
                        refined, refinement = refine_rejected_joint_patch(
                            base_graph=base_graph, patch=current_patch,
                            gate_evidence=local, chat_fn=self.chat_fn,
                        )
                        save_json(refinement_path, refinement)
                        save_json(refined_path, refined.to_dict())
                    else:
                        print(
                            "[graphopt stage resume] reused precise Local Gate refinement "
                            f"joint={candidate_index} patch={patch_digest[:12]}",
                            flush=True,
                        )
                    if refined.edits:
                        current_patch, refined_manifest = group_epoch_joint_patch(refined)
                        save_json(
                            attempt_dir / "refined_joint_manifest.json",
                            refined_manifest,
                        )
                        final_record["precise_refinement_triggered"] = True
                        continue
                break
            if final_record is None:
                raise RuntimeError("local Gate failed to produce a record")
            local_records.append(final_record)

        # Materialize every locally accepted joint group together only after
        # all local decisions are complete. A graph-level apply conflict turns
        # that group into an explicit rejection instead of a partial commit.
        candidate_graph = base_graph.copy()
        accepted_records: list[dict[str, Any]] = []
        for patch_item, record in locally_accepted:
            trial = candidate_graph.copy()
            report = apply_patch(trial, patch_item)
            if report.get("n_failed"):
                record["accepted"] = False
                record["action"] = "reject_combined_materialization_conflict"
                record["combined_apply_report"] = report
                for row in record.get("edit_attribution") or []:
                    row["keep"] = False
                self.evolution_cache.consume_rejected_proposals(
                    evo.edit_plan, list(patch_item.edits)
                )
                continue
            candidate_graph = trial
            record["combined_apply_report"] = report
            accepted_records.append(record)
            self.evolution_cache.consume_accepted_proposals(
                evo.edit_plan, list(patch_item.edits)
            )
        accepted_ids = {id(record) for record in accepted_records}
        for record in local_records:
            if id(record) in accepted_ids:
                continue
            rejected_patch = GraphPatch.from_dict(record.get("joint_patch") or {})
            self.evolution_cache.consume_rejected_proposals(
                evo.edit_plan, list(rejected_patch.edits)
            )

        self.graph = candidate_graph
        self.evolution_cache.ensure_graph_keys(self.graph)
        self.evolution_cache.save(self.out / "evolution_cache.json")
        epoch_summary.update({
            "action": "epoch_local_joint_gates_complete",
            "accepted": bool(accepted_records),
            "n_joint_candidates": len(local_records),
            "n_joint_candidates_accepted": len(accepted_records),
            "n_edits": sum(
                int(record.get("n_edits") or 0) for record in accepted_records
            ),
            "local_gate_records": local_records,
        })
        save_json(update_dir / "epoch_local_gate_summary.json", epoch_summary)
        save_json(update_dir / "small_gate_records.json", local_records)
        locally_admitted = GraphPatch(
            reasoning="all locally effective epoch joint proposals",
            edits=[
                GraphEdit.from_dict(row["edit"])
                for record in accepted_records
                for row in (record.get("edit_attribution") or [])
                if row.get("keep") and isinstance(row.get("edit"), dict)
            ],
        )
        save_json(update_dir / "local_gate_accepted_patch.json", locally_admitted.to_dict())
        save_json(update_dir / "component_raw_patch.json", proposed.to_dict())
        save_skill_json(candidate_graph, str(update_dir / "candidate_graph.json"))
        recorder.flush()
        return self._run_grouped_big_gate(
            epoch=epoch, base_graph=base_graph, base_results=base_results,
            base_score=base_score, small_gate_records=local_records,
            grouped_batches=grouped_batches,
        )

    def _run_small_group_gate(
        self,
        *,
        epoch: int,
        batch_index: int,
        grouped_batch: GroupedBatch,
        allow_execution_child: bool,
    ) -> dict[str, Any]:
        """Update from one train batch and run its mapped validation Gate once."""
        batch_dir = self.out / "epochs" / f"epoch_{epoch:02d}" / "batch_updates" / f"batch_{batch_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        recorder = StepRecorder(batch_dir)
        token = epoch * 10000 + batch_index
        train_batch = build_exact_split_batch(
            self.dataloader, "train", grouped_batch.train_ids, seed=self.seed + token
        )
        val_batch = build_exact_split_batch(
            self.dataloader, "valid_seen", grouped_batch.val_ids,
            seed=self.seed + token + 500000,
        )
        reference_val: list[dict[str, Any]] = []
        ref_val_hard = ref_val_soft = ref_val_gate = 0.0
        reference_val_tags: list[str] = []
        if self.use_small_gate:
            (
                reference_val,
                ref_val_hard,
                ref_val_soft,
                ref_val_gate,
                _,
                reference_val_tags,
            ) = self._run_gate_repeats(
                self.graph,
                f"epoch_{epoch:02d}_small_g_pre_val_b{batch_index:04d}",
                batch=val_batch,
            )
        train_results, train_hard, train_soft, train_gate, _ = self._rollout(
            self.graph, "train", f"epoch_{epoch:02d}_small_g_train_b{batch_index:04d}",
            batch=train_batch, seed_offset=token,
        )
        same_group_rightcases = _attach_same_group_rightcases(
            grouped_batch, train_results,
            enabled=self.evolution_cfg.use_positive_context,
        )
        save_json(batch_dir / "same_group_rightcases.json", {
            "schema_version": "graphopt-same-group-rightcases-v1",
            "persistent": False,
            "teacher_curator": False,
            "groups": same_group_rightcases,
        })
        evidence = [
            row for row in train_results
            if not row.get("exclude_from_evolution")
            and float(row.get("hard") or 0.0) < 1.0 - 1e-12
        ]
        record: dict[str, Any] = {
            "epoch": epoch, "batch_in_epoch": batch_index,
            "train_case_ids": list(grouped_batch.train_ids),
            "validation_case_ids": list(grouped_batch.val_ids),
            "train_size": len(grouped_batch.train_ids),
            "validation_size": len(grouped_batch.val_ids),
            "tail_batch": grouped_batch.tail,
            "train_hard": train_hard, "train_soft": train_soft,
            "train_gate": train_gate, "n_evidence_cases": len(evidence),
        }
        save_json(batch_dir / "group_mapping.json", grouped_batch.to_dict())
        if not evidence:
            empty_patch = GraphPatch(
                reasoning="no eligible training evidence", edits=[]
            )
            save_json(
                batch_dir / "component_raw_patch.json", empty_patch.to_dict()
            )
            save_json(
                batch_dir / "gate_input_patch.json", empty_patch.to_dict()
            )
            record.update(action="skip_no_eligible_training_evidence", accepted=False, n_edits=0)
            save_json(batch_dir / "small_gate.json", record)
            recorder.flush()
            return record

        mode = self.reflect_mode if self.chat_fn else "template"
        batch_evolution_cfg = self.evolution_cfg
        if not allow_execution_child:
            batch_evolution_cfg = copy.copy(self.evolution_cfg)
            batch_evolution_cfg.max_execution_child_candidates_per_epoch = 0
        if self.update_strategy == "evidence":
            evo = run_evolution_step(
                self.graph, evidence, cfg=batch_evolution_cfg, cache=self.evolution_cache,
                chat_fn=self.chat_fn, mode=mode, meta_context=format_meta(self.meta),
                step_dir=batch_dir, step=token, recorder=recorder,
                update_protocol=self.update_protocol,
            )
        else:
            evo = run_ablation_evolution(
                self.graph, evidence, strategy=self.update_strategy,
                cache=self.evolution_cache, chat_fn=self.chat_fn, mode=mode,
                meta_context=format_meta(self.meta), step=token,
                random_seed=self.seed + token,
                random_sample_rate=self.ablation_random_sample_rate,
                recorder=recorder,
            )
        self.evolution_cache = evo.cache or self.evolution_cache
        proposed_patch = GraphPatch(
            reasoning=evo.reasoning, edits=list(evo.patch_edits)
        )
        save_json(batch_dir / "proposed_patch.json", proposed_patch.to_dict())
        execution_child_proposed = any(
            edit.edit_kind in {"execution_detail", "execution_detail_child"}
            for edit in proposed_patch.edits
        )
        record["execution_child_candidate_allowed"] = allow_execution_child
        record["execution_child_proposed"] = execution_child_proposed
        record["n_edits_proposed"] = len(proposed_patch.edits)
        patch, protected_root_updates = drop_protected_root_updates(
            proposed_patch, self.protected_root_node_ids
        )
        if protected_root_updates:
            record["protected_root_updates_dropped"] = protected_root_updates
            protected_set = set(protected_root_updates)
            self.evolution_cache.consume_rejected_proposals(
                evo.edit_plan,
                [
                    edit for edit in proposed_patch.edits
                    if edit.op == "update_node" and str(edit.node_id) in protected_set
                ],
            )
        patch, dropped_weight_refreshes = drop_orphan_weight_refreshes(patch)
        if dropped_weight_refreshes:
            record["orphan_weight_refreshes_dropped"] = dropped_weight_refreshes
        # Freeze the structurally valid proposal stream before Gate logic.
        # Component shadows never feed back into training.
        component_raw_patch = GraphPatch(
            reasoning=patch.reasoning, edits=list(patch.edits)
        )
        save_json(
            batch_dir / "component_raw_patch.json",
            component_raw_patch.to_dict(),
        )
        record["update_protocol"] = self.update_protocol
        original_candidate_patch = GraphPatch(
            reasoning=patch.reasoning, edits=list(patch.edits)
        )
        patch, _, _ = limit_patch_to_atomic_groups(
            patch, self.max_atomic_groups_per_small_gate
        )
        for edit in patch.edits:
            edit.evidence_items = []
        record["legacy_trace_evidence_used_for_gate"] = False

        selected_indices = set()
        selected_group_id = None
        if patch.edits:
            for group_id, indices in atomic_edit_groups(original_candidate_patch.edits):
                group_edits = [original_candidate_patch.edits[index] for index in indices]
                if group_edits == patch.edits:
                    selected_indices.update(indices)
                    selected_group_id = str(group_id)
                    break
        deferred_patch = GraphPatch(
            reasoning=original_candidate_patch.reasoning,
            edits=[
                edit for index, edit in enumerate(original_candidate_patch.edits)
                if index not in selected_indices
            ],
        )
        deferred_group_ids = [
            str(group_id)
            for group_id, _ in atomic_edit_groups(original_candidate_patch.edits)
            if str(group_id) != selected_group_id
        ]
        save_json(batch_dir / "deferred_patch.json", deferred_patch.to_dict())
        record["deferred_atomic_group_ids"] = deferred_group_ids
        record["n_atomic_groups_tested"] = len(atomic_edit_groups(patch.edits))
        record["n_atomic_groups_deferred"] = len(deferred_group_ids)
        save_json(batch_dir / "patch.json", patch.to_dict())
        save_json(
            batch_dir / "unlimited_candidate_patch.json",
            original_candidate_patch.to_dict(),
        )
        save_json(batch_dir / "gate_input_patch.json", patch.to_dict())

        if not patch.edits:
            record.update(
                action=(
                    "skip_orphan_weight_refreshes"
                    if dropped_weight_refreshes else "skip_no_edits"
                ),
                accepted=False,
                n_edits=0,
            )
            save_json(batch_dir / "small_gate.json", record)
            self.evolution_cache.save(self.out / "evolution_cache.json")
            recorder.flush()
            return record

        candidate = self.graph.copy()
        apply_report = apply_patch(candidate, patch)
        applied = [int(index) for index in apply_report.get("applied_indices") or []]
        effective = GraphPatch(
            reasoning=patch.reasoning, edits=[patch.edits[index] for index in applied]
        )
        save_json(batch_dir / "apply_report.json", apply_report)
        save_json(batch_dir / "effective_patch.json", effective.to_dict())
        if not effective.edits:
            self.evolution_cache.consume_rejected_proposals(evo.edit_plan, list(patch.edits))
            self.evolution_cache.save(self.out / "evolution_cache.json")
            record.update(action="skip_no_effective_edits", accepted=False, n_edits=0,
                          n_edits_proposed=len(patch.edits))
            save_json(batch_dir / "small_gate.json", record)
            recorder.flush()
            return record

        save_skill_json(self.graph, str(batch_dir / "pre_batch_graph.json"))
        if not self.use_small_gate:
            # E ablation: keep the structured updater and the epoch big Gate,
            # but remove the online A/(A+B) decision and its extra rollouts.
            self.graph = candidate
            self.evolution_cache.consume_accepted_proposals(
                evo.edit_plan, effective.edits
            )
            self.evolution_cache.ensure_graph_keys(self.graph)
            self.evolution_cache.save(self.out / "evolution_cache.json")
            self.evolution_cache.save(batch_dir / "evolution_cache.json")
            record.update({
                "action": "accept_small_gate_disabled",
                "accepted": True,
                "small_gate_policy": "disabled_by_ablation",
                "reference_graph": "pre_batch_graph.json",
                "candidate_graph_semantics": "pre_batch_graph_plus_current_batch_patch",
                "n_edits": len(effective.edits),
                "edit_ops": [edit.op for edit in effective.edits],
                "rollout_tags": [
                    f"epoch_{epoch:02d}_small_g_train_b{batch_index:04d}"
                ],
                "gate_comparisons": 0,
                "graph_rollout_passes": 0,
                "partial_rollback": False,
            })
            save_json(batch_dir / "small_gate.json", record)
            save_skill_json(candidate, str(batch_dir / "candidate_graph.json"))
            save_skill_json(self.graph, str(batch_dir / "committed_graph.json"))
            recorder.flush()
            return record

        gate_train_ids = list(map(str, grouped_batch.train_ids))
        gate_train_batch = train_batch
        gate_reference_train = list(train_results)
        gate_reference_train_hard = train_hard
        gate_reference_train_soft = train_soft
        gate_reference_train_tags: list[str] = []
        if (
            self.use_train_tiebreak
            and gate_train_ids != list(map(str, grouped_batch.train_ids))
        ):
            gate_train_batch = build_exact_split_batch(
                self.dataloader, "train", gate_train_ids, seed=self.seed + token
            )
            (
                gate_reference_train,
                gate_reference_train_hard,
                gate_reference_train_soft,
                _,
                _,
            ) = self._rollout(
                self.graph, "train",
                f"epoch_{epoch:02d}_small_g_reference_related_train_b{batch_index:04d}",
                batch=gate_train_batch, seed_offset=token,
            )
            gate_reference_train_tags = [
                f"epoch_{epoch:02d}_small_g_reference_related_train_b{batch_index:04d}"
            ]

        def run_related_candidate_train(
            graph: SkillGraph, edits: list[GraphEdit], suffix: str
        ) -> tuple[list[dict[str, Any]], float, float, float, list[str]]:
            # Candidate usage discovery must see every train case in this batch;
            # per-edit auditing later narrows and expands exact fixed groups.
            del edits
            related_train_ids = list(gate_train_ids)
            if not related_train_ids:
                return [], 0.0, 0.0, 0.0, []
            # Reuse the *same materialized batch* and rollout seed as the
            # reference graph.  Rebuilding with a different seed confounds the
            # graph edit with item/choice ordering and model sampling context.
            related_batch = gate_train_batch
            rows, hard, soft, gate, _ = self._rollout(
                graph, "train",
                f"epoch_{epoch:02d}_small_g_{suffix}_related_train_b{batch_index:04d}",
                batch=related_batch, seed_offset=token,
            )
            rows = [row for row in rows if str(row.get("id")) in set(related_train_ids)]
            _require_same_case_ids(
                [row for row in gate_reference_train if str(row.get("id")) in set(related_train_ids)],
                rows,
            )
            return rows, hard, soft, gate, [
                f"epoch_{epoch:02d}_small_g_{suffix}_related_train_b{batch_index:04d}"
            ]

        (
            candidate_val,
            cand_val_hard,
            cand_val_soft,
            cand_val_gate,
            _,
            candidate_val_tags,
        ) = self._run_gate_repeats(
            candidate,
            f"epoch_{epoch:02d}_small_g_candidate_val_b{batch_index:04d}",
            batch=val_batch,
        )
        _require_same_case_ids(reference_val, candidate_val)
        # Before/after graph usage is part of the per-edit measurement. The
        # environment still decides: usage selects case scope; hard transitions
        # supply the votes; the teacher can only veto a script-eligible edit.
        save_json(batch_dir / "small_gate_reanalysis.json", {
            "schema_version": "graphopt-gate-usage-input-v3",
            "used_for_gate_scope_selection": True,
            "decision_authority": "paired_environment_hard_transitions",
            "reference_case_ids": [str(row.get("id")) for row in reference_val],
            "candidate_case_ids": [str(row.get("id")) for row in candidate_val],
        })
        candidate_hard_audit = _hard_transition_summary(reference_val, candidate_val)
        group_audits = _group_hard_audit(
            grouped_batch, reference_val, candidate_val,
        )
        if self.use_train_tiebreak:
            (
                candidate_train,
                candidate_train_hard,
                candidate_train_soft,
                candidate_train_gate,
                candidate_train_tags,
            ) = run_related_candidate_train(candidate, effective.edits, "candidate")
        else:
            candidate_train = []
            candidate_train_hard = candidate_train_soft = candidate_train_gate = 0.0
            candidate_train_tags = []
        if self.selective_gate:
            edit_attribution, kept_indices, retry_indices = _mapped_small_gate_attribution(
                grouped_batch, self.graph, candidate, effective.edits,
                reference_val, candidate_val,
                gate_reference_train if self.use_train_tiebreak else [], candidate_train,
                retry_round=0,
                chat_fn=(self.chat_fn if self.use_teacher_veto else None),
                require_nonnegative_combined_on_validation_gain=(
                    self.require_nonnegative_combined_on_validation_gain
                ),
            )
        else:
            # Global-Gate ablation: usage/audits remain observable, but no edit
            # is filtered. The complete B is decided only by the final whole-
            # candidate validation/train formula below.
            diagnostic_audits = build_atomic_edit_audits(
                grouped_batch=grouped_batch, base_graph=self.graph,
                edits=effective.edits, reference_val=reference_val,
                candidate_val=candidate_val,
                reference_train=(gate_reference_train if self.use_train_tiebreak else []),
                candidate_train=candidate_train,
                require_nonnegative_combined_on_validation_gain=(
                    self.require_nonnegative_combined_on_validation_gain
                ),
            )
            diagnostic_by_index = {
                index: audit for audit in diagnostic_audits
                for index in audit["edit_indices"]
            }
            kept_indices = set(range(len(effective.edits)))
            retry_indices = set()
            edit_attribution = [{
                "edit_index": index,
                "atomic_group_id": diagnostic_by_index[index]["atomic_group_id"],
                "decision_scope": "global_candidate_diagnostic_only",
                "individual_isolation": False,
                "environment_usage_statistics": diagnostic_by_index[index],
                "teacher_review": None,
                "teacher_audit_status": "disabled_by_global_gate_ablation",
                "keep": True,
                "request_reoptimization": False,
                "attribution_decision": "not_used_global_candidate_only",
                "attribution_basis": "diagnostic_only",
                "edit": edit.to_dict(),
            } for index, edit in enumerate(effective.edits)]
        reoptimization_performed = False
        first_round_attribution = list(edit_attribution)
        teacher_audit_artifact = {
            "schema_version": "graphopt-per-edit-audit-artifact-v1",
            "status": (
                edit_attribution[0].get("teacher_audit_status")
                if edit_attribution else "no_effective_atomic_groups"
            ),
            "audits": [],
            "reviews": [],
            "kept_group_ids": [],
        }
        seen_audit_groups: set[str] = set()
        for row in edit_attribution:
            group_id = str(row.get("atomic_group_id"))
            if group_id in seen_audit_groups:
                continue
            seen_audit_groups.add(group_id)
            teacher_audit_artifact["audits"].append(
                row.get("environment_usage_statistics")
            )
            if row.get("teacher_review") is not None:
                teacher_audit_artifact["reviews"].append(row["teacher_review"])
            if row.get("keep"):
                teacher_audit_artifact["kept_group_ids"].append(group_id)
        save_json(
            batch_dir / "small_gate_edit_teacher_audit.json",
            teacher_audit_artifact,
        )

        candidate_hard_audit = _hard_transition_summary(reference_val, candidate_val)
        group_audits = _group_hard_audit(grouped_batch, reference_val, candidate_val)
        all_indices = set(range(len(effective.edits)))
        rolled_indices = sorted(all_indices - kept_indices)
        final_graph = candidate
        final_val = candidate_val
        final_val_hard = cand_val_hard
        final_val_soft = cand_val_soft
        final_val_gate = cand_val_gate
        partial_val_tags: list[str] = []
        partial_train_tags: list[str] = []
        final_train = candidate_train
        if kept_indices and rolled_indices:
            final_graph = self.graph.copy()
            selected_patch = GraphPatch(
                reasoning="mapped_validation_partial_small_gate",
                edits=[
                    edit for index, edit in enumerate(effective.edits)
                    if index in kept_indices
                ],
            )
            partial_report = apply_patch(final_graph, selected_patch)
            if partial_report.get("n_failed"):
                raise RuntimeError(
                    "small Gate partial graph failed to materialize: "
                    + "; ".join(str(x) for x in partial_report.get("warnings") or [])
                )
            save_json(batch_dir / "partial_apply_report.json", partial_report)
            (
                final_val,
                final_val_hard,
                final_val_soft,
                final_val_gate,
                _,
                partial_val_tags,
            ) = self._run_gate_repeats(
                final_graph,
                f"epoch_{epoch:02d}_small_g_partial_val_b{batch_index:04d}",
                batch=val_batch,
            )
            _require_same_case_ids(reference_val, final_val)
            if self.use_train_tiebreak:
                (
                    final_train, _, _, _, partial_train_tags
                ) = run_related_candidate_train(
                    final_graph,
                    [
                        edit for index, edit in enumerate(effective.edits)
                        if index in kept_indices
                    ],
                    "partial",
                )
            else:
                final_train, partial_train_tags = [], []
        kept_edits = [
            edit for index, edit in enumerate(effective.edits)
            if index in kept_indices
        ]
        kept_related_validation_ids = (
            set(map(str, grouped_batch.val_ids))
            if not self.selective_gate else {
                str(case_id)
                for row in edit_attribution
                if int(row.get("edit_index", -1)) in kept_indices
                for case_id in (
                    (row.get("environment_usage_statistics") or {}).get(
                        "related_validation_case_ids"
                    ) or []
                )
            }
        )
        kept_related_train_ids = (
            set(map(str, grouped_batch.train_ids))
            if not self.selective_gate else {
                str(case_id)
                for row in edit_attribution
                if int(row.get("edit_index", -1)) in kept_indices
                for case_id in (
                    (row.get("environment_usage_statistics") or {}).get(
                        "related_train_case_ids"
                    ) or []
                )
            }
        )
        final_reference_val = [
            row for row in reference_val
            if str(row.get("id")) in kept_related_validation_ids
        ]
        final_candidate_val = [
            row for row in final_val
            if str(row.get("id")) in kept_related_validation_ids
        ]
        final_hard_audit = (
            _hard_transition_summary(final_reference_val, final_candidate_val)
            if final_reference_val or final_candidate_val
            else {
                "n_eligible": 0, "reference_hard_successes": 0.0,
                "candidate_hard_successes": 0.0, "hard_success_delta": 0.0,
                "improved_case_ids": [], "regressed_case_ids": [],
                "unchanged_case_ids": [], "n_improved": 0, "n_regressed": 0,
                "hard_net_case_gain": 0,
            }
        )
        final_reference_train = [
            row for row in gate_reference_train if str(row.get("id")) in kept_related_train_ids
        ] if self.use_train_tiebreak else []
        final_candidate_train = [
            row for row in final_train if str(row.get("id")) in kept_related_train_ids
        ]
        final_train_audit = (
            _hard_transition_summary(final_reference_train, final_candidate_train)
            if final_reference_train or final_candidate_train
            else {
                "n_eligible": 0, "reference_hard_successes": 0.0,
                "candidate_hard_successes": 0.0, "hard_success_delta": 0.0,
                "improved_case_ids": [], "regressed_case_ids": [],
                "unchanged_case_ids": [], "n_improved": 0, "n_regressed": 0,
                "hard_net_case_gain": 0,
            }
        )
        accepted = bool(
            kept_indices
            and self.pipeline_policy.accept_small_candidate(
                final_hard_audit, final_train_audit,
                use_train_tiebreak=self.use_train_tiebreak,
            )
        )
        if accepted:
            action = (
                "accept_partial_unified_gate"
                if rolled_indices else "accept_unified_gate"
            )
            self.graph = final_graph
            accepted_edits = [
                edit for index, edit in enumerate(effective.edits)
                if index in kept_indices
            ]
            rejected_edits = [
                edit for index, edit in enumerate(effective.edits)
                if index in rolled_indices
            ]
            self.evolution_cache.consume_accepted_proposals(evo.edit_plan, accepted_edits)
            if rejected_edits:
                self.evolution_cache.consume_rejected_proposals(evo.edit_plan, rejected_edits)
            self.evolution_cache.ensure_graph_keys(self.graph)
        else:
            atomic_audit = (
                (edit_attribution[0].get("environment_usage_statistics") or {})
                if edit_attribution else {}
            )
            script_decision = str(atomic_audit.get("script_decision") or "")
            teacher_review = (
                edit_attribution[0].get("teacher_review")
                if edit_attribution else None
            )
            if (
                atomic_audit.get("script_keep")
                and isinstance(teacher_review, dict)
                and teacher_review.get("decision") == "ROLLBACK"
            ):
                action = "reject_teacher_veto"
            else:
                action = {
                    "ROLLBACK_TRACE_SOURCE_UNOBSERVED": (
                        "reject_trace_source_unobserved"
                    ),
                    "ROLLBACK_TRACE_SOURCE_NOT_REPAIRED": (
                        "reject_trace_source_not_repaired"
                    ),
                    "ROLLBACK_TRACE_PROTECTED_SUCCESS_REGRESSION": (
                        "reject_trace_protected_success_regression"
                    ),
                    "ROLLBACK_COMBINED_NET_REGRESSION": (
                        "reject_combined_net_regression"
                    ),
                    "ROLLBACK_NO_REPORTED_USAGE_OR_GROUP": (
                        "reject_no_reported_usage_or_group"
                    ),
                    "ROLLBACK_UNCOVERED_UNKNOWN_USAGE": (
                        "reject_uncovered_unknown_usage"
                    ),
                    "ROLLBACK_NET_NOT_POSITIVE": (
                        "reject_mapped_validation_not_improved"
                    ),
                }.get(script_decision, "reject_combined_gate_not_improved")
            self.evolution_cache.consume_rejected_proposals(
                evo.edit_plan, effective.edits
            )
            feedback_val_ids = set(
                map(str, atomic_audit.get("related_validation_case_ids") or [])
            )
            feedback_reference_val = [
                row for row in reference_val
                if not feedback_val_ids or str(row.get("id")) in feedback_val_ids
            ]
            feedback_candidate_val = [
                row for row in candidate_val
                if not feedback_val_ids or str(row.get("id")) in feedback_val_ids
            ]
            feedback_audit = _hard_transition_summary(
                feedback_reference_val, feedback_candidate_val
            )
            feedback = _format_small_gate_rejection(
                effective, feedback_reference_val, feedback_candidate_val,
                feedback_audit, decision=action, atomic_edit_audit=atomic_audit,
            )
            self.evolution_cache.set_pending_bad_case_prompt(feedback)
            (batch_dir / "small_gate_rejection_prompt.txt").write_text(
                feedback, encoding="utf-8"
            )
            record["pending_bad_case_prompt_set"] = True
        self.evolution_cache.save(self.out / "evolution_cache.json")
        self.evolution_cache.save(batch_dir / "evolution_cache.json")
        rollout_tags = [
            f"epoch_{epoch:02d}_small_g_train_b{batch_index:04d}",
            *reference_val_tags,
            *gate_reference_train_tags,
            *candidate_val_tags,
            *candidate_train_tags,
            *partial_val_tags,
            *partial_train_tags,
        ]
        record.update({
            "action": action,
            "accepted": accepted,
            "small_gate_policy": (
                ("per_edit_validation_net_then_train_tie_teacher_veto_and_combined_rerun"
                 if self.use_train_tiebreak else
                 "per_edit_validation_net_teacher_veto_and_combined_rerun")
                if self.selective_gate else
                "global_candidate_validation_then_train_tie"
            ),
            "reoptimization_performed": reoptimization_performed,
            "retry_round": 0,
            "first_round_edit_attribution": first_round_attribution,
            "final_train_audit": final_train_audit,
            "accepted_edit_source_train_case_ids": (
                sorted({str(case_id) for edit in kept_edits for case_id in edit.source_case_ids})
                if accepted else []
            ),
            "accepted_related_train_case_ids": (
                sorted(kept_related_train_ids) if accepted else []
            ),
            "accepted_related_validation_case_ids": (
                sorted(kept_related_validation_ids) if accepted else []
            ),
            "reference_graph": "pre_batch_graph.json",
            "candidate_graph_semantics": "pre_batch_graph_plus_current_batch_patch",
            "reference_hard": (
                final_hard_audit["reference_hard_successes"] / max(final_hard_audit["n_eligible"], 1)
            ),
            "candidate_hard": (
                final_hard_audit["candidate_hard_successes"]
                / max(final_hard_audit["n_eligible"], 1)
            ),
            "final_hard": final_val_hard,
            "reference_hard_successes": final_hard_audit["reference_hard_successes"],
            "candidate_hard_successes": candidate_hard_audit["candidate_hard_successes"],
            "final_hard_successes": final_hard_audit["candidate_hard_successes"],
            "hard_success_delta": final_hard_audit["hard_success_delta"],
            "n_improved": final_hard_audit["n_improved"],
            "n_regressed": final_hard_audit["n_regressed"],
            "improved_case_ids": final_hard_audit["improved_case_ids"],
            "regressed_case_ids": final_hard_audit["regressed_case_ids"],
            "reference_train_hard": gate_reference_train_hard,
            "reference_train_soft": gate_reference_train_soft,
            "gate_train_case_ids": list(gate_train_ids),
            "reference_validation_hard": ref_val_hard,
            "reference_validation_soft": ref_val_soft,
            "reference_validation_gate": ref_val_gate,
            "candidate_validation_hard": cand_val_hard,
            "candidate_validation_soft": cand_val_soft,
            "candidate_validation_gate": cand_val_gate,
            "final_validation_hard": final_val_hard,
            "final_validation_soft": final_val_soft,
            "final_validation_gate": final_val_gate,
            "group_hard_audits": group_audits,
            "edit_attribution": edit_attribution,
            "kept_edit_indices": sorted(kept_indices) if accepted else [],
            "rolled_back_edit_indices": (
                rolled_indices if accepted else sorted(all_indices)
            ),
            "n_edits": len(effective.edits),
            "n_edits_kept": len(kept_indices) if accepted else 0,
            "n_edits_rolled_back": (
                len(rolled_indices) if accepted else len(effective.edits)
            ),
            "edit_ops": [edit.op for edit in effective.edits],
            "rollout_tags": rollout_tags,
            "gate_comparisons": 2 if partial_val_tags else 1,
            "graph_rollout_passes": 3 if partial_val_tags else 2,
            "partial_rollback": bool(accepted and rolled_indices),
        })
        save_json(batch_dir / "gate_reference_train_results.json", gate_reference_train)
        save_json(batch_dir / "gate_reference_results.json", reference_val)
        save_json(batch_dir / "gate_candidate_results.json", candidate_val)
        save_json(batch_dir / "gate_candidate_related_train_results.json", candidate_train)
        if partial_val_tags:
            save_json(batch_dir / "gate_partial_results.json", final_val)
        save_json(batch_dir / "small_gate_attribution.json", {
            "edit_attribution": edit_attribution,
            "kept_edit_indices": sorted(kept_indices) if accepted else [],
            "rolled_back_edit_indices": (
                rolled_indices if accepted else sorted(all_indices)
            ),
        })
        save_json(batch_dir / "small_gate.json", record)
        save_skill_json(candidate, str(batch_dir / "candidate_graph.json"))
        if partial_val_tags:
            save_skill_json(final_graph, str(batch_dir / "partial_candidate_graph.json"))
        save_skill_json(self.graph, str(batch_dir / "committed_graph.json"))
        recorder.flush()
        return record

    def run_epoch(
        self,
        step: int,
        epoch: int,
        *,
        batches: list[Any] | None = None,
        expected_train_size: int = 0,
    ) -> dict[str, Any]:
        """Run one epoch-level GraphOpt update.

        ``self.graph`` stays frozen while every rollout batch in the planned
        epoch shard is collected. No case, batch, or group of batches can mutate
        the graph. Only after the complete shard has been collected do we analyze,
        create one patch, and run exactly one complete-candidate Gate.
        """
        step_dir = self.out / "steps" / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        recorder = StepRecorder(step_dir)
        t0 = time.time()
        meta_ctx = format_meta(self.meta)

        all_train: list[dict[str, Any]] = []
        batch_stats: list[dict] = []
        rollout_tags: list[str] = []

        epoch_batches = list(batches) if batches is not None else [None]
        for batch_index, batch in enumerate(epoch_batches, start=1):
            tag = f"epoch_{epoch:02d}_train_b{batch_index:04d}"
            rollout_tags.append(tag)
            train_res, th, ts, tg, _ = self._rollout(
                self.graph,
                "train",
                tag,
                batch=batch,
                seed_offset=epoch * 100_000 + batch_index,
            )
            all_train.extend(train_res)
            batch_stats.append(
                {
                    "batch_in_epoch": batch_index,
                    "tag": tag,
                    "n": len(train_res),
                    "hard": th,
                    "soft": ts,
                    "gate": tg,
                }
            )

        # One epoch shard is a single evidence set. Silently losing or repeating
        # a benchmark case would bias graph statistics, so fail before analysis.
        case_ids = [str(r.get("id")) for r in all_train]
        if expected_train_size > 0 and len(case_ids) != len(set(case_ids)):
            duplicates = sorted({cid for cid in case_ids if case_ids.count(cid) > 1})
            raise ValueError(f"epoch {epoch} contains duplicate train case ids: {duplicates[:10]}")
        if expected_train_size > 0 and len(all_train) != expected_train_size:
            raise ValueError(
                f"epoch {epoch} rollout is incomplete: expected {expected_train_size} "
                f"train cases, got {len(all_train)}"
            )

        excluded_train = [
            row for row in all_train if bool(row.get("exclude_from_evolution"))
        ]
        train_evidence = [
            row for row in all_train if not bool(row.get("exclude_from_evolution"))
        ]
        exclusion_manifest = {
            "schema_version": "graphopt-training-evidence-exclusions-v1",
            "epoch": epoch,
            "count_policy": "excluded cases do not enter Analyzer or graph updates; metric eligibility is controlled independently",
            "raw_rollout_cases": len(all_train),
            "eligible_evidence_cases": len(train_evidence),
            "excluded_cases": [
                {
                    "id": row.get("id"),
                    "reason": row.get("evolution_exclusion_reason") or row.get("exclusion_reason"),
                    "phase": row.get("phase"),
                    "fail_reason": row.get("fail_reason"),
                    "abnormal_response_steps": row.get("abnormal_response_steps") or [],
                    "n_turns": row.get("n_turns"),
                    "step_limit_excluded": bool(row.get("step_limit_excluded")),
                    "overlength_threshold": row.get("overlength_threshold"),
                }
                for row in excluded_train
            ],
        }
        recorder.store.save(
            "excluded_train_cases",
            stage="training_evidence_filter",
            inputs={"raw_rollout_cases": len(all_train)},
            outputs=exclusion_manifest,
        )

        _, rv = recorder.store.save(
            "rollout_train",
            stage="rollout_train",
            inputs={
                "tags": rollout_tags,
                "rollout_batches": len(epoch_batches),
                "update_boundary": (
                    "epoch_shard"
                    if self.cfg.get("_active_train_schedule") == "disjoint_epoch_shards"
                    else "full_epoch"
                ),
                "step": step,
                "epoch": epoch,
            },
            outputs={
                "case_ids": [r.get("id") for r in all_train],
                "evidence_case_ids": [r.get("id") for r in train_evidence],
                "excluded_case_ids": [r.get("id") for r in excluded_train],
                "batches": batch_stats,
            },
        )
        recorder.record(
            "rollout_train",
            files=[f"rollout_train_v{rv}.json", "artifact_index.json"]
            + [f"../rollouts/{t}/rollout_v*.json" for t in rollout_tags],
            n_cases=len(train_evidence),
            n_rollout_cases=len(all_train),
            n_excluded_cases=len(excluded_train),
            calls_llm=True,
            version=rv,
        )

        if not train_evidence:
            rec = {
                "step": step,
                "epoch": epoch,
                "action": "skip_no_eligible_training_evidence",
                "current_score": self.score,
                "accepted": False,
                "n_edits": 0,
                "n_cases": len(train_evidence),
                "n_rollout_cases": len(all_train),
                "n_evidence_cases": 0,
                "n_excluded_cases": len(excluded_train),
                "rollout_batches": batch_stats,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            self.history.append(rec)
            save_json(step_dir / "gate.json", rec)
            save_json(step_dir / "gate_committed_results.json", self._last_sel_results)
            recorder.store.save("gate", stage="gate", inputs={"n_edits": 0}, outputs=rec)
            recorder.record(
                "gate_skip_no_evidence",
                files=["gate_v*.json", "excluded_train_cases_v*.json"],
                action=rec["action"],
            )
            recorder.flush()
            save_skill_json(
                self.graph, str(self.out / "graphs" / f"graph_step{step:04d}.json")
            )
            return rec

        mode = self.reflect_mode if self.chat_fn else "template"
        if self.update_strategy == "evidence":
            evo = run_evolution_step(
                self.graph,
                train_evidence,
                cfg=self.evolution_cfg,
                cache=self.evolution_cache,
                chat_fn=self.chat_fn,
                mode=mode,
                meta_context=meta_ctx,
                step_dir=step_dir,
                step=step,
                recorder=recorder,
                update_protocol=self.update_protocol,
            )
        else:
            evo = run_ablation_evolution(
                self.graph,
                train_evidence,
                strategy=self.update_strategy,
                cache=self.evolution_cache,
                chat_fn=self.chat_fn,
                mode=mode,
                meta_context=meta_ctx,
                step=step,
                random_seed=self.seed + step,
                random_sample_rate=self.ablation_random_sample_rate,
                recorder=recorder,
            )
        self.evolution_cache = evo.cache or self.evolution_cache
        self.evolution_cache.save(self.out / "evolution_cache.json")
        self.evolution_cache.save(step_dir / "evolution_cache.json")
        clipped = GraphPatch(reasoning=evo.reasoning, edits=list(evo.patch_edits))

        if not clipped.edits:
            rec = {
                "step": step,
                "epoch": epoch,
                "action": "skip_no_edits",
                "current_score": self.score,
                "accepted": False,
                "n_edits": 0,
                "n_cases": len(train_evidence),
                "n_rollout_cases": len(all_train),
                "n_evidence_cases": len(train_evidence),
                "n_excluded_cases": len(excluded_train),
                "rollout_batches": batch_stats,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            self.history.append(rec)
            save_json(step_dir / "gate.json", rec)
            save_json(step_dir / "gate_committed_results.json", self._last_sel_results)
            recorder.store.save("gate", stage="gate", inputs={"n_edits": 0}, outputs=rec)
            recorder.record("gate_skip_no_edits", files=["gate_v*.json"], action="skip_no_edits")
            recorder.flush()
            save_skill_json(self.graph, str(self.out / "graphs" / f"graph_step{step:04d}.json"))
            return rec

        (step_dir / "patch.json").write_text(
            json.dumps(clipped.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        cand = self.graph.copy()
        report = apply_patch(cand, clipped)
        proposed_count = len(clipped.edits)
        applied_indices = [int(i) for i in report.get("applied_indices") or []]
        clipped = GraphPatch(
            reasoning=clipped.reasoning,
            edits=[clipped.edits[i] for i in applied_indices],
        )
        (step_dir / "effective_patch.json").write_text(
            json.dumps(clipped.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        candidate_bundle = {
            "schema_version": "graphopt-candidate-edit-bundle-v1",
            "reference_epoch": max(0, epoch - 1),
            "candidate_epoch": epoch,
            "comparison_scope": "immediately_previous_checkpoint_only",
            "n_edits": len(clipped.edits),
            "ordered_edits": [
                {
                    "edit_index": index,
                    "label": f"edit{index}",
                    "edit": edit.to_dict(),
                }
                for index, edit in enumerate(clipped.edits)
            ],
        }
        (step_dir / "candidate_edit_bundle.json").write_text(
            json.dumps(candidate_bundle, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        recorder.store.save(
            "apply_report",
            stage="apply_patch",
            inputs={"n_edits": proposed_count},
            outputs=report,
        )
        save_skill_json(cand, str(step_dir / "candidate_graph.json"))
        recorder.record(
            "apply_patch",
            files=[
                "apply_report_v*.json", "candidate_graph.json", "effective_patch.json",
                "candidate_edit_bundle.json",
            ],
            n_edits=proposed_count,
            n_applied=len(clipped.edits),
        )

        if not clipped.edits:
            rec = {
                "step": step,
                "epoch": epoch,
                "action": "skip_no_effective_edits",
                "current_score": self.score,
                "accepted": False,
                "n_edits": 0,
                "n_edits_proposed": proposed_count,
                "n_edits_failed_apply": int(report.get("n_failed") or 0),
                "n_cases": len(train_evidence),
                "n_rollout_cases": len(all_train),
                "n_evidence_cases": len(train_evidence),
                "n_excluded_cases": len(excluded_train),
                "rollout_batches": batch_stats,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            self.history.append(rec)
            save_json(step_dir / "gate.json", rec)
            save_json(step_dir / "gate_committed_results.json", self._last_sel_results)
            recorder.store.save(
                "gate",
                stage="gate",
                inputs={"n_edits": 0},
                outputs=rec,
            )
            recorder.record(
                "gate_skip_no_effective_edits",
                files=["gate_v*.json", "apply_report_v*.json", "effective_patch.json"],
                action="skip_no_effective_edits",
            )
            recorder.flush()
            save_skill_json(
                self.graph,
                str(self.out / "graphs" / f"graph_step{step:04d}.json"),
            )
            return rec

        gate_rollout_tags: list[str] = []
        prev_val = list(self._last_sel_results)
        if not prev_val:
            raise RuntimeError(
                "Gate reference results are missing. Run the one-time baseline before "
                "epoch updates; GraphOpt will not re-rollout the current graph inside Gate."
            )
        (step_dir / "gate_reference_results.json").write_text(
            json.dumps(prev_val, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        val_res, vh, vs, vg, val_skill, val_tags = self._run_gate_repeats(
            cand, f"step_{step:04d}_val"
        )
        gate_rollout_tags.extend(val_tags)
        if self.use_gate:
            _require_same_case_ids(prev_val, val_res)

        save_json(step_dir / "gate_candidate_results.json", val_res)
        candidate_hard_audit = _hard_transition_summary(prev_val, val_res)
        direct_source_train_ids = list(dict.fromkeys(
            str(case_id) for edit in clipped.edits
            for case_id in edit.source_case_ids if str(case_id)
        ))
        # Ungrouped schedules use the whole update batch as the sibling-
        # protection group for every edit that reports usage/source.
        related_train_ids = list(dict.fromkeys(
            str(row.get("id")) for row in all_train if str(row.get("id"))
        ))
        related_batch = (
            build_exact_split_batch(
                self.dataloader, "train", related_train_ids,
                seed=self.seed + step + 700000,
            ) if self.dataloader is not None and related_train_ids else None
        )

        def evaluate_related_train(graph: SkillGraph, suffix: str) -> list[dict[str, Any]]:
            if not related_train_ids:
                return []
            rows, _, _, _, _ = self._rollout(
                graph, "train", f"step_{step:04d}_{suffix}_related_train",
                batch=related_batch, seed_offset=step + 700000,
            )
            wanted = set(related_train_ids)
            rows = [row for row in rows if str(row.get("id")) in wanted]
            return rows

        if self.use_gate and self.use_train_tiebreak:
            reference_related_train = evaluate_related_train(
                self.graph, "reference"
            )
            candidate_related_train = evaluate_related_train(cand, "candidate")
            _require_same_case_ids(reference_related_train, candidate_related_train)
            candidate_train_audit = _hard_transition_summary(
                reference_related_train, candidate_related_train
            )
        else:
            reference_related_train = []
            candidate_related_train = []
            candidate_train_audit = {
                "hard_net_case_gain": 0, "source": "disabled_by_ablation"
            }

        full_candidate_gate_passed = bool(
            not self.use_gate
            or self.pipeline_policy.accept_complete_candidate(
                candidate_hard_audit,
                candidate_train_audit,
                allow_validation_tie=self.big_gate_allow_hard_tie,
                use_train_tiebreak=self.use_train_tiebreak,
            )
        )

        audits = build_atomic_edit_audits(
            grouped_batch=None,
            base_graph=self.graph,
            edits=clipped.edits,
            reference_val=prev_val,
            candidate_val=val_res,
            reference_train=(
                reference_related_train
                if self.use_gate and self.use_train_tiebreak else []
            ),
            candidate_train=candidate_related_train,
            require_nonnegative_combined_on_validation_gain=(
                self.require_nonnegative_combined_on_validation_gain
            ),
            allow_validation_tie=self.big_gate_allow_hard_tie,
        )
        teacher_artifact = {
            "schema_version": "graphopt-per-edit-audit-artifact-v1",
            "status": "diagnostic_only_complete_candidate_big_gate",
            "audits": audits,
            "reviews": [],
            "kept_group_ids": (
                [str(audit["atomic_group_id"]) for audit in audits]
                if full_candidate_gate_passed else []
            ),
            "decision_authority": "complete_candidate_environment_hard_transition",
            "teacher_used_for_decision": False,
            "partial_rescue_supported": False,
        }
        kept_indices = (
            list(range(len(clipped.edits))) if full_candidate_gate_passed else []
        )
        save_json(step_dir / "per_edit_gate_audits.json", audits)
        save_json(step_dir / "per_edit_teacher_audit.json", teacher_artifact)

        all_indices = set(range(len(clipped.edits)))
        rolled_indices = sorted(all_indices - set(kept_indices))
        accepted_final = bool(full_candidate_gate_passed)
        final_graph: SkillGraph | None = cand if accepted_final else None
        final_val_res = list(val_res if accepted_final else prev_val)
        if accepted_final:
            final_vh, final_vs, final_vg, final_skill = vh, vs, vg, val_skill
            final_related_train = list(candidate_related_train)
        else:
            final_vh, final_vs = _mean_result_scores(
                prev_val, mixed_weight=self.mixed_weight
            )
            final_vg = self.score
            final_skill = self._skill_for(self.graph)[0]
            final_related_train = list(reference_related_train)
            kept_indices = []
            rolled_indices = list(range(len(clipped.edits)))

        final_related_train_ids = set(related_train_ids)
        final_reference_train = list(reference_related_train)
        final_candidate_train = list(final_related_train)
        final_train_audit = (
            _hard_transition_summary(final_reference_train, final_candidate_train)
            if self.use_gate and self.use_train_tiebreak
            and (final_reference_train or final_candidate_train)
            else {"hard_net_case_gain": 0, "n_eligible": 0,
                  "n_improved": 0, "n_regressed": 0,
                  "improved_case_ids": [], "regressed_case_ids": [],
                  "source": "disabled_by_ablation"}
        )
        evaluated_final_hard_audit = _hard_transition_summary(
            prev_val, final_val_res
        )

        is_best = accepted_final and final_vg > self.best_score + 1e-12
        decision = GateDecision(
            accepted=accepted_final,
            action=(
                "accept_new_best" if accepted_final and is_best
                else "accept" if accepted_final else "reject"
            ),
            current_score=self.score,
            candidate_score=vg,
            best_score=final_vg if accepted_final and is_best else self.best_score,
            best_step=step if accepted_final and is_best else self.best_step,
            reason=(
                "gate_disabled" if not self.use_gate
                else "complete_candidate_passed"
                if accepted_final else "complete_candidate_rejected"
            ),
        )
        sg = None
        gate_reanalysis = audits
        attr = {
            "schema_version": "graphopt-complete-big-gate-attribution-v1",
            "scope": "full_validation_and_whole_update_train_group",
            "environment_decides": True,
            "decision_granularity": "complete_candidate_only",
            "teacher_can_veto_only": False,
            "full_candidate_gate_passed": full_candidate_gate_passed,
            "partial_rescue_attempted": False,
            "teacher_used_for_decision": False,
            "audits": audits,
            "teacher_audit": teacher_artifact,
            "kept_edit_indices": kept_indices if accepted_final else [],
            "rolled_back_edit_indices": rolled_indices,
            "candidate_validation_audit": candidate_hard_audit,
            "candidate_train_audit": candidate_train_audit,
            "committed_validation_audit": evaluated_final_hard_audit,
            "committed_train_audit": final_train_audit,
        }
        recorder.store.save(
            "gate_attribution", stage="complete_candidate_big_gate",
            inputs={"n_edits": len(clipped.edits)}, outputs=attr,
        )
        save_json(step_dir / "gate_attribution.json", attr)
        if self.use_gate and not accepted_final:
            failure_evidence = _gate_failure_evidence(
                clipped, prev_val, val_res, candidate_hard_audit,
                stage="big_gate_complete_candidate_rejected",
            )
            feedback = _format_small_gate_rejection(
                clipped, prev_val, val_res, candidate_hard_audit
            )
            self.evolution_cache.set_pending_bad_case_prompt(feedback)
            save_json(step_dir / "bad_case_summary.json", failure_evidence)
            (step_dir / "bad_case_prompt.txt").write_text(feedback, encoding="utf-8")
        save_json(step_dir / "gate_candidate_related_train_results.json", candidate_related_train)
        final_hard_audit = _hard_transition_summary(prev_val, final_val_res)
        train_hard = sum(float(row.get("hard") or 0.0) for row in train_evidence) / max(
            len(train_evidence), 1
        )
        rec = {
            "step": step,
            "epoch": epoch,
            "train_score": train_hard,
            "val_hard": final_vh,
            "val_soft": final_vs,
            "val_score": final_vg,
            "candidate_val_score": vg,
            "current_score": final_vg if decision.accepted else self.score,
            "accepted": decision.accepted if self.use_gate else bool(final_graph),
            "action": sg.action if sg else decision.action,
            "reason": sg.reason if sg else decision.reason,
            "edit_ops": [e.op for e in clipped.edits],
            "n_edits": len(clipped.edits),
            "n_edits_proposed": proposed_count,
            "n_edits_failed_apply": int(report.get("n_failed") or 0),
            "n_edits_kept": len(kept_indices) if decision.accepted else 0,
            "n_edits_rolled_back": (
                len(clipped.edits) - len(kept_indices)
                if decision.accepted else len(clipped.edits)
            ),
            "n_cases": len(train_evidence),
            "n_rollout_cases": len(all_train),
            "n_evidence_cases": len(train_evidence),
            "n_excluded_cases": len(excluded_train),
            "warnings": report.get("warnings"),
            "rollout_batches": batch_stats,
            "elapsed_sec": round(time.time() - t0, 2),
            "ablation_mode": self.ablation_mode,
            "update_strategy": self.update_strategy,
            "use_gate": self.use_big_gate,
            "use_small_gate": self.use_small_gate,
            "use_big_gate": self.use_big_gate,
            "use_train_tiebreak": self.use_train_tiebreak,
            "selective_gate": self.selective_gate,
            "use_teacher_veto": self.use_teacher_veto,
            "ablation_random_sample_rate": self.ablation_random_sample_rate,
            "pending_bad_case_prompt_set": bool(
                self.use_gate and not decision.accepted
            ),
            "big_gate_policy": (
                "complete_candidate_validation_then_combined_train_protection"
                if self.big_gate_allow_hard_tie
                else "complete_candidate_validation_then_train_tie"
                if self.use_train_tiebreak
                else "complete_candidate_validation_net"
            ),
            "full_candidate_gate_passed": full_candidate_gate_passed,
            "partial_rescue_attempted": False,
            "big_gate_allow_hard_tie": self.big_gate_allow_hard_tie,
            "edit_source_train_case_ids": direct_source_train_ids,
            "related_train_case_ids": sorted(final_related_train_ids),
            "candidate_related_train_audit": candidate_train_audit,
            "final_related_train_audit": final_train_audit,
            "candidate_update_validation_audit": candidate_hard_audit,
            "committed_hard_audit": final_hard_audit,
            "gate_passes": len(gate_rollout_tags),
            "max_gate_passes": 1,
        }
        self.history.append(rec)
        recorder.store.save("gate", stage="gate", inputs={"n_edits": len(clipped.edits)}, outputs=rec)
        gate_rollout_out = {
            "tags": gate_rollout_tags,
            "candidate_val_score": vg,
            "final_val_score": final_vg,
            "gate_repeats": GATE_REPEATS,
            "gate_passes": len(gate_rollout_tags),
            "max_gate_passes": 1,
            "action": rec.get("action"),
        }
        recorder.store.save(
            "gate_rollouts",
            stage="gate_rollouts",
            inputs={"tags": gate_rollout_tags},
            outputs=gate_rollout_out,
        )
        (step_dir / "gate.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

        gate_files = [
            "gate_v*.json", "gate_rollouts_v*.json", "artifact_index.json",
            "gate_reference_results.json", "candidate_edit_bundle.json",
        ]
        if (step_dir / "gate_attribution.json").exists() or recorder.store.versions("gate_attribution"):
            gate_files.append("gate_attribution_v*.json")
        if recorder.store.versions("pending_bad_case_prompt"):
            gate_files.extend([
                "pending_bad_case_prompt_v*.json", "bad_case_summary.json",
                "bad_case_prompt.txt",
            ])
        recorder.record(
            "gate",
            files=gate_files + [f"../rollouts/{t}/rollout_v*.json" for t in gate_rollout_tags],
            action=rec.get("action"),
            accepted=rec.get("accepted"),
            calls_llm=True,
        )
        recorder.flush()

        accepted = rec["accepted"]
        if accepted and final_graph is not None:
            accepted_indices = list(kept_indices)
            accepted_edits = [clipped.edits[i] for i in accepted_indices]
            self.evolution_cache.consume_accepted_proposals(evo.edit_plan, accepted_edits)
            # The Gate has now made one terminal decision for the complete
            # candidate; every planned cluster is therefore consumed together.
            # during transactional apply. Consume the whole remaining plan so
            # no mature rejected proposal can reappear unchanged. Probe-deferred
            # execution children were removed from edit_plan earlier.
            self.evolution_cache.consume_planned_proposals(evo.edit_plan)
            self.graph = final_graph
            self.evolution_cache.reconcile_graph(
                self.graph, source_graph="big_gate_final_graph"
            )
            self.evolution_cache.save(self.out / "evolution_cache.json")
            self.evolution_cache.save(step_dir / "evolution_cache.json")
            self.score = final_vg
            self._last_sel_results = final_val_res
            if final_vg >= self.best_score - 1e-12:
                # Accepted ties must update the deployable best graph too.
                self.best_score = max(self.best_score, final_vg)
                self.best = final_graph.copy()
                self.best_step = step
                save_skill_json(self.best, str(self.out / "best_graph.json"))
                (self.out / "best_skill.md").write_text(final_skill, encoding="utf-8")
        else:
            self.evolution_cache.consume_rejected_proposals(
                evo.edit_plan, list(clipped.edits)
            )
            self.evolution_cache.reconcile_graph(
                self.graph, source_graph="big_gate_rollback"
            )
            self.evolution_cache.save(self.out / "evolution_cache.json")
            self.evolution_cache.save(step_dir / "evolution_cache.json")

        save_json(step_dir / "gate_committed_results.json", self._last_sel_results)
        save_skill_json(self.graph, str(self.out / "graphs" / f"graph_step{step:04d}.json"))
        print(
            json.dumps(
                {
                    "step": step,
                    "val_score": final_vg,
                    "accepted": accepted,
                    "action": rec["action"],
                    "n_edits": len(clipped.edits),
                    "n_edits_kept": rec["n_edits_kept"],
                    "n_cases": len(train_evidence),
                    "n_rollout_cases": len(all_train),
                    "n_excluded_cases": len(excluded_train),
                    "rollout_batches": len(epoch_batches),
                    "update_boundary": (
                        "epoch_shard"
                        if self.cfg.get("_active_train_schedule") == "disjoint_epoch_shards"
                        else "full_epoch"
                    ),
                },
                ensure_ascii=False,
            )
        )
        return rec

    def step(
        self,
        step: int,
        epoch: int,
        *,
        batches: list[Any] | None = None,
        expected_train_size: int = 0,
    ) -> dict[str, Any]:
        """Compatibility alias; one historical ``step`` now means one epoch."""
        return self.run_epoch(
            step,
            epoch,
            batches=batches,
            expected_train_size=expected_train_size,
        )

    def run(self) -> dict[str, Any]:
        if self.experiment_mode in {
            "graphopt_best", "no_skill", "initial_skill", "skillaa_md"
        }:
            return self._run_evaluation_only()

        epochs = max(1, int(self.cfg.get("epochs") or self.cfg.get("num_epochs") or 1))
        if (
            self.update_protocol == "case_complete_v1"
            and self.ablation_mode == "g_full"
            and self.run_epoch_test
            and epochs != 3
        ):
            raise ValueError("the full main GraphOpt run requires exactly 3 epochs")
        stop_after_epoch = int(self.cfg.get("stop_after_epoch") or epochs)
        if not 1 <= stop_after_epoch <= epochs:
            raise ValueError(
                f"stop_after_epoch must be in [1, {epochs}], got {stop_after_epoch}"
            )
        if stop_after_epoch < self.resume_completed_epochs:
            raise ValueError(
                "stop_after_epoch cannot precede the restored checkpoint: "
                f"stop={stop_after_epoch} restored={self.resume_completed_epochs}"
            )
        train_size, rollout_batches_per_epoch, _ = resolve_steps(self.cfg, self.dataloader)
        shard_train = bool(self.cfg.get("shard_train_across_epochs", True))
        grouped_active = bool(
            self.grouped_batch_gate and self.dataloader is not None
        )
        no_validation = bool(self.cfg.get("no_validation_split", False))
        single_full_pool = bool(self.cfg.get("single_full_pool_input", False))
        grouped_schedule: list[list[GroupedBatch]] = []
        if grouped_active:
            grouped_schedule = build_grouped_schedule(
                list(getattr(self.dataloader, "train_items", []) or []),
                list(self.dataloader.get_split_items("valid_seen") or []),
                split_dir=self.cfg.get("split_dir"), batch_size=self.train_batch_size,
                single_full_pool=single_full_pool,
                num_epochs=epochs, seed=self.seed, train_size=train_size,
            )
            shard_sizes = [
                sum(len(batch.train_ids) for batch in epoch_batches)
                for epoch_batches in grouped_schedule
            ]
            rollout_batches_per_epoch = max(len(items) for items in grouped_schedule)
            if no_validation:
                for epoch_index, epoch_batches in enumerate(
                    grouped_schedule, start=1
                ):
                    scheduled_train_ids = [
                        str(case_id) for batch in epoch_batches
                        for case_id in batch.train_ids
                    ]
                    scheduled_val_ids = [
                        str(case_id) for batch in epoch_batches
                        for case_id in batch.val_ids
                    ]
                    if scheduled_val_ids:
                        raise RuntimeError(
                            "no-validation grouped schedule contains validation "
                            f"IDs in epoch {epoch_index}"
                        )
                    if (
                        len(scheduled_train_ids) != shard_sizes[epoch_index - 1]
                        or len(scheduled_train_ids) != len(set(scheduled_train_ids))
                    ):
                        raise RuntimeError(
                            "no-validation grouped schedule must cover each "
                            f"update ID exactly once in epoch {epoch_index}"
                        )
            save_json(
                self.out / "grouped_schedule.json",
                {
                    "schema_version": "graphopt-grouped-schedule-v1",
                    "update_cases_per_group": 4,
                    "batch_size": self.batch_size,
                    "batch_size_scope": self.batch_size_scope,
                    "train_batch_size": self.train_batch_size,
                    "validation_batch_size": self.validation_batch_size,
                    "epochs": [
                        [batch.to_dict() for batch in epoch_batches]
                        for epoch_batches in grouped_schedule
                    ],
                },
            )
        else:
            shard_sizes = _train_shard_sizes(train_size, epochs) if shard_train and train_size else []
            if shard_sizes:
                rollout_batches_per_epoch = max(
                    1, (max(shard_sizes) + self.train_batch_size - 1) // self.train_batch_size
                )
        self.cfg["_active_train_schedule"] = (
            "full_epoch_collect_synthesize_joint_local_gate_big_gate" if grouped_active
            else "disjoint_epoch_shards" if shard_sizes else "full_epoch"
        )
        if self.accumulation != 1:
            print(
                "[graphopt] warning: train.accumulation is a legacy option and is ignored; "
                "grouped GraphOpt collects one frozen epoch before proposal synthesis."
            )
        collection_detail = (
            f"update_pool_cases/epoch={shard_sizes or 'n/a'} "
            if single_full_pool
            else (
                f"batch_size={self.batch_size} "
                f"batch_size_scope={self.batch_size_scope} "
                f"train_batch_size={self.train_batch_size} "
                f"mapped_val_batch_size="
                f"{self.validation_batch_size if grouped_active else 'n/a'} "
            )
        )
        print(
            f"[graphopt] train_size={train_size or 'n/a'} "
            f"train_schedule={self.cfg['_active_train_schedule']} "
            f"{'epoch_train_sizes' if grouped_active else 'shard_sizes'}={shard_sizes or 'n/a'} "
            f"rollout_batches/epoch<={rollout_batches_per_epoch} "
            f"updates/epoch={1 if grouped_active else 1} "
            f"local_joint_gates/epoch=proposal_dependent "
            f"big_gates/epoch={1 if self.use_big_gate else 0} "
            f"{collection_detail}"
            f"evolution x={self.evolution_cfg.node_support_threshold} "
            f"y={self.evolution_cfg.edge_support_threshold} "
            f"per_node_update_points={self.evolution_cfg.node_merge_top_k} "
            f"execution_children/epoch<={self.evolution_cfg.max_execution_child_candidates_per_epoch} "
            f"execution_merge_evidence<={self.evolution_cfg.execution_merge_evidence_cap} "
            f"execution_clusters/parent<={self.evolution_cfg.max_active_execution_children_per_parent} "
            f"patch_cap=none"
        )

        if self.score < 0 and not self._restore_incomplete_baseline():
            self._baseline()

        self._finish_pending_meta()
        self._finish_pending_epoch_tests()

        for ep in range(self.resume_completed_epochs + 1, stop_after_epoch + 1):
            self._epoch_start_graph_text = format_graph(self.graph)
            epoch_prev_sel = list(self._last_sel_results)
            epoch_dir = self.out / "epochs" / f"epoch_{ep:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            if grouped_active:
                grouped_batches = grouped_schedule[ep - 1]
                save_json(
                    epoch_dir / "batches.json",
                    [
                        {
                            "batch_in_epoch": index + 1,
                            **batch.to_dict(),
                            "train_schedule": "full_epoch_collect_synthesize_joint_local_gate_big_gate",
                            "shard_size": shard_sizes[ep - 1],
                        }
                        for index, batch in enumerate(grouped_batches)
                    ],
                )
                self.run_grouped_epoch(epoch=ep, grouped_batches=grouped_batches)
            else:
                epoch_batches: list[Any]
                expected_epoch_size = train_size
                if self.dataloader is not None:
                    if shard_sizes:
                        epoch_batches = plan_disjoint_train_shard(
                            self.dataloader, epoch=ep, num_epochs=epochs,
                            batch_size=self.train_batch_size, seed=self.seed,
                            train_size=train_size,
                        )
                        expected_epoch_size = shard_sizes[ep - 1]
                    else:
                        epoch_batches = self.dataloader.plan_train_epoch(
                            epoch=ep, steps_per_epoch=rollout_batches_per_epoch,
                            accumulation=1, batch_size=self.train_batch_size,
                            seed=self.seed, out_root=str(self.out),
                        )
                    save_json(
                        epoch_dir / "batches.json",
                        [
                            {
                                "batch_in_epoch": i + 1,
                                "batch_size": _batch_size_for_logging(b, self.train_batch_size),
                                "case_ids": _batch_case_ids(b),
                                "train_schedule": (
                                    "disjoint_epoch_shards" if shard_sizes else "full_epoch"
                                ),
                                "shard_size": expected_epoch_size,
                            }
                            for i, b in enumerate(epoch_batches)
                        ],
                    )
                else:
                    epoch_batches = [None] * rollout_batches_per_epoch
                self.run_epoch(
                    ep, ep, batches=epoch_batches,
                    expected_train_size=expected_epoch_size,
                )

            # Gate is the formal graph boundary. Persist it before advisory
            # Meta so SIGTERM can resume Meta/test without replaying this epoch.
            self._checkpoint_gate_before_meta(
                ep,
                previous_selection_results=epoch_prev_sel,
                previous_graph_text=self._epoch_start_graph_text,
            )
            self._finish_pending_meta()

            current_record = (
                self.history[-1]
                if self.history and self.history[-1].get("epoch") == ep
                else None
            )
            if self.run_epoch_test:
                if current_record is None:
                    raise RuntimeError(f"epoch {ep} is missing its Gate record")
                epoch_test, _ = self._run_epoch_test_repeats(ep, self.graph)
                current_record["epoch_test"] = self._epoch_test_record(
                    ep, epoch_test
                )
            # Materialize the cumulative Gate decisions before snapshotting this epoch.
            with open(self.out / "gate_history.jsonl", "w", encoding="utf-8") as history_file:
                for record in self.history:
                    history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            write_epoch_versions_doc(self.out, self.history)
            archive_dir = archive_epoch_artifacts(
                self.out,
                epoch=ep,
                step=ep,
                decision=self.history[-1] if self.history else None,
            )
            self.epoch_archives.append(str(archive_dir))
            if self.history and self.history[-1].get("epoch") == ep:
                self.history[-1]["epoch_archive"] = str(archive_dir)
            self._save_trainer_state(ep)

        completed_epochs = max(
            [self.resume_completed_epochs]
            + [int(record.get("epoch") or 0) for record in self.history]
        )
        run_complete = completed_epochs >= epochs
        final: dict[str, Any] = {}
        if run_complete and bool(
            self.cfg.get("run_final_test") or self.cfg.get("eval_test")
        ):
            final, _ = self._run_final_test_repeats(self.best)
            self._run_component_ablation_suite()

        summary = {
            # final_test.json is the single aggregation source of truth.  The
            # identical values in summary.json keep strict postcheck and result
            # collection on one contract instead of two diverging write paths.
            **final,
            "experiment_mode": self.experiment_mode,
            "ablation_mode": self.ablation_mode,
            "update_strategy": self.update_strategy,
            "update_protocol": self.update_protocol,
            "badcase_analysis_protocol": self.badcase_analysis_protocol,
            "renderer_protocol": self.renderer_protocol,
            "persist_proposal_pool": self.evolution_cfg.persist_proposal_pool,
            "update_protocol_semantics_version": (
                UPDATE_PROTOCOL_SEMANTICS_VERSION
                if self.update_protocol == "case_complete_v1" else None
            ),
            "candidate_policy": "all_epoch_joint_groups",
            "initial_graph_sha256": self.initial_graph_file_sha256,
            "use_gate": self.use_big_gate,
            "use_small_gate": self.use_small_gate,
            "use_big_gate": self.use_big_gate,
            "use_train_tiebreak": self.use_train_tiebreak,
            "selective_gate": self.selective_gate,
            "use_teacher_veto": self.use_teacher_veto,
            "teacher_profile": self.cfg.get("teacher_profile"),
            "teacher_model_base": self.cfg.get("teacher_model_base"),
            "student_model_base": self.cfg.get("student_model_base"),
            "best_step": self.best_step,
            "best_epoch": self.best_step,
            "best_score": self.best_score,
            "current_score": self.score,
            "out_root": str(self.out),
            "best_graph": str(self.out / "best_graph.json"),
            "best_skill_md": str(self.out / "best_skill.md"),
            "accumulation": self.accumulation,
            "node_merge_top_k": self.evolution_cfg.node_merge_top_k,
            "teacher_analysis_policy": "failed cases receive same-group contrasts; touched nodes also receive all-success semantic guards",
            "analyst_workers": self.evolution_cfg.analyst_workers,
            "success_guard_workers": self.evolution_cfg.success_guard_workers,
            "teacher_request_timeout_seconds": max(
                1, int(self.cfg.get("teacher_request_timeout") or 300)
            ),
            "positive_example_cap": self.evolution_cfg.positive_example_cap,
            "use_positive_context": self.evolution_cfg.use_positive_context,
            "failure_example_cap": self.evolution_cfg.failure_example_cap,
            "rightcase_context_policy": "all_current_epoch_correct_uses_for_touched_nodes_summarized_into_full_coverage_guards",
            "rightcase_persistent": False,
            "rightcase_teacher_curator": False,
            "max_execution_child_candidates_per_epoch": self.evolution_cfg.max_execution_child_candidates_per_epoch,
            "execution_merge_evidence_cap": self.evolution_cfg.execution_merge_evidence_cap,
            "max_active_execution_children_per_parent": self.evolution_cfg.max_active_execution_children_per_parent,
            "ablation_random_sample_rate": self.ablation_random_sample_rate,
            "max_ops_semantics": "legacy CLI alias for node_merge_top_k; no patch-wide cap",
            "epoch_archives": self.epoch_archives,
            "epoch_versions_doc": str(self.out / "epoch_versions.md"),
            "train_size": train_size,
            "update_validation_size": 0 if no_validation else len(self._last_sel_results),
            "update_pool_size": (
                train_size
                if no_validation
                else train_size + len(self._last_sel_results)
            ),
            "update_pool_policy": (
                "single_complete_update_pool"
                if no_validation else "full_train_plus_update_validation"
            ),
            "big_gate_decision_split": (
                "complete_update_pool"
                if no_validation else "train_plus_validation"
            ),
            "test_used_for_gate": False,
            "test_used_for_graph_selection": False,
            "train_schedule": self.cfg.get("_active_train_schedule"),
            "grouped_batch_gate": grouped_active,
            "train_per_validation": (
                None if no_validation else (3 if grouped_active else None)
            ),
            "batch_size": None if single_full_pool else self.batch_size,
            "batch_size_scope": (
                None if single_full_pool else self.batch_size_scope
            ),
            "train_batch_size": (
                None if single_full_pool else self.train_batch_size
            ),
            "validation_batch_size": (
                None
                if no_validation or single_full_pool or not grouped_active
                else self.validation_batch_size
            ),
            "train_shard_sizes": [] if grouped_active else shard_sizes,
            "epoch_train_sizes": shard_sizes if grouped_active else [],
            "completed_train_shard_sizes": (
                shard_sizes[:completed_epochs] if shard_sizes and not grouped_active else []
            ),
            "completed_epoch_train_sizes": (
                shard_sizes[:completed_epochs] if grouped_active else []
            ),
            "completed_epochs": completed_epochs,
            "run_complete": run_complete,
            "stop_after_epoch": stop_after_epoch,
            "total_training_rollout_cases": sum(
                int(record.get("n_rollout_cases", record.get("n_cases") or 0))
                for record in self.history
            ),
            "total_training_evidence_cases": sum(
                int(record.get("n_evidence_cases", record.get("n_cases") or 0))
                for record in self.history
            ),
            "total_excluded_training_cases": sum(
                int(record.get("n_excluded_cases") or 0) for record in self.history
            ),
            "steps_per_epoch": rollout_batches_per_epoch if grouped_active else 1,
            "rollout_batches_per_epoch": rollout_batches_per_epoch,
            "updates_per_epoch": 1,
            "small_gates_per_epoch": 0,
            "local_joint_gates_per_epoch": (
                "proposal_dependent" if grouped_active else 0
            ),
            "local_gate_mandatory_for_grouped_graphopt": grouped_active,
            "big_gates_per_epoch": 1 if self.use_big_gate else 0,
            "candidate_gates_per_epoch": (
                "proposal_dependent_plus_one_big_gate"
                if grouped_active else (1 if self.use_big_gate else 0)
            ),
            "max_big_gate_passes": self.max_big_gate_passes,
            "num_epochs": epochs,
            "update_boundary": (
                "full_train_plus_update_validation_synthesis_then_joint_affected_scope_local_gates_then_full_train_validation_big_gate"
                if grouped_active else "epoch_shard" if shard_sizes else "full_epoch"
            ),
            "run_epoch_test": self.run_epoch_test,
            "epoch_test_repeats": self.epoch_test_repeats,
            "epoch_test_role": (
                f"every_committed_epoch_{self.epoch_test_repeats}_repeats_diagnostic_only"
                if self.run_epoch_test else "disabled"
            ),
            "run_component_ablation_tests": self.run_component_ablation_tests,
            "component_ablation_test_repeats": (
                1 if self.run_component_ablation_tests else 0
            ),
            "component_ablation_summary": (
                str(self.out / "component_ablation" / "summary.json")
                if self.run_component_ablation_tests else None
            ),
            "component_ablation_role": (
                "diagnostic_only_not_used_by_gate_or_optimizer"
                if self.run_component_ablation_tests else "disabled"
            ),
            "final_test_repeats": self.final_test_repeats,
            "final_test_aggregation": (
                "single_terminal_graph_evaluation"
                if self.final_test_repeats == 1
                else "min_max_mean_plus_half_range"
            ),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(self.out / "gate_history.jsonl", "w", encoding="utf-8") as f:
            for h in self.history:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
        return summary
