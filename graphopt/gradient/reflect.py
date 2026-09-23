"""Reflect: trajectories + experience ledger → discrete graph edits.

Minibatch dispatch mirrors ``graphskillaa.gradient.reflect.run_minibatch_reflect``:
fail/success split, shuffle, split by ``minibatch_size``, one optimizer call per group.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from graphopt.optimizer.experience_ledger import ExperienceLedger
from graphopt.types import GraphEdit, GraphPatch, SkillGraph, normalize_edge_type

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

_HARD_EPS = 1e-9


def _is_failure(r: dict[str, Any]) -> bool:
    """GraphSkillAA ``run_minibatch_reflect`` failure partition."""
    return not r.get("hard") or float(r.get("hard") or 0) < _HARD_EPS


def _is_success(r: dict[str, Any]) -> bool:
    return bool(r.get("hard")) and float(r.get("hard") or 0) >= _HARD_EPS


def _split_minibatches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(batch_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _shuffle_for_minibatch(items: list[dict[str, Any]], seed: int | None) -> list[dict[str, Any]]:
    ordered = list(items)
    if seed is None:
        return ordered
    random.Random(seed).shuffle(ordered)
    return ordered


def format_graph(graph: SkillGraph) -> str:
    """Structured view of the JSON rule graph for the optimizer (not Agent prompt)."""
    lines = [
        "## SkillGraph rule nodes (JSON state — edit these)",
        "Nodes are reusable procedural skills, not objects, actions, or case memories.",
        "Categories: task=end-to-end task recipe; general=cross-task procedure; "
        "search_control=state/action filter; recovery=narrow fallback; X=learned rule.",
        "Each node: id | category | prereq-level | title | meaning/when/how/avoid",
        "Permanent Protocol is immutable and not listed here.",
    ]
    for n in sorted(graph.nodes.values(), key=lambda x: (x.category, x.level, x.id)):
        desc = (
            f"Meaning: {n.meaning} When: {n.when_to_use} Do: {n.how_to_use} "
            f"Avoid: {'; '.join(n.avoid)}"
        ).replace("\n", " ").strip()
        lines.append(f"- `{n.id}` | {n.category} | L{n.level} | **{n.title}**: {desc}")
    lines.append("")
    lines.append("## SkillGraph edges (relations)")
    lines.append("prereq=necessary ordering; enhance=optional help without required order; "
                 "co_occur=symmetric retrieval hint (do not propose or mark wrong).")
    for e in graph.edges:
        typ = normalize_edge_type(e.type)
        lines.append(
            f"- `{e.id}` | `{e.src}` -[{typ} w={e.weight or e.w}]-> `{e.dst}` "
            f"| Why: {e.rationale or 'no rationale recorded'}"
        )
    return "\n".join(lines)


def format_rollouts(results: list[dict[str, Any]], max_items: int | None = None) -> str:
    """Format rollout dicts for one reflect minibatch (all items by default)."""
    ordered = sorted(results, key=lambda r: float(r.get("hard") or 0))
    if max_items is not None:
        ordered = ordered[:max_items]
    parts = []
    for r in ordered:
        parts.append(
            json.dumps(
                {
                    "id": r.get("id"),
                    "task_type": r.get("task_type"),
                    "task": (r.get("task_description") or "")[:280],
                    "hard": r.get("hard"),
                    "soft": r.get("soft"),
                    "fail_reason": r.get("fail_reason"),
                    "trajectory": str(r.get("trajectory") or "")[:1000],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return "\n\n".join(parts)


def filter_edits(patch: GraphPatch) -> GraphPatch:
    allowed = {
        "add_node",
        "delete_node",
        "update_node",
        "add_edge",
        "delete_edge",
        "change_edge_type",
    }
    kept = []
    for e in patch.edits:
        if e.op not in allowed:
            continue
        if e.is_edge_op and ("co_occur" in (e.edge_type, e.new_edge_type)):
            continue
        if e.edge_type:
            e.edge_type = normalize_edge_type(e.edge_type)
        if e.new_edge_type:
            e.new_edge_type = normalize_edge_type(e.new_edge_type)
        kept.append(e)
    return GraphPatch(edits=kept, reasoning=patch.reasoning)


def template_reflect(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    ledger: ExperienceLedger | None = None,
) -> GraphPatch:
    """Fallback when no optimizer API: turn open ledger notes into rule-graph edits."""
    del results
    ledger = ledger or ExperienceLedger()
    edits: list[GraphEdit] = []
    open_notes = sorted(
        (n for n in ledger.notes.values() if n.status == "open"),
        key=lambda n: (-n.n_fail, -n.n_seen),
    )
    for note in open_notes[:3]:
        target = next((nid for nid in note.related_node_ids if nid in graph.nodes), "")
        if target:
            edits.append(
                GraphEdit(
                    op="update_node",
                    node_id=target,
                    how_to_use=(
                        graph.nodes[target].how_to_use.rstrip()
                        + f" ALSO: {note.summary}"
                    ),
                    source_type="failure",
                    experience_note_id=note.id,
                    reasoning=f"ledger:{note.id}×{note.n_fail}",
                )
            )
        else:
            edits.append(
                GraphEdit(
                    op="add_node",
                    name=note.summary[:72],
                    meaning=note.summary,
                    when_to_use="When the matching failure pattern recurs.",
                    how_to_use=note.summary,
                    avoid=["Ignoring this recurring failure mode."],
                    category="general",
                    source_type="failure",
                    experience_note_id=note.id,
                    reasoning=f"ledger:{note.id}×{note.n_fail}",
                )
            )
    return GraphPatch(reasoning="ledger_template", edits=edits)


def reflect(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    *,
    chat_fn=None,
    meta_context: str = "",
    mode: str = "template",
    source_hint: str = "",
    ledger: ExperienceLedger | None = None,
) -> GraphPatch:
    """One minibatch group → GraphPatch (GraphSkillAA analyst-minibatch analogue)."""
    ledger = ledger or ExperienceLedger()
    if mode != "teacher" or chat_fn is None:
        return filter_edits(template_reflect(graph, results, ledger))

    system = (PROMPTS / "reflect.md").read_text(encoding="utf-8")
    hint = f"\nFocus: {source_hint}\n" if source_hint else ""
    user = (
        ((meta_context + "\n\n") if meta_context else "")
        + format_graph(graph)
        + "\n\n"
        + ledger.format_for_optimizer()
        + hint
        + "\n\n## Rollouts (failures first)\n"
        + format_rollouts(results)
        + "\n\nPropose discrete graph edits ONLY from the allowed edit space. No co_occur. "
        "Do not put experience ledger text into the skill document."
    )
    response, _ = chat_fn(
        system=system,
        user=user,
        max_completion_tokens=16384,
        retries=3,
        stage="graph_reflect",
    )
    try:
        from graphopt.json_utils import extract_json

        obj = extract_json(response) or {}
    except Exception:
        obj = {}
    return filter_edits(GraphPatch.from_dict(obj if isinstance(obj, dict) else {}))


def _run_reflect_group(
    graph: SkillGraph,
    batch: list[dict[str, Any]],
    *,
    kind: str,
    chat_fn,
    meta_context: str,
    mode: str,
    ledger: ExperienceLedger | None,
) -> GraphPatch:
    hint = (
        "failure trajectories"
        if kind == "failure"
        else "success trajectories — merge duplicates / reinforce useful edges"
    )
    patch = reflect(
        graph,
        batch,
        chat_fn=chat_fn,
        meta_context=meta_context,
        mode=mode,
        source_hint=hint,
        ledger=ledger,
    )
    for e in patch.edits:
        if not e.source_type:
            e.source_type = "failure" if kind == "failure" else "success"
    return patch


def reflect_minibatch(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    *,
    chat_fn=None,
    meta_context: str = "",
    mode: str = "template",
    failure_only: bool = False,
    ledger: ExperienceLedger | None = None,
    minibatch_size: int = 8,
    random_seed: int | None = None,
    patches_dir: str | Path | None = None,
    workers: int = 1,
) -> tuple[list[GraphPatch], list[GraphPatch]]:
    """GraphSkillAA ``run_minibatch_reflect`` analogue for SkillGraph."""
    failures = [r for r in results if _is_failure(r)]
    successes = [r for r in results if _is_success(r)] if not failure_only else []

    failures = _shuffle_for_minibatch(failures, random_seed)
    successes = _shuffle_for_minibatch(
        successes, None if random_seed is None else random_seed + 1
    )

    fail_batches = _split_minibatches(failures, minibatch_size)
    succ_batches = _split_minibatches(successes, minibatch_size)

    patches_path = Path(patches_dir) if patches_dir else None
    if patches_path is not None:
        patches_path.mkdir(parents=True, exist_ok=True)

    use_teacher = mode == "teacher" and chat_fn is not None

    if not use_teacher:
        fail_patches: list[GraphPatch] = []
        if failures:
            p = filter_edits(template_reflect(graph, failures, ledger))
            for e in p.edits:
                if not e.source_type:
                    e.source_type = "failure"
            fail_patches = [p]
            if patches_path is not None:
                (patches_path / "minibatch_fail_000.json").write_text(
                    json.dumps(p.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        return fail_patches, []

    fail_results: dict[int, GraphPatch] = {}
    succ_results: dict[int, GraphPatch] = {}
    pending: list[tuple[str, int, list[dict[str, Any]], str]] = []

    for idx, batch in enumerate(fail_batches):
        tag = f"minibatch_fail_{idx:03d}"
        path = patches_path / f"{tag}.json" if patches_path else None
        if path is not None and path.is_file():
            fail_results[idx] = GraphPatch.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        else:
            pending.append(("failure", idx, batch, tag))

    for idx, batch in enumerate(succ_batches):
        tag = f"minibatch_succ_{idx:03d}"
        path = patches_path / f"{tag}.json" if patches_path else None
        if path is not None and path.is_file():
            succ_results[idx] = GraphPatch.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        else:
            pending.append(("success", idx, batch, tag))

    results_by_key: dict[tuple[str, int], GraphPatch] = {}
    if pending:
        max_workers = max(1, int(workers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    _run_reflect_group,
                    graph,
                    batch,
                    kind=kind,
                    chat_fn=chat_fn,
                    meta_context=meta_context,
                    mode=mode,
                    ledger=ledger,
                ): (kind, idx, tag)
                for kind, idx, batch, tag in pending
            }
            for fut in as_completed(futs):
                kind, idx, tag = futs[fut]
                patch = fut.result()
                results_by_key[(kind, idx)] = patch
                if patches_path is not None:
                    (patches_path / f"{tag}.json").write_text(
                        json.dumps(patch.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    fail_patches = [
        fail_results.get(i) or results_by_key[("failure", i)] for i in range(len(fail_batches))
    ]
    succ_patches = [
        succ_results.get(i) or results_by_key[("success", i)] for i in range(len(succ_batches))
    ]

    return fail_patches, succ_patches
