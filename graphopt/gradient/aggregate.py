"""Aggregate: merge mini-batch graph patches (SkillAA Aggregate analogue).

Failure-driven patches take priority. Teacher mode uses hierarchical LLM merge;
fallback uses **consensus merge** for nodes.

Structural edges (prereq / enhance) use **global support merge**:
- count optimizer proposals across all patches in this step
- keep iff support >= ``edge_merge_min_support`` (default k=3)
- assign strength quintiles (20% buckets × 5 tiers) within each edge type
- existing edges with support < k are deleted; weak new proposals are dropped
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from graphopt.types import GraphEdit, GraphPatch, float_to_weight, normalize_edge_type

if TYPE_CHECKING:
    from graphopt.types import SkillGraph

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

_TOP_VARIANTS = 3
_STRUCTURAL_EDGE_TYPES = frozenset({"prereq", "enhance"})
# Top 20% → strong … bottom 20% → weak
_QUINTILE_W = (0.85, 0.70, 0.55, 0.40, 0.25)


def _key(e: GraphEdit) -> tuple:
    if e.is_node_op:
        return (e.op, e.node_id or e.name)
    return (e.op, e.src, e.dst, e.edge_type, e.new_edge_type)


def _as_patch(p: GraphPatch | dict) -> GraphPatch:
    return p if isinstance(p, GraphPatch) else GraphPatch.from_dict(p)


def _is_structural_edge_edit(e: GraphEdit) -> bool:
    if e.op == "add_edge":
        return normalize_edge_type(e.edge_type) in _STRUCTURAL_EDGE_TYPES
    if e.op in {"delete_edge", "change_edge_type"}:
        return normalize_edge_type(e.edge_type) in _STRUCTURAL_EDGE_TYPES
    return False


def _edge_triple(src: str, dst: str, edge_type: str) -> tuple[str, str, str]:
    return (src, dst, normalize_edge_type(edge_type))


def _existing_structural_edge_keys(graph: "SkillGraph | None") -> set[tuple[str, str, str]]:
    if graph is None:
        return set()
    keys: set[tuple[str, str, str]] = set()
    for e in graph.edges:
        typ = normalize_edge_type(e.type)
        if typ in _STRUCTURAL_EDGE_TYPES:
            keys.add((e.src, e.dst, typ))
    return keys


def _normalize_sig(text: str) -> str:
    return " ".join((text or "").lower().split())[:320]


def _update_node_signature(e: GraphEdit) -> str:
    parts: list[str] = []
    for name in ("how_to_use", "when_to_use", "meaning", "title"):
        v = getattr(e, name, "") or ""
        if v.strip():
            parts.append(f"{name}={_normalize_sig(v)}")
    if parts:
        return ";".join(parts)
    return f"reasoning={_normalize_sig(e.reasoning)}"


def _support_threshold(n_patches: int) -> int:
    if n_patches <= 2:
        return 1
    return 2


def _score_variant(edits: list[GraphEdit]) -> tuple[int, int, int]:
    n_fail = sum(1 for e in edits if e.source_type == "failure")
    return (n_fail, len(edits), len(edits) - n_fail)


def _pick_representative(edits: list[GraphEdit]) -> GraphEdit:
    ordered = sorted(
        edits,
        key=lambda e: (
            0 if e.source_type == "failure" else 1,
            -len(e.how_to_use or e.when_to_use or e.meaning or e.reasoning or ""),
        ),
    )
    return ordered[0]


def _quintile_w(rank_index: int, total: int) -> float:
    """rank_index 0 = highest support within an edge-type pool."""
    if total <= 0:
        return _QUINTILE_W[2]
    if total == 1:
        return _QUINTILE_W[0]
    bucket = min(4, (rank_index * 5) // total)
    return _QUINTILE_W[bucket]


def _merge_structural_edges_global(
    patches: list[GraphPatch],
    graph: "SkillGraph | None",
    *,
    min_support: int,
) -> list[GraphEdit]:
    """Global edge reconciliation for prereq / enhance (optimizer proposals + existing graph)."""
    k = max(1, int(min_support))
    proposals: dict[tuple[str, str, str], list[GraphEdit]] = defaultdict(list)
    for p in patches:
        seen_in_patch: set[tuple[str, str, str]] = set()
        for e in p.edits:
            if e.op == "add_edge" and normalize_edge_type(e.edge_type) in _STRUCTURAL_EDGE_TYPES:
                if e.src and e.dst:
                    key = _edge_triple(e.src, e.dst, e.edge_type)
                    if key not in seen_in_patch:
                        proposals[key].append(e)
                        seen_in_patch.add(key)

    existing = _existing_structural_edge_keys(graph)
    survivors: list[tuple[tuple[str, str, str], int, list[GraphEdit]]] = []
    for key in set(proposals):
        support = len(proposals.get(key, []))
        if support >= k and key not in existing:
            survivors.append((key, support, proposals.get(key, [])))

    out: list[GraphEdit] = []

    by_type: dict[str, list[tuple[tuple[str, str, str], int, list[GraphEdit]]]] = {
        "prereq": [],
        "enhance": [],
    }
    for key, support, eds in survivors:
        by_type[key[2]].append((key, support, eds))

    for etype in ("prereq", "enhance"):
        pool = sorted(by_type[etype], key=lambda x: (-x[1], x[0][0], x[0][1]))
        n = len(pool)
        for i, (key, support, eds) in enumerate(pool):
            src, dst, et = key
            rep = _pick_representative(eds) if eds else GraphEdit(
                op="add_edge", src=src, dst=dst, edge_type=et
            )
            if support >= 3 * k:
                w = 0.85
            elif support >= 2 * k:
                w = 0.80
            else:
                w = 0.60
            tier = float_to_weight(w)
            out.append(
                replace(
                    rep,
                    op="add_edge",
                    src=src,
                    dst=dst,
                    edge_type=et,
                    w=w,
                    source_type=rep.source_type or "success",
                    reasoning=(
                        f"edge_global:{etype} support={support} rank={i + 1}/{n} "
                        f"tier={tier} (k={k}) | {rep.reasoning}"
                    )[:500],
                )
            )
    return out


def _synthesize_update_node(
    node_id: str,
    ranked: list[tuple[str, list[GraphEdit]]],
    *,
    max_variants: int = _TOP_VARIANTS,
) -> GraphEdit:
    top = ranked[:max_variants]
    primary = _pick_representative(top[0][1])
    if len(top) == 1:
        return primary

    base = (primary.how_to_use or "").rstrip()
    seen = {_normalize_sig(base)} if base else set()
    extras: list[str] = []
    for _, eds in top[1:]:
        rep = _pick_representative(eds)
        ht = (rep.how_to_use or "").strip()
        if not ht:
            continue
        sig = _normalize_sig(ht)
        if sig in seen:
            continue
        seen.add(sig)
        extras.append(ht)

    how = base
    for ex in extras:
        how = f"{how} ALSO: {ex}" if how else ex

    n_support = sum(len(eds) for _, eds in top)
    return replace(
        primary,
        node_id=node_id,
        how_to_use=how,
        reasoning=(
            f"consensus:{node_id} support={n_support} variants={len(top)} "
            f"| {primary.reasoning}".strip(" |")
        )[:500],
    )


def _merge_update_nodes_by_consensus(
    edits: list[GraphEdit],
    *,
    n_patches: int,
) -> list[GraphEdit]:
    by_node: dict[str, list[GraphEdit]] = defaultdict(list)
    for e in edits:
        if e.op == "update_node" and e.node_id:
            by_node[e.node_id].append(e)

    min_support = _support_threshold(n_patches)
    merged: list[GraphEdit] = []
    for node_id, group in by_node.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        buckets: dict[str, list[GraphEdit]] = defaultdict(list)
        for e in group:
            buckets[_update_node_signature(e)].append(e)

        ranked = sorted(buckets.items(), key=lambda kv: _score_variant(kv[1]), reverse=True)
        kept: list[tuple[str, list[GraphEdit]]] = [ranked[0]]
        for item in ranked[1:_TOP_VARIANTS]:
            if len(item[1]) >= min_support:
                kept.append(item)

        merged.append(_synthesize_update_node(node_id, kept))
    return merged


def _merge_other_edits_by_consensus(
    edits: list[GraphEdit],
    *,
    n_patches: int,
) -> list[GraphEdit]:
    buckets: dict[tuple, list[GraphEdit]] = defaultdict(list)
    for e in edits:
        buckets[_key(e)].append(e)

    min_support = _support_threshold(n_patches)
    merged: list[GraphEdit] = []
    for key, group in buckets.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        ranked = sorted(group, key=lambda e: _score_variant([e]), reverse=True)
        winner = ranked[0]
        if len(ranked) > 1 and _score_variant([ranked[1]])[1] >= min_support:
            winner = replace(
                winner,
                reasoning=(
                    f"consensus:{key} support={len(group)} | {winner.reasoning}"
                ).strip(" |")[:500],
            )
        merged.append(winner)
    return merged


def _consensus_merge_non_edge(
    patches: list[GraphPatch],
    *,
    failure_first: bool = True,
) -> list[GraphEdit]:
    ordered = list(patches)
    if failure_first:
        ordered.sort(
            key=lambda p: 0 if any(e.source_type == "failure" for e in p.edits) else 1
        )

    node_updates: list[GraphEdit] = []
    other_edits: list[GraphEdit] = []
    for p in ordered:
        for e in p.edits:
            if _is_structural_edge_edit(e):
                continue
            if e.op == "update_node" and e.node_id:
                node_updates.append(e)
            else:
                other_edits.append(e)

    n_patches = len(ordered)
    merged = _merge_update_nodes_by_consensus(node_updates, n_patches=n_patches)
    merged.extend(_merge_other_edits_by_consensus(other_edits, n_patches=n_patches))
    merged.sort(key=lambda e: (0 if e.source_type == "failure" else 1, 0 if e.is_node_op else 1))
    return merged


def _consensus_merge(
    patches: list[GraphPatch],
    graph: "SkillGraph | None" = None,
    *,
    edge_merge_min_support: int = 3,
    failure_first: bool = True,
) -> GraphPatch:
    reasons: list[str] = []
    for p in patches:
        if p.reasoning:
            reasons.append(p.reasoning)

    edits = _consensus_merge_non_edge(patches, failure_first=failure_first)
    edits.extend(
        _merge_structural_edges_global(
            patches, graph, min_support=edge_merge_min_support
        )
    )
    edits.sort(key=lambda e: (0 if e.source_type == "failure" else 1, 0 if e.is_node_op else 1))
    return GraphPatch(edits=edits, reasoning=" | ".join(reasons)[:2000])


def _dedup(patches: list[GraphPatch], *, failure_first: bool = True) -> GraphPatch:
    return _consensus_merge(patches, graph=None, failure_first=failure_first)


def _llm_merge(
    graph_text: str,
    patches: list[GraphPatch],
    chat_fn,
    meta_context: str = "",
) -> GraphPatch | None:
    try:
        from graphopt.json_utils import extract_json
    except Exception:
        return None
    system = (PROMPTS / "merge.md").read_text(encoding="utf-8")
    payload = [p.to_dict() for p in patches]
    user = (
        (meta_context + "\n\n" if meta_context else "")
        + f"## Current Skill Graph\n{graph_text}\n\n"
        + f"## Patches to merge ({len(patches)})\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    try:
        resp, _ = chat_fn(
            system=system,
            user=user,
            max_completion_tokens=8192,
            retries=2,
            stage="graph_merge",
        )
        obj = extract_json(resp) or {}
        if isinstance(obj, dict) and (obj.get("edits") or obj.get("graph_edits")):
            return GraphPatch.from_dict(obj)
    except Exception:
        return None
    return None


def _finalize_merged_patch(
    patches: list[GraphPatch],
    edits: list[GraphEdit],
    graph: "SkillGraph | None",
    edge_merge_min_support: int,
) -> GraphPatch:
    """Strip structural edge ops from LLM/consensus output; apply global edge policy."""
    non_edge = [e for e in edits if not _is_structural_edge_edit(e)]
    edge_edits = _merge_structural_edges_global(
        patches, graph, min_support=edge_merge_min_support
    )
    merged = non_edge + edge_edits
    merged.sort(key=lambda e: (0 if e.source_type == "failure" else 1, 0 if e.is_node_op else 1))
    reasons = [p.reasoning for p in patches if p.reasoning]
    return GraphPatch(edits=merged, reasoning=" | ".join(reasons)[:2000])


def merge_patches(
    patches: list[GraphPatch | dict],
    *,
    graph_text: str = "",
    graph: "SkillGraph | None" = None,
    chat_fn=None,
    meta_context: str = "",
    batch_size: int = 4,
    edge_merge_min_support: int = 3,
) -> GraphPatch:
    """Merge patches from multiple mini-batches into one GraphPatch."""
    clean = [_as_patch(p) for p in patches if p]
    clean = [p for p in clean if p.edits]
    if not clean:
        return GraphPatch(reasoning="no patches", edits=[])

    k = max(1, int(edge_merge_min_support))

    if len(clean) == 1:
        return _consensus_merge(clean, graph, edge_merge_min_support=k)

    if chat_fn is not None and graph_text:
        level = list(clean)
        while len(level) > 1:
            nxt: list[GraphPatch] = []
            for i in range(0, len(level), max(1, batch_size)):
                group = level[i : i + batch_size]
                if len(group) == 1:
                    nxt.append(group[0])
                    continue
                merged = _llm_merge(graph_text, group, chat_fn, meta_context=meta_context)
                if merged is not None:
                    nxt.append(
                        _finalize_merged_patch(group, merged.edits, graph, k)
                    )
                else:
                    nxt.append(_consensus_merge(group, graph, edge_merge_min_support=k))
            level = nxt
        return level[0]

    return _consensus_merge(clean, graph, edge_merge_min_support=k)
