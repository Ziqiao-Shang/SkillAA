"""Graph usage extraction and deterministic edit touch-key helpers.

This module contains no Gate acceptance or rollback policy. Small-Gate scope
measurement lives in ``graphopt.evaluation.edit_gate``; complete-candidate Big-Gate
admission is owned by each environment's ``pipeline.py``.
"""

from __future__ import annotations

import re
from typing import Any

from graphopt.optimizer.skill import apply_edit
from graphopt.types import GraphEdit, SkillGraph, normalize_edge_type

_HARD_OK = 1e-9

_RE_ACTIVE_NODES = re.compile(r"Active nodes=\[([^\]]*)\]", re.IGNORECASE)
_RE_ACTIVE_EDGES = re.compile(r"Active edges=\[([^\]]*)\]", re.IGNORECASE)
_RE_WRONG_NODES = re.compile(r"Nodes wrong:\s*\[([^\]]*)\]", re.IGNORECASE)
_RE_WRONG_EDGES = re.compile(r"Edges wrong:\s*\[([^\]]*)\]", re.IGNORECASE)
_RE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _hard_ok(hard: float) -> bool:
    return float(hard or 0) >= _HARD_OK


def _parse_id_list(blob: str) -> list[str]:
    return _RE_ID.findall(blob or "")


def extract_graph_refs(result: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Parse node/edge ids referenced in rollout trajectory / readout."""
    refs = result.get("graph_refs")
    if "graph_refs" in result and isinstance(refs, dict):
        node_fields = ("used_nodes", "correct_nodes", "wrong_nodes", "missed_relevant_nodes", "execution_lapse_nodes")
        edge_fields = ("used_edges", "correct_edges", "wrong_edges", "missed_relevant_edges", "execution_lapse_edges")
        nodes = {str(x) for key in node_fields for x in (refs.get(key) or [])}
        edges = {str(x) for key in edge_fields for x in (refs.get(key) or [])}
        return nodes, edges
    text = " ".join(
        [
            str(result.get("trajectory") or ""),
            str(result.get("fail_reason") or ""),
        ]
    )
    nodes: set[str] = set()
    edges: set[str] = set()
    for pat in (_RE_ACTIVE_NODES, _RE_WRONG_NODES):
        for m in pat.finditer(text):
            nodes.update(_parse_id_list(m.group(1)))
    for pat in (_RE_ACTIVE_EDGES, _RE_WRONG_EDGES):
        for m in pat.finditer(text):
            edges.update(_parse_id_list(m.group(1)))
    # Fallback: any token that looks like Gxx / Exxx / Hxx / Txx
    for tok in _RE_ID.findall(text):
        if tok[0] in "GHEXT" and any(c.isdigit() for c in tok):
            if tok.startswith("E"):
                edges.add(tok)
            else:
                nodes.add(tok)
    return nodes, edges


def _validated_graph_refs(
    result: dict[str, Any], graph: SkillGraph | None
) -> tuple[set[str], set[str]]:
    """Reject an edge citation unless its two skill endpoints were active."""
    nodes, edges = extract_graph_refs(result)
    if graph is None:
        return nodes, edges
    edge_map = {edge.id: edge for edge in graph.edges if edge.id}
    return nodes, {
        edge_id for edge_id in edges
        if edge_id in edge_map
        and edge_map[edge_id].src in nodes
        and edge_map[edge_id].dst in nodes
    }


def build_flip_pairs(
    prev_results: list[dict[str, Any]],
    curr_results: list[dict[str, Any]],
    *,
    metric: str = "mixed",
    mixed_weight: float = 0.8,
    eps: float = 1e-12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (improved, regressed) cases using the configured gate metric."""
    prev = {str(r.get("id")): r for r in prev_results if r.get("id") is not None}
    curr = {str(r.get("id")): r for r in curr_results if r.get("id") is not None}
    prev_ids = set(prev)
    curr_ids = set(curr)
    if len(prev) != len(prev_results) or len(curr) != len(curr_results):
        raise ValueError("validation results contain missing or duplicate case ids")
    if prev_ids != curr_ids:
        missing = sorted(prev_ids - curr_ids)
        extra = sorted(curr_ids - prev_ids)
        raise ValueError(
            f"validation case ids are not aligned; missing_in_candidate={missing}, "
            f"extra_in_candidate={extra}"
        )

    def _value(result: dict[str, Any]) -> float:
        hard = float(result.get("hard") or 0)
        soft = hard if result.get("soft") is None else float(result["soft"])
        if metric == "soft":
            return soft
        if metric == "mixed":
            w = max(0.0, min(1.0, float(mixed_weight)))
            return (1.0 - w) * hard + w * soft
        return hard

    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    for rid in sorted(prev_ids):
        ph = float(prev[rid].get("hard") or 0)
        ch = float(curr[rid].get("hard") or 0)
        if metric == "hard" and not _hard_ok(ph) and _hard_ok(ch):
            improved.append({"id": rid, "prev": prev[rid], "curr": curr[rid]})
        elif metric == "hard" and _hard_ok(ph) and not _hard_ok(ch):
            regressed.append({"id": rid, "prev": prev[rid], "curr": curr[rid]})
        elif metric != "hard":
            delta = _value(curr[rid]) - _value(prev[rid])
            if delta > eps:
                improved.append({"id": rid, "prev": prev[rid], "curr": curr[rid]})
            elif delta < -eps:
                regressed.append({"id": rid, "prev": prev[rid], "curr": curr[rid]})
    return improved, regressed


def _touch_keys_for_edit(graph: SkillGraph, edit: GraphEdit) -> set[str]:
    op = edit.op
    keys: set[str] = set()
    if op in {"update_node", "delete_node"} and edit.node_id:
        keys.add(f"node:{edit.node_id}")
    elif op == "add_node":
        before = set(graph.nodes)
        g = graph.copy()
        apply_edit(g, edit)
        for nid in set(g.nodes) - before:
            keys.add(f"node:{nid}")
    elif op in {"add_edge", "delete_edge", "change_edge_type"}:
        et = normalize_edge_type(edit.edge_type or "prereq")
        if edit.src and edit.dst:
            # An edge edit may change when either endpoint is activated even
            # when the old edge ID was not cited. Include both endpoint users
            # so their successful behavior becomes Local-Gate counter-evidence.
            keys.update((f"node:{edit.src}", f"node:{edit.dst}"))
            keys.add(f"edge:{edit.src}->{edit.dst}:{et}")
            for edge in graph.edges:
                if (
                    edge.src == edit.src
                    and edge.dst == edit.dst
                    and normalize_edge_type(edge.type) == et
                    and edge.id
                ):
                    keys.add(f"edge_id:{edge.id}")
    return keys


def simulate_edit_touch_keys(base: SkillGraph, edits: list[GraphEdit]) -> list[set[str]]:
    """Keys touched by each edit when applied sequentially on *base*.

    For additions, include the materialized node/edge IDs as well as the
    requested relation. This lets rollout sidecars cite the exact new object.
    """
    graph = base.copy()
    out: list[set[str]] = []
    for edit in edits:
        before_nodes = set(graph.nodes)
        before_edges = {edge.id for edge in graph.edges if edge.id}
        keys = _touch_keys_for_edit(graph, edit)
        apply_edit(graph, edit)
        keys.update(f"node:{node_id}" for node_id in set(graph.nodes) - before_nodes)
        keys.update(
            f"edge_id:{edge_id}"
            for edge_id in {edge.id for edge in graph.edges if edge.id} - before_edges
        )
        out.append(keys)
    return out


def format_harmful_hints_for_prompt(hints: list[str]) -> str:
    """Return prompt block only when hints exist; otherwise empty string."""
    clean = [h.strip() for h in hints if h and h.strip()]
    if not clean:
        return ""
    lines = [
        "## Rejected graph edits (one-shot — do not repeat)",
        *[f"- {h}" for h in clean],
        "",
    ]
    return "\n".join(lines)
