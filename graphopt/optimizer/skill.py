"""Load / save / apply SkillGraph edits on the evolvable rule graph.

The immutable Agent protocol is a separate Markdown template and therefore is
never part of the edit space. Optimizer experience lives in ``ExperienceLedger``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from graphopt.types import (
    Edge,
    GraphEdit,
    GraphPatch,
    SkillGraph,
    SkillNode,
    float_to_weight,
    normalize_edge_type,
)

TOP_LEVEL_KEYS = frozenset({"stable_rule_graph"})
RULE_GRAPH_KEYS = frozenset({"frozen", "nodes", "edges", "common_mistakes"})
NODE_KEYS = frozenset(
    {
        "id", "title", "section", "category", "node_type", "level",
        "when_to_use", "how_to_use", "avoid", "stats", "active", "frozen",
    }
)
NODE_STATS_KEYS = frozenset({"n_use", "n_succ", "p_hat"})
EDGE_KEYS = frozenset(
    {"id", "source", "target", "type", "weight", "rationale", "active", "frozen"}
)
EDGE_TYPES = frozenset({"prereq", "enhance", "co_occur"})
EDGE_WEIGHTS = frozenset({"medium", "strong"})
NODE_SECTIONS = frozenset(
    {"task_types", "general_principles", "search_control", "specialized_recovery"}
)
NODE_CATEGORIES = frozenset(
    {"task", "general", "search_control", "recovery", "learned"}
)
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_graph_json_shape(data: dict[str, Any]) -> None:
    """Require one fixed persisted node schema without a ``meaning`` key."""
    if set(data) != TOP_LEVEL_KEYS:
        raise ValueError(
            f"SkillGraph top-level keys must be {sorted(TOP_LEVEL_KEYS)}, got {sorted(data)}"
        )
    graph = data.get("stable_rule_graph")
    if not isinstance(graph, dict) or set(graph) != RULE_GRAPH_KEYS:
        got = sorted(graph) if isinstance(graph, dict) else type(graph).__name__
        raise ValueError(f"stable_rule_graph keys must be {sorted(RULE_GRAPH_KEYS)}, got {got}")
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("stable_rule_graph.nodes and edges must be lists")
    if not isinstance(graph.get("common_mistakes"), list):
        raise ValueError("stable_rule_graph.common_mistakes must be a list")
    if not isinstance(graph.get("frozen"), bool):
        raise ValueError("stable_rule_graph.frozen must be a boolean")
    if any(not isinstance(item, str) for item in graph["common_mistakes"]):
        raise ValueError("stable_rule_graph.common_mistakes must contain only strings")
    node_ids: list[str] = []
    node_active: dict[str, bool] = {}
    for i, node in enumerate(graph["nodes"]):
        if not isinstance(node, dict) or set(node) != NODE_KEYS:
            got = sorted(node) if isinstance(node, dict) else type(node).__name__
            raise ValueError(f"node[{i}] keys must be {sorted(NODE_KEYS)}, got {got}")
        stats = node.get("stats")
        if not isinstance(stats, dict) or set(stats) != NODE_STATS_KEYS:
            got = sorted(stats) if isinstance(stats, dict) else type(stats).__name__
            raise ValueError(f"node[{i}].stats keys must be {sorted(NODE_STATS_KEYS)}, got {got}")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not ID_PATTERN.fullmatch(node_id):
            raise ValueError(f"node[{i}].id is invalid: {node_id!r}")
        node_ids.append(node_id)
        for key in ("title", "section", "category", "node_type"):
            if not isinstance(node.get(key), str) or not node[key].strip():
                raise ValueError(f"node[{i}].{key} must be a non-empty string")
        for key in ("when_to_use", "how_to_use"):
            if not isinstance(node[key], str):
                raise ValueError(f"node[{i}].{key} must be a string")
        if node["section"] not in NODE_SECTIONS:
            raise ValueError(f"node[{i}].section must be one of {sorted(NODE_SECTIONS)}")
        if node["category"] not in NODE_CATEGORIES:
            raise ValueError(f"node[{i}].category must be one of {sorted(NODE_CATEGORIES)}")
        if node["node_type"] != "rule":
            raise ValueError(f"node[{i}].node_type must be 'rule'")
        if isinstance(node.get("level"), bool) or not isinstance(node.get("level"), int) or node["level"] < 0:
            raise ValueError(f"node[{i}].level must be a non-negative integer")
        if not isinstance(node["avoid"], list) or any(
            not isinstance(item, str) for item in node["avoid"]
        ):
            raise ValueError(f"node[{i}].avoid must be a string list")
        if not any(node[key].strip() for key in ("when_to_use", "how_to_use")) \
                and not any(str(item).strip() for item in node["avoid"]):
            raise ValueError(f"node[{i}] must contain at least one non-empty semantic field")
        for key in ("active", "frozen"):
            if not isinstance(node.get(key), bool):
                raise ValueError(f"node[{i}].{key} must be a boolean")
        for key in ("n_use", "n_succ"):
            value = stats[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"node[{i}].stats.{key} must be a non-negative integer")
        if stats["n_succ"] > stats["n_use"]:
            raise ValueError(f"node[{i}].stats.n_succ cannot exceed n_use")
        p_hat = stats["p_hat"]
        if p_hat is not None and (
            isinstance(p_hat, bool)
            or not isinstance(p_hat, (int, float))
            or not 0.0 <= float(p_hat) <= 1.0
        ):
            raise ValueError(f"node[{i}].stats.p_hat must be null or in [0, 1]")
        node_active[str(node_id)] = node["active"]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("node ids must be unique")

    edge_ids: list[str] = []
    edge_triples: set[tuple[str, str, str]] = set()
    structural_pairs: dict[tuple[str, str], str] = {}
    valid_nodes = set(node_ids)
    for i, edge in enumerate(graph["edges"]):
        if not isinstance(edge, dict) or set(edge) != EDGE_KEYS:
            got = sorted(edge) if isinstance(edge, dict) else type(edge).__name__
            raise ValueError(f"edge[{i}] keys must be {sorted(EDGE_KEYS)}, got {got}")
        edge_id = edge.get("id")
        if not isinstance(edge_id, str) or not ID_PATTERN.fullmatch(edge_id):
            raise ValueError(f"edge[{i}].id is invalid: {edge_id!r}")
        edge_ids.append(edge_id)
        source, target = edge.get("source"), edge.get("target")
        if source not in valid_nodes or target not in valid_nodes:
            raise ValueError(f"edge[{i}] has missing endpoints: {source!r}->{target!r}")
        if source == target:
            raise ValueError(f"edge[{i}] self-loop is not allowed: {source!r}")
        if edge.get("type") not in EDGE_TYPES:
            raise ValueError(f"edge[{i}].type must be one of {sorted(EDGE_TYPES)}")
        if edge.get("weight") not in EDGE_WEIGHTS:
            raise ValueError(f"edge[{i}].weight must be one of {sorted(EDGE_WEIGHTS)}")
        if not isinstance(edge.get("rationale"), str):
            raise ValueError(f"edge[{i}].rationale must be a string")
        for key in ("active", "frozen"):
            if not isinstance(edge.get(key), bool):
                raise ValueError(f"edge[{i}].{key} must be a boolean")
        if edge["active"] and (not node_active[str(source)] or not node_active[str(target)]):
            raise ValueError(
                f"active edge[{i}] cannot reference an inactive endpoint: {source}->{target}"
            )
        triple = (str(source), str(target), str(edge.get("type")))
        if triple in edge_triples:
            raise ValueError(f"duplicate edge relation is not allowed: {triple}")
        edge_triples.add(triple)
        if edge.get("type") in {"prereq", "enhance"}:
            pair = (str(source), str(target))
            previous = structural_pairs.get(pair)
            if previous and previous != edge.get("type"):
                raise ValueError(
                    f"one directed node pair cannot be both {previous} and "
                    f"{edge.get('type')}: {source}->{target}"
                )
            structural_pairs[pair] = str(edge.get("type"))
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("edge ids must be unique")

    # Only prerequisite edges impose order, so only they must form a DAG.
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in graph["edges"]:
        if edge.get("type") != "prereq":
            continue
        adjacency[str(edge["source"])].append(str(edge["target"]))
        indegree[str(edge["target"])] += 1
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    seen = 0
    while queue:
        current = queue.pop()
        seen += 1
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if seen != len(node_ids):
        raise ValueError("prerequisite edges must be acyclic")


def load_graph(path: str | Path) -> SkillGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_graph_json_shape(data)
    g = SkillGraph.from_dict(data)
    recompute_levels(g)
    return g


def save_graph(graph: SkillGraph, path: str | Path) -> None:
    recompute_levels(graph)
    data = graph.to_dict()
    validate_graph_json_shape(data)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return s[:48] or "skill"


def recompute_levels(graph: SkillGraph) -> None:
    preds: dict[str, set[str]] = {nid: set() for nid in graph.nodes}
    for e in graph.edges:
        typ = normalize_edge_type(e.type)
        # Only a true prerequisite imposes execution order. Enhancement is an
        # optional composition/retrieval cue and must not inflate topo levels.
        if typ == "prereq" and e.src in preds and e.dst in preds:
            preds[e.dst].add(e.src)
    memo: dict[str, int] = {}

    def level(nid: str, stack: set[str]) -> int:
        if nid in memo:
            return memo[nid]
        if nid in stack:
            return 0
        stack.add(nid)
        memo[nid] = 0 if not preds[nid] else 1 + max(level(p, stack) for p in preds[nid])
        stack.remove(nid)
        return memo[nid]

    for nid in graph.nodes:
        graph.nodes[nid].level = level(nid, set())


def _next_rule_id(graph: SkillGraph, prefix: str = "X") -> str:
    nums = []
    for nid in graph.nodes:
        if nid.startswith(prefix):
            m = re.search(r"(\d+)$", nid)
            if m:
                nums.append(int(m.group(1)))
    return f"{prefix}{(max(nums) + 1) if nums else 1:02d}"


def apply_edit(graph: SkillGraph, edit: GraphEdit) -> list[str]:
    edit.sanitize_model_visible_fields()
    op = edit.op

    if op == "add_node":
        nid = edit.node_id or _slug(edit.name or edit.how_to_use or edit.description)
        if nid in graph.nodes or not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", nid):
            nid = _next_rule_id(graph, "X")
        base, k = nid, 2
        while nid in graph.nodes:
            nid = f"{base}_{k}"
            k += 1
        if not (edit.name or edit.description or edit.how_to_use or edit.avoid):
            return ["add_node skipped: empty"]
        node = SkillNode(
            id=nid,
            title=edit.name or nid,
            meaning="",
            when_to_use=edit.when_to_use or "When the matching failure pattern appears.",
            how_to_use=edit.how_to_use or edit.description or edit.name or nid,
            avoid=list(edit.avoid or []),
            category=edit.category or "general",
            section=(
                graph.nodes[edit.parent_node].section
                if edit.parent_node in graph.nodes
                else "general_principles"
            ),
            node_type="rule",
            graph_name="stable_rule_graph",
            frozen=False,
            active=True,
        )
        if edit.description and not edit.how_to_use:
            node.description = edit.description
        graph.nodes[nid] = node
        return []

    if op == "delete_node":
        nid = edit.node_id
        if nid not in graph.nodes:
            return [f"delete_node missing {nid}"]
        del graph.nodes[nid]
        graph.edges = [e for e in graph.edges if e.src != nid and e.dst != nid]
        return []

    if op == "update_node":
        nid = edit.node_id
        if nid not in graph.nodes:
            return [f"update_node missing {nid}"]
        if not any(
            (
                edit.when_to_use,
                edit.how_to_use,
                edit.description,
                edit.category,
                edit.avoid is not None,
            )
        ):
            return [f"update_node empty {nid}"]
        n = graph.nodes[nid]
        # Title is the stable human-readable identity paired with node_id.
        # Rewriters may change only the substantive procedural fields.
        if edit.when_to_use:
            n.when_to_use = edit.when_to_use
        if edit.how_to_use:
            n.how_to_use = edit.how_to_use
        if edit.avoid is not None:
            n.avoid = list(edit.avoid)
        if edit.description:
            n.description = edit.description
        if edit.category:
            n.category = edit.category
        n.frozen = False
        return []

    if op == "merge_nodes":
        return ["merge_nodes is disabled: GraphOpt merges opinions, never graph nodes"]

    if op == "add_edge":
        et = normalize_edge_type(edit.edge_type or "prereq")
        if not 0.0 <= float(edit.w) <= 1.0:
            return [f"add_edge weight must be in [0,1], got {edit.w}"]
        if et == "co_occur":
            # Optimizers cannot create co-occurrence structure, but the
            # statistics-owned weight pass may refresh an existing edge.
            for e in graph.edges:
                if e.src == edit.src and e.dst == edit.dst and normalize_edge_type(e.type) == et:
                    e.w = edit.w
                    e.weight = float_to_weight(e.w)
                    e.frozen = False
                    return []
            return ["add_edge: co_occur is statistics-owned"]
        if et not in {"prereq", "enhance"}:
            return [f"add_edge bad type {et}"]
        if edit.src not in graph.nodes or edit.dst not in graph.nodes:
            return [f"add_edge missing endpoints {edit.src}->{edit.dst}"]
        if edit.src == edit.dst:
            return [f"add_edge self-loop {edit.src}->{edit.dst}"]
        for e in graph.edges:
            if e.src == edit.src and e.dst == edit.dst and normalize_edge_type(e.type) == et:
                # This operation also serves as the explicit weight refresh
                # for an existing relation. The new epoch may provide weaker
                # evidence, so the weight must be allowed to decrease.
                e.w = edit.w
                e.weight = float_to_weight(e.w)
                if edit.rationale:
                    e.rationale = edit.rationale
                e.frozen = False
                return []
            if (
                e.src == edit.src and e.dst == edit.dst
                and normalize_edge_type(e.type) in {"prereq", "enhance"}
                and normalize_edge_type(e.type) != et
            ):
                return [
                    f"add_edge conflicting structural relation on "
                    f"{edit.src}->{edit.dst}: existing={normalize_edge_type(e.type)}, new={et}"
                ]
        graph.edges.append(
            Edge(
                src=edit.src,
                dst=edit.dst,
                type=et,
                w=edit.w,
                weight=float_to_weight(edit.w),
                rationale=edit.rationale or edit.reasoning,
                graph_name="stable_rule_graph",
                frozen=False,
                active=True,
            )
        )
        return []

    if op == "delete_edge":
        et = normalize_edge_type(edit.edge_type or "prereq")
        if et == "co_occur":
            return ["delete_edge: co_occur topology is statistics-owned"]
        before = len(graph.edges)
        graph.edges = [
            e
            for e in graph.edges
            if not (e.src == edit.src and e.dst == edit.dst and normalize_edge_type(e.type) == et)
        ]
        return [] if len(graph.edges) < before else ["delete_edge not found"]

    if op == "change_edge_type":
        old_t = normalize_edge_type(edit.edge_type or "prereq")
        new_t = normalize_edge_type(edit.new_edge_type)
        if old_t == "co_occur" or new_t == "co_occur":
            return ["change_edge_type: co_occur is statistics-owned"]
        if new_t not in {"prereq", "enhance"}:
            return [f"change_edge_type bad {new_t}"]
        for e in graph.edges:
            if e.src == edit.src and e.dst == edit.dst and normalize_edge_type(e.type) == old_t:
                if any(
                    other is not e
                    and other.src == edit.src and other.dst == edit.dst
                    and normalize_edge_type(other.type) == new_t
                    for other in graph.edges
                ):
                    return ["change_edge_type would duplicate an existing relation"]
                e.type = new_t
                if edit.rationale:
                    e.rationale = edit.rationale
                e.frozen = False
                return []
        return ["change_edge_type not found"]

    return [f"unknown op {op}"]


def _apply_patch_nonatomic(graph: SkillGraph, patch: GraphPatch | dict[str, Any]) -> dict[str, Any]:
    if isinstance(patch, dict):
        patch = GraphPatch.from_dict(patch)
    warnings: list[str] = []
    applied_indices: list[int] = []
    failed_indices: list[int] = []
    edit_reports: list[dict[str, Any]] = []
    for index, e in enumerate(patch.edits):
        before = graph.copy()
        edit_warnings = apply_edit(graph, e)
        if not edit_warnings:
            try:
                graph.sync_data_from_views()
                validate_graph_json_shape(graph.to_dict())
            except ValueError as exc:
                graph.data = before.data
                graph.nodes = before.nodes
                graph.edges = before.edges
                graph.active_max_level = before.active_max_level
                graph.meta = before.meta
                edit_warnings = [f"{e.op} would make graph invalid: {exc}"]
        warnings.extend(edit_warnings)
        if edit_warnings:
            failed_indices.append(index)
        else:
            applied_indices.append(index)
        edit_reports.append(
            {"edit_index": index, "applied": not edit_warnings, "warnings": edit_warnings}
        )
    recompute_levels(graph)
    graph.sync_data_from_views()
    return {
        "n_edits": len(patch.edits),
        "n_applied": len(applied_indices),
        "n_failed": len(failed_indices),
        "applied_indices": applied_indices,
        "failed_indices": failed_indices,
        "edit_reports": edit_reports,
        "warnings": warnings,
    }


def atomic_edit_groups(edits: list[GraphEdit]) -> list[tuple[str, list[int]]]:
    """Infer dependency groups while preserving proposal order."""
    added = {e.node_id for e in edits if e.op == "add_node" and e.node_id}
    groups: list[tuple[str, list[int]]] = []
    positions: dict[str, int] = {}
    for index, edit in enumerate(edits):
        if edit.group_id:
            key = f"explicit:{edit.group_id}"
        else:
            dependencies = sorted(
                node_id for node_id in added
                if edit.node_id == node_id or edit.src == node_id or edit.dst == node_id
            )
            key = f"new-node:{'+'.join(dependencies)}" if dependencies else f"edit:{index}"
        if key not in positions:
            positions[key] = len(groups)
            groups.append((key, []))
        groups[positions[key]][1].append(index)
    return groups


def apply_patch(graph: SkillGraph, patch: GraphPatch | dict[str, Any]) -> dict[str, Any]:
    """Apply each dependency group transactionally.

    A failed edit rolls back its whole group. Independent groups may still be
    applied, which lets Selective Gate reason about them separately.
    """
    if isinstance(patch, dict):
        patch = GraphPatch.from_dict(patch)
    applied: list[int] = []
    failed: list[int] = []
    warnings: list[str] = []
    edit_reports: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    for group_id, indices in atomic_edit_groups(patch.edits):
        trial = graph.copy()
        subset = GraphPatch(
            reasoning=f"atomic:{group_id}", edits=[patch.edits[i] for i in indices]
        )
        report = _apply_patch_nonatomic(trial, subset)
        group_warnings = list(report.get("warnings") or [])
        ok = not group_warnings and int(report.get("n_applied") or 0) == len(indices)
        if ok:
            graph.data = trial.data
            graph.nodes = trial.nodes
            graph.edges = trial.edges
            graph.active_max_level = trial.active_max_level
            graph.meta = trial.meta
            applied.extend(indices)
        else:
            failed.extend(indices)
            warnings.extend(f"{group_id}: {warning}" for warning in group_warnings)
        for index in indices:
            edit_reports.append({"edit_index": index, "group_id": group_id,
                "applied": ok, "warnings": group_warnings})
        group_reports.append({"group_id": group_id, "edit_indices": indices,
            "applied": ok, "warnings": group_warnings})
    recompute_levels(graph)
    graph.sync_data_from_views()
    return {"n_edits": len(patch.edits), "n_applied": len(applied),
        "n_failed": len(failed), "applied_indices": applied,
        "failed_indices": failed, "edit_reports": edit_reports,
        "group_reports": group_reports, "warnings": warnings}
