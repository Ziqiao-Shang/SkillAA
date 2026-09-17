"""Statistical co-occurrence + node use/success counters on the rule graph.

Co-occurrence is computed on **successful** trajectories only.

For nodes A and B, let ``n_both`` be how many success trajectories mention both,
and ``n_succ(A)`` / ``n_succ(B)`` how many success trajectories mention each alone
or together. The co-occurrence rate is:

    rate(A, B) = n_both / min(n_succ(A), n_succ(B))

Rates at or above 70% map to strong; lower admitted rates map to medium.
Pairs below 25% or with fewer than ``min_co_both`` joint hits do not get an edge.
"""

from __future__ import annotations

from typing import Any

from graphopt.types import Edge, SkillGraph, float_to_weight, normalize_edge_type

# User-defined tier thresholds (see types.FLOAT_TO_WEIGHT).
MIN_CO_BOTH = 2
MIN_CO_RATE = 0.25


def _hit_nodes(graph: SkillGraph, text: str) -> list[str]:
    t = text.lower()
    return sorted(
        i
        for i, n in graph.nodes.items()
        if i.lower() in t or n.name.lower() in t
    )


def co_occurrence_rate(n_both: int, n_a: int, n_b: int) -> float:
    """Symmetric co-occurrence rate in [0, 1]."""
    if n_both <= 0:
        return 0.0
    denom = min(n_a, n_b)
    if denom <= 0:
        return 0.0
    return min(1.0, n_both / denom)


def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def _parse_pair_key(key: str) -> tuple[str, str]:
    a, b = key.split("|", 1)
    return a, b


def update_node_stats(graph: SkillGraph, results: list[dict[str, Any]]) -> None:
    for r in results:
        text = str(r.get("trajectory") or "")
        hits = _hit_nodes(graph, text)
        ok = float(r.get("hard") or 0) >= 0.5
        for nid in hits:
            n = graph.nodes[nid]
            n.n_use += 1
            if ok:
                n.n_succ += 1


def update_cooccur_from_rollouts(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    *,
    min_co_both: int = MIN_CO_BOTH,
    min_rate: float = MIN_CO_RATE,
    stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Refresh co-occur edges and return graph-external cumulative statistics."""
    update_node_stats(graph, results)
    stats = stats if stats is not None else {}

    for r in results:
        if float(r.get("hard") or 0) < 0.5:
            continue
        hits = _hit_nodes(graph, str(r.get("trajectory") or ""))
        for i, a in enumerate(hits):
            for b in hits[i + 1 :]:
                key = _pair_key(a, b)
                entry = stats.setdefault(key, {"n_both": 0})
                entry["n_both"] = int(entry.get("n_both") or 0) + 1

    non_co = [
        e for e in graph.edges if normalize_edge_type(e.type) != "co_occur"
    ]
    co_edges: list[Edge] = []

    for key, entry in stats.items():
        n_both = int(entry.get("n_both") or 0)
        if n_both < min_co_both:
            continue
        a, b = _parse_pair_key(key)
        if a not in graph.nodes or b not in graph.nodes:
            continue
        n_a = graph.nodes[a].n_succ
        n_b = graph.nodes[b].n_succ
        rate = co_occurrence_rate(n_both, n_a, n_b)
        entry["rate"] = round(rate, 4)
        entry["n_a"] = n_a
        entry["n_b"] = n_b
        if rate < min_rate:
            continue
        label = float_to_weight(rate)
        co_edges.append(
            Edge(
                src=a,
                dst=b,
                type="co_occur",
                w=rate,
                weight=label,
                rationale=(
                    f"co_occur rate={rate:.1%} "
                    f"(joint={n_both}, n({a})={n_a}, n({b})={n_b})"
                ),
                graph_name="stable_rule_graph",
                frozen=False,
                active=True,
            )
        )

    graph.edges = non_co + co_edges
    graph.sync_data_from_views()
    return stats
