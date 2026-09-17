"""Aggregate case-level binary statistics onto graph nodes/edges."""

from __future__ import annotations

from graphopt.evolution.types import CaseAnalysis, UsageStats
from graphopt.types import SkillGraph, normalize_edge_type


def _bump(
    stats: UsageStats,
    *,
    used: bool = False,
    correct: bool = False,
    wrong: bool = False,
    retrieval_missed: bool = False,
) -> None:
    if used:
        stats.used += 1
    if correct:
        stats.correct += 1
    if wrong:
        stats.wrong += 1
    if retrieval_missed:
        stats.retrieval_missed += 1


def aggregate_statistics(
    graph: SkillGraph,
    analyses: list[CaseAnalysis],
) -> tuple[dict[str, UsageStats], dict[str, UsageStats]]:
    """Count the current epoch only (+1 at most once per case per stat).

    ``used/correct/wrong`` deliberately reset to zero at every evolution
    boundary. Proposal support may use a separate cross-epoch pool, but graph
    usage statistics never inherit values from an earlier epoch.
    """
    node_stats = {nid: UsageStats() for nid in graph.nodes}
    edge_stats = {e.id: UsageStats() for e in graph.edges if e.id}

    for a in analyses:
        used_n = set(a.used_nodes)
        for nid in used_n:
            if nid not in node_stats:
                node_stats[nid] = UsageStats()
            _bump(node_stats[nid], used=True)
        for nid in set(a.correct_nodes):
            if nid in node_stats:
                _bump(node_stats[nid], correct=True)
        for nid in set(a.wrong_nodes):
            if nid in node_stats:
                _bump(node_stats[nid], wrong=True)
        for nid in set(a.missed_relevant_nodes):
            if nid in node_stats:
                _bump(node_stats[nid], retrieval_missed=True)

        used_e = set(a.used_edges)
        for eid in used_e:
            if eid not in edge_stats:
                edge_stats[eid] = UsageStats()
            _bump(edge_stats[eid], used=True)
        for eid in set(a.correct_edges):
            if eid in edge_stats:
                _bump(edge_stats[eid], correct=True)
        for eid in set(a.wrong_edges):
            if eid in edge_stats:
                _bump(edge_stats[eid], wrong=True)
        for eid in set(a.missed_relevant_edges):
            if eid in edge_stats:
                _bump(edge_stats[eid], retrieval_missed=True)

    return node_stats, edge_stats


def apply_statistics_to_graph(
    graph: SkillGraph,
    node_stats: dict[str, UsageStats],
    edge_stats: dict[str, UsageStats],
) -> None:
    """In-memory only (evolution weighting within a step). Not written to skill JSON."""
    for nid, st in node_stats.items():
        if nid in graph.nodes:
            n = graph.nodes[nid]
            n.stat_used = st.used
            n.stat_correct = st.correct
            n.stat_wrong = st.wrong
    for e in graph.edges:
        if e.id and e.id in edge_stats:
            st = edge_stats[e.id]
            e.stat_used = st.used
            e.stat_correct = st.correct
            e.stat_wrong = st.wrong


def statistics_to_json(
    node_stats: dict[str, UsageStats],
    edge_stats: dict[str, UsageStats],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        "nodes": {k: v.to_dict() for k, v in node_stats.items()},
        "edges": {k: v.to_dict() for k, v in edge_stats.items()},
    }
