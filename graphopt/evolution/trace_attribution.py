"""Deterministic reasoning-trace evidence for graph-element attribution.

The student-authored reasoning trace is useful only when its graph citations can
be checked independently.  This module binds the trace to the strict
``graph_usage`` sidecar and the current graph before any teacher is allowed to
name a faulty node or edge.
"""

from __future__ import annotations

import re
from typing import Any

from graphopt.types import SkillGraph


TRACE_EVIDENCE_SCHEMA = "graphopt-trace-evidence-v1"

_BRACKETED_ID_RE = re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_]*)\s*\]")


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _citation_excerpts(trace: str, element_ids: list[str]) -> dict[str, list[str]]:
    """Return short, verbatim trace clauses for every cited graph element."""
    excerpts: dict[str, list[str]] = {element_id: [] for element_id in element_ids}
    clauses = [
        clause.strip()
        for clause in re.split(r"(?<=[.!?;。！？；])\s+|\n+", trace)
        if clause.strip()
    ]
    for clause in clauses:
        cited = {
            match.group(1) for match in _BRACKETED_ID_RE.finditer(clause)
        }
        for element_id in cited & set(excerpts):
            if clause not in excerpts[element_id]:
                excerpts[element_id].append(clause[:1000])
    return excerpts


def build_trace_evidence(
    graph: SkillGraph,
    result: dict[str, Any],
    *,
    trace: str | None = None,
    trace_status: str | None = None,
) -> dict[str, Any]:
    """Build a graph-bound certificate from one rollout row.

    ``VERIFIED`` means all of the following are true:

    * the trace was authored in the student response;
    * the graph-usage sidecar passed its strict JSON/ID validation;
    * the deduplicated cited node and edge sets exactly equal the sidecar sets;
    * every cited edge has both endpoints in the cited node set.

    A post-hoc trace remains useful diagnostic prose but can never receive a
    verified certificate.
    """
    raw_trace = (
        result.get("semantic_reasoning_trace") if trace is None else trace
    )
    semantic_trace = str(raw_trace or "").strip()
    raw_status = (
        result.get("semantic_reasoning_trace_status")
        if trace_status is None else trace_status
    )
    semantic_status = str(raw_status or "not_requested")
    refs = result.get("graph_refs")
    refs = refs if isinstance(refs, dict) else {}
    usage_status = str(refs.get("status") or "missing_student_usage")
    usage_nodes = _ordered_unique([
        str(value) for value in (refs.get("used_nodes") or [])
    ])
    usage_edges = _ordered_unique([
        str(value) for value in (refs.get("used_edges") or [])
    ])

    valid_nodes = set(map(str, graph.nodes))
    edge_map = {str(edge.id): edge for edge in graph.edges if edge.id}
    valid_edges = set(edge_map)
    raw_citations = [
        match.group(1) for match in _BRACKETED_ID_RE.finditer(semantic_trace)
    ]
    cited_nodes = _ordered_unique([
        value for value in raw_citations if value in valid_nodes
    ])
    cited_edges = _ordered_unique([
        value for value in raw_citations if value in valid_edges
    ])
    unknown_ids = _ordered_unique([
        value
        for value in raw_citations
        if (
            any(character.isdigit() for character in value)
            and value not in valid_nodes
            and value not in valid_edges
        )
    ])

    cited_node_set = set(cited_nodes)
    cited_edge_set = set(cited_edges)
    usage_node_set = set(usage_nodes)
    usage_edge_set = set(usage_edges)
    edge_endpoint_errors = []
    for edge_id in cited_edges:
        edge = edge_map[edge_id]
        missing = [
            node_id for node_id in (str(edge.src), str(edge.dst))
            if node_id not in cited_node_set
        ]
        if missing:
            edge_endpoint_errors.append({
                "edge_id": edge_id,
                "source": str(edge.src),
                "target": str(edge.dst),
                "missing_cited_endpoints": missing,
            })

    missing_from_usage_nodes = sorted(cited_node_set - usage_node_set)
    missing_from_usage_edges = sorted(cited_edge_set - usage_edge_set)
    unmentioned_usage_nodes = sorted(usage_node_set - cited_node_set)
    unmentioned_usage_edges = sorted(usage_edge_set - cited_edge_set)
    verified = bool(
        semantic_status == "validated_student_trace"
        and semantic_trace
        and usage_status == "validated_student_usage"
        and cited_nodes
        and not unknown_ids
        and not missing_from_usage_nodes
        and not missing_from_usage_edges
        and not unmentioned_usage_nodes
        and not unmentioned_usage_edges
        and not edge_endpoint_errors
    )

    if verified:
        status = "VERIFIED"
        reason = "student trace and graph_usage are graph-bound and exactly aligned"
    elif semantic_status != "validated_student_trace":
        status = "UNAVAILABLE" if not semantic_trace else "UNVERIFIED"
        reason = (
            "no validated student-authored reasoning trace"
            if not semantic_trace
            else f"trace provenance {semantic_status!r} cannot certify attribution"
        )
    elif usage_status != "validated_student_usage":
        status = "UNVERIFIED"
        reason = f"graph_usage status is {usage_status!r}"
    elif not cited_nodes:
        status = "UNVERIFIED"
        reason = "trace cites no known graph node"
    else:
        status = "UNVERIFIED"
        reason = "trace citations and graph_usage do not satisfy the exact graph contract"

    cited_elements = [*cited_nodes, *cited_edges]
    return {
        "schema_version": TRACE_EVIDENCE_SCHEMA,
        "status": status,
        "verified": verified,
        "reason": reason,
        "trace_provenance": semantic_status,
        "graph_usage_status": usage_status,
        "cited_nodes": cited_nodes,
        "cited_edges": cited_edges,
        "usage_nodes": usage_nodes,
        "usage_edges": usage_edges,
        "unknown_cited_ids": unknown_ids,
        "missing_from_usage_nodes": missing_from_usage_nodes,
        "missing_from_usage_edges": missing_from_usage_edges,
        "unmentioned_usage_nodes": unmentioned_usage_nodes,
        "unmentioned_usage_edges": unmentioned_usage_edges,
        "edge_endpoint_errors": edge_endpoint_errors,
        "citation_excerpts": _citation_excerpts(
            semantic_trace, cited_elements
        ),
    }


def attach_trace_evidence(
    graph: SkillGraph, results: list[dict[str, Any]]
) -> None:
    """Attach deterministic trace evidence to rollout rows in place."""
    for result in results:
        result["trace_evidence"] = build_trace_evidence(graph, result)

