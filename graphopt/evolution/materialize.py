"""Materialize a trainable rule graph JSON from base graph + evolution patch."""

from __future__ import annotations

from graphopt.optimizer.skill import apply_patch, save_graph
from graphopt.types import GraphPatch, SkillGraph


def materialize_skill_graph(base: SkillGraph, patch: GraphPatch) -> SkillGraph:
    """Apply patch on a copy; output conforms to ``initial.json`` key layout."""
    out = base.copy()
    report = apply_patch(out, patch)
    if report.get("n_failed"):
        raise ValueError(
            "patch failed to materialize atomically: "
            + "; ".join(str(item) for item in report.get("warnings") or [])
        )
    return out


def save_skill_json(graph: SkillGraph, path: str) -> None:
    """Persist the full SkillGraph JSON document."""
    save_graph(graph, path)
