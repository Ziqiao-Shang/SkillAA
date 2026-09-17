"""Batch-local compact failure examples for graph-edit synthesis.

The former persistent teacher-curated right/bad casebook was removed. Same-group
successful siblings are attached directly to each failed train case by Trainer
and consumed by Case Analyzer; this module only compacts current-batch failures
for downstream node synthesis.
"""

from __future__ import annotations

from typing import Any

def build_failure_casebook_examples(
    analyses: list[Any],
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Attribute compact failed training cases to the existing nodes they concern."""
    from graphopt.evolution.experience_quality import compact_failure_example

    result_by_id = {str(row.get("id") or ""): row for row in results}
    by_node: dict[str, list[dict[str, Any]]] = {}
    for analysis in analyses:
        if analysis.success or analysis.analysis_partial:
            continue
        result = result_by_id.get(str(analysis.case_id), {})
        if not result:
            continue
        targets = set(str(x) for x in (
            list(analysis.wrong_nodes)
            + list(analysis.missed_relevant_nodes)
            + list(analysis.execution_lapse_nodes)
        ) if str(x))
        targets.update(
            str(proposal.target_node)
            for proposal in analysis.node_revision_proposals
            if str(proposal.target_node)
        )
        targets.update(
            str(proposal.target_node)
            for proposal in analysis.retrieval_revision_proposals
            if str(proposal.target_node)
        )
        base = compact_failure_example(dict(result))
        base["failure_type"] = str(analysis.failure_type or "")
        for node_id in sorted(targets):
            related = [
                proposal for proposal in analysis.node_revision_proposals
                if str(proposal.target_node) == node_id
            ]
            item = dict(base)
            if related:
                item.update({
                    "observable_state": " | ".join(dict.fromkeys(
                        str(x.observable_state) for x in related if str(x.observable_state)
                    )),
                    "bad_action": " | ".join(dict.fromkeys(
                        str(x.bad_action) for x in related if str(x.bad_action)
                    )),
                    "better_action": " | ".join(dict.fromkeys(
                        str(x.better_action) for x in related if str(x.better_action)
                    )),
                    "semantic_delta": " | ".join(dict.fromkeys(
                        str(x.semantic_delta) for x in related if str(x.semantic_delta)
                    )),
                })
            by_node.setdefault(node_id, []).append(item)
    return by_node

