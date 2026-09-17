"""Graph evolution engine — case attribution → merge → threshold → edit plan → patch."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from graphopt.debug.artifacts import (
    ArtifactStore, StepRecorder, save_json, save_llm_call,
)
from graphopt.evolution.cache import EvolutionCache
from graphopt.evolution.casebook import build_failure_casebook_examples
from graphopt.evolution.case_analyzer import (
    ATTRIBUTION_BOUNDARY_PROTOCOL,
    analyze_cases,
    badcase_analysis_protocol,
)
from graphopt.evolution.config import EvolutionConfig
from graphopt.evolution.materialize import materialize_skill_graph, save_skill_json
from graphopt.evolution.merge import (
    OPINION_MERGE_TIMEOUT_SECONDS,
    _is_retryable_opinion_merge_error,
    _allocate_new_node_id,
    _merge_activation_edge_records,
    _rank_and_cap_node_revisions,
    _similar,
    semantic_merge_execution_lapses,
    semantic_merge_proposals,
)
from graphopt.evolution.execution_child import synthesize_execution_child
from graphopt.evolution.exact_refs import exact_reference_json
from graphopt.evolution.node_rewrite import (
    reinforce_node_execution,
    rewrite_node,
    rewrite_toxic_node,
    rewrite_node_when_to_use,
)
from graphopt.evolution.new_node_synthesis import synthesize_new_node
from graphopt.evolution.proposal_pool import (
    active_proposal_pool_buckets,
    semantic_recluster_proposal_pool,
)
from graphopt.evolution.stats import (
    aggregate_statistics,
    apply_statistics_to_graph,
    statistics_to_json,
)
from graphopt.evolution.types import (
    CaseAnalysis, EvolutionResult, GraphEditPlan, MergedProposal, UsageStats,
)
from graphopt.evolution.edge_weights import assign_edge_weights, weight_float_to_label
from graphopt.evolution.experience_quality import (
    extract_actions,
    select_failure_examples,
    select_positive_examples,
)
from graphopt.types import GraphEdit, GraphPatch, SkillGraph, normalize_edge_type

STRUCTURAL = frozenset({"prereq", "enhance"})


def _merge_with_cached_pool(
    current: dict[str, list[MergedProposal]],
    cache: EvolutionCache,
    *,
    failed_node_merges: set[str] | None = None,
) -> dict[str, list[MergedProposal]]:
    """Add prior, not-yet-materialized proposal support to this step's merge.

    The cache stores evidence from completed steps.  Keeping this combination
    here (before thresholding) makes ``persist_proposal_pool`` real rather than
    merely archival.  Current-step evidence is ingested separately afterwards,
    which avoids counting a cumulative total again on the next step.
    """
    # Prior support must enrich a separate planning view. A shallow copy would
    # mutate ``current`` and cause cached evidence to be ingested again.
    failed_nodes = failed_node_merges or set()
    out = {key: [deepcopy(value) for value in values] for key, values in current.items()}

    def add_prior(bucket: str, prior: dict[str, Any]) -> None:
        kind = str(prior.get("kind") or "")
        content = str(prior.get("content") or prior.get("merged_proposal") or "")
        target_node = str(prior.get("target_node") or "")
        source = str(prior.get("source") or "")
        target = str(prior.get("target") or "")
        relation = str(prior.get("relation") or "")
        old_edge = str(prior.get("target_old_edge") or "")
        rationale = str(prior.get("rationale") or "")
        parent_node = str(prior.get("parent_node") or "")
        activation_edges = _merge_activation_edge_records(
            [dict(item) for item in (prior.get("activation_edges") or [])]
        )
        semantic_key = str(prior.get("semantic_key") or "")
        operation = str(prior.get("operation") or "PATCH").upper()
        toxic_text = str(prior.get("toxic_text") or "")
        case_ids = list(dict.fromkeys(str(x) for x in (prior.get("source_case_ids") or [])))
        evidence_items = [
            dict(item) for item in (prior.get("evidence_items") or [])
            if isinstance(item, dict)
        ]
        support = len(case_ids)
        if support <= 0:
            return
        for item in out.setdefault(bucket, []):
            same = False
            if bucket in {"node_revisions", "retrieval_revisions"}:
                same = (
                    item.target_node == target_node
                    and _similar(item.content, content)
                    and (
                        bucket == "retrieval_revisions"
                        or (item.operation == operation and item.toxic_text == toxic_text)
                    )
                )
            elif bucket == "new_nodes":
                # Keep current and cached specialist opinions separate until
                # the complete-case semantic reclusterer sees both.
                same = False
            else:
                same = (
                    item.source == source
                    and item.target == target
                    and normalize_edge_type(item.relation) == normalize_edge_type(relation)
                    and item.target_old_edge == old_edge
                )
            if same:
                if bucket == "new_nodes":
                    if content and content not in item.content:
                        item.content = f"{item.content}\n{content}".strip()
                    item.activation_edges = _merge_activation_edge_records(
                        item.activation_edges, activation_edges
                    )
                item.source_case_ids = list(dict.fromkeys(item.source_case_ids + case_ids))
                item.support = len(item.source_case_ids)
                evidence_by_case = {
                    str(evidence.get("case_id") or index): evidence
                    for index, evidence in enumerate(
                        [*item.evidence_items, *evidence_items]
                    )
                }
                item.evidence_items = list(evidence_by_case.values())
                return
        out[bucket].append(
            MergedProposal(
                kind=kind or {
                    "node_revisions": "node_revision",
                    "retrieval_revisions": "retrieval_revision",
                    "new_nodes": "new_node",
                }.get(bucket, "edge"),
                content=content,
                target_node=target_node,
                source=source,
                target=target,
                relation=relation,
                target_old_edge=old_edge,
                rationale=rationale,
                support=support,
                source_case_ids=case_ids,
                operation=operation,
                toxic_text=toxic_text,
                semantic_key=semantic_key,
                parent_node=parent_node,
                activation_edges=activation_edges,
                evidence_items=evidence_items,
            )
        )

    for entry in cache.nodes.values():
        for prior in entry.merged_revisions:
            if str(prior.get("target_node") or "") in failed_nodes:
                continue
            add_prior("node_revisions", prior)
        for prior in entry.merged_retrieval_revisions:
            if str(prior.get("target_node") or "") in failed_nodes:
                continue
            add_prior("retrieval_revisions", prior)
    for prior in cache.new_node_candidates:
        add_prior("new_nodes", prior)
    for entry in cache.edges.values():
        for prior in entry.merged_corrections:
            add_prior("edges", prior)
    for prior in cache.new_edge_candidates:
            add_prior("edges", prior)
    return out


def _proposal_evidence_items(
    proposals: list[MergedProposal],
) -> list[dict[str, Any]]:
    """Deduplicate graph-bound case evidence while preserving source order."""
    evidence_by_case: dict[str, dict[str, Any]] = {}
    anonymous = 0
    for proposal in proposals:
        for evidence in proposal.evidence_items:
            case_id = str(evidence.get("case_id") or "")
            if not case_id:
                anonymous += 1
                case_id = f"anonymous:{anonymous}"
            evidence_by_case[case_id] = dict(evidence)
    return list(evidence_by_case.values())


def _edge_map(graph: SkillGraph) -> dict[str, Any]:
    return {e.id: e for e in graph.edges if e.id}

def _execution_children(graph: SkillGraph, parent_node: str) -> list[str]:
    """Return active execution-detail children explicitly linked to a parent."""
    children: list[str] = []
    for edge in graph.edges:
        if (
            edge.active
            and edge.src == parent_node
            and normalize_edge_type(edge.type) == "enhance"
            and str(edge.rationale or "").startswith("execution_detail_child:")
            and edge.dst in graph.nodes
            and graph.nodes[edge.dst].active
        ):
            children.append(edge.dst)
    return sorted(set(children))

def _execution_child_keys(graph: SkillGraph, parent_node: str) -> set[str]:
    prefix = f"execution_detail_child:{parent_node}:"
    return {
        str(edge.rationale)[len(prefix):]
        for edge in graph.edges
        if (
            edge.active
            and edge.src == parent_node
            and normalize_edge_type(edge.type) == "enhance"
            and str(edge.rationale or "").startswith(prefix)
        )
    }

def _is_execution_child(graph: SkillGraph, node_id: str) -> bool:
    """Prevent recursive execution-child chains."""
    return any(
        edge.active
        and edge.dst == node_id
        and normalize_edge_type(edge.type) == "enhance"
        and str(edge.rationale or "").startswith("execution_detail_child:")
        for edge in graph.edges
    )


def _allocate_execution_child_id(
    graph: SkillGraph, parent_node: str, reserved: set[str]
) -> str:
    prefix = f"{parent_node}E"
    index = 1
    while f"{prefix}{index}" in graph.nodes or f"{prefix}{index}" in reserved:
        index += 1
    node_id = f"{prefix}{index}"
    reserved.add(node_id)
    return node_id


def _build_edit_plan(
    graph: SkillGraph,
    merged: dict[str, list[MergedProposal]],
    node_stats: dict,
    edge_stats: dict,
    cfg: EvolutionConfig,
    *,
    seed: int = 0,
) -> GraphEditPlan:
    x, y = cfg.node_support_threshold, cfg.edge_support_threshold
    # Train support ranks and explains candidates; it is not an admission
    # Gate. One eligible case is enough to materialize a candidate for the
    # independent mapped-validation small Gate.
    node_min_support = max(1, x)
    plan = GraphEditPlan()
    emap = _edge_map(graph)

    # Every genuinely new specialist is atomic with all evidence-grounded
    # activation edges. A standalone node is never a valid candidate.
    reserved_new_ids: set[str] = set()
    for m in merged.get("new_nodes") or []:
        if m.support < node_min_support:
            continue
        raw_activation_edges = list(m.activation_edges)
        parent = str(m.parent_node or "")
        if not raw_activation_edges and parent in graph.nodes:
            raw_activation_edges = [{
                "source": parent,
                "relation": normalize_edge_type(m.relation or "enhance"),
                "reason": m.rationale or f"semantic_specialist_for:{parent}",
                "source_case_ids": list(m.source_case_ids),
            }]
        activation_edges: list[dict[str, Any]] = []
        for raw in raw_activation_edges:
            source = str(raw.get("source") or "")
            relation = normalize_edge_type(str(raw.get("relation") or ""))
            edge_cases = sorted(
                set(map(str, raw.get("source_case_ids") or []))
                & set(map(str, m.source_case_ids))
            )
            reason = str(raw.get("reason") or "").strip()
            if source in graph.nodes and relation == "enhance" and edge_cases and reason:
                activation_edges.append({
                    "source": source,
                    "relation": "enhance",
                    "reason": reason,
                    "source_case_ids": edge_cases,
                })
        if not activation_edges:
            continue
        node_id = _allocate_new_node_id(graph, reserved_new_ids)
        parents = sorted({str(item["source"]) for item in activation_edges})
        group_id = f"semantic-specialist:{node_id}"
        evidence_items = _proposal_evidence_items([m])
        plan.add_nodes.append({
            "node_id": node_id,
            "candidate_id": node_id,
            "support": m.support,
            "content": m.content,
            "source_case_ids": list(m.source_case_ids),
            "evidence_items": evidence_items,
            "kind": "semantic_specialist",
            "parent_node": parents[0],
            "group_id": group_id,
        })
        for activation in activation_edges:
            edge_cases = list(activation["source_case_ids"])
            case_set = set(edge_cases)
            edge_evidence = [
                item for item in evidence_items
                if str(item.get("case_id") or "") in case_set
            ] or evidence_items
            plan.add_edges.append({
                "source": activation["source"],
                "target": node_id,
                "relation": "enhance",
                "kind": "semantic_specialist",
                "parent_node": activation["source"],
                "group_id": group_id,
                "support": len(edge_cases),
                "source_case_ids": edge_cases,
                "evidence_items": edge_evidence,
                "rationale": activation["reason"],
            })

    added_node_ids = {str(item["node_id"]) for item in plan.add_nodes}

    eligible_by_node: dict[tuple[str, str, str], list[MergedProposal]] = {}
    for proposal in merged.get("node_revisions") or []:
        if proposal.support >= node_min_support and (
            proposal.target_node in graph.nodes or proposal.target_node in added_node_ids
        ):
            operation = str(proposal.operation or "PATCH").upper()
            toxic_text = proposal.toxic_text if operation == "REWRITE" else ""
            eligible_by_node.setdefault(
                (proposal.target_node, operation, toxic_text), []
            ).append(proposal)

    for node_id, operation, toxic_text in sorted(
        eligible_by_node,
        key=lambda key: (key[0], 0 if key[1] == "REWRITE" else 1, key[2]),
    ):
        selected = sorted(
            eligible_by_node[(node_id, operation, toxic_text)],
            key=lambda item: (-item.support, item.content.casefold()),
        )[: cfg.node_merge_top_k]
        source_case_ids = list(
            dict.fromkeys(
                case_id
                for proposal in selected
                for case_id in proposal.source_case_ids
            )
        )
        selected_json = [proposal.to_dict() for proposal in selected]
        merged_revision = "\n".join(
            f"{index}. [support={proposal.support}] {proposal.content}"
            for index, proposal in enumerate(selected, start=1)
        )
        plan.update_nodes.append(
            {
                "node_id": node_id,
                "revision_scope": operation.lower(),
                "toxic_text": toxic_text,
                "support": max(proposal.support for proposal in selected),
                "aggregate_support": len(source_case_ids),
                "merged_revision": merged_revision,
                "selected_revisions": selected_json,
                "source_case_ids": source_case_ids,
                "evidence_items": _proposal_evidence_items(selected),
            }
        )

    # Retrieval misses prove the skill is relevant but its activation wording
    # was insufficient. They are intentionally kept separate from content
    # defects: after x semantically merged votes, only `when_to_use` may change.
    retrieval_by_node: dict[str, list[MergedProposal]] = {}
    for proposal in merged.get("retrieval_revisions") or []:
        if (
            proposal.support >= node_min_support
            and proposal.target_node in graph.nodes
        ):
            retrieval_by_node.setdefault(proposal.target_node, []).append(proposal)
    for node_id in sorted(retrieval_by_node):
        selected = sorted(
            retrieval_by_node[node_id],
            key=lambda item: (-item.support, item.content.casefold()),
        )[: cfg.node_merge_top_k]
        source_case_ids = list(
            dict.fromkeys(
                case_id
                for proposal in selected
                for case_id in proposal.source_case_ids
            )
        )
        plan.update_nodes.append(
            {
                "node_id": node_id,
                "revision_scope": "when_to_use",
                "support": max(proposal.support for proposal in selected),
                "aggregate_support": len(source_case_ids),
                "merged_revision": "\n".join(
                    f"{index}. [support={proposal.support}] {proposal.content}"
                    for index, proposal in enumerate(selected, start=1)
                ),
                "selected_revisions": [p.to_dict() for p in selected],
                "source_case_ids": source_case_ids,
                "evidence_items": _proposal_evidence_items(selected),
            }
        )

    # A cited-but-not-executed rule is not rewritten. Rank distinct-case lapse
    # evidence and create a narrow child that is probed before the full Gate.
    already_updated = {str(item["node_id"]) for item in plan.update_nodes}
    execution_ranked = sorted(
        list(merged.get("execution_reinforcements") or []),
        key=lambda proposal: (
            -int(proposal.support), proposal.target_node, proposal.content.casefold()
        ),
    )
    execution_count = 0
    for proposal in execution_ranked:
        parent = str(proposal.target_node)
        semantic_key = str(proposal.semantic_key or "")
        if (
            execution_count >= cfg.max_execution_child_candidates_per_epoch
            or proposal.support < node_min_support
            or parent not in graph.nodes
            or _is_execution_child(graph, parent)
            or parent in already_updated
            or len(_execution_children(graph, parent))
            >= cfg.max_active_execution_children_per_parent
            or (semantic_key and semantic_key in _execution_child_keys(graph, parent))
        ):
            continue
        child_id = _allocate_execution_child_id(graph, parent, reserved_new_ids)
        group_id = f"execution-child:{parent}:{child_id}"
        source_case_ids = list(dict.fromkeys(
            str(case_id) for case_id in proposal.source_case_ids
        ))
        plan.add_nodes.append({
            "node_id": child_id,
            "candidate_id": child_id,
            "kind": "execution_detail",
            "parent_node": parent,
            "group_id": group_id,
            "support": len(source_case_ids),
            "aggregate_support": len(source_case_ids),
            "content": proposal.content,
            "source_case_ids": source_case_ids,
            "semantic_key": semantic_key,
            "evidence_items": [dict(item) for item in proposal.evidence_items],
        })
        plan.add_edges.append({
            "source": parent,
            "target": child_id,
            "relation": "enhance",
            "kind": "execution_detail",
            "parent_node": parent,
            "group_id": group_id,
            "support": len(source_case_ids),
            "source_case_ids": source_case_ids,
            "semantic_key": semantic_key,
            "rationale": f"execution_detail_child:{parent}:{semantic_key}",
        })
        added_node_ids.add(child_id)
        execution_count += 1

    edge_candidates: list[MergedProposal] = list(merged.get("edges") or [])
    for m in edge_candidates:
        if m.support < y:
            continue
        old_id = m.target_old_edge
        old = emap.get(old_id) if old_id else None
        entry = {
            "source": m.source,
            "target": m.target,
            "relation": m.relation,
            "support": m.support,
            "source_case_ids": m.source_case_ids,
            "rationale": m.rationale,
            "evidence_items": _proposal_evidence_items([m]),
        }
        if old and old_id:
            st = edge_stats.get(old_id)
            c = st.correct if st else old.stat_correct
            w = st.wrong if st else old.stat_wrong
            if w > c and m.support >= y:
                plan.replace_edges.append(
                    {
                        "old_edge": old_id,
                        "reason": "wrong > correct",
                        "new_edge": entry,
                        "support": m.support,
                    }
                )
            else:
                # A correction proposal is evidence about the named old edge,
                # not permission to create a second, conflicting relation. If
                # the old edge is not wrong more often than correct, retain it
                # and discard this correction for the current epoch.
                plan.keep_edges.append(old_id)
        else:
            plan.add_edges.append(entry)

    node_deletion_threshold = cfg.delete_used_threshold
    if cfg.delete_never_used_node:
        protected_nodes = {item["node_id"] for item in plan.update_nodes}
        # A node created in this patch has no rollout usage yet. Protect it for
        # this round; from the next epoch it is an ordinary node and may be
        # removed if that next epoch's used count is below d.
        protected_nodes.update(added_node_ids)
        for item in plan.add_edges + plan.replace_edges:
            ne = item.get("new_edge") or item
            protected_nodes.add(str(ne.get("source") or ""))
            protected_nodes.add(str(ne.get("target") or ""))
        # used counts come only from this epoch's complete case pool (all
        # rollout batches); graph usage statistics never carry across epochs.
        for nid, st in node_stats.items():
            if nid in protected_nodes:
                continue
            # A missed-relevant node is useful evidence whose trigger failed.
            # Treating it as ordinary used=0 would delete the very skill that
            # should have been retrieved.
            if (
                nid in graph.nodes
                and nid.startswith("X")
                and st.retrieval_missed == 0
                and st.used < node_deletion_threshold
            ):
                plan.delete_nodes.append(nid)

    if cfg.delete_never_used_edge:
        protected_edges = {item["old_edge"] for item in plan.replace_edges if item.get("old_edge")}
        for item in plan.add_edges + plan.replace_edges:
            ne = item.get("new_edge") or item
            protected_edges.add(f"new:{ne.get('source')}->{ne.get('target')}:{ne.get('relation')}")
        for eid, st in edge_stats.items():
            if eid in protected_edges:
                continue
            e = emap.get(eid)
            if not e:
                continue
            if normalize_edge_type(e.type) not in STRUCTURAL:
                continue
            if st.retrieval_missed == 0 and st.used < cfg.edge_delete_used_threshold:
                plan.delete_edges.append(eid)

    assign_edge_weights(
        graph,
        plan,
        edge_stats,
        structural_min_used=cfg.edge_support_threshold,
        seed=seed,
    )
    return plan


def _plan_to_patch(
    graph: SkillGraph,
    plan: GraphEditPlan,
    *,
    chat_fn=None,
    mode: str = "template",
    delete_used_threshold: int = 4,
    store: ArtifactStore | None = None,
) -> GraphPatch:
    edits: list[GraphEdit] = []
    delete_edits: list[GraphEdit] = []

    # Patch order is intentional: add genuinely-new nodes first, then rewrite
    # both old and just-added nodes, and delete low-use old nodes last.
    for item in plan.add_nodes:
        fields = item.get("node_fields") or synthesize_new_node(
            str(item["node_id"]), str(item["content"]), int(item.get("support") or 0),
            [str(case_id) for case_id in (item.get("source_case_ids") or [])],
            chat_fn=chat_fn, mode=mode, store=store,
        )
        if fields is None:
            continue
        edits.append(
            GraphEdit(
                op="add_node",
                node_id=str(item["node_id"]),
                name=str(fields["title"]),
                meaning="",
                when_to_use=str(fields["when_to_use"]),
                how_to_use=str(fields["how_to_use"]),
                avoid=list(fields["avoid"]),
                category="learned",
                source_type="failure",
                reasoning=f"new_node support={item.get('support')}",
                group_id=str(item.get("group_id") or ""),
                edit_kind=str(item.get("kind") or ""),
                parent_node=str(item.get("parent_node") or ""),
                source_case_ids=[str(x) for x in (item.get("source_case_ids") or [])],
                evidence_items=[
                    dict(evidence)
                    for evidence in (item.get("evidence_items") or [])
                    if isinstance(evidence, dict)
                ],
            )
        )

    rewrite_graph = graph.copy()
    for edit in edits:
        from graphopt.optimizer.skill import apply_edit

        apply_edit(rewrite_graph, edit)

    for item in plan.update_nodes:
        nid = item["node_id"]
        fields = item.get("node_fields")
        if not fields:
            continue
        edits.append(
            GraphEdit(
                op="update_node",
                node_id=nid,
                how_to_use=fields.get("how_to_use", ""),
                when_to_use=fields.get("when_to_use", ""),
                avoid=(list(fields["avoid"]) if "avoid" in fields else None),
                source_type="failure",
                reasoning=f"evolution support={item.get('support')}",
                edit_kind=str(item.get("revision_scope") or ""),
                source_case_ids=[
                    str(case_id) for case_id in (item.get("source_case_ids") or [])
                ],
                evidence_items=[
                    dict(evidence)
                    for evidence in (item.get("evidence_items") or [])
                    if isinstance(evidence, dict)
                ],
            )
        )

    for eid in plan.delete_edges:
        e = _edge_map(graph).get(eid)
        if e:
            delete_edits.append(
                GraphEdit(
                    op="delete_edge",
                    src=e.src,
                    dst=e.dst,
                    edge_type=e.type,
                    source_type="failure",
                    reasoning="never used or replaced",
                )
            )

    for item in plan.replace_edges:
        old = _edge_map(graph).get(item["old_edge"])
        ne = item.get("new_edge") or item
        if old:
            edits.append(
                GraphEdit(
                    op="delete_edge",
                    src=old.src,
                    dst=old.dst,
                    edge_type=old.type,
                    group_id=f"replace-edge:{item['old_edge']}",
                    source_type="failure",
                    reasoning=item.get("reason", "wrong > correct"),
                    source_case_ids=[
                        str(value) for value in (ne.get("source_case_ids") or [])
                    ],
                    evidence_items=[
                        dict(evidence)
                        for evidence in (ne.get("evidence_items") or [])
                        if isinstance(evidence, dict)
                    ],
                )
            )
        edits.append(
            GraphEdit(
                op="add_edge",
                src=str(ne["source"]),
                dst=str(ne["target"]),
                edge_type=normalize_edge_type(str(ne.get("relation") or "prereq")),
                group_id=f"replace-edge:{item['old_edge']}",
                w=plan.edge_weights.get(item.get("old_edge", ""), 0.8),
                rationale=str(ne.get("rationale") or item.get("rationale") or item.get("reason") or ""),
                source_type="failure" if item.get("reason") else "success",
                reasoning=f"corrective support={ne.get('support')}",
                source_case_ids=[
                    str(value) for value in (ne.get("source_case_ids") or [])
                ],
                evidence_items=[
                    dict(evidence)
                    for evidence in (ne.get("evidence_items") or [])
                    if isinstance(evidence, dict)
                ],
            )
        )

    for item in plan.add_edges:
        key = f"new:{item['source']}->{item['target']}:{item.get('relation')}"
        w = plan.edge_weights.get(key, item.get("support", 3) / max(item.get("support", 3), 3))
        edits.append(
            GraphEdit(
                op="add_edge",
                src=str(item["source"]),
                dst=str(item["target"]),
                edge_type=normalize_edge_type(str(item.get("relation") or "prereq")),
                w=float(w) if isinstance(w, (int, float)) else 0.6,
                rationale=str(item.get("rationale") or ""),
                source_type=(
                    "failure"
                    if str(item.get("kind") or "") in {
                        "execution_detail", "semantic_specialist"
                    }
                    else "success"
                ),
                reasoning=f"edge support={item.get('support' )}",
                group_id=str(item.get("group_id") or ""),
                edit_kind=str(item.get("kind") or ""),
                parent_node=str(item.get("parent_node") or ""),
                source_case_ids=[str(x) for x in (item.get("source_case_ids") or [])],
                evidence_items=[
                    dict(evidence)
                    for evidence in (item.get("evidence_items") or [])
                    if isinstance(evidence, dict)
                ],
            )
        )

    for nid in plan.delete_nodes:
        if nid in graph.nodes:
            delete_edits.append(
                GraphEdit(
                    op="delete_node",
                    node_id=nid,
                    source_type="failure",
                    reasoning=f"used<{delete_used_threshold} in case pool",
                )
            )

    edits.extend(delete_edits)

    # Refresh weights for surviving edges (prereq / enhance / co_occur).
    # Never emit a refresh for an edge/node that this same patch removes.
    removed_nodes = set(plan.delete_nodes)
    removed_edges = set(plan.delete_edges)
    removed_edges.update(
        str(item.get("old_edge") or "") for item in plan.replace_edges
    )
    for eid, w in plan.edge_weights.items():
        if eid.startswith("new:"):
            continue
        if eid in removed_edges:
            continue
        e = _edge_map(graph).get(eid)
        if not e:
            continue
        if e.src in removed_nodes or e.dst in removed_nodes:
            continue
        typ = normalize_edge_type(e.type)
        if typ not in STRUCTURAL and typ != "co_occur":
            continue
        target_label = weight_float_to_label(float(w), edge_type=typ)
        current_label = str(e.weight or weight_float_to_label(float(e.w), edge_type=typ))
        if abs(float(w) - float(e.w)) <= 1e-9 and target_label == current_label:
            continue
        edits.append(
            GraphEdit(
                op="add_edge",
                src=e.src,
                dst=e.dst,
                edge_type=e.type,
                w=float(w),
                source_type="success",
                reasoning=f"weight refresh {current_label}->{target_label}",
            )
        )

    return GraphPatch(reasoning="graph_evolution", edits=edits)


def _prepare_new_nodes(
    plan: GraphEditPlan,
    *,
    graph: SkillGraph | None = None,
    case_examples_by_id: dict[str, dict[str, Any]] | None = None,
    successful_examples: list[dict[str, Any]] | None = None,
    positive_examples_by_node: dict[str, list[dict[str, Any]]] | None = None,
    success_guards_by_node: dict[str, dict[str, Any]] | None = None,
    failure_examples_by_node: dict[str, list[dict[str, Any]]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
    positive_example_cap: int = 10,
    failure_example_cap: int = 6,
) -> list[dict[str, Any]]:
    """Synthesize mature new nodes before cache consumption and patch creation."""
    ready: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    case_examples_by_id = case_examples_by_id or {}
    successful_examples = successful_examples or []
    positive_examples_by_node = positive_examples_by_node or {}
    success_guards_by_node = success_guards_by_node or {}
    failure_examples_by_node = failure_examples_by_node or {}
    for item in plan.add_nodes:
        raw_failures = [
            dict(case_examples_by_id[case_id])
            for case_id in item.get("source_case_ids") or []
            if case_id in case_examples_by_id
        ]
        complete_source_cases = [
            dict(evidence)
            for evidence in (item.get("evidence_items") or [])
            if isinstance(evidence, dict)
        ]
        failures = complete_source_cases or select_failure_examples(
            raw_failures, cap=failure_example_cap
        )
        if str(item.get("kind") or "") == "execution_detail":
            parent = str(item.get("parent_node") or "")
            failures = complete_source_cases or [
                dict(example)
                for example in failure_examples_by_node.get(parent, [])[:failure_example_cap]
            ] or failures
            positives = select_positive_examples(
                parent,
                list(positive_examples_by_node.get(parent) or []),
                raw_failures,
                cap=positive_example_cap,
            )
            if parent in success_guards_by_node:
                positives.append(dict(success_guards_by_node[parent]))
            fields = (
                synthesize_execution_child(
                    graph, parent, str(item["node_id"]), str(item["content"]),
                    int(item.get("support") or 0),
                    [str(case_id) for case_id in (item.get("source_case_ids") or [])],
                    lapse_evidence=list(item.get("evidence_items") or []),
                    failure_examples=failures,
                    success_examples=positives,
                    chat_fn=chat_fn,
                    mode=mode,
                    store=store,
                )
                if graph is not None else None
            )
            if fields is None:
                abstained.append(dict(item))
                continue
            ready.append({
                **item,
                "node_fields": fields,
                "positive_case_ids": [
                    str(example.get("case_id") or "") for example in positives
                ],
            })
            continue
        task_types = {
            str(example.get("task_type") or "")
            for example in failures
            if str(example.get("task_type") or "")
        }
        comparable_successes = select_positive_examples(
            str(item["node_id"]),
            [
                dict(example)
                for example in successful_examples
                if not task_types
                or str(example.get("task_type") or "") in task_types
            ],
            raw_failures,
            cap=positive_example_cap,
        )
        parent = str(item.get("parent_node") or "")
        if parent in success_guards_by_node:
            comparable_successes.append(dict(success_guards_by_node[parent]))
        fields = synthesize_new_node(
            str(item["node_id"]),
            str(item["content"]),
            int(item.get("support") or 0),
            [str(case_id) for case_id in (item.get("source_case_ids") or [])],
            graph=graph,
            failure_examples=failures,
            success_examples=comparable_successes,
            chat_fn=chat_fn,
            mode=mode,
            store=store,
        )
        if fields is None:
            abstained.append(dict(item))
            continue
        ready.append({**item, "node_fields": fields})
    rejected_ids = {str(item.get("node_id") or "") for item in abstained}
    if rejected_ids:
        plan.add_edges = [
            item for item in plan.add_edges
            if str(item.get("target") or "") not in rejected_ids
        ]
        # Edge weights are assigned before teacher synthesis. If synthesis
        # abstains, remove the weight for the now-removed atomic child edge too.
        surviving_new_edge_keys: set[str] = set()
        for edge_item in plan.add_edges + plan.replace_edges:
            new_edge = edge_item.get("new_edge") or edge_item
            source = str(new_edge.get("source") or "")
            target = str(new_edge.get("target") or "")
            relation = normalize_edge_type(
                str(new_edge.get("relation") or "prereq")
            )
            if source and target:
                surviving_new_edge_keys.add(f"new:{source}->{target}:{relation}")
        plan.edge_weights = {
            key: value
            for key, value in plan.edge_weights.items()
            if not key.startswith("new:") or key in surviving_new_edge_keys
        }
    plan.add_nodes = ready
    return abstained


def _prepare_node_updates(
    graph: SkillGraph,
    plan: GraphEditPlan,
    *,
    positive_examples_by_node: dict[str, list[dict[str, Any]]] | None = None,
    success_guards_by_node: dict[str, dict[str, Any]] | None = None,
    case_examples_by_id: dict[str, dict[str, Any]] | None = None,
    failure_examples_by_node: dict[str, list[dict[str, Any]]] | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
    positive_example_cap: int = 10,
    failure_example_cap: int = 6,
) -> list[dict[str, Any]]:
    """Rewrite nodes before plan persistence and evidence consumption."""
    ready: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    positive_examples_by_node = positive_examples_by_node or {}
    success_guards_by_node = success_guards_by_node or {}
    failure_examples_by_node = failure_examples_by_node or {}
    case_examples_by_id = case_examples_by_id or {}
    working_graph = graph.copy()
    structural_context_by_node = _plan_structural_context_by_node(graph, plan)
    for item in plan.update_nodes:
        node_id = str(item["node_id"])
        positive_pool = list(positive_examples_by_node.get(node_id) or [])
        source_failures = [
            dict(case_examples_by_id[case_id])
            for case_id in item.get("source_case_ids") or []
            if case_id in case_examples_by_id
        ]
        complete_source_cases = [
            dict(evidence)
            for evidence in (item.get("evidence_items") or [])
            if isinstance(evidence, dict)
        ]
        failure_examples = complete_source_cases or [
            dict(example)
            for example in failure_examples_by_node.get(node_id, [])[:failure_example_cap]
        ] or select_failure_examples(source_failures, cap=failure_example_cap)
        positive_examples = select_positive_examples(
            node_id, positive_pool, failure_examples, cap=positive_example_cap
        )
        if node_id in success_guards_by_node:
            positive_examples.append(dict(success_guards_by_node[node_id]))
        scope = str(item.get("revision_scope") or "full")
        if scope == "execution_reinforcement":
            fields = reinforce_node_execution(
                working_graph, node_id, str(item["merged_revision"]),
                positive_examples=positive_examples,
                failure_examples=failure_examples, chat_fn=chat_fn,
                mode=mode, store=store,
            )
        elif scope == "when_to_use":
            fields = rewrite_node_when_to_use(
                working_graph,
                node_id,
                str(item["merged_revision"]),
                positive_examples=positive_examples,
                failure_examples=failure_examples,
                chat_fn=chat_fn,
                mode=mode,
                store=store,
            )
        elif scope == "rewrite":
            fields = rewrite_toxic_node(
                working_graph, node_id, str(item.get("toxic_text") or ""),
                str(item["merged_revision"]), positive_examples=positive_examples,
                failure_examples=failure_examples, chat_fn=chat_fn, mode=mode,
                store=store,
            )
        else:
            fields = rewrite_node(
                working_graph,
                node_id,
                str(item["merged_revision"]),
                positive_examples=positive_examples,
                failure_examples=failure_examples,
                chat_fn=chat_fn,
                mode=mode,
                store=store,
            )
        if fields is None:
            abstained.append(dict(item))
            continue
        fields = dict(fields)
        fields.pop("meaning", None)
        node = graph.nodes.get(node_id)
        if node is not None:
            unchanged = (
                fields.get("when_to_use", node.when_to_use) == node.when_to_use
                and fields.get("how_to_use", node.how_to_use) == node.how_to_use
                and fields.get("avoid", list(node.avoid)) == list(node.avoid)
            )
            if unchanged:
                continue
        ready.append({
            **item,
            "positive_pool_count": len(positive_pool),
            "positive_case_count": sum(
                str(example.get("task_type") or "") != "aggregate_success_guard"
                for example in positive_examples
            ),
            "positive_rejected_or_deferred_count": len(positive_pool) - sum(
                str(example.get("task_type") or "") != "aggregate_success_guard"
                for example in positive_examples
            ),
            "positive_case_ids": [
                str(example["case_id"]) for example in positive_examples
                if str(example.get("task_type") or "") != "aggregate_success_guard"
            ],
            "protected_success_count": int(
                (success_guards_by_node.get(node_id) or {}).get(
                    "protected_success_count", 0
                )
            ),
            "node_fields": fields,
        })
        updated = working_graph.nodes.get(node_id)
        if updated is not None:
            for key, value in fields.items():
                if key == "avoid":
                    updated.avoid = list(value)
                elif hasattr(updated, key):
                    setattr(updated, key, value)
    # Multiple operations on one node are not independently materializable:
    # later fields contain the earlier changes through working_graph. Collapse
    # them into the final combined node and audit that exact deployed meaning.
    combined: list[dict[str, Any]] = []
    by_node: dict[str, list[dict[str, Any]]] = {}
    for item in ready:
        by_node.setdefault(str(item["node_id"]), []).append(item)
    for node_id, items in by_node.items():
        final_item = dict(items[-1])
        source_case_ids = list(dict.fromkeys(
            str(case_id)
            for item in items
            for case_id in (item.get("source_case_ids") or [])
        ))
        evidence_by_case = {
            str(evidence.get("case_id") or f"anonymous:{index}"): dict(evidence)
            for index, evidence in enumerate(
                evidence
                for item in items
                for evidence in (item.get("evidence_items") or [])
                if isinstance(evidence, dict)
            )
        }
        positive_pool = list(positive_examples_by_node.get(node_id) or [])
        source_failures = [
            dict(case_examples_by_id[case_id])
            for case_id in source_case_ids
            if case_id in case_examples_by_id
        ]
        complete_source_cases = list(evidence_by_case.values())
        failure_examples = complete_source_cases or [
            dict(example)
            for example in failure_examples_by_node.get(node_id, [])[:failure_example_cap]
        ] or select_failure_examples(source_failures, cap=failure_example_cap)
        positive_examples = select_positive_examples(
            node_id, positive_pool, failure_examples, cap=positive_example_cap
        )
        if node_id in success_guards_by_node:
            positive_examples.append(dict(success_guards_by_node[node_id]))
        components = [{
            "revision_scope": item.get("revision_scope"),
            "toxic_text": item.get("toxic_text"),
            "merged_revision": item.get("merged_revision"),
            "source_case_ids": item.get("source_case_ids") or [],
        } for item in items]
        components.extend(
            dict(item) for item in structural_context_by_node.get(node_id, [])
        )
        # Preserve the cumulatively composed candidate here. Cross-node and
        # node/edge conflicts are resolved later by the true joint semantic
        # synthesizer, which sees the complete concrete edit component at once.
        deployed = working_graph.nodes[node_id]
        original = graph.nodes[node_id]
        reconciled_fields: dict[str, Any] = {}
        if deployed.when_to_use != original.when_to_use:
            reconciled_fields["when_to_use"] = deployed.when_to_use
        if deployed.how_to_use != original.how_to_use:
            reconciled_fields["how_to_use"] = deployed.how_to_use
        if list(deployed.avoid) != list(original.avoid):
            reconciled_fields["avoid"] = list(deployed.avoid)
        final_item["node_fields"] = reconciled_fields
        final_item.update({
            "revision_scope": "combined" if len(items) > 1 else items[0].get("revision_scope"),
            "component_updates": components,
            "source_case_ids": source_case_ids,
            "evidence_items": list(evidence_by_case.values()),
            "positive_pool_count": len(positive_pool),
            "positive_case_count": sum(
                str(example.get("task_type") or "") != "aggregate_success_guard"
                for example in positive_examples
            ),
            "positive_rejected_or_deferred_count": len(positive_pool) - sum(
                str(example.get("task_type") or "") != "aggregate_success_guard"
                for example in positive_examples
            ),
            "positive_case_ids": [
                str(example["case_id"]) for example in positive_examples
                if str(example.get("task_type") or "") != "aggregate_success_guard"
            ],
            "protected_success_count": int(
                (success_guards_by_node.get(node_id) or {}).get(
                    "protected_success_count", 0
                )
            ),
            "aggregate_support": len(source_case_ids),
            "support": max(int(item.get("support") or 0) for item in items),
        })
        combined.append(final_item)
    plan.update_nodes = combined
    return abstained


def _successful_node_examples(
    analyses: list[Any],
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Collect every successful, correct use with its complete source result."""
    result_by_id = {str(result.get("id") or ""): result for result in results}
    by_node: dict[str, list[dict[str, Any]]] = {}
    for analysis in analyses:
        if not analysis.success:
            continue
        result = result_by_id.get(str(analysis.case_id), {})
        actions = extract_actions(str(result.get("trajectory") or ""))
        trajectory = str(result.get("trajectory") or "")
        environment = str(result.get("environment") or "searchqa")
        example = {
            "case_id": str(analysis.case_id),
            "environment": environment,
            "task_type": str(result.get("task_type") or ""),
            "task_description": str(result.get("task_description") or ""),
            "action_path": actions,
            "n_turns": int(result.get("n_turns") or len(actions)),
            "semantic_reasoning_trace": str(
                result.get("semantic_reasoning_trace") or ""
            ),
            "semantic_reasoning_trace_status": str(
                result.get("semantic_reasoning_trace_status") or "not_requested"
            ),
        }
        question = (
            result.get("question") or result.get("instruction")
            or result.get("task_description") or ""
        )
        student_answer = (
            result.get("predicted_answer") or result.get("predicted_text")
            or result.get("final_answer") or result.get("response") or ""
        )
        reference_answer = (
            result.get("reference_answer") or result.get("gold_answer")
            or result.get("training_reference") or ""
        )
        if question:
            example["question"] = str(question)
        if student_answer:
            example["student_answer"] = str(student_answer)
        if reference_answer:
            example["reference_answer"] = str(reference_answer)
        if isinstance(result.get("graph_refs"), dict):
            example["graph_refs"] = dict(result["graph_refs"])
        evidence = trajectory or str(result.get("response") or "")
        example["evidence_excerpt"] = evidence
        example["source_result"] = deepcopy(result)
        for node_id in dict.fromkeys(str(node) for node in analysis.correct_nodes):
            by_node.setdefault(node_id, []).append(example)
    return by_node


def _plan_related_existing_node_ids(
    graph: SkillGraph, plan: GraphEditPlan,
) -> set[str]:
    """Nodes whose semantics participate in the planned composite edits."""
    related = {
        str(item.get("node_id") or "") for item in plan.update_nodes
    }
    related.update(
        str(item.get("parent_node") or "") for item in plan.add_nodes
    )
    for item in plan.add_edges:
        related.update((
            str(item.get("source") or ""), str(item.get("target") or ""),
        ))
    edge_map = _edge_map(graph)
    for item in plan.replace_edges:
        old = edge_map.get(str(item.get("old_edge") or ""))
        if old is not None:
            related.update((str(old.src), str(old.dst)))
        new_edge = item.get("new_edge") or item
        related.update((
            str(new_edge.get("source") or ""),
            str(new_edge.get("target") or ""),
        ))
    for edge_id in plan.delete_edges:
        edge = edge_map.get(str(edge_id))
        if edge is not None:
            related.update((str(edge.src), str(edge.dst)))
    related.update(str(node_id) for node_id in plan.delete_nodes)
    return {node_id for node_id in related if node_id in graph.nodes}


def _plan_structural_context_by_node(
    graph: SkillGraph, plan: GraphEditPlan,
) -> dict[str, list[dict[str, Any]]]:
    """Expose every incident edge change to same-node semantic reconciliation."""
    context: dict[str, list[dict[str, Any]]] = {}

    def add(nodes: tuple[str, ...], payload: dict[str, Any]) -> None:
        for node_id in nodes:
            if node_id in graph.nodes:
                context.setdefault(node_id, []).append(dict(payload))

    for item in plan.add_edges:
        source, target = str(item.get("source") or ""), str(item.get("target") or "")
        add((source, target), {
            "revision_scope": "connected_edge_add",
            "structural_edit": dict(item),
            "source_case_ids": list(item.get("source_case_ids") or []),
        })
    edge_map = _edge_map(graph)
    for item in plan.replace_edges:
        old = edge_map.get(str(item.get("old_edge") or ""))
        new_edge = item.get("new_edge") or item
        nodes = {
            str(new_edge.get("source") or ""),
            str(new_edge.get("target") or ""),
        }
        if old is not None:
            nodes.update((str(old.src), str(old.dst)))
        add(tuple(sorted(nodes)), {
            "revision_scope": "connected_edge_replace",
            "structural_edit": dict(item),
            "source_case_ids": list(new_edge.get("source_case_ids") or []),
        })
    for edge_id in plan.delete_edges:
        edge = edge_map.get(str(edge_id))
        if edge is not None:
            add((str(edge.src), str(edge.dst)), {
                "revision_scope": "connected_edge_delete",
                "structural_edit": {"edge_id": str(edge_id)},
                "source_case_ids": [],
            })
    return context


def _summarize_related_successes(
    graph: SkillGraph,
    examples_by_node: dict[str, list[dict[str, Any]]],
    related_node_ids: set[str],
    *,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
    chunk_size: int = 12,
    max_workers: int = 16,
) -> dict[str, dict[str, Any]]:
    """Explain every related right case and compress it into node guards.

    Raw right cases are never treated as generic praise. Each is converted into
    an observable protected condition/action, then all chunk invariants are
    supplied to node synthesis so every related success influences the joint
    edit even when only a few raw examples fit in the prompt.
    """
    from graphopt.json_utils import extract_json

    guards: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    size = max(1, int(chunk_size))
    system = """Summarize why successful graph-guided cases succeeded. For the named
node, use only the original task, answer/output, semantic reasoning trace, graph_usage,
and observed action path. Return strict JSON with keys cases and protected_invariants.
cases must contain exactly one item per input case with keys case_id, why_successful,
protected_condition, and protected_action. protected_invariants items must contain
condition, action, reason, and case_ids. Every input case_id must be cited by at least
one invariant. State observable semantics, not generic praise. These invariants will
constrain a joint graph edit, so preserve successful behavior without blocking a narrow
bad-case fix."""

    def load_completed_chunks(
        expected_work: list[tuple[str, int, list[dict[str, Any]]]],
    ) -> dict[tuple[str, int], dict[str, Any]]:
        """Recover only structurally complete chunks from this artifact store."""
        if store is None:
            return {}
        expected = {
            (node_id, start): tuple(str(item["case_id"]) for item in chunk)
            for node_id, start, chunk in expected_work
        }
        recovered: dict[tuple[str, int], dict[str, Any]] = {}
        for entry in store.versions("success_mechanism_chunk"):
            path = store.base_dir / str(entry.get("file") or "")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            inputs = dict(payload.get("inputs") or {})
            output = payload.get("outputs")
            key = (
                str(inputs.get("node_id") or ""),
                int(inputs.get("chunk_start", -1)),
            )
            case_ids = tuple(
                str(value) for value in (inputs.get("case_ids") or [])
            )
            if expected.get(key) != case_ids or not isinstance(output, dict):
                continue
            node_id = key[0]
            mechanisms = output.get("case_mechanisms") or []
            invariants = output.get("protected_invariants") or []
            mechanism_ids = {
                str(row.get("case_id") or "")
                for row in mechanisms if isinstance(row, dict)
            }
            invariant_ids = {
                str(case_id)
                for row in invariants if isinstance(row, dict)
                for case_id in (row.get("case_ids") or [])
            }
            if (
                output.get("case_id") != f"__all_related_successes__:{node_id}"
                or output.get("task_type") != "aggregate_success_guard"
                or int(output.get("protected_success_count") or -1) != len(case_ids)
                or tuple(
                    str(value)
                    for value in (output.get("protected_success_case_ids") or [])
                ) != case_ids
                or mechanism_ids != set(case_ids)
                or invariant_ids != set(case_ids)
            ):
                continue
            recovered[key] = output
        return recovered

    # All-success coverage can be much larger than the failure set (for example,
    # one high-frequency normalization node may have hundreds of right cases).
    # Chunk prompts are independent evidence summaries, so execute them in a
    # bounded pool and merge their outputs in deterministic node/chunk order.
    # This preserves exact coverage while preventing one slow HTTP request from
    # making the whole epoch look stalled.
    if mode == "teacher" and chat_fn is not None and int(max_workers) > 1:
        prepared: dict[str, list[dict[str, Any]]] = {}
        work: list[tuple[str, int, list[dict[str, Any]]]] = []
        for node_id in sorted(related_node_ids):
            examples = [dict(item) for item in examples_by_node.get(node_id) or []]
            deduped = {str(item.get("case_id") or ""): item for item in examples}
            deduped.pop("", None)
            examples = [deduped[key] for key in sorted(deduped)]
            if not examples:
                continue
            prepared[node_id] = examples
            for start in range(0, len(examples), size):
                work.append((node_id, start, examples[start : start + size]))
        if len(work) > 1:
            workers = min(max(1, int(max_workers)), len(work))
            completed = load_completed_chunks(work)
            pending = [
                item for item in work if (item[0], item[1]) not in completed
            ]
            print(
                f"[graphopt success guards] nodes={len(prepared)} "
                f"chunks={len(work)} workers={workers} "
                f"cases={sum(len(values) for values in prepared.values())} "
                f"reused={len(completed)} pending={len(pending)}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _summarize_related_successes,
                        graph, {node_id: chunk}, {node_id},
                        chat_fn=chat_fn, mode=mode, store=None,
                        chunk_size=size, max_workers=1,
                    ): (node_id, start, chunk)
                    for node_id, start, chunk in pending
                }
                for count, future in enumerate(
                    as_completed(futures), start=len(completed) + 1
                ):
                    node_id, start, chunk = futures[future]
                    chunk_guard = future.result()[node_id]
                    completed[(node_id, start)] = chunk_guard
                    print(
                        f"[graphopt success guards] completed={count}/{len(work)} "
                        f"node={node_id} cases={len(chunk)}",
                        flush=True,
                    )
                    if store is not None:
                        store.save(
                            "success_mechanism_chunk",
                            stage="success_mechanism_summary_chunk",
                            inputs={
                                "node_id": node_id,
                                "chunk_start": start,
                                "case_ids": [
                                    str(item["case_id"]) for item in chunk
                                ],
                            },
                            outputs=chunk_guard,
                        )
            combined_guards: dict[str, dict[str, Any]] = {}
            combined_artifacts: dict[str, Any] = {}
            for node_id, examples in prepared.items():
                chunk_guards = [
                    completed[(node_id, start)]
                    for start in range(0, len(examples), size)
                ]
                mechanisms = [
                    dict(row) for guard in chunk_guards
                    for row in (guard.get("case_mechanisms") or [])
                ]
                invariants = [
                    dict(row) for guard in chunk_guards
                    for row in (guard.get("protected_invariants") or [])
                ]
                case_ids = [str(item["case_id"]) for item in examples]
                guard = {
                    "case_id": f"__all_related_successes__:{node_id}",
                    "task_type": "aggregate_success_guard",
                    "task_description": (
                        f"Preserve all {len(examples)} successful uses related to {node_id}."
                    ),
                    "evidence_excerpt": json.dumps(invariants, ensure_ascii=False),
                    "protected_success_count": len(examples),
                    "protected_success_case_ids": case_ids,
                    "protected_invariants": invariants,
                    "case_mechanisms": mechanisms,
                    "protected_success_cases": examples,
                }
                combined_guards[node_id] = guard
                combined_artifacts[node_id] = {
                    "node_id": node_id,
                    "n_cases": len(examples),
                    "case_mechanisms": mechanisms,
                    "protected_invariants": invariants,
                    "parallel_chunks": len(chunk_guards),
                }
            if store is not None and combined_artifacts:
                store.save(
                    "success_mechanism_summaries",
                    stage="success_mechanism_summary",
                    inputs={
                        "related_node_ids": sorted(related_node_ids),
                        "workers": workers,
                        "chunks": len(work),
                    },
                    outputs=combined_artifacts,
                )
            return combined_guards

    for node_id in sorted(related_node_ids):
        examples = [dict(item) for item in examples_by_node.get(node_id) or []]
        deduped = {str(item.get("case_id") or ""): item for item in examples}
        deduped.pop("", None)
        examples = [deduped[key] for key in sorted(deduped)]
        if not examples:
            continue
        case_rows: list[dict[str, Any]] = []
        invariants: list[dict[str, Any]] = []
        attempts_log: list[dict[str, Any]] = []
        for start in range(0, len(examples), size):
            chunk = examples[start : start + size]
            expected_ids = [str(item["case_id"]) for item in chunk]
            compact = [{
                key: item.get(key) for key in (
                    "case_id", "task_type", "task_description", "question",
                    "student_answer", "reference_answer", "action_path",
                    "semantic_reasoning_trace", "graph_refs", "evidence_excerpt",
                    "source_result",
                ) if item.get(key) not in (None, "", [], {})
            } for item in chunk]
            parsed: dict[str, Any] | None = None
            if mode == "teacher" and chat_fn is not None:
                user = exact_reference_json({
                    "node": graph.nodes[node_id].to_dict(),
                    "successful_cases": compact,
                })
                for attempt in range(1, 3):
                    response = ""
                    usage: Any = None
                    try:
                        response, usage = chat_fn(
                            system=system, user=user, max_completion_tokens=4096,
                            retries=2, stage="success_mechanism_summary",
                        )
                        obj = extract_json(response)
                        if not isinstance(obj, dict) or set(obj) != {
                            "cases", "protected_invariants",
                        }:
                            raise ValueError("success summary requires cases and protected_invariants")
                        rows = obj["cases"]
                        inv = obj["protected_invariants"]
                        if not isinstance(rows, list) or not isinstance(inv, list):
                            raise ValueError("success summary fields must be lists")
                        required_case = {
                            "case_id", "why_successful", "protected_condition",
                            "protected_action",
                        }
                        if (
                            {str(row.get("case_id") or "") for row in rows
                             if isinstance(row, dict)} != set(expected_ids)
                            or len(rows) != len(expected_ids)
                            or any(not isinstance(row, dict) or set(row) != required_case for row in rows)
                            or any(
                                not isinstance(row[key], str) or not row[key].strip()
                                for row in rows for key in required_case - {"case_id"}
                            )
                        ):
                            raise ValueError("success summary must explain every case exactly once")
                        required_inv = {"condition", "action", "reason", "case_ids"}
                        if any(not isinstance(row, dict) or set(row) != required_inv for row in inv):
                            raise ValueError("protected invariant has invalid fields")
                        cited: set[str] = set()
                        for row in inv:
                            ids = row["case_ids"]
                            if (
                                not isinstance(ids, list) or not ids
                                or any(str(case_id) not in expected_ids for case_id in ids)
                                or any(
                                    not isinstance(row[key], str) or not row[key].strip()
                                    for key in ("condition", "action", "reason")
                                )
                            ):
                                raise ValueError("protected invariant evidence is invalid")
                            cited.update(map(str, ids))
                        if cited != set(expected_ids):
                            raise ValueError("every successful case must support an invariant")
                        parsed = obj
                        attempts_log.append({
                            "chunk_start": start, "attempt": attempt, "usage": usage,
                            "status": "complete",
                        })
                        break
                    except Exception as exc:
                        attempts_log.append({
                            "chunk_start": start, "attempt": attempt,
                            "usage": usage, "response": response or None,
                            "status": "invalid", "error": str(exc),
                        })
            if parsed is None:
                fallback_rows = []
                for item in chunk:
                    trace = str(item.get("semantic_reasoning_trace") or "").strip()
                    evidence = str(item.get("evidence_excerpt") or "").strip()
                    action = "; ".join(map(str, item.get("action_path") or []))
                    fallback_rows.append({
                        "case_id": str(item["case_id"]),
                        "why_successful": (trace or evidence or "The recorded node use produced the correct result.")[-800:],
                        "protected_condition": str(item.get("task_description") or "the recorded observable case state")[:500],
                        "protected_action": (action or "preserve the recorded successful decision path")[:500],
                    })
                parsed = {
                    "cases": fallback_rows,
                    "protected_invariants": [{
                        "condition": "the recorded successful conditions in these cited cases",
                        "action": "preserve their successful node-guided decision paths",
                        "reason": "deterministic fallback after no valid semantic summary",
                        "case_ids": expected_ids,
                    }],
                }
            case_rows.extend(dict(row) for row in parsed["cases"])
            invariants.extend(dict(row) for row in parsed["protected_invariants"])

        guard = {
            "case_id": f"__all_related_successes__:{node_id}",
            "task_type": "aggregate_success_guard",
            "task_description": (
                f"Preserve all {len(examples)} successful uses related to {node_id}."
            ),
            "evidence_excerpt": json.dumps(invariants, ensure_ascii=False),
            "protected_success_count": len(examples),
            "protected_success_case_ids": [str(item["case_id"]) for item in examples],
            "protected_invariants": invariants,
            "case_mechanisms": case_rows,
            "protected_success_cases": examples,
        }
        guards[node_id] = guard
        artifacts[node_id] = {
            "node_id": node_id, "n_cases": len(examples),
            "case_mechanisms": case_rows, "protected_invariants": invariants,
            "attempts": attempts_log,
        }
    if store is not None and artifacts:
        store.save(
            "success_mechanism_summaries", stage="success_mechanism_summary",
            inputs={"related_node_ids": sorted(related_node_ids)},
            outputs=artifacts,
        )
    return guards


def _execution_reinforcements(
    analyses: list[Any],
    cache: EvolutionCache,
    *,
    graph: SkillGraph | None = None,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
    evidence_cap: int = 64,
    include_cached_evidence: bool = True,
) -> list[MergedProposal]:
    """Build execution-semantic clusters from the current train batch.

    Cached proposals are accepted only for legacy callers that explicitly ask
    for them. Formal grouped training validates a one-case candidate instead
    of accumulating train votes across graph states.
    """
    evidence_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    current_ids: set[str] = set()
    current_parents: set[str] = set()

    def add(raw: dict[str, Any], *, current: bool) -> None:
        parent = str(raw.get("target_node") or "")
        case_id = str(raw.get("case_id") or "")
        proposal = str(raw.get("proposal") or "").strip()
        if not parent or not case_id or not proposal:
            return
        evidence_by_case[(parent, case_id)] = {
            "target_node": parent,
            "case_id": case_id,
            "proposal": proposal,
            "reason": str(raw.get("reason") or ""),
            "observable_state": str(raw.get("observable_state") or ""),
            "bad_action": str(raw.get("bad_action") or ""),
            "better_action": str(raw.get("better_action") or ""),
            "is_current": current,
        }
        if current:
            current_ids.add(case_id)
            current_parents.add(parent)

    for node_id, entry in cache.nodes.items() if include_cached_evidence else []:
        for item in entry.execution_reinforcement_proposals:
            evidence_items = list(item.get("evidence_items") or [])
            if evidence_items:
                for evidence in evidence_items:
                    add(
                        {
                            **dict(evidence),
                            "target_node": str(
                                evidence.get("target_node") or node_id
                            ),
                        },
                        current=False,
                    )
                continue
            # Backward-compatible expansion of the old parent-level cache.
            case_ids = [
                str(value) for value in (item.get("source_case_ids") or [])
                if str(value)
            ]
            reminders = [
                str(value).strip() for value in (item.get("proposals") or [])
                if str(value).strip()
            ]
            if not reminders and item.get("proposal"):
                reminders = [str(item.get("proposal")).strip()]
            for index, case_id in enumerate(case_ids):
                add(
                    {
                        **item,
                        "target_node": node_id,
                        "case_id": case_id,
                        "proposal": (
                            reminders[index]
                            if index < len(reminders)
                            else (reminders[-1] if reminders else "")
                        ),
                    },
                    current=False,
                )

    for analysis in analyses:
        targets = set(analysis.execution_lapse_nodes)
        for proposal in analysis.node_revision_proposals:
            if proposal.target_node not in targets:
                continue
            add(proposal.to_dict(), current=True)

    by_parent: dict[str, list[dict[str, Any]]] = {}
    for (parent, _), item in evidence_by_case.items():
        if parent in current_parents:
            by_parent.setdefault(parent, []).append(item)
    for values in by_parent.values():
        values.sort(
            key=lambda item: (
                0 if item.get("is_current") else 1,
                str(item.get("case_id") or ""),
            )
        )

    # One bounded merge task gets evidence from several parents without letting
    # the largest bucket erase smaller but potentially coherent clusters.
    parents = sorted(
        by_parent,
        key=lambda parent: (-len(by_parent[parent]), parent),
    )
    selected: list[dict[str, Any]] = []
    cursor = 0
    cap = max(1, int(evidence_cap))
    while len(selected) < cap:
        advanced = False
        for parent in parents:
            values = by_parent[parent]
            if cursor < len(values):
                selected.append(values[cursor])
                advanced = True
                if len(selected) >= cap:
                    break
        if not advanced:
            break
        cursor += 1

    merged = semantic_merge_execution_lapses(
        selected,
        graph,
        chat_fn=chat_fn,
        mode=mode,
        store=store,
    )
    # A cluster needs one current case so its exact G0 baseline can be probed.
    return [
        proposal
        for proposal in merged
        if set(proposal.source_case_ids) & current_ids
    ]



def _edit_key(e: GraphEdit) -> tuple:
    if e.op == "update_node":
        return (e.op, e.node_id or "")
    if e.op == "add_node":
        return (e.op, _sig(e.meaning or e.how_to_use or e.name or ""))
    if e.op in ("add_edge", "delete_edge", "change_edge_type"):
        return (e.op, e.src or "", e.dst or "", normalize_edge_type(e.edge_type or ""))
    if e.op == "delete_node":
        return (e.op, e.node_id or "")
    return (e.op, id(e))


def _sig(text: str) -> str:
    return " ".join((text or "").lower().split())[:200]


def _is_empty_update_node(e: GraphEdit) -> bool:
    """Match the update payload contract enforced by optimizer.skill.apply_edit."""
    return e.op == "update_node" and not any(
        (
            e.when_to_use,
            e.how_to_use,
            e.description,
            e.category,
            e.avoid is not None,
        )
    )


def _dedupe_edits(edits: list[GraphEdit]) -> list[GraphEdit]:
    """Drop semantic no-ops and exact duplicates; retain substantive changes."""
    positions: dict[tuple, int] = {}
    out: list[GraphEdit] = []
    for e in edits:
        # Teacher planning metadata (for example ``reasoning`` and
        # ``source_type``) does not mutate a node. Letting such a metadata-only
        # update reach atomic materialization aborts the whole otherwise-valid
        # patch, so normalize it away here. ``avoid=[]`` is intentionally not
        # empty: it explicitly clears the node's avoid list.
        if _is_empty_update_node(e):
            continue
        key = _edit_key(e)
        if key in positions:
            if e.op == "update_node":
                # A later same-node update contains the cumulative final fields.
                out[positions[key]] = e
            continue
        positions[key] = len(out)
        out.append(e)
    return out


def _joint_patch_components(edits: list[GraphEdit]) -> list[list[int]]:
    """Return transitive semantic components by source case or incident node."""
    semantic_indices = [
        index for index, edit in enumerate(edits)
        if not (
            edit.op == "add_edge"
            and str(edit.reasoning or "").startswith("weight refresh ")
        )
    ]
    parent = {index: index for index in semantic_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    first_group: dict[str, int] = {}
    first_source: dict[str, int] = {}
    first_node: dict[str, int] = {}
    for index in semantic_indices:
        edit = edits[index]
        if edit.group_id:
            if edit.group_id in first_group:
                union(first_group[edit.group_id], index)
            else:
                first_group[edit.group_id] = index
        for case_id in dict.fromkeys(map(str, edit.source_case_ids)):
            if not case_id:
                continue
            if case_id in first_source:
                union(first_source[case_id], index)
            else:
                first_source[case_id] = index
        for node_id in {
            str(value) for value in (edit.node_id, edit.src, edit.dst)
            if str(value or "")
        }:
            if node_id in first_node:
                union(first_node[node_id], index)
            else:
                first_node[node_id] = index
    components: dict[int, list[int]] = {}
    for index in semantic_indices:
        components.setdefault(find(index), []).append(index)
    return list(components.values())


def _requires_joint_semantic_synthesis(
    edits: list[GraphEdit], indices: list[int],
) -> bool:
    # Every concrete edit must cross the same failure/success semantic boundary.
    # Skipping a one-edit/one-case component let broad base-node rewrites bypass
    # the only stage that can convert them into narrow specialists.
    del edits
    return bool(indices)


def _joint_component_id(edits: list[GraphEdit], indices: list[int]) -> str:
    payload = json.dumps(
        [edits[index].to_dict() for index in indices],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return "joint-semantic-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _joint_component_payload(
    graph: SkillGraph,
    edits: list[GraphEdit],
    indices: list[int],
    success_guards_by_node: dict[str, dict[str, Any]],
    available_new_node_ids: list[str],
    prior_local_gate_feedback: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str], list[str], list[int]]:
    incident_nodes = sorted({
        str(value) for index in indices
        for value in (edits[index].node_id, edits[index].src, edits[index].dst)
        if str(value or "")
    })
    failure_ids = sorted({
        str(case_id) for index in indices
        for case_id in edits[index].source_case_ids if str(case_id)
    })
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for index in indices:
        for position, evidence in enumerate(edits[index].evidence_items):
            case_id = str(evidence.get("case_id") or "")
            key = case_id or f"edit-{index}-evidence-{position}"
            evidence_by_id.setdefault(key, dict(evidence))
    guard_nodes = sorted(
        node_id for node_id in incident_nodes if node_id in success_guards_by_node
    )
    original_edges = [
        edge.to_dict() for edge in graph.edges
        if edge.src in incident_nodes or edge.dst in incident_nodes
    ]
    high_exposure_profiles: list[dict[str, Any]] = []
    for index in indices:
        edit = edits[index]
        node_id = str(edit.node_id or "")
        if edit.op != "update_node" or node_id not in success_guards_by_node:
            continue
        guard = success_guards_by_node[node_id]
        protected_ids = {
            str(case_id)
            for case_id in (guard.get("protected_success_case_ids") or [])
            if str(case_id)
        }
        protected_ids.update(
            str(item.get("case_id") or "")
            for item in (guard.get("case_mechanisms") or [])
            if str(item.get("case_id") or "")
        )
        protected_count = max(
            len(protected_ids),
            int(guard.get("protected_success_count") or guard.get("n_cases") or 0),
        )
        source_ids = sorted({
            str(case_id) for case_id in edit.source_case_ids if str(case_id)
        })
        if source_ids and protected_count >= max(12, 3 * len(source_ids)):
            high_exposure_profiles.append({
                "edit_index": index,
                "node_id": node_id,
                "failure_support_count": len(source_ids),
                "failure_case_ids": source_ids,
                "protected_success_count": protected_count,
                "review_guidance": (
                    "Prefer a field-local additive revision that preserves every "
                    "protected invariant. Use a specialist only when the evidence "
                    "shows a genuine conditional conflict that cannot be expressed "
                    "safely in the existing node."
                ),
            })
    high_exposure_indices = [row["edit_index"] for row in high_exposure_profiles]
    relevant_prior_feedback = [
        row for row in (prior_local_gate_feedback or [])
        if set(map(str, row.get("incident_node_ids") or [])).intersection(incident_nodes)
    ]
    payload = {
        "original_subgraph": {
            "nodes": {
                node_id: graph.nodes[node_id].to_dict()
                for node_id in incident_nodes if node_id in graph.nodes
            },
            "incident_edges": original_edges,
        },
        "proposed_edits": [
            {"edit_index": index, **edits[index].to_dict()} for index in indices
        ],
        "complete_failure_evidence": list(evidence_by_id.values()),
        "failure_case_ids": failure_ids,
        "all_related_success_guards": {
            node_id: success_guards_by_node[node_id] for node_id in guard_nodes
        },
        "high_exposure_update_policy": {
            "review_edit_indices": high_exposure_indices,
            "profiles": high_exposure_profiles,
            "policy": (
                "high exposure requires stronger success-preservation review, "
                "but does not prohibit a grounded field-local base-node update"
            ),
        },
        "prior_rejected_local_gate_attempts": relevant_prior_feedback,
        "proposed_add_node_ids": sorted({
            str(edits[index].node_id) for index in indices
            if edits[index].op == "add_node" and str(edits[index].node_id)
        }),
        "available_specialist_node_ids": list(available_new_node_ids),
        "available_additional_specialist_node_ids": list(available_new_node_ids),
    }
    return payload, failure_ids, guard_nodes, high_exposure_indices


def _validate_joint_semantic_response(
    obj: Any,
    *,
    graph: SkillGraph,
    edits: list[GraphEdit],
    indices: list[int],
    failure_ids: list[str],
    guard_nodes: list[str],
    success_guard_case_ids: list[str],
    available_new_node_ids: list[str],
    high_exposure_update_indices: list[int],
) -> tuple[dict[int, GraphEdit], list[GraphEdit], dict[str, Any]]:
    required = {
        "decision", "reason", "failure_case_ids", "protected_guard_node_ids",
        "conflict_resolutions", "resolved_edits", "additional_edits",
    }
    if not isinstance(obj, dict) or set(obj) != required:
        raise ValueError("joint synthesis output has the wrong top-level schema")
    raw_decision = str(obj["decision"] or "").upper()
    decision = "APPLY" if raw_decision == "KEEP" else raw_decision
    if decision not in {"APPLY", "ABSTAIN"}:
        raise ValueError("joint synthesis decision must be APPLY or ABSTAIN")
    if not isinstance(obj["reason"], str) or not obj["reason"].strip():
        raise ValueError("joint synthesis reason must be non-empty")
    if obj["failure_case_ids"] != failure_ids:
        raise ValueError("joint synthesis must acknowledge every failure case exactly")
    if obj["protected_guard_node_ids"] != guard_nodes:
        raise ValueError(
            "joint synthesis must copy required_protected_guard_node_ids exactly; "
            f"expected={guard_nodes!r}, "
            f"received={obj['protected_guard_node_ids']!r}"
        )
    resolutions = obj["conflict_resolutions"]
    if not isinstance(resolutions, list):
        raise ValueError("conflict_resolutions must be a list")
    allowed_evidence = set(failure_ids) | set(success_guard_case_ids)
    for row in resolutions:
        if not isinstance(row, dict) or set(row) != {
            "issue", "resolution", "evidence_case_ids",
        }:
            raise ValueError("invalid conflict resolution row")
        if (
            not isinstance(row["issue"], str) or not row["issue"].strip()
            or not isinstance(row["resolution"], str) or not row["resolution"].strip()
            or not isinstance(row["evidence_case_ids"], list)
        ):
            raise ValueError("invalid conflict resolution evidence")
        invalid_evidence = sorted({
            str(case_id) for case_id in row["evidence_case_ids"]
            if str(case_id) not in allowed_evidence
        })
        if invalid_evidence:
            raise ValueError(
                "invalid conflict resolution evidence IDs: "
                + ", ".join(invalid_evidence)
                + "; use exact IDs from allowed_conflict_evidence_case_ids"
            )
    rows = obj["resolved_edits"]
    if not isinstance(obj["additional_edits"], list):
        raise ValueError("additional_edits must be a list")
    row_fields = {
        "edit_index", "keep", "when_to_use", "how_to_use", "avoid", "rationale",
    }
    if (
        not isinstance(rows, list) or len(rows) != len(indices)
        or {int(row.get("edit_index", -1)) for row in rows if isinstance(row, dict)}
        != set(indices)
        or any(not isinstance(row, dict) or set(row) != row_fields for row in rows)
    ):
        raise ValueError("joint synthesis must resolve every edit index exactly once")
    by_index = {int(row["edit_index"]): row for row in rows}
    # Exposure is advisory evidence for semantic review, not an edit-type ban.
    # Protected successes and paired Gates decide whether a field-local update is safe.
    del high_exposure_update_indices
    kept: dict[int, GraphEdit] = {}
    kept_add_nodes: set[str] = set()
    for index in indices:
        edit = edits[index]
        row = by_index[index]
        if not isinstance(row["keep"], bool):
            raise ValueError("resolved edit keep must be boolean")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("every resolved edit needs a rationale")
        node_fields = (row["when_to_use"], row["how_to_use"], row["avoid"])
        if not row["keep"]:
            if any(value is not None for value in node_fields):
                raise ValueError("dropped edits must use null node fields")
            continue
        resolved = deepcopy(edit)
        if edit.op in {"add_node", "update_node"}:
            when, how, avoid = node_fields
            if (
                not isinstance(when, str)
                or not isinstance(how, str) or not how.strip()
                or not isinstance(avoid, list)
                or any(not isinstance(value, str) for value in avoid)
                or (edit.op == "add_node" and not when.strip())
            ):
                raise ValueError(
                    "kept node edits need type-valid complete when/how/avoid fields; "
                    "an update may preserve an originally empty when_to_use"
                )
            if edit.op == "update_node":
                original = graph.nodes.get(edit.node_id)
                if original is None:
                    raise ValueError("joint synthesis update target is missing")
                if (
                    original.when_to_use.strip()
                    and original.when_to_use.strip() not in when
                ):
                    when = (
                        original.when_to_use.rstrip()
                        + "\nJoint refinement: " + when.strip()
                    ).strip()
                if (
                    original.how_to_use.strip()
                    and original.how_to_use.strip() not in how
                ):
                    how = (
                        original.how_to_use.rstrip()
                        + "\nJoint refinement: " + how.strip()
                    ).strip()
                avoid = list(dict.fromkeys([*original.avoid, *avoid]))
            else:
                kept_add_nodes.add(edit.node_id)
            resolved.when_to_use = when.strip()
            resolved.how_to_use = how.strip()
            resolved.avoid = list(avoid)
        elif any(value is not None for value in node_fields):
            raise ValueError("edge/delete edits must use null node fields")
        resolved.group_id = ""
        resolved.reasoning = (
            str(resolved.reasoning or "").rstrip()
            + " | joint_semantic_resolution: " + row["rationale"].strip()
        ).strip(" |")
        kept[index] = resolved
    if decision == "ABSTAIN":
        if kept or obj["additional_edits"]:
            raise ValueError("ABSTAIN cannot keep or add edits")
        return {}, [], obj
    available_nodes = set(graph.nodes) | kept_add_nodes
    for edit in kept.values():
        if edit.op == "add_edge" and (edit.src not in available_nodes or edit.dst not in available_nodes):
            raise ValueError("joint synthesis kept an edge with a dropped endpoint")

    additional_rows = obj["additional_edits"]
    additional_fields = {
        "op", "node_id", "name", "src", "dst", "edge_type",
        "when_to_use", "how_to_use", "avoid", "rationale",
        "source_case_ids",
    }
    if (
        not isinstance(additional_rows, list) or len(additional_rows) > 4
        or any(
            not isinstance(row, dict) or set(row) != additional_fields
            for row in additional_rows
        )
    ):
        raise ValueError("additional_edits must be a bounded strict edit list")
    evidence_by_case = {
        str(item.get("case_id") or ""): dict(item)
        for index in indices for item in edits[index].evidence_items
        if str(item.get("case_id") or "")
    }
    additional: list[GraphEdit] = []
    additional_node_ids: set[str] = set()
    allowed_ids = set(available_new_node_ids)
    for row in additional_rows:
        op = str(row["op"] or "")
        source_ids = list(map(str, row["source_case_ids"] or []))
        if not source_ids:
            raise ValueError("additional edit needs at least one failure evidence ID")
        duplicate_source_ids = sorted({
            case_id for case_id in source_ids if source_ids.count(case_id) > 1
        })
        if duplicate_source_ids:
            raise ValueError(
                "additional edit has duplicate failure evidence IDs: "
                + ", ".join(duplicate_source_ids)
            )
        invalid_source_ids = sorted(set(source_ids) - set(failure_ids))
        if invalid_source_ids:
            raise ValueError(
                "additional edit has invalid failure evidence IDs: "
                + ", ".join(invalid_source_ids)
                + "; use exact IDs from allowed_conflict_evidence_case_ids.failure"
            )
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("additional edit needs a non-empty rationale")
        if op == "add_node":
            node_id = str(row["node_id"] or "")
            if node_id not in allowed_ids or node_id in additional_node_ids:
                raise ValueError("specialist node must use one supplied unused ID")
            if any(row[key] not in ("", None) for key in ("src", "dst")):
                raise ValueError("additional node cannot contain edge fields")
            redundant_edge_type = str(row["edge_type"] or "").strip()
            if (
                redundant_edge_type
                and normalize_edge_type(redundant_edge_type)
                not in {"prereq", "enhance"}
            ):
                raise ValueError("additional node contains an invalid redundant edge type")
            if (
                not isinstance(row["name"], str) or not row["name"].strip()
                or not isinstance(row["when_to_use"], str) or not row["when_to_use"].strip()
                or not isinstance(row["how_to_use"], str) or not row["how_to_use"].strip()
                or not isinstance(row["avoid"], list)
                or any(not isinstance(value, str) for value in row["avoid"])
            ):
                raise ValueError("additional specialist node needs complete semantics")
            additional_node_ids.add(node_id)
            additional.append(GraphEdit(
                op="add_node", node_id=node_id, name=row["name"].strip(),
                when_to_use=row["when_to_use"].strip(),
                how_to_use=row["how_to_use"].strip(), avoid=list(row["avoid"]),
                category="learned", source_type="failure",
                edit_kind="joint_specialist", rationale=row["rationale"].strip(),
                reasoning="joint semantic specialist",
                source_case_ids=source_ids,
                evidence_items=[
                    evidence_by_case[case_id] for case_id in source_ids
                    if case_id in evidence_by_case
                ],
            ))
        elif op != "add_edge":
            raise ValueError("additional edit op must be add_node or add_edge")
    all_available_nodes = available_nodes | additional_node_ids
    for row in additional_rows:
        if str(row["op"] or "") != "add_edge":
            continue
        source_ids = list(map(str, row["source_case_ids"] or []))
        src, dst = str(row["src"] or ""), str(row["dst"] or "")
        edge_type = normalize_edge_type(str(row["edge_type"] or ""))
        edge_when = row["when_to_use"]
        edge_how = row["how_to_use"]
        edge_avoid = row["avoid"]
        if (
            row["node_id"] not in ("", None) or row["name"] not in ("", None)
            or edge_when is not None and not isinstance(edge_when, str)
            or edge_how is not None and not isinstance(edge_how, str)
            or edge_avoid is not None and (
                not isinstance(edge_avoid, list)
                or any(not isinstance(value, str) for value in edge_avoid)
            )
            or src not in all_available_nodes or dst not in all_available_nodes
            or src == dst or edge_type not in {"prereq", "enhance"}
        ):
            raise ValueError("additional specialist edge is invalid")
        activation_semantics = [
            value.strip() for value in (edge_when, edge_how)
            if isinstance(value, str) and value.strip()
        ]
        activation_semantics.extend(
            value.strip() for value in (edge_avoid or []) if value.strip()
        )
        additional.append(GraphEdit(
            op="add_edge", src=src, dst=dst, edge_type=edge_type, w=0.7,
            source_type="failure", edit_kind="joint_specialist",
            rationale=str(row["rationale"]).strip(),
            reasoning=(
                "joint semantic specialist activation"
                + (": " + " | ".join(activation_semantics) if activation_semantics else "")
            ),
            source_case_ids=source_ids,
            evidence_items=[
                evidence_by_case[case_id] for case_id in source_ids
                if case_id in evidence_by_case
            ],
        ))
    for node_id in additional_node_ids:
        if not any(
            edit.op == "add_edge" and node_id in {edit.src, edit.dst}
            for edit in additional
        ):
            raise ValueError("every specialist node requires an activation edge")
    if not kept and not additional:
        raise ValueError("APPLY must keep or add at least one edit")
    return kept, additional, obj


def synthesize_joint_semantic_patch(
    graph: SkillGraph,
    patch: GraphPatch,
    success_guards_by_node: dict[str, dict[str, Any]],
    *,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
    prior_local_gate_feedback: list[dict[str, Any]] | None = None,
    max_component_edits: int = 0,
) -> tuple[GraphPatch, dict[str, Any]]:
    """Regenerate every multi-edit component from failures, successes, nodes, and edges."""
    components = _joint_patch_components(patch.edits)
    audit: dict[str, Any] = {
        "schema_version": "graphopt-joint-semantic-synthesis-v3",
        "policy": "one_component_one_joint_generation_with_full_failure_and_success_guards",
        "components": [],
    }
    if mode != "teacher" or chat_fn is None:
        audit["status"] = "template_passthrough"
        return GraphPatch(reasoning=patch.reasoning, edits=list(patch.edits)), audit
    from graphopt.json_utils import extract_json

    system = """You are the final GraphOpt joint semantic synthesizer. One component contains
all proposed node edits, new specialist nodes, and incident edge edits that share a source
case or touch a common graph node. Read the original subgraph, every complete failure packet,
and every all-related-success guard together. Return one coherent component, not independent
per-edit opinions. Preserve original node semantics verbatim. If failures require opposite
actions, write observable conditional branches. Keep a specialist and its activation edge
together. Drop an edge or edit that contradicts the resolved node meaning or protected success.
Every add_node ID already present in proposed_edits is valid and may be kept with that exact ID;
it does not need to appear in available_specialist_node_ids. Keep its supplied activation edge
when the node remains useful. available_specialist_node_ids and
available_additional_specialist_node_ids are identical reserves used only when creating a new
add_node inside additional_edits. Preserve coverage of every distinct evidence-backed reusable error family in the component; do not drop one merely because another family has more support. Prefer preserving an evidence-grounded update_node when it clarifies a specific existing field
(when_to_use, how_to_use, or avoid) without contradicting protected successes. High exposure is
a reason to inspect all protected invariants carefully, not a reason to force a child node.
When two evidence-backed semantics genuinely require incompatible behavior under distinguishable
observable conditions, and one conditional branch cannot safely express them in the existing
node, use one reserved additional specialist node ID and add its activation edge. Do not create
a specialist merely to avoid editing a highly used base node. Prior rejected Local Gate attempts
are measured counterexamples: do not repeat their tested semantics.
A new specialist without an edge, or an edge without failure evidence, is invalid.
ABSTAIN only when no evidence-grounded observable separator can resolve a real contradiction.
Never invent node IDs, endpoints, facts, or extra edits.

Never write a concrete training-case identifier into any model-visible graph text, including name, when_to_use, how_to_use, avoid, or rationale. Describe the reusable semantic evidence instead. Exact identifiers belong only in failure_case_ids, source_case_ids, and evidence_case_ids.

Return strict JSON with exactly: decision, reason, failure_case_ids,
protected_guard_node_ids, conflict_resolutions, resolved_edits, additional_edits. decision must be exactly "APPLY" when returning a resolved candidate or "ABSTAIN" when no evidence-grounded resolution exists. Never use KEEP, ACCEPT, REJECT, or another label. Copy the supplied sorted
failure_case_ids exactly, and copy protected_guard_node_ids exactly from
required_protected_guard_node_ids; do not infer guard nodes from original_subgraph.
conflict_resolutions contains objects
with issue, resolution, evidence_case_ids. Each evidence_case_ids list may use only the supplied failure case IDs or case IDs inside all_related_success_guards; include both sides when resolving a failure-versus-success conflict. Use only exact IDs copied from allowed_conflict_evidence_case_ids. Never invent a placeholder,
sentinel, aggregate label, or synthetic ID such as __all_related_successes__. For a protected
success side, cite one or more representative real success case IDs from that explicit list;
the complete all_related_success_guards still constrains the semantic resolution.
resolved_edits contains exactly one row for each input edit_index with edit_index, keep,
when_to_use, how_to_use, avoid, rationale. For a kept
add_node or update_node, return the complete final when_to_use/how_to_use/avoid. An update_node may return an empty when_to_use string only when that original node field is already empty; add_node when_to_use must be non-empty. The runtime deterministically prepends any missing original when_to_use/how_to_use text and unions original avoid entries, so focus the returned complete fields on an observable, non-contradictory refinement rather than risking accidental deletion of base semantics. For edge edits,
delete_node, or any dropped edit, set all three node fields to null. Structural endpoints and
relation types of supplied edits are fixed; use keep=false to remove one. additional_edits may
contain at most four strict objects with op, node_id, name, src, dst, edge_type, when_to_use,
how_to_use, avoid, rationale, source_case_ids. Only add_node and add_edge are allowed. For an additional add_edge, node_id and name must
be null or empty. Its when_to_use/how_to_use may be null or strings and avoid may be null or a
list of strings; these optional activation details are retained in edge audit reasoning. Use only
supplied specialist IDs, prereq/enhance edges, and cited failure_case_ids. Use [] when none."""

    synthesized_by_index: dict[int, GraphEdit] = {}
    generated_edits: list[GraphEdit] = []
    consumed: set[int] = set()
    reserved_new_ids = set(graph.nodes) | {
        str(edit.node_id) for edit in patch.edits
        if edit.op == "add_node" and str(edit.node_id)
    }
    component_work: list[dict[str, Any]] = []
    for indices in components:
        if not _requires_joint_semantic_synthesis(patch.edits, indices):
            continue
        component_id = _joint_component_id(patch.edits, indices)
        available_new_node_ids = [
            _allocate_new_node_id(graph, reserved_new_ids) for _ in range(2)
        ]
        payload, failure_ids, guard_nodes, high_exposure_indices = _joint_component_payload(
            graph, patch.edits, indices, success_guards_by_node,
            available_new_node_ids, prior_local_gate_feedback,
        )
        success_guard_case_ids = sorted({
            str(case_id)
            for node_id in guard_nodes
            for case_id in [
                *[
                    item.get("case_id")
                    for item in (
                        success_guards_by_node[node_id].get("case_mechanisms") or []
                    )
                ],
                *[
                    value
                    for invariant in (
                        success_guards_by_node[node_id].get("protected_invariants") or []
                    )
                    for value in (invariant.get("case_ids") or [])
                ],
            ]
            if str(case_id or "")
        })
        payload["allowed_conflict_evidence_case_ids"] = {
            "failure": list(failure_ids),
            "protected_success": list(success_guard_case_ids),
        }
        payload["required_protected_guard_node_ids"] = list(guard_nodes)
        component_work.append({
            "indices": list(indices),
            "component_id": component_id,
            "available_new_node_ids": available_new_node_ids,
            "payload": payload,
            "failure_ids": failure_ids,
            "guard_nodes": guard_nodes,
            "high_exposure_indices": high_exposure_indices,
            "success_guard_case_ids": success_guard_case_ids,
            "oversized": bool(
                max_component_edits and len(indices) > max_component_edits
            ),
        })

    def run_component(
        work: dict[str, Any], *, deferred_retry: bool,
    ) -> tuple[
        dict[int, GraphEdit] | None,
        list[GraphEdit] | None,
        dict[str, Any] | None,
        list[dict[str, Any]],
        bool,
    ]:
        indices = list(work["indices"])
        component_id = str(work["component_id"])
        payload = dict(work["payload"])
        failure_ids = list(work["failure_ids"])
        guard_nodes = list(work["guard_nodes"])
        success_guard_case_ids = list(work["success_guard_case_ids"])
        available_new_node_ids = list(work["available_new_node_ids"])
        high_exposure_indices = list(work["high_exposure_indices"])
        user = exact_reference_json(payload)
        attempts: list[dict[str, Any]] = []
        attempt_range = range(1, 2) if deferred_retry else range(1, 4)
        for attempt in attempt_range:
            response = ""
            usage: Any = None
            try:
                response, usage = chat_fn(
                    system=system,
                    user=user,
                    max_completion_tokens=8192,
                    retries=2,
                    stage="joint_semantic_synthesis",
                    timeout=OPINION_MERGE_TIMEOUT_SECONDS,
                )
                obj = extract_json(response)
                resolved, additional, parsed = _validate_joint_semantic_response(
                    obj,
                    graph=graph,
                    edits=patch.edits,
                    indices=indices,
                    failure_ids=failure_ids,
                    guard_nodes=guard_nodes,
                    success_guard_case_ids=success_guard_case_ids,
                    available_new_node_ids=available_new_node_ids,
                    high_exposure_update_indices=high_exposure_indices,
                )
                trial = GraphPatch(
                    reasoning="joint semantic trial",
                    edits=[*resolved.values(), *additional],
                )
                if trial.edits:
                    materialize_skill_graph(graph, trial)
                attempts.append({
                    "attempt": attempt,
                    "phase": "deferred_retry" if deferred_retry else "first_pass",
                    "status": "complete",
                    "usage": usage,
                })
                if store is not None:
                    save_llm_call(
                        store,
                        component_id,
                        stage="joint_semantic_synthesis",
                        system=system,
                        user=user,
                        response=response,
                        usage=usage,
                        parsed=parsed,
                    )
                return resolved, additional, parsed, attempts, False
            except Exception as exc:
                infrastructure = _is_retryable_opinion_merge_error(exc)
                attempts.append({
                    "attempt": attempt,
                    "phase": "deferred_retry" if deferred_retry else "first_pass",
                    "status": "infrastructure_failure" if infrastructure else "invalid",
                    "usage": usage,
                    "response": response or None,
                    "error": str(exc),
                })
                if store is not None:
                    save_llm_call(
                        store,
                        component_id,
                        stage="joint_semantic_synthesis",
                        system=system,
                        user=user,
                        response=response or None,
                        usage=usage,
                        error=(
                            ("deferred retry" if deferred_retry else f"attempt {attempt}/3")
                            + f": {exc}"
                        ),
                    )
                if infrastructure:
                    if deferred_retry:
                        raise RuntimeError(
                            "Joint opinion merge infrastructure failure repeated "
                            "after the deferred retry; stopping experiment: "
                            f"component={component_id}: {exc}"
                        ) from exc
                    return None, None, None, attempts, True
                user = (
                    exact_reference_json(payload)
                    + "\n\nPrevious output was invalid: " + str(exc)
                    + "\nReturn the complete strict JSON again."
                )
        return None, None, None, attempts, False

    def finalize_component(
        work: dict[str, Any],
        resolved: dict[int, GraphEdit] | None,
        additional: list[GraphEdit] | None,
        parsed: dict[str, Any] | None,
        attempts: list[dict[str, Any]],
    ) -> None:
        indices = list(work["indices"])
        component_id = str(work["component_id"])
        consumed.update(indices)
        if work["oversized"]:
            status = "oversized_component_passthrough_to_local_gate"
        elif resolved is None or parsed is None:
            # Malformed semantic output may use the deterministic safety path,
            # but an infrastructure timeout may not: those are deferred above.
            resolved = {
                index: deepcopy(patch.edits[index])
                for index in indices
            }
            additional = []
            status = "invalid_three_times_passthrough_to_local_gate"
        else:
            status = (
                "applied" if resolved or additional
                else "teacher_abstained_component"
            )
        additional = list(additional or [])
        resolved = dict(resolved or {})
        for edit in [*resolved.values(), *additional]:
            edit.group_id = component_id
        synthesized_by_index.update(resolved)
        generated_edits.extend(additional)
        audit["components"].append({
            "component_id": component_id,
            "edit_indices": indices,
            "failure_case_ids": list(work["failure_ids"]),
            "protected_guard_node_ids": list(work["guard_nodes"]),
            "status": status,
            "resolved": parsed,
            "attempts": attempts,
            "n_kept": len(resolved),
            "n_added": len(additional),
        })

    deferred_components: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for work in component_work:
        if work["oversized"]:
            indices = list(work["indices"])
            resolved = {
                index: deepcopy(patch.edits[index])
                for index in indices
            }
            parsed = {
                "decision": "DETERMINISTIC_SAFE_FALLBACK",
                "reason": (
                    f"component has {len(indices)} edits above the dataset-local "
                    f"limit {max_component_edits}; preserve ordinary edits and "
                    "preserve grounded edits for paired Local Gate measurement"
                ),
            }
            finalize_component(
                work,
                resolved,
                [],
                parsed,
                [{
                    "attempt": 0,
                    "status": "skipped_oversized_component",
                    "n_edits": len(indices),
                    "max_component_edits": max_component_edits,
                }],
            )
            continue
        resolved, additional, parsed, attempts, timed_out = run_component(
            work, deferred_retry=False
        )
        if timed_out:
            deferred_components.append((work, attempts))
            continue
        finalize_component(work, resolved, additional, parsed, attempts)

    # All ordinary components finish before any timeout is replayed. Retry each
    # timed-out component exactly once; a second timeout stops the experiment.
    for work, first_attempts in deferred_components:
        print(
            "[graphopt joint opinion merge] deferred retry "
            f"component={work['component_id']}",
            flush=True,
        )
        resolved, additional, parsed, retry_attempts, timed_out = run_component(
            work, deferred_retry=True
        )
        if timed_out:  # run_component raises first; this is defensive.
            raise RuntimeError(
                "Joint opinion merge timed out twice; stopping experiment: "
                f"component={work['component_id']}"
            )
        finalize_component(
            work,
            resolved,
            additional,
            parsed,
            [*first_attempts, *retry_attempts],
        )

    output: list[GraphEdit] = []
    for index, edit in enumerate(patch.edits):
        if index not in consumed:
            output.append(edit)
            continue
        replacement = synthesized_by_index.get(index)
        if replacement is not None:
            output.append(replacement)
    output.extend(generated_edits)
    audit["status"] = "complete"
    audit["n_input_edits"] = len(patch.edits)
    audit["n_output_edits"] = len(output)
    if store is not None:
        store.save(
            "joint_semantic_synthesis", stage="joint_semantic_synthesis",
            inputs={"n_components": len(components)}, outputs=audit,
        )
    return GraphPatch(reasoning=patch.reasoning, edits=output), audit



def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def _compact_local_gate_case(row: dict[str, Any]) -> dict[str, Any]:
    """Retain causal evidence needed by a later joint retry, without full blobs."""
    compact: dict[str, Any] = {}
    for key in (
        "id", "question", "task_description", "predicted_answer", "gold_answers",
        "reference_answer", "hard", "semantic_reasoning_trace", "reasoning_trace",
        "graph_usage", "graph_refs", "evaluator_feedback",
    ):
        value = row.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    original = row.get("original_task")
    if isinstance(original, dict):
        compact["original_task"] = {
            key: value for key, value in original.items()
            if key in {"id", "question", "task_description", "answer", "answers"}
            and value not in (None, "", [], {})
        }
    if "semantic_reasoning_trace" not in compact and "reasoning_trace" not in compact:
        response = row.get("response")
        if isinstance(response, str) and response.strip():
            compact["response"] = response[-4000:]
    return compact


def _load_prior_local_gate_feedback(step_dir: Path) -> list[dict[str, Any]]:
    """Load rejected measured attempts so resumed joint synthesis cannot repeat them."""
    feedback: list[dict[str, Any]] = []
    for gate_path in sorted((step_dir / "local_gates").glob("**/local_gate.json")):
        patch_path = gate_path.with_name("tested_patch.json")
        if not patch_path.is_file():
            continue
        try:
            gate = _json_file(gate_path)
            tested = GraphPatch.from_dict(_json_file(patch_path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if bool(gate.get("accepted")):
            continue
        incident = sorted({
            str(value)
            for edit in tested.edits
            for value in (edit.node_id, edit.src, edit.dst)
            if str(value or "")
        })
        pairs_by_category: dict[str, list[dict[str, Any]]] = {
            "effective": [], "harmful": [], "unresolved_source": [],
        }
        for pair in gate.get("paired_cases") or []:
            if not isinstance(pair, dict):
                continue
            category = str(pair.get("category") or "")
            if category not in pairs_by_category or len(pairs_by_category[category]) >= 6:
                continue
            before = pair.get("before") if isinstance(pair.get("before"), dict) else {}
            after = pair.get("after") if isinstance(pair.get("after"), dict) else {}
            pairs_by_category[category].append({
                "case_id": str(pair.get("case_id") or ""),
                "category": category,
                "before": _compact_local_gate_case(before),
                "after": _compact_local_gate_case(after),
            })
        feedback.append({
            "artifact": str(gate_path.relative_to(step_dir)),
            "incident_node_ids": incident,
            "tested_edits": [{
                "op": edit.op, "node_id": edit.node_id,
                "src": edit.src, "dst": edit.dst,
                "when_to_use": edit.when_to_use,
                "how_to_use": edit.how_to_use, "avoid": list(edit.avoid or []),
                "source_case_ids": list(edit.source_case_ids),
            } for edit in tested.edits],
            "n_effective": int(gate.get("n_effective") or 0),
            "n_ineffective": int(gate.get("n_ineffective") or 0),
            "measured_counterexamples": [
                pair for category in ("harmful", "unresolved_source", "effective")
                for pair in pairs_by_category[category]
            ],
        })
    return feedback


def _stage_results_signature(results: list[dict[str, Any]]) -> str:
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


def _pre_joint_resume_files(step_dir: Path) -> dict[str, Path]:
    return {
        "inputs": step_dir / "evolution_inputs.json",
        "analyses": step_dir / "case_analysis.json",
        "statistics": step_dir / "graph_statistics.json",
        "merged": step_dir / "merged_proposals.json",
        "plan": step_dir / "graph_edit_plan.json",
        "patch": step_dir / "patch_before_dedupe.json",
        "success_guards": step_dir / "success_mechanism_summaries.json",
        "checkpoint": step_dir / "pre_joint_checkpoint.json",
    }


def _graph_digest(graph: SkillGraph) -> str:
    raw = json.dumps(
        graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def can_resume_pre_joint_stage(
    step_dir: str | Path,
    *,
    graph: SkillGraph,
    results: list[dict[str, Any]],
    step: int,
    allow_legacy_v18_reconstruction: bool = False,
) -> tuple[bool, str]:
    """Validate an exact pre-joint frontier without trusting file presence alone."""
    root = Path(step_dir)
    files = _pre_joint_resume_files(root)
    required = [
        files["inputs"], files["analyses"], files["statistics"], files["merged"],
        files["plan"], files["patch"], files["success_guards"],
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return False, "missing pre-joint artifacts: " + ", ".join(missing)
    try:
        inputs = _json_file(files["inputs"])
        patch = GraphPatch.from_dict(_json_file(files["patch"]))
        analyses = _json_file(files["analyses"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return False, f"invalid pre-joint artifact: {exc}"
    expected_ids = [str(row.get("id") or "") for row in results]
    stored_ids = [str(value or "") for value in (inputs.get("case_ids") or [])]
    if int(inputs.get("step") or -1) != int(step):
        return False, "evolution step identity changed"
    if stored_ids != expected_ids:
        return False, "evolution case IDs/order changed"
    if len(analyses) != len(results):
        return False, "case analysis coverage changed"
    if not patch.edits:
        return False, "pre-joint patch is empty"
    checkpoint = files["checkpoint"]
    if checkpoint.is_file():
        record = _json_file(checkpoint)
        if str(record.get("graph_sha256") or "") != _graph_digest(graph):
            return False, "epoch-start graph changed"
        if str(record.get("results_signature") or "") != _stage_results_signature(results):
            return False, "epoch results content changed"
        return True, "audited_pre_v20_pre_joint_checkpoint"
    if allow_legacy_v18_reconstruction and step == 10000:
        return True, "audited_v18_epoch1_artifact_reconstruction"
    return False, "pre_joint_checkpoint.json is required"


def _resume_evolution_from_pre_joint(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    *,
    cfg: EvolutionConfig,
    evo_cache: EvolutionCache,
    chat_fn: Any,
    mode: str,
    step_dir: Path,
    step: int,
    recorder: StepRecorder | None,
    store: ArtifactStore,
    allow_legacy_v18_reconstruction: bool,
) -> EvolutionResult:
    ok, source = can_resume_pre_joint_stage(
        step_dir, graph=graph, results=results, step=step,
        allow_legacy_v18_reconstruction=allow_legacy_v18_reconstruction,
    )
    if not ok:
        raise RuntimeError(f"unsafe pre-joint resume refused: {source}")
    files = _pre_joint_resume_files(step_dir)
    analyses = [CaseAnalysis.from_dict(item) for item in _json_file(files["analyses"])]
    stats_json = _json_file(files["statistics"])
    node_stats = {
        str(key): UsageStats.from_dict(value)
        for key, value in (stats_json.get("nodes") or {}).items()
    }
    edge_stats = {
        str(key): UsageStats.from_dict(value)
        for key, value in (stats_json.get("edges") or {}).items()
    }
    merged_json = _json_file(files["merged"])
    merged = {
        str(bucket): [MergedProposal.from_dict(item) for item in (items or [])]
        for bucket, items in merged_json.items()
    }
    plan = GraphEditPlan.from_dict(_json_file(files["plan"]))
    patch = GraphPatch.from_dict(_json_file(files["patch"]))
    success_guards = {
        str(key): dict(value)
        for key, value in _json_file(files["success_guards"]).items()
    }
    checkpoint = files["checkpoint"]
    if checkpoint.is_file():
        checkpoint_data = _json_file(checkpoint)
        cached = checkpoint_data.get("cache_state_before_joint")
        if isinstance(cached, dict):
            evo_cache = EvolutionCache.from_dict(cached)
        step_merged_json = checkpoint_data.get("step_merged") or merged_json
        step_merged = {
            str(bucket): [MergedProposal.from_dict(item) for item in (items or [])]
            for bucket, items in step_merged_json.items()
        }
    else:
        evidence_ids = {
            analysis.case_id for analysis in analyses
            if not analysis.success and (
                analysis.node_revision_proposals
                or analysis.retrieval_revision_proposals
                or analysis.new_node_proposals
            )
        }
        focus = {
            analysis.case_id: min(
                proposal.first_wrong_step
                for proposal in analysis.node_revision_proposals
                if proposal.first_wrong_step >= 0
            )
            for analysis in analyses
            if any(p.first_wrong_step >= 0 for p in analysis.node_revision_proposals)
        }
        evo_cache.ingest_case_examples(
            [row for row in results if str(row.get("id") or "") in evidence_ids],
            focus_steps_by_case=focus,
        )
        step_merged = merged
    if not checkpoint.is_file():
        store.save(
            "pre_joint_checkpoint", stage="pre_joint_checkpoint",
            inputs={
                "graph_sha256": _graph_digest(graph),
                "results_signature": _stage_results_signature(results),
                "step": step,
                "reconstructed_from": source,
            },
            outputs={
                "schema_version": "graphopt-pre-joint-checkpoint-v1",
                "graph_sha256": _graph_digest(graph),
                "results_signature": _stage_results_signature(results),
                "step": step,
                "step_merged": {
                    key: [proposal.to_dict() for proposal in values]
                    for key, values in step_merged.items()
                },
                "cache_state_before_joint": evo_cache.to_dict(),
                "reconstructed_from": source,
            },
        )
    patch.edits = _dedupe_edits(patch.edits)
    store.save(
        "stage_resume_manifest", stage="stage_resume",
        inputs={"requested_stage": "joint_semantic_synthesis"},
        outputs={
            "schema_version": "graphopt-stage-resume-v1",
            "source": source,
            "reused_through": "patch_before_dedupe",
            "recomputed_from": "joint_semantic_synthesis",
            "graph_sha256": _graph_digest(graph),
            "results_signature": _stage_results_signature(results),
            "n_cases_reused": len(results),
            "n_analyses_reused": len(analyses),
            "n_patch_edits_reused": len(patch.edits),
        },
    )
    print(
        f"[graphopt stage resume] reused collection/analysis/rewrite through "
        f"patch_before_dedupe ({len(results)} cases, {len(patch.edits)} edits); "
        "restarting at joint_semantic_synthesis",
        flush=True,
    )
    prior_local_gate_feedback = _load_prior_local_gate_feedback(step_dir)
    store.save(
        "prior_local_gate_feedback", stage="stage_resume",
        inputs={"n_rejected_attempts": len(prior_local_gate_feedback)},
        outputs={
            "schema_version": "graphopt-prior-local-gate-feedback-v1",
            "policy": "reuse_rejected_measured_attempts_as_joint_counterexamples",
            "attempts": prior_local_gate_feedback,
        },
    )
    patch, _ = synthesize_joint_semantic_patch(
        graph, patch, success_guards, chat_fn=chat_fn, mode=mode, store=store,
        prior_local_gate_feedback=prior_local_gate_feedback,
        max_component_edits=cfg.max_joint_semantic_component_edits,
    )
    patch.edits = _dedupe_edits(patch.edits)
    evo_cache.ingest_step(
        step=step, graph=graph, analyses=analyses, node_stats=node_stats,
        edge_stats=edge_stats, merged=step_merged, plan=plan,
        persist_proposal_pool=cfg.persist_proposal_pool,
    )
    if cfg.persist_proposal_pool:
        evo_cache.reconcile_semantic_proposal_pool(merged)
    evo_cache.save(step_dir / "evolution_cache.json")
    store.save(
        "patch", stage="patch_final", inputs={"n_edits": len(patch.edits)},
        outputs=patch.to_dict(),
    )
    if patch.edits:
        next_graph = materialize_skill_graph(graph, patch)
        save_skill_json(next_graph, str(step_dir / "skill_graph_next.json"))
        store.save(
            "skill_graph_next", stage="materialize",
            inputs={"n_edits": len(patch.edits)}, outputs=next_graph.to_dict(),
        )
    if recorder:
        recorder.record(
            "stage_resume", files=["stage_resume_manifest_v*.json"],
            reused_through="patch_before_dedupe",
            recomputed_from="joint_semantic_synthesis",
        )
        recorder.record(
            "patch_and_cache",
            files=["patch_v*.json", "joint_semantic_synthesis_v*.json"],
            n_edits=len(patch.edits),
        )
        recorder.flush()
    return EvolutionResult(
        patch_edits=patch.edits, case_analyses=analyses,
        graph_statistics=stats_json, merged_proposals=merged_json,
        edit_plan=plan, reasoning=patch.reasoning,
        node_stats=node_stats, edge_stats=edge_stats, cache=evo_cache,
    )


def run_evolution_step(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    *,
    cfg: EvolutionConfig | None = None,
    cache: EvolutionCache | None = None,
    chat_fn=None,
    mode: str = "template",
    meta_context: str = "",
    step_dir: str | Path | None = None,
    step: int = 0,
    recorder: StepRecorder | None = None,
    update_protocol: str = "case_complete",
    resume_from_stage: str = "",
    allow_legacy_v18_reconstruction: bool = False,
) -> EvolutionResult:
    """Full pipeline on frozen G_t; returns patch without mutating graph."""
    update_protocol = str(update_protocol or "case_complete").strip().lower()
    if update_protocol == "case_complete_v1":
        update_protocol = "case_complete"
    if update_protocol not in {"legacy", "case_complete", "causal"}:
        raise ValueError("update_protocol must be legacy, case_complete, or causal")
    cfg = cfg or EvolutionConfig()
    step_dir = Path(step_dir) if step_dir else None
    if recorder is None and step_dir:
        recorder = StepRecorder(step_dir)
    evo_cache = cache or EvolutionCache.from_graph(graph)
    evo_cache.ensure_graph_keys(graph)

    pending_before = list(evo_cache.pending_harmful_hints)
    pending_bad_case_before = evo_cache.pending_bad_case_prompt
    harmful_prompt = evo_cache.consume_harmful_prompt()
    gate_experience_prompt = evo_cache.format_gate_experiences_for_prompt()
    analysis_context = "\n\n".join(
        part for part in (meta_context, gate_experience_prompt, harmful_prompt)
        if str(part or "").strip()
    ).strip()

    store = recorder.store if recorder else (ArtifactStore(step_dir) if step_dir else None)

    if resume_from_stage:
        if resume_from_stage != "joint_semantic_synthesis":
            raise ValueError(
                "supported resume_from_stage is joint_semantic_synthesis"
            )
        if step_dir is None or store is None:
            raise ValueError("stage resume requires step_dir and ArtifactStore")
        return _resume_evolution_from_pre_joint(
            graph, results, cfg=cfg, evo_cache=evo_cache, chat_fn=chat_fn,
            mode=mode, step_dir=step_dir, step=step, recorder=recorder,
            store=store,
            allow_legacy_v18_reconstruction=allow_legacy_v18_reconstruction,
        )

    if store is not None:
        evo_in = {
            "step": step,
            "mode": mode,
            "update_protocol": update_protocol,
            "badcase_analysis_protocol": badcase_analysis_protocol(update_protocol),
            "attribution_boundary_protocol": ATTRIBUTION_BOUNDARY_PROTOCOL,
            "n_cases": len(results),
            "case_ids": [r.get("id") for r in results],
            "meta_context_len": len(meta_context or ""),
            "pending_harmful_hints_before": pending_before,
            "pending_bad_case_prompt_before": pending_bad_case_before or None,
            "harmful_prompt_consumed": harmful_prompt or None,
            "prior_gate_experience_count": len(evo_cache.gate_experiences),
            "prior_gate_experience_injected": bool(gate_experience_prompt),
        }
        store.save(
            "evolution_inputs",
            stage="evolution_inputs",
            inputs={"results_count": len(results), "meta_context": meta_context[:500] if meta_context else ""},
            outputs=evo_in,
        )
        if harmful_prompt or pending_before or pending_bad_case_before:
            store.save(
                "harmful_hints_consumed",
                stage="harmful_hints",
                inputs={
                    "pending_before": pending_before,
                    "pending_bad_case_prompt_before": pending_bad_case_before or None,
                },
                outputs={"injected_prompt": harmful_prompt or ""},
            )
        if recorder:
            recorder.record(
                "evolution_inputs",
                files=["evolution_inputs_v*.json", "artifact_index.json"]
                + (["harmful_hints_consumed_v*.json"] if harmful_prompt or pending_before or pending_bad_case_before else []),
                n_cases=len(results),
                mode=mode,
            )

    analyses = analyze_cases(
        graph,
        results,
        chat_fn=chat_fn,
        mode=mode,
        meta_context=analysis_context,
        store=store,
        max_workers=cfg.analyst_workers,
        analyze_successes_with_teacher=False,
        update_protocol=update_protocol,
        attribution_adjudication_cache=evo_cache.attribution_adjudications,
    )
    evidence_case_ids: set[str] = set()
    for analysis in analyses:
        if analysis.success:
            continue
        if (
            analysis.node_revision_proposals
            or analysis.retrieval_revision_proposals
            or analysis.new_node_proposals
        ):
            evidence_case_ids.add(str(analysis.case_id))
    focus_steps_by_case: dict[str, int] = {}
    for analysis in analyses:
        steps = [
            proposal.first_wrong_step
            for proposal in analysis.node_revision_proposals
            if proposal.first_wrong_step >= 0
        ]
        if steps:
            focus_steps_by_case[str(analysis.case_id)] = min(steps)
    evo_cache.ingest_case_examples(
        [
            result for result in results
            if str(result.get("id") or "") in evidence_case_ids
        ],
        focus_steps_by_case=focus_steps_by_case,
    )
    if store is not None:
        input_case_ids = [str(result.get("id") or "case") for result in results]
        analyzed_case_ids = [analysis.case_id for analysis in analyses]
        analyzed_set = set(analyzed_case_ids)
        dropped_case_ids = [case_id for case_id in input_case_ids if case_id not in analyzed_set]
        store.save(
            "case_analysis",
            stage="case_analyze",
            inputs={"n_cases": len(results)},
            outputs=[a.to_dict() for a in analyses],
        )
        store.save(
            "case_analysis_status",
            stage="case_analyze_status",
            inputs={"input_case_ids": input_case_ids},
            outputs={
                "n_input": len(input_case_ids),
                "n_analyzed": len(analyzed_case_ids),
                "n_dropped": len(dropped_case_ids),
                "n_partial": sum(1 for analysis in analyses if analysis.analysis_partial),
                "n_teacher_analyzed": (
                    len(analyses)
                    if mode == "teacher" and chat_fn is not None else 0
                ),
                "n_success_heuristic": 0,
                "n_template_analyzed": (
                    len(analyses)
                    if mode != "teacher" or chat_fn is None else 0
                ),
                "attribution_boundary_protocol": ATTRIBUTION_BOUNDARY_PROTOCOL,
                "n_retrieval_miss": sum(
                    analysis.failure_type == "RETRIEVAL_MISS"
                    for analysis in analyses
                ),
                "n_execution_lapse": sum(
                    analysis.failure_type == "EXECUTION_LAPSE"
                    for analysis in analyses
                ),
                "n_unattributed": sum(
                    analysis.failure_type == "UNATTRIBUTED"
                    for analysis in analyses
                ),
                "analyzed_case_ids": analyzed_case_ids,
                "dropped_case_ids": dropped_case_ids,
                "partial_policy": "two invalid teacher outputs => diagnostic statistics only; no graph edits",
            },
        )
        if recorder:
            recorder.record(
                "case_analyze",
                files=[
                    "case_analysis_v*.json",
                    "case_analysis_status_v*.json",
                    "llm/case_analyze/*_v*.json",
                    "llm/attribution_boundary_adjudicate/*_v*.json",
                    "debug/case_analyze/*_v*.json",
                    "debug/attribution_boundary_adjudicate_cache/*_v*.json",
                ],
                n_cases=len(analyses),
                n_dropped=len(dropped_case_ids),
                llm=(
                    mode == "teacher" and chat_fn is not None
                    and any(not analysis.success for analysis in analyses)
                ),
            )

    node_stats, edge_stats = aggregate_statistics(
        graph,
        analyses,
    )
    stats_json = statistics_to_json(node_stats, edge_stats)
    if store is not None:
        store.save(
            "graph_statistics",
            stage="aggregate_statistics",
            inputs={"n_analyses": len(analyses)},
            outputs=stats_json,
        )
        if recorder:
            recorder.record("aggregate_statistics", files=["graph_statistics_v*.json"])

    failed_node_merges: set[str] = set()
    defect_analyses = deepcopy(analyses)
    for analysis in defect_analyses:
        lapse_targets = set(analysis.execution_lapse_nodes)
        if lapse_targets:
            analysis.node_revision_proposals = [
                proposal for proposal in analysis.node_revision_proposals
                if proposal.target_node not in lapse_targets
            ]
        if analysis.failure_type == "EXECUTION_LAPSE":
            analysis.retrieval_revision_proposals = []
            analysis.edge_correction_proposals = []
    # One evolution step exposes the complete update pool at once. Semantic
    # merger internals may still organize opinions by target node/type, but no
    # dataset-level 256/156 evidence windows or rolling batch summaries exist.
    step_merged = semantic_merge_proposals(
        defect_analyses, graph, case_results=results,
        case_evidence_analyses=analyses, chat_fn=chat_fn, mode=mode,
        node_top_k=cfg.node_merge_top_k, store=store,
        failed_node_merges=failed_node_merges,
    )
    if store is not None:
        store.save(
            "opinion_input_manifest", stage="single_complete_update_pool",
            inputs={
                "input_mode": "single_complete_update_pool",
                "case_count": len(results),
                "case_ids": [str(row.get("id") or "") for row in results],
            },
            outputs={
                "batching_enabled": False,
                "rolling_batch_merge_enabled": False,
                "all_cases_visible_in_one_evolution_step": True,
            },
        )
    merged = (
        _merge_with_cached_pool(
            step_merged,
            evo_cache,
            failed_node_merges=failed_node_merges,
        )
        if cfg.persist_proposal_pool
        else step_merged
    )
    if cfg.persist_proposal_pool:
        merged = semantic_recluster_proposal_pool(
            merged,
            active_buckets=active_proposal_pool_buckets(step_merged),
            chat_fn=chat_fn,
            mode=mode,
            store=store,
            stage="rolling_cached_opinion_merge",
            include_source_cases=False,
        )
    merged["execution_reinforcements"] = _execution_reinforcements(
        analyses,
        evo_cache,
        graph=graph,
        chat_fn=chat_fn,
        mode=mode,
        store=store,
        evidence_cap=cfg.execution_merge_evidence_cap,
        include_cached_evidence=cfg.persist_proposal_pool,
    )
    if failed_node_merges:
        merged["node_revisions"] = [
            proposal
            for proposal in (merged.get("node_revisions") or [])
            if proposal.target_node not in failed_node_merges
        ]
    merged["node_revisions"] = _rank_and_cap_node_revisions(
        list(merged.get("node_revisions") or []),
        cfg.node_merge_top_k,
    )
    merged_json = {k: [m.to_dict() for m in v] for k, v in merged.items()}
    if store is not None:
        store.save(
            "merged_proposals",
            stage="semantic_merge",
            inputs={
                "n_analyses": len(analyses),
                "node_merge_mode": "teacher" if mode == "teacher" and chat_fn else "template",
                "node_merge_top_k": cfg.node_merge_top_k,
                "edge_merge_mode": "exact_script",
                "statistics_source": "raw_case_analyses",
            },
            outputs=merged_json,
        )
        if recorder:
            recorder.record(
                "semantic_merge",
                files=[
                    "merged_proposals_v*.json",
                    "llm/node_proposal_merge/*_v*.json",
                    "debug/node_proposal_merge/*_v*.json",
                ],
                calls_llm=mode == "teacher" and chat_fn is not None,
                node_merge_top_k=cfg.node_merge_top_k,
                edge_merge_mode="exact_script",
            )

    plan = _build_edit_plan(graph, merged, node_stats, edge_stats, cfg, seed=step)
    # A dropped Case Analysis contributes neither usage nor correctness evidence.
    # Therefore its absence must never be interpreted as `used=0`: if even one
    # Experience failed both teacher attempts, suppress low-usage deletion for
    # this epoch while keeping evidence-backed add/update/replace operations.
    if len(analyses) != len(results) or any(a.analysis_partial for a in analyses):
        plan.delete_nodes = []
        plan.delete_edges = []
    current_failure_examples = build_failure_casebook_examples(analyses, results)
    # Every successful use of a node touched by this epoch is positive
    # counter-evidence. Raw examples remain bounded in rewrite prompts, while
    # semantic guards summarize *all* related successes and are always passed.
    for entry in evo_cache.nodes.values():
        entry.positive_examples = []
        entry.failure_examples = []
    positive_examples_by_node = (
        _successful_node_examples(analyses, results)
        if cfg.use_positive_context else {}
    )
    related_node_ids = _plan_related_existing_node_ids(graph, plan)
    success_guards_by_node = (
        _summarize_related_successes(
            graph, positive_examples_by_node, related_node_ids,
            chat_fn=chat_fn, mode=mode, store=store,
            max_workers=cfg.success_guard_workers,
        )
        if cfg.use_positive_context else {}
    )
    current_successful_examples: dict[str, dict[str, Any]] = {}
    for examples in positive_examples_by_node.values():
        for example in examples:
            current_successful_examples.setdefault(
                str(example.get("case_id") or ""), dict(example)
            )
    current_successful_examples.pop("", None)
    abstained_new_nodes = _prepare_new_nodes(
        plan,
        graph=graph,
        case_examples_by_id=evo_cache.case_examples,
        successful_examples=(
            list(current_successful_examples.values())
            if cfg.use_positive_context else []
        ),
        positive_examples_by_node=positive_examples_by_node,
        success_guards_by_node=success_guards_by_node,
        failure_examples_by_node=current_failure_examples,
        chat_fn=chat_fn,
        mode=mode,
        store=store,
        positive_example_cap=cfg.positive_example_cap,
        failure_example_cap=cfg.failure_example_cap,
    )
    abstained_node_updates = _prepare_node_updates(
        graph,
        plan,
        positive_examples_by_node=positive_examples_by_node,
        success_guards_by_node=success_guards_by_node,
        case_examples_by_id=evo_cache.case_examples,
        failure_examples_by_node=current_failure_examples,
        chat_fn=chat_fn,
        mode=mode,
        store=store,
        positive_example_cap=cfg.positive_example_cap,
        failure_example_cap=cfg.failure_example_cap,
    )
    if store is not None:
        store.save(
            "graph_edit_plan",
            stage="edit_plan",
            inputs={"thresholds": {"x": cfg.node_support_threshold, "y": cfg.edge_support_threshold}},
            outputs=plan.to_dict(),
        )
        if recorder:
            recorder.record("edit_plan", files=["graph_edit_plan_v*.json"])

    patch = _plan_to_patch(
        graph,
        plan,
        chat_fn=chat_fn,
        mode=mode,
        delete_used_threshold=cfg.delete_used_threshold,
        store=store,
    )
    before_dedupe = list(patch.edits)
    if store is not None:
        store.save(
            "patch_before_dedupe",
            stage="patch",
            inputs={"n_edits_raw": len(before_dedupe)},
            outputs=patch.to_dict(),
        )

    # Patch size is not globally capped. Per-node scope is controlled earlier by
    # node_merge_top_k; here we only remove exact duplicate edits.
    patch.edits = _dedupe_edits(patch.edits)

    if store is not None and len(before_dedupe) != len(patch.edits):
        store.save(
            "patch_dedupe_report",
            stage="patch_dedupe",
            inputs={"policy": "exact_edit_dedup_only"},
            outputs={
                "n_before": len(before_dedupe),
                "n_after": len(patch.edits),
                "dropped_ops": [e.op for e in before_dedupe if e not in patch.edits],
            },
        )
        if recorder:
            recorder.record(
                "patch_dedupe",
                files=["patch_dedupe_report_v*.json", "patch_before_dedupe_v*.json"],
            )


    if store is not None:
        store.save(
            "pre_joint_checkpoint",
            stage="pre_joint_checkpoint",
            inputs={
                "graph_sha256": _graph_digest(graph),
                "results_signature": _stage_results_signature(results),
                "step": step,
            },
            outputs={
                "schema_version": "graphopt-pre-joint-checkpoint-v1",
                "graph_sha256": _graph_digest(graph),
                "results_signature": _stage_results_signature(results),
                "step": step,
                "step_merged": {
                    key: [proposal.to_dict() for proposal in values]
                    for key, values in step_merged.items()
                },
                "cache_state_before_joint": evo_cache.to_dict(),
            },
        )
        if recorder:
            recorder.record(
                "pre_joint_checkpoint",
                files=["pre_joint_checkpoint_v*.json"],
                n_edits=len(patch.edits),
            )

    # This is the actual cross-edit semantic composition boundary. Every
    # connected node/edge component is regenerated once from its complete
    # failure evidence plus every related successful-use guard. The later
    # Trainer grouping is an execution/Gate boundary, not a substitute for
    # semantic synthesis.
    patch, _joint_semantic_audit = synthesize_joint_semantic_patch(
        graph, patch, success_guards_by_node, chat_fn=chat_fn, mode=mode,
        store=store,
        max_component_edits=cfg.max_joint_semantic_component_edits,
    )
    patch.edits = _dedupe_edits(patch.edits)

    evo_cache.ingest_step(
        step=step,
        graph=graph,
        analyses=analyses,
        node_stats=node_stats,
        edge_stats=edge_stats,
        # Store this step's evidence, not the already cumulative view.
        merged=step_merged,
        plan=plan,
        persist_proposal_pool=cfg.persist_proposal_pool,
    )
    if cfg.persist_proposal_pool:
        evo_cache.reconcile_semantic_proposal_pool(merged)
    if abstained_new_nodes:
        # Invalid/unsafe synthesis is terminal for this mature proposal; do not
        # pay to regenerate the same candidate in a later epoch.
        evo_cache.consume_rejected_proposals(
            GraphEditPlan(add_nodes=abstained_new_nodes), []
        )
    if abstained_node_updates:
        # A teacher-declared unsafe/redundant update is a final decision for
        # this mature cluster. Retire it so stale evidence cannot force the
        # same low-quality experience into every later epoch.
        evo_cache.consume_rejected_proposals(
            GraphEditPlan(update_nodes=abstained_node_updates), []
        )
    # Evidence stays staged until Trainer knows which edits Gate accepted.
    # Rejected evidence must remain available for a revised proposal.
    if step_dir:
        evo_cache.save(step_dir / "evolution_cache.json")
    if store is not None:
        store.save(
            "patch",
            stage="patch_final",
            inputs={"n_edits": len(patch.edits)},
            outputs=patch.to_dict(),
        )
        if patch.edits:
            next_graph = materialize_skill_graph(graph, patch)
            if step_dir:
                save_skill_json(next_graph, str(step_dir / "skill_graph_next.json"))
            store.save(
                "skill_graph_next",
                stage="materialize",
                inputs={"n_edits": len(patch.edits)},
                outputs=next_graph.to_dict(),
            )
        if recorder:
            recorder.record(
                "patch_and_cache",
                files=[
                    "patch_v*.json",
                    "artifact_index.json",
                    "joint_semantic_synthesis_v*.json",
                    "llm/joint_semantic_synthesis/*_v*.json",
                    "llm/node_rewrite/*_v*.json",
                    "debug/node_rewrite/*_v*.json",
                ],
                n_edits=len(patch.edits),
            )
            recorder.flush()

    return EvolutionResult(
        patch_edits=patch.edits,
        case_analyses=analyses,
        graph_statistics=statistics_to_json(node_stats, edge_stats),
        merged_proposals={k: [m.to_dict() for m in v] for k, v in merged.items()},
        edit_plan=plan,
        reasoning=patch.reasoning,
        node_stats=node_stats,
        edge_stats=edge_stats,
        cache=evo_cache,
    )
