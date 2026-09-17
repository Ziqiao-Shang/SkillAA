"""Cross-batch semantic clustering for staged graph-edit opinions."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from graphopt.debug.artifacts import ArtifactStore, save_llm_call, save_template_call
from graphopt.evolution.exact_refs import exact_reference_json
from graphopt.evolution.merge import (
    OPINION_MERGE_TIMEOUT_SECONDS, PROMPTS,
    _is_retryable_opinion_merge_error, _merge_activation_edge_records,
    _merge_texts, _proposal_pool_bucket, _similar,
)
from graphopt.evolution.types import MergedProposal


def _merge_exact_edges(
    proposals: list[MergedProposal],
) -> list[MergedProposal]:
    """Merge identical edge opinions while preserving distinct-case support."""
    buckets: dict[tuple[str, str, str, str], list[MergedProposal]] = defaultdict(list)
    for proposal in proposals:
        buckets[(
            proposal.target_old_edge, proposal.source, proposal.target, proposal.relation,
        )].append(proposal)
    merged: list[MergedProposal] = []
    for _, group in sorted(buckets.items()):
        first = group[0]
        case_ids = list(dict.fromkeys(
            case_id for proposal in group for case_id in proposal.source_case_ids
        ))
        merged.append(MergedProposal(
            kind="edge", content=first.content, source=first.source,
            target=first.target, relation=first.relation,
            target_old_edge=first.target_old_edge,
            rationale=" ".join(dict.fromkeys(
                proposal.rationale for proposal in group if proposal.rationale
            ))[:1200],
            support=len(case_ids), source_case_ids=case_ids,
        ))
    return merged


def active_proposal_pool_buckets(
    merged: dict[str, list[MergedProposal]],
) -> set[str]:
    """Return buckets that received evidence in the current batch."""
    active: set[str] = set()
    for name in ("node_revisions", "retrieval_revisions", "new_nodes"):
        for proposal in merged.get(name) or []:
            bucket = _proposal_pool_bucket(proposal)
            if bucket:
                active.add(bucket)
    return active


def semantic_recluster_proposal_pool(
    merged: dict[str, list[MergedProposal]],
    *,
    active_buckets: set[str],
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
    stage: str = "opinion_merge",
    include_source_cases: bool = True,
) -> dict[str, list[MergedProposal]]:
    """Re-cluster current plus cached opinions without inventing support.

    The teacher may only partition opaque IDs. Python protects node/operation
    buckets and reconstructs all text and distinct-case provenance.
    """
    out = {name: list(values) for name, values in merged.items()}
    all_by_bucket: dict[str, list[MergedProposal]] = defaultdict(list)
    for name in ("node_revisions", "retrieval_revisions", "new_nodes"):
        for proposal in merged.get(name) or []:
            bucket = _proposal_pool_bucket(proposal)
            if bucket:
                all_by_bucket[bucket].append(proposal)
    selected = {
        bucket: proposals
        for bucket, proposals in all_by_bucket.items()
        if bucket in active_buckets and len(proposals) > 1
    }
    if not selected:
        out["edges"] = _merge_exact_edges(out.get("edges") or [])
        return out

    opinions: list[dict[str, Any]] = []
    by_id: dict[str, tuple[str, MergedProposal]] = {}
    for bucket_index, (bucket, proposals) in enumerate(
        sorted(selected.items()), start=1
    ):
        for proposal_index, proposal in enumerate(proposals, start=1):
            opinion_id = f"pool_{bucket_index:03d}_{proposal_index:04d}"
            by_id[opinion_id] = (bucket, proposal)
            opinions.append({
                "opinion_id": opinion_id,
                "bucket": bucket,
                "content": proposal.content,
                "evidence_count": len(set(proposal.source_case_ids)),
                "source_cases": (
                    [dict(item) for item in proposal.evidence_items]
                    if include_source_cases else []
                ),
                "activation_edges": [dict(item) for item in proposal.activation_edges],
            })
    expected_ids = set(by_id)

    def fallback_groups() -> list[tuple[list[str], str]]:
        groups: list[list[str]] = []
        for opinion_id in sorted(expected_ids):
            bucket, proposal = by_id[opinion_id]
            for group in groups:
                other_bucket, other = by_id[group[0]]
                if other_bucket == bucket and _similar(
                    proposal.content, other.content
                ):
                    group.append(opinion_id)
                    break
            else:
                groups.append([opinion_id])
        return [
            (ids, _merge_texts([by_id[opinion_id][1].content for opinion_id in ids]))
            for ids in groups
        ]

    groups: list[tuple[list[str], str]] = []
    if mode == "teacher" and chat_fn is not None:
        system = (PROMPTS / "proposal_pool_merge.md").read_text(encoding="utf-8")
        user = exact_reference_json({"opinions": opinions})
        from graphopt.json_utils import extract_json

        def parse(response: str) -> list[tuple[list[str], str]]:
            obj = extract_json(response)
            if not isinstance(obj, dict) or set(obj) != {"clusters"}:
                raise ValueError("proposal-pool merge must contain exactly clusters")
            raw_clusters = obj["clusters"]
            if not isinstance(raw_clusters, list) or not raw_clusters:
                raise ValueError("clusters must be a non-empty list")
            seen: set[str] = set()
            parsed: list[tuple[list[str], str]] = []
            for raw in raw_clusters:
                if not isinstance(raw, dict) or set(raw) != {"opinion_ids", "synthesis"}:
                    raise ValueError("each cluster must contain opinion_ids and synthesis")
                ids = raw["opinion_ids"]
                if (
                    not isinstance(ids, list)
                    or not ids
                    or any(not isinstance(value, str) for value in ids)
                    or len(ids) != len(set(ids))
                ):
                    raise ValueError(
                        "opinion_ids must be a non-empty unique string list"
                    )
                if set(ids) - expected_ids or set(ids) & seen:
                    raise ValueError("cluster has unknown or repeated opinion_ids")
                if len({by_id[opinion_id][0] for opinion_id in ids}) != 1:
                    raise ValueError("one cluster cannot cross a protected bucket")
                synthesis = raw["synthesis"]
                if not isinstance(synthesis, str) or not synthesis.strip():
                    raise ValueError("synthesis must be a non-empty modification opinion")
                if len(synthesis) > 3000:
                    raise ValueError("synthesis exceeds 3000 characters")
                seen.update(ids)
                parsed.append((ids, synthesis.strip()))
            if seen != expected_ids:
                raise ValueError("every opinion_id must appear exactly once")
            return parsed

        current_user = user
        for attempt in range(1, 3):
            response = ""
            usage: Any = None
            try:
                response, usage = chat_fn(
                    system=system,
                    user=current_user,
                    max_completion_tokens=8192,
                    retries=2,
                    stage=stage,
                    timeout=OPINION_MERGE_TIMEOUT_SECONDS,
                )
                groups = parse(response)
                if store is not None:
                    save_llm_call(
                        store,
                        "all_cached_proposals",
                        stage=stage,
                        system=system,
                        user=current_user,
                        response=response,
                        usage=usage,
                        parsed={
                            "clusters": [
                                {"opinion_ids": ids, "synthesis": synthesis}
                                for ids, synthesis in groups
                            ]
                        },
                    )
                break
            except Exception as exc:
                if store is not None:
                    save_llm_call(
                        store,
                        "all_cached_proposals",
                        stage=stage,
                        system=system,
                        user=current_user,
                        response=response or None,
                        usage=usage,
                        error=f"attempt {attempt}/2: {exc}",
                    )
                if _is_retryable_opinion_merge_error(exc):
                    if attempt == 2:
                        raise RuntimeError(
                            "Proposal-pool opinion merge timed out again; "
                            "stopping experiment"
                        ) from exc
                    continue
                current_user = user + (
                    "\n\nReturn strict JSON. Include every opinion_id exactly once; "
                    "never put different bucket values in one cluster; include one grounded "
                    "synthesis for every cluster."
                )
    if not groups:
        groups = fallback_groups()
        if store is not None:
            save_template_call(
                store,
                "all_cached_proposals",
                stage=f"{stage}_fallback",
                inputs={"opinions": opinions},
                outputs={
                    "clusters": [{"opinion_ids": ids, "synthesis": synthesis}
                                for ids, synthesis in groups]
                },
            )

    rebuilt: dict[str, list[MergedProposal]] = defaultdict(list)
    output_name = {
        "node_revision": "node_revisions",
        "retrieval_revision": "retrieval_revisions",
        "new_node": "new_nodes",
    }
    for ids, synthesis in groups:
        proposals = [by_id[opinion_id][1] for opinion_id in ids]
        first = proposals[0]
        case_ids = list(dict.fromkeys(
            case_id
            for proposal in proposals
            for case_id in proposal.source_case_ids
        ))
        evidence_items: list[dict[str, Any]] = []
        evidence_seen: set[str] = set()
        for proposal in proposals:
            for evidence in proposal.evidence_items:
                signature = json.dumps(
                    evidence, ensure_ascii=False, sort_keys=True
                )
                if signature not in evidence_seen:
                    evidence_seen.add(signature)
                    evidence_items.append(dict(evidence))
        rebuilt[output_name[first.kind]].append(MergedProposal(
            kind=first.kind,
            content=synthesis,
            target_node=first.target_node,
            source=first.source,
            target=first.target,
            relation=first.relation,
            target_old_edge=first.target_old_edge,
            rationale=" ".join(dict.fromkeys(
                proposal.rationale
                for proposal in proposals
                if proposal.rationale
            ))[:1200],
            support=len(case_ids),
            source_case_ids=case_ids,
            operation=first.operation,
            toxic_text=first.toxic_text,
            semantic_key=first.semantic_key,
            parent_node=first.parent_node,
            activation_edges=_merge_activation_edge_records(
                *[proposal.activation_edges for proposal in proposals]
            ),
            evidence_items=evidence_items,
        ))

    for name in ("node_revisions", "retrieval_revisions", "new_nodes"):
        untouched = [
            proposal
            for proposal in (merged.get(name) or [])
            if _proposal_pool_bucket(proposal) not in selected
        ]
        out[name] = untouched + rebuilt[name]
    out["edges"] = _merge_exact_edges(out.get("edges") or [])
    return out
