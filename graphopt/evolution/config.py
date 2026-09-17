"""Graph evolution configuration (case-semantic attribution pipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvolutionConfig:
    # Candidate generation is intentionally single-case. The mapped
    # validation small Gate, not a train-evidence vote count, decides whether
    # an edit may enter the formal graph.
    node_support_threshold: int = 1
    edge_support_threshold: int = 1
    delete_used_threshold: int = 4  # node d — delete old node when used < d
    edge_delete_used_threshold: int = 4  # edge deletion threshold (separate)
    delete_never_used_node: bool = False
    delete_never_used_edge: bool = False
    edge_weight_quantiles: int = 2
    # Historical semantic opinions must not be accumulated to unlock an edit.
    # EvolutionCache remains useful for positive examples and Gate feedback.
    persist_proposal_pool: bool = False
    edge_tie_policy: str = "keep_only"  # legacy; non-wrong old edge is retained alone
    # Field-local limit: retain one semantically merged body revision per node.
    # Retrieval-trigger revisions are ranked separately, so one body change and
    # one when_to_use change may coexist for the same node.
    node_merge_top_k: int = 1
    # Legacy artifact field; fixed to one because execution candidates are
    # admitted by mapped validation rather than train-vote accumulation.
    execution_reinforcement_threshold: int = 1
    # Applied once in the single evolution call for the frozen epoch.
    # The legacy field name is retained for config compatibility.
    max_execution_child_candidates_per_epoch: int = 1
    max_active_execution_children_per_parent: int = 4
    execution_merge_evidence_cap: int = 64
    analyst_workers: int = 16  # concurrent teacher Case Analyzer calls
    success_guard_workers: int = 16  # concurrent all-success summary chunks
    # Zero preserves strict teacher synthesis. A dataset pipeline may set a
    # positive ceiling when a transitive component cannot fit the complete
    # per-edit response contract; oversized components then use the same safe
    # fallback used after three invalid teacher responses.
    max_joint_semantic_component_edits: int = 0
    # Bounded per-rewrite examples keep the rewriter focused; complete success guards
    # still constrain joint synthesis and paired Gate scopes.
    positive_example_cap: int = 2
    failure_example_cap: int = 3
    # Ablation-only switch: successes are still scored by Gate, but are not
    # shown to the teacher while synthesizing node semantics.
    use_positive_context: bool = True

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> "EvolutionConfig":
        c = dict(cfg or {})
        # One eligible trace may propose a candidate; the epoch-wide local paired Gate
        # supplies independent acceptance evidence over every affected case.
        x = 1
        y = 1
        k = int(c.get("delete_used_threshold") or c.get("delete_threshold") or 4)
        return cls(
            node_support_threshold=x,
            edge_support_threshold=y,
            delete_used_threshold=k,
            edge_delete_used_threshold=int(c.get("edge_delete_used_threshold") or 4),
            delete_never_used_node=bool(c.get("delete_never_used_node", False)),
            delete_never_used_edge=bool(c.get("delete_never_used_edge", False)),
            edge_weight_quantiles=int(c.get("edge_weight_quantiles") or 2),
            persist_proposal_pool=False,
            edge_tie_policy=str(c.get("edge_tie_policy") or "keep_only"),
            node_merge_top_k=max(1, int(c.get("node_merge_top_k") or 1)),
            execution_reinforcement_threshold=1,
            max_execution_child_candidates_per_epoch=max(
                0,
                int(
                    c.get("max_execution_child_candidates_per_epoch", 1)
                ),
            ),
            max_active_execution_children_per_parent=max(
                1, int(c.get("max_active_execution_children_per_parent") or 4)
            ),
            execution_merge_evidence_cap=max(
                1, int(c.get("execution_merge_evidence_cap") or 64)
            ),
            analyst_workers=max(1, int(c.get("analyst_workers") or 16)),
            success_guard_workers=max(
                1, int(c.get("success_guard_workers") or 16)
            ),
            max_joint_semantic_component_edits=max(
                0, int(c.get("max_joint_semantic_component_edits") or 0)
            ),
            positive_example_cap=max(1, int(c.get("positive_example_cap") or 2)),
            failure_example_cap=max(1, int(c.get("failure_example_cap") or 3)),
            use_positive_context=bool(c.get("use_positive_context", True)),
        )
