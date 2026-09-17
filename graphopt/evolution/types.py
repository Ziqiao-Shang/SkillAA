"""Data structures for case-level semantic attribution and graph evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageStats:
    used: int = 0
    correct: int = 0
    wrong: int = 0
    retrieval_missed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "used": self.used,
            "correct": self.correct,
            "wrong": self.wrong,
            "retrieval_missed": self.retrieval_missed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "UsageStats":
        d = d or {}
        return cls(
            used=int(d.get("used") or 0),
            correct=int(d.get("correct") or 0),
            wrong=int(d.get("wrong") or 0),
            retrieval_missed=int(d.get("retrieval_missed") or 0),
        )


@dataclass
class NodeRevisionProposal:
    target_node: str
    proposal: str
    reason: str = ""
    case_id: str = ""
    operation: str = "PATCH"
    toxic_text: str = ""
    first_wrong_step: int = -1
    observable_state: str = ""
    bad_action: str = ""
    better_action: str = ""
    semantic_delta: str = ""

    def to_dict(self) -> dict[str, Any]:
        record = {
            "target_node": self.target_node,
            "proposal": self.proposal,
            "reason": self.reason,
            "case_id": self.case_id,
            "operation": self.operation,
            "toxic_text": self.toxic_text,
            "first_wrong_step": self.first_wrong_step,
            "observable_state": self.observable_state,
            "bad_action": self.bad_action,
            "better_action": self.better_action,
            "semantic_delta": self.semantic_delta,
        }
        return record


@dataclass
class RetrievalRevisionProposal:
    """A trigger-only revision for a correct skill that was not retrieved."""

    target_node: str
    proposed_when_to_use: str
    reason: str = ""
    case_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_node": self.target_node,
            "proposed_when_to_use": self.proposed_when_to_use,
            "reason": self.reason,
            "case_id": self.case_id,
        }


@dataclass
class EdgeCorrectionProposal:
    target_edge: str
    source: str
    target: str
    relation: str
    reason: str = ""
    case_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_edge": self.target_edge,
            "proposal": {
                "source": self.source,
                "target": self.target,
                "relation": self.relation,
            },
            "reason": self.reason,
            "case_id": self.case_id,
        }


@dataclass
class NewNodeProposal:
    temp_id: str
    content: str
    case_id: str = ""
    parent_node: str = ""
    relation: str = "enhance"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "temp_id": self.temp_id,
            "content": self.content,
            "case_id": self.case_id,
            "parent_node": self.parent_node,
            "relation": self.relation,
            "reason": self.reason,
        }


@dataclass
class NewEdgeProposal:
    source: str
    target: str
    relation: str
    reason: str = ""
    case_id: str = ""
    target_old_edge: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "reason": self.reason,
            "case_id": self.case_id,
        }
        if self.target_old_edge:
            d["target_old_edge"] = self.target_old_edge
        return d


@dataclass
class CaseAnalysis:
    case_id: str
    success: bool
    failure_type: str = "UNATTRIBUTED"
    analysis_partial: bool = False
    badcase_summary: dict[str, str] = field(default_factory=dict)
    same_sample_analysis: dict[str, Any] = field(default_factory=dict)
    root_cause_code: str = "INSUFFICIENT_EVIDENCE"
    trace_evidence: dict[str, Any] = field(default_factory=dict)
    attributed_nodes: list[str] = field(default_factory=list)
    attributed_edges: list[str] = field(default_factory=list)
    used_nodes: list[str] = field(default_factory=list)
    correct_nodes: list[str] = field(default_factory=list)
    wrong_nodes: list[str] = field(default_factory=list)
    used_edges: list[str] = field(default_factory=list)
    correct_edges: list[str] = field(default_factory=list)
    wrong_edges: list[str] = field(default_factory=list)
    missed_relevant_nodes: list[str] = field(default_factory=list)
    missed_relevant_edges: list[str] = field(default_factory=list)
    execution_lapse_nodes: list[str] = field(default_factory=list)
    execution_lapse_edges: list[str] = field(default_factory=list)
    node_revision_proposals: list[NodeRevisionProposal] = field(default_factory=list)
    retrieval_revision_proposals: list[RetrievalRevisionProposal] = field(default_factory=list)
    edge_correction_proposals: list[EdgeCorrectionProposal] = field(default_factory=list)
    new_node_proposals: list[NewNodeProposal] = field(default_factory=list)
    new_edge_proposals: list[NewEdgeProposal] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.success and self.failure_type == "UNATTRIBUTED":
            self.failure_type = "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        record = {
            "case_id": self.case_id,
            "success": self.success,
            "failure_type": self.failure_type,
            "analysis_partial": self.analysis_partial,
            "badcase_summary": dict(self.badcase_summary),
            "trace_attribution": {
                "root_cause_code": self.root_cause_code,
                "attributed_nodes": list(self.attributed_nodes),
                "attributed_edges": list(self.attributed_edges),
                "evidence": dict(self.trace_evidence),
            },
            "existing_graph_usage": {
                "used_nodes": list(self.used_nodes),
                "used_edges": list(self.used_edges),
                "correct_nodes": list(self.correct_nodes),
                "correct_edges": list(self.correct_edges),
                "wrong_nodes": list(self.wrong_nodes),
                "wrong_edges": list(self.wrong_edges),
                "missed_relevant_nodes": list(self.missed_relevant_nodes),
                "missed_relevant_edges": list(self.missed_relevant_edges),
                "execution_lapse_nodes": list(self.execution_lapse_nodes),
                "execution_lapse_edges": list(self.execution_lapse_edges),
            },
            "node_revision_proposals": [p.to_dict() for p in self.node_revision_proposals],
            "retrieval_revision_proposals": [
                p.to_dict() for p in self.retrieval_revision_proposals
            ],
            "edge_correction_proposals": [p.to_dict() for p in self.edge_correction_proposals],
            "new_node_proposals": [p.to_dict() for p in self.new_node_proposals],
            "new_edge_proposals": [p.to_dict() for p in self.new_edge_proposals],
        }
        if self.same_sample_analysis:
            record["same_sample_analysis"] = dict(self.same_sample_analysis)
        return record


    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CaseAnalysis":
        d = d or {}
        attribution = d.get("trace_attribution") or {}
        usage = d.get("existing_graph_usage") or {}
        return cls(
            case_id=str(d.get("case_id") or ""), success=bool(d.get("success")),
            failure_type=str(d.get("failure_type") or "UNATTRIBUTED"),
            analysis_partial=bool(d.get("analysis_partial")),
            badcase_summary=dict(d.get("badcase_summary") or {}),
            same_sample_analysis=dict(d.get("same_sample_analysis") or {}),
            root_cause_code=str(attribution.get("root_cause_code") or "INSUFFICIENT_EVIDENCE"),
            trace_evidence=dict(attribution.get("evidence") or {}),
            attributed_nodes=list(attribution.get("attributed_nodes") or []),
            attributed_edges=list(attribution.get("attributed_edges") or []),
            used_nodes=list(usage.get("used_nodes") or []),
            correct_nodes=list(usage.get("correct_nodes") or []),
            wrong_nodes=list(usage.get("wrong_nodes") or []),
            used_edges=list(usage.get("used_edges") or []),
            correct_edges=list(usage.get("correct_edges") or []),
            wrong_edges=list(usage.get("wrong_edges") or []),
            missed_relevant_nodes=list(usage.get("missed_relevant_nodes") or []),
            missed_relevant_edges=list(usage.get("missed_relevant_edges") or []),
            execution_lapse_nodes=list(usage.get("execution_lapse_nodes") or []),
            execution_lapse_edges=list(usage.get("execution_lapse_edges") or []),
            node_revision_proposals=[NodeRevisionProposal(**item) for item in (d.get("node_revision_proposals") or [])],
            retrieval_revision_proposals=[RetrievalRevisionProposal(**item) for item in (d.get("retrieval_revision_proposals") or [])],
            edge_correction_proposals=[EdgeCorrectionProposal(
                target_edge=str(item.get("target_edge") or ""),
                source=str((item.get("proposal") or {}).get("source") or ""),
                target=str((item.get("proposal") or {}).get("target") or ""),
                relation=str((item.get("proposal") or {}).get("relation") or ""),
                reason=str(item.get("reason") or ""), case_id=str(item.get("case_id") or ""),
            ) for item in (d.get("edge_correction_proposals") or [])],
            new_node_proposals=[NewNodeProposal(**item) for item in (d.get("new_node_proposals") or [])],
            new_edge_proposals=[NewEdgeProposal(**item) for item in (d.get("new_edge_proposals") or [])],
        )


@dataclass
class MergedProposal:
    kind: str  # node_revision | new_node | edge
    content: str = ""
    target_node: str = ""
    source: str = ""
    target: str = ""
    relation: str = ""
    target_old_edge: str = ""
    rationale: str = ""
    support: int = 0
    source_case_ids: list[str] = field(default_factory=list)
    operation: str = "PATCH"
    toxic_text: str = ""
    semantic_key: str = ""
    parent_node: str = ""
    activation_edges: list[dict[str, Any]] = field(default_factory=list)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "support": self.support,
            "source_case_ids": list(self.source_case_ids),
        }
        if self.target_node:
            d["target_node"] = self.target_node
        if self.kind == "node_revision":
            d["operation"] = self.operation
            if self.toxic_text:
                d["toxic_text"] = self.toxic_text
        if self.content:
            d["content"] = self.content
            d["merged_proposal"] = self.content
        if self.source and self.target:
            d["source"] = self.source
            d["target"] = self.target
            d["relation"] = self.relation
        if self.target_old_edge:
            d["target_old_edge"] = self.target_old_edge
        if self.rationale:
            d["rationale"] = self.rationale
        if self.semantic_key:
            d["semantic_key"] = self.semantic_key
        if self.parent_node:
            d["parent_node"] = self.parent_node
        if self.activation_edges:
            d["activation_edges"] = [dict(item) for item in self.activation_edges]
        if self.evidence_items:
            d["evidence_items"] = [dict(item) for item in self.evidence_items]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MergedProposal":
        d = d or {}
        return cls(
            kind=str(d.get("kind") or ""),
            content=str(d.get("content") or d.get("merged_proposal") or ""),
            target_node=str(d.get("target_node") or ""),
            source=str(d.get("source") or ""),
            target=str(d.get("target") or ""),
            relation=str(d.get("relation") or ""),
            target_old_edge=str(d.get("target_old_edge") or ""),
            rationale=str(d.get("rationale") or ""),
            support=int(d.get("support") or 0),
            source_case_ids=[str(x) for x in (d.get("source_case_ids") or [])],
            operation=str(d.get("operation") or "PATCH"),
            toxic_text=str(d.get("toxic_text") or ""),
            semantic_key=str(d.get("semantic_key") or ""),
            parent_node=str(d.get("parent_node") or ""),
            activation_edges=[dict(x) for x in (d.get("activation_edges") or [])],
            evidence_items=[dict(x) for x in (d.get("evidence_items") or [])],
        )



@dataclass
class GraphEditPlan:
    update_nodes: list[dict[str, Any]] = field(default_factory=list)
    add_nodes: list[dict[str, Any]] = field(default_factory=list)
    keep_edges: list[str] = field(default_factory=list)
    replace_edges: list[dict[str, Any]] = field(default_factory=list)
    add_edges: list[dict[str, Any]] = field(default_factory=list)
    delete_nodes: list[str] = field(default_factory=list)
    delete_edges: list[str] = field(default_factory=list)
    edge_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_nodes": self.update_nodes,
            "add_nodes": self.add_nodes,
            "keep_edges": self.keep_edges,
            "replace_edges": self.replace_edges,
            "add_edges": self.add_edges,
            "delete_nodes": self.delete_nodes,
            "delete_edges": self.delete_edges,
            "edge_weights": self.edge_weights,
        }


    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GraphEditPlan":
        d = d or {}
        return cls(
            update_nodes=[dict(x) for x in (d.get("update_nodes") or [])],
            add_nodes=[dict(x) for x in (d.get("add_nodes") or [])],
            keep_edges=[str(x) for x in (d.get("keep_edges") or [])],
            replace_edges=[dict(x) for x in (d.get("replace_edges") or [])],
            add_edges=[dict(x) for x in (d.get("add_edges") or [])],
            delete_nodes=[str(x) for x in (d.get("delete_nodes") or [])],
            delete_edges=[str(x) for x in (d.get("delete_edges") or [])],
            edge_weights={str(k): float(v) for k, v in (d.get("edge_weights") or {}).items()},
        )


@dataclass
class EvolutionResult:
    patch_edits: list[Any]
    case_analyses: list[CaseAnalysis]
    graph_statistics: dict[str, Any]
    merged_proposals: dict[str, Any]
    edit_plan: GraphEditPlan
    reasoning: str = ""
    node_stats: dict[str, Any] | None = None
    edge_stats: dict[str, Any] | None = None
    cache: Any | None = None
