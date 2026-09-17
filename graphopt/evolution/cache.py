"""Evolution cache — heuristic stats and proposals separate from the rule graph JSON.

The agent-facing skill file (``initial.json`` format) keeps its original keys only.
All case-level ``used/correct/wrong/retrieval_missed`` counts and revision proposals live here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphopt.evolution.exact_refs import exact_reference_json
from graphopt.evolution.experience_quality import select_positive_examples
from graphopt.evolution.types import CaseAnalysis, GraphEditPlan, MergedProposal, UsageStats
from graphopt.types import SkillGraph

CACHE_SCHEMA = "skillgraph-evolution-cache-v1"


@dataclass
class NodeCacheEntry:
    stats: UsageStats = field(default_factory=UsageStats)
    revision_proposals: list[dict[str, Any]] = field(default_factory=list)
    merged_revisions: list[dict[str, Any]] = field(default_factory=list)
    retrieval_revision_proposals: list[dict[str, Any]] = field(default_factory=list)
    merged_retrieval_revisions: list[dict[str, Any]] = field(default_factory=list)
    execution_reinforcement_proposals: list[dict[str, Any]] = field(default_factory=list)
    positive_examples: list[dict[str, Any]] = field(default_factory=list)
    failure_examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "revision_proposals": list(self.revision_proposals),
            "merged_revisions": list(self.merged_revisions),
            "retrieval_revision_proposals": list(self.retrieval_revision_proposals),
            "merged_retrieval_revisions": list(self.merged_retrieval_revisions),
            "execution_reinforcement_proposals": list(self.execution_reinforcement_proposals),
            "positive_examples": list(self.positive_examples),
            "failure_examples": list(self.failure_examples),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "NodeCacheEntry":
        d = d or {}
        return cls(
            stats=UsageStats.from_dict(d.get("stats")),
            revision_proposals=list(d.get("revision_proposals") or []),
            merged_revisions=list(d.get("merged_revisions") or []),
            retrieval_revision_proposals=list(
                d.get("retrieval_revision_proposals") or []
            ),
            merged_retrieval_revisions=list(
                d.get("merged_retrieval_revisions") or []
            ),
            execution_reinforcement_proposals=list(
                d.get("execution_reinforcement_proposals") or []
            ),
            positive_examples=list(d.get("positive_examples") or []),
            failure_examples=list(d.get("failure_examples") or []),
        )


@dataclass
class EdgeCacheEntry:
    stats: UsageStats = field(default_factory=UsageStats)
    correction_proposals: list[dict[str, Any]] = field(default_factory=list)
    merged_corrections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "correction_proposals": list(self.correction_proposals),
            "merged_corrections": list(self.merged_corrections),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "EdgeCacheEntry":
        d = d or {}
        return cls(
            stats=UsageStats.from_dict(d.get("stats")),
            correction_proposals=list(d.get("correction_proposals") or []),
            merged_corrections=list(d.get("merged_corrections") or []),
        )


@dataclass
class EvolutionCache:
    """Cumulative evolution evidence keyed by existing node/edge ids."""

    schema_version: str = CACHE_SCHEMA
    source_graph: str = ""
    updated_step: int = 0
    nodes: dict[str, NodeCacheEntry] = field(default_factory=dict)
    edges: dict[str, EdgeCacheEntry] = field(default_factory=dict)
    new_node_candidates: list[dict[str, Any]] = field(default_factory=list)
    new_edge_candidates: list[dict[str, Any]] = field(default_factory=list)
    case_analyses: list[dict[str, Any]] = field(default_factory=list)
    merged_proposals: dict[str, Any] = field(default_factory=dict)
    edit_plan: dict[str, Any] = field(default_factory=dict)
    step_history: list[dict[str, Any]] = field(default_factory=list)
    pending_harmful_hints: list[str] = field(default_factory=list)
    pending_bad_case_prompt: str = ""
    case_examples: dict[str, dict[str, Any]] = field(default_factory=dict)
    causal_scope_rejections: list[dict[str, Any]] = field(default_factory=list)
    candidate_scope_outcomes: list[dict[str, Any]] = field(default_factory=list)
    gate_experiences: list[dict[str, Any]] = field(default_factory=list)
    attribution_adjudications: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph: SkillGraph, *, source_graph: str = "") -> "EvolutionCache":
        cache = cls(source_graph=source_graph)
        for nid in graph.nodes:
            cache.nodes.setdefault(nid, NodeCacheEntry())
        for e in graph.edges:
            if e.id:
                cache.edges.setdefault(e.id, EdgeCacheEntry())
        return cache

    @classmethod
    def load(cls, path: str | Path | None) -> "EvolutionCache | None":
        p = Path(path) if path else None
        if not p or not p.is_file():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(obj)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "EvolutionCache":
        if not isinstance(obj, dict):
            raise TypeError("evolution cache must be a JSON object")
        schema = str(obj.get("schema_version") or "")
        if schema != CACHE_SCHEMA:
            raise ValueError(
                f"unsupported evolution cache schema {schema!r}; expected {CACHE_SCHEMA!r}"
            )
        if not isinstance(obj.get("nodes") or {}, dict) or not isinstance(
            obj.get("edges") or {}, dict
        ):
            raise TypeError("evolution cache nodes/edges must be JSON objects")
        nodes = {k: NodeCacheEntry.from_dict(v) for k, v in (obj.get("nodes") or {}).items()}
        edges = {k: EdgeCacheEntry.from_dict(v) for k, v in (obj.get("edges") or {}).items()}
        return cls(
            schema_version=str(obj.get("schema_version") or CACHE_SCHEMA),
            source_graph=str(obj.get("source_graph") or ""),
            updated_step=int(obj.get("updated_step") or 0),
            nodes=nodes,
            edges=edges,
            new_node_candidates=list(obj.get("new_node_candidates") or []),
            new_edge_candidates=list(obj.get("new_edge_candidates") or []),
            case_analyses=list(obj.get("case_analyses") or []),
            merged_proposals=dict(obj.get("merged_proposals") or {}),
            edit_plan=dict(obj.get("edit_plan") or {}),
            step_history=list(obj.get("step_history") or []),
            pending_harmful_hints=list(obj.get("pending_harmful_hints") or []),
            pending_bad_case_prompt=str(obj.get("pending_bad_case_prompt") or ""),
            case_examples={
                str(key): dict(value)
                for key, value in (obj.get("case_examples") or {}).items()
                if isinstance(value, dict)
            },
            causal_scope_rejections=[
                dict(item)
                for item in (obj.get("causal_scope_rejections") or [])
                if isinstance(item, dict)
            ],
            candidate_scope_outcomes=[
                dict(item)
                for item in (obj.get("candidate_scope_outcomes") or [])
                if isinstance(item, dict)
            ],
            gate_experiences=[
                dict(item)
                for item in (obj.get("gate_experiences") or [])
                if isinstance(item, dict)
            ],
            attribution_adjudications={
                str(key): dict(value)
                for key, value in (
                    obj.get("attribution_adjudications") or {}
                ).items()
                if isinstance(value, dict)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_graph": self.source_graph,
            "updated_step": self.updated_step,
            "nodes": {k: v.to_dict() for k, v in sorted(self.nodes.items())},
            "edges": {k: v.to_dict() for k, v in sorted(self.edges.items())},
            "new_node_candidates": list(self.new_node_candidates),
            "new_edge_candidates": list(self.new_edge_candidates),
            "case_analyses": list(self.case_analyses),
            "merged_proposals": dict(self.merged_proposals),
            "edit_plan": dict(self.edit_plan),
            "step_history": list(self.step_history),
            "pending_harmful_hints": list(self.pending_harmful_hints),
            "pending_bad_case_prompt": self.pending_bad_case_prompt,
            "case_examples": {
                key: dict(value) for key, value in sorted(self.case_examples.items())
            },
            "causal_scope_rejections": [
                dict(item) for item in self.causal_scope_rejections
            ],
            "candidate_scope_outcomes": [
                dict(item) for item in self.candidate_scope_outcomes
            ],
            "gate_experiences": [
                dict(item) for item in self.gate_experiences
            ],
            "attribution_adjudications": {
                key: dict(value)
                for key, value in sorted(
                    self.attribution_adjudications.items()
                )
            },
        }

    def has_causal_scope_rejection(
        self,
        *,
        graph_sha256: str,
        edit_scope: str,
        target_nodes: list[str],
    ) -> bool:
        target_key = sorted({str(node_id) for node_id in target_nodes if str(node_id)})
        return any(
            str(item.get("graph_sha256") or "") == str(graph_sha256)
            and str(item.get("edit_scope") or "") == str(edit_scope)
            and sorted(map(str, item.get("target_nodes") or [])) == target_key
            for item in self.causal_scope_rejections
        )

    def record_causal_scope_rejection(
        self,
        *,
        graph_sha256: str,
        edit_scope: str,
        target_nodes: list[str],
        confirmed_cause: str,
        reason: str = "",
        certificate_id: str = "",
        step: int = 0,
    ) -> dict[str, Any]:
        target_key = sorted({str(node_id) for node_id in target_nodes if str(node_id)})
        for item in self.causal_scope_rejections:
            if (
                str(item.get("graph_sha256") or "") == str(graph_sha256)
                and str(item.get("edit_scope") or "") == str(edit_scope)
                and sorted(map(str, item.get("target_nodes") or [])) == target_key
            ):
                item["confirmed_cause"] = str(confirmed_cause)
                item["reason"] = str(reason)
                item["certificate_id"] = str(certificate_id)
                item["last_step"] = int(step)
                item["count"] = int(item.get("count") or 1) + 1
                return item
        item = {
            "graph_sha256": str(graph_sha256),
            "edit_scope": str(edit_scope),
            "target_nodes": target_key,
            "confirmed_cause": str(confirmed_cause),
            "reason": str(reason),
            "certificate_id": str(certificate_id),
            "first_step": int(step),
            "last_step": int(step),
            "count": 1,
        }
        self.causal_scope_rejections.append(item)
        self.causal_scope_rejections = self.causal_scope_rejections[-256:]
        return item

    def candidate_scope_outcome_count(
        self,
        *,
        graph_sha256: str,
        edit_scope: str,
        target_nodes: list[str],
    ) -> int:
        """Return prior failed-Gate count for this scope on this exact graph."""
        target_key = sorted({
            str(node_id) for node_id in target_nodes if str(node_id)
        })
        return sum(
            int(item.get("count") or 1)
            for item in self.candidate_scope_outcomes
            if str(item.get("graph_sha256") or "") == str(graph_sha256)
            and str(item.get("edit_scope") or "") == str(edit_scope)
            and sorted(map(str, item.get("target_nodes") or [])) == target_key
        )

    def record_candidate_scope_outcome(
        self,
        *,
        graph_sha256: str,
        edit_scope: str,
        target_nodes: list[str],
        action: str,
        step: int = 0,
    ) -> dict[str, Any]:
        """Remember a Gate-rejected scope for ranking, never as a hard veto."""
        target_key = sorted({
            str(node_id) for node_id in target_nodes if str(node_id)
        })
        action_key = str(action)
        for item in self.candidate_scope_outcomes:
            if (
                str(item.get("graph_sha256") or "") == str(graph_sha256)
                and str(item.get("edit_scope") or "") == str(edit_scope)
                and sorted(map(str, item.get("target_nodes") or [])) == target_key
                and str(item.get("action") or "") == action_key
            ):
                item["last_step"] = int(step)
                item["count"] = int(item.get("count") or 1) + 1
                return item
        item = {
            "graph_sha256": str(graph_sha256),
            "edit_scope": str(edit_scope),
            "target_nodes": target_key,
            "action": action_key,
            "first_step": int(step),
            "last_step": int(step),
            "count": 1,
        }
        self.candidate_scope_outcomes.append(item)
        self.candidate_scope_outcomes = self.candidate_scope_outcomes[-256:]
        return item

    def ingest_case_examples(
        self,
        results: list[dict[str, Any]],
        *,
        focus_steps_by_case: dict[str, int] | None = None,
    ) -> None:
        """Persist compact rollout evidence so mature cross-epoch edits keep their cases."""
        focus_steps_by_case = focus_steps_by_case or {}
        for result in results:
            case_id = str(result.get("id") or "")
            if not case_id:
                continue
            self.case_examples[case_id] = {
                "case_id": case_id,
                "hard": result.get("hard"),
                "soft": result.get("soft"),
                "task_type": result.get("task_type"),
                "task_description": result.get("task_description"),
                "fail_reason": result.get("fail_reason"),
                "trajectory": result.get("trajectory"),
                "student_step_limit_failure": bool(
                    result.get("student_step_limit_failure")
                ),
                "training_reference_plan": (
                    dict(result["training_reference_plan"])
                    if isinstance(result.get("training_reference_plan"), dict)
                    else None
                ),
                "focus_step": focus_steps_by_case.get(case_id),
            }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_graph_keys(self, graph: SkillGraph) -> None:
        for nid in graph.nodes:
            self.nodes.setdefault(nid, NodeCacheEntry())
        for e in graph.edges:
            if e.id:
                self.edges.setdefault(e.id, EdgeCacheEntry())

    def ingest_positive_examples(
        self,
        examples_by_node: dict[str, list[dict[str, Any]]],
        *,
        cap_per_node: int = 10,
    ) -> None:
        """Persist successful correct-use counterexamples, deduplicated by case ID."""
        for node_id, examples in examples_by_node.items():
            entry = self.nodes.setdefault(node_id, NodeCacheEntry())
            by_case = {
                str(item.get("case_id") or ""): item
                for item in entry.positive_examples
                if str(item.get("case_id") or "")
            }
            for example in examples:
                case_id = str(example.get("case_id") or "")
                if case_id:
                    by_case[case_id] = dict(example)
            entry.positive_examples = select_positive_examples(
                node_id,
                [by_case[key] for key in sorted(by_case)],
                [],
                cap=max(1, int(cap_per_node)),
            )

    def set_node_casebooks(
        self,
        successes_by_node: dict[str, list[dict[str, Any]]],
        failures_by_node: dict[str, list[dict[str, Any]]],
        *,
        success_cap: int = 10,
        failure_cap: int = 6,
    ) -> None:
        """Atomically replace teacher-curated node casebooks."""
        for node_id in sorted(set(successes_by_node) | set(failures_by_node)):
            entry = self.nodes.setdefault(node_id, NodeCacheEntry())
            entry.positive_examples = [
                dict(item) for item in successes_by_node.get(node_id, [])[:success_cap]
            ]
            entry.failure_examples = [
                dict(item) for item in failures_by_node.get(node_id, [])[:failure_cap]
            ]

    def reconcile_graph(self, graph: SkillGraph, *, source_graph: str = "") -> None:
        """Drop stale references and add missing entries for the active graph."""
        valid_nodes = set(graph.nodes)
        valid_edges = {e.id for e in graph.edges if e.id}
        self.nodes = {nid: entry for nid, entry in self.nodes.items() if nid in valid_nodes}
        self.edges = {eid: entry for eid, entry in self.edges.items() if eid in valid_edges}
        self.ensure_graph_keys(graph)
        self.new_edge_candidates = [
            item
            for item in self.new_edge_candidates
            if str(item.get("source") or "") in valid_nodes
            and str(item.get("target") or "") in valid_nodes
        ]
        for nid, entry in self.nodes.items():
            entry.revision_proposals = [
                p for p in entry.revision_proposals if str(p.get("target_node") or nid) == nid
            ]
            entry.merged_revisions = [
                p for p in entry.merged_revisions if str(p.get("target_node") or nid) == nid
            ]
            entry.retrieval_revision_proposals = [
                p
                for p in entry.retrieval_revision_proposals
                if str(p.get("target_node") or nid) == nid
            ]
            entry.merged_retrieval_revisions = [
                p
                for p in entry.merged_retrieval_revisions
                if str(p.get("target_node") or nid) == nid
            ]
            entry.execution_reinforcement_proposals = [
                p for p in entry.execution_reinforcement_proposals
                if str(p.get("target_node") or nid) == nid
            ]
        for eid, entry in self.edges.items():
            entry.correction_proposals = [
                p for p in entry.correction_proposals if str(p.get("target_edge") or eid) == eid
            ]
            entry.merged_corrections = [
                p
                for p in entry.merged_corrections
                if str(p.get("target_old_edge") or eid) == eid
                and str(p.get("source") or "") in valid_nodes
                and str(p.get("target") or "") in valid_nodes
            ]
        if source_graph:
            self.source_graph = source_graph

    def ingest_step(
        self,
        *,
        step: int,
        graph: SkillGraph,
        analyses: list[CaseAnalysis],
        node_stats: dict[str, UsageStats],
        edge_stats: dict[str, UsageStats],
        merged: dict[str, list[MergedProposal]],
        plan: GraphEditPlan,
        persist_proposal_pool: bool = False,
    ) -> None:
        """Update runtime protection/feedback state after one evolution step.

        Proposal fields are retained only in the explicit legacy persistence
        mode. Formal grouped training generates candidates from the current
        frozen epoch and therefore clears opinion pools after writing step artifacts.
        """
        self.ensure_graph_keys(graph)
        self.updated_step = step

        for nid, st in node_stats.items():
            entry = self.nodes.setdefault(nid, NodeCacheEntry())
            entry.stats = UsageStats(
                st.used, st.correct, st.wrong, st.retrieval_missed
            )

        for eid, st in edge_stats.items():
            entry = self.edges.setdefault(eid, EdgeCacheEntry())
            entry.stats = UsageStats(
                st.used, st.correct, st.wrong, st.retrieval_missed
            )

        for a in analyses:
            rec = a.to_dict()
            rec["step"] = step
            self.case_analyses.append(rec)
            if a.success:
                continue
            # Lapse opinions always belong to the reinforcement channel, also
            # when another defect makes the case failure_type MIXED.
            lapse_targets = set(a.execution_lapse_nodes)
            for p in a.node_revision_proposals:
                nid = p.target_node
                entry = self.nodes.setdefault(nid, NodeCacheEntry())
                if nid in lapse_targets:
                    self._upsert_execution_reinforcement(entry, {**p.to_dict(), "step": step})
                else:
                    entry.revision_proposals.append({**p.to_dict(), "step": step})
            for p in a.retrieval_revision_proposals:
                nid = p.target_node
                entry = self.nodes.setdefault(nid, NodeCacheEntry())
                entry.retrieval_revision_proposals.append(
                    {**p.to_dict(), "step": step}
                )
            for p in a.edge_correction_proposals:
                eid = p.target_edge
                if eid:
                    entry = self.edges.setdefault(eid, EdgeCacheEntry())
                    entry.correction_proposals.append({**p.to_dict(), "step": step})

        def evidence_ids(m: MergedProposal) -> list[str]:
            # Support is global distinct-case support. Epoch belongs in the
            # surrounding record, never in the identity used for voting.
            return list(dict.fromkeys(str(cid) for cid in m.source_case_ids))

        for m in merged.get("node_revisions") or []:
            nid = m.target_node
            if not nid:
                continue
            entry = self.nodes.setdefault(nid, NodeCacheEntry())
            old = next((
                x for x in entry.merged_revisions
                if x.get("content") == m.content
                and str(x.get("operation") or "PATCH").upper() == m.operation
                and str(x.get("toxic_text") or "") == m.toxic_text
            ), None)
            if old is None:
                payload = {**m.to_dict(), "step": step}
                payload["source_case_ids"] = evidence_ids(m)
                entry.merged_revisions.append(payload)
            else:
                ids = list(dict.fromkeys(list(old.get("source_case_ids") or []) + evidence_ids(m)))
                old.update({**m.to_dict(), "step": step, "source_case_ids": ids, "support": len(ids)})

        for m in merged.get("retrieval_revisions") or []:
            nid = m.target_node
            if not nid:
                continue
            entry = self.nodes.setdefault(nid, NodeCacheEntry())
            old = next(
                (
                    x
                    for x in entry.merged_retrieval_revisions
                    if x.get("content") == m.content
                ),
                None,
            )
            if old is None:
                payload = {**m.to_dict(), "step": step}
                payload["source_case_ids"] = evidence_ids(m)
                entry.merged_retrieval_revisions.append(payload)
            else:
                ids = list(
                    dict.fromkeys(
                        list(old.get("source_case_ids") or []) + evidence_ids(m)
                    )
                )
                old.update(
                    {
                        **m.to_dict(),
                        "step": step,
                        "source_case_ids": ids,
                        "support": len(ids),
                    }
                )

        edge_merged = merged.get("edges") or []
        for m in edge_merged:
            old = m.target_old_edge
            payload = {**m.to_dict(), "step": step, "source_case_ids": evidence_ids(m)}
            if old:
                entry = self.edges.setdefault(old, EdgeCacheEntry())
                old_item = next(
                    (x for x in entry.merged_corrections if x.get("content") == m.content), None
                )
                if old_item is None:
                    entry.merged_corrections.append(payload)
                else:
                    ids = list(dict.fromkeys(list(old_item.get("source_case_ids") or []) + evidence_ids(m)))
                    old_item.update({**payload, "source_case_ids": ids, "support": len(ids)})
            else:
                self._upsert_new_edge_candidate(payload, persist=persist_proposal_pool)

        self.merged_proposals = {k: [x.to_dict() for x in v] for k, v in merged.items()}
        self.edit_plan = plan.to_dict()

        new_nodes = [
            {**m.to_dict(), "source_case_ids": evidence_ids(m)}
            for m in (merged.get("new_nodes") or [])
        ]
        if persist_proposal_pool:
            self._merge_new_node_candidates(new_nodes, step=step)
        else:
            self.new_node_candidates = []
            self.new_edge_candidates = []
            self.merged_proposals = {}
            self.edit_plan = {}
            for entry in self.nodes.values():
                entry.revision_proposals = []
                entry.merged_revisions = []
                entry.retrieval_revision_proposals = []
                entry.merged_retrieval_revisions = []
                entry.execution_reinforcement_proposals = []
            for entry in self.edges.values():
                entry.correction_proposals = []
                entry.merged_corrections = []

        self.step_history.append(
            {
                "step": step,
                "n_cases": len(analyses),
                "n_failures": sum(1 for a in analyses if not a.success),
                "edit_plan_summary": {
                    "update_nodes": len(plan.update_nodes),
                    "add_nodes": len(plan.add_nodes),
                    "add_edges": len(plan.add_edges),
                    "replace_edges": len(plan.replace_edges),
                    "delete_nodes": len(plan.delete_nodes),
                    "delete_edges": len(plan.delete_edges),
                },
            }
        )

        for nid in plan.delete_nodes:
            self.nodes.pop(nid, None)
        for eid in plan.delete_edges:
            self.edges.pop(eid, None)

    @staticmethod
    def _upsert_execution_reinforcement(entry: NodeCacheEntry, payload: dict[str, Any]) -> None:
        """Store one lossless execution record per distinct parent/case pair."""
        node_id = str(payload.get("target_node") or "")
        proposal = str(payload.get("proposal") or "").strip()
        case_id = str(payload.get("case_id") or "")
        if not node_id or not proposal or not case_id:
            return
        evidence = {
            "target_node": node_id,
            "case_id": case_id,
            "proposal": proposal,
            "reason": str(payload.get("reason") or ""),
            "observable_state": str(payload.get("observable_state") or ""),
            "bad_action": str(payload.get("bad_action") or ""),
            "better_action": str(payload.get("better_action") or ""),
        }
        stored = {
            **payload,
            "semantic_signature": f"execution_case:{node_id}:{case_id}",
            "source_case_ids": [case_id],
            "proposals": [proposal],
            "evidence_items": [evidence],
            "support": 1,
        }
        for index, item in enumerate(entry.execution_reinforcement_proposals):
            if case_id in {
                str(value) for value in (item.get("source_case_ids") or [])
            }:
                entry.execution_reinforcement_proposals[index] = stored
                return
        entry.execution_reinforcement_proposals.append(stored)

    def _upsert_new_edge_candidate(self, payload: dict[str, Any], *, persist: bool) -> None:
        key = (payload.get("source"), payload.get("target"), payload.get("relation"))
        for i, c in enumerate(self.new_edge_candidates):
            ck = (c.get("source"), c.get("target"), c.get("relation"))
            if ck == key:
                if persist:
                    c.update(payload)
                    ids = list(dict.fromkeys(list(c.get("source_case_ids") or []) + list(payload.get("source_case_ids") or [])))
                    c["source_case_ids"] = ids
                    c["support"] = len(ids)
                else:
                    self.new_edge_candidates[i] = payload
                return
        self.new_edge_candidates.append(payload)

    def _merge_new_node_candidates(self, items: list[dict[str, Any]], *, step: int) -> None:
        for item in items:
            payload = {**item, "step": step}
            sig = " ".join((item.get("content") or "").lower().split())[:280]
            parent_node = str(item.get("parent_node") or "")
            found = False
            for c in self.new_node_candidates:
                cs = " ".join((c.get("content") or "").lower().split())[:280]
                if cs == sig and str(c.get("parent_node") or "") == parent_node:
                    previous_content = str(c.get("content") or "").strip()
                    current_content = str(payload.get("content") or "").strip()
                    c.update(payload)
                    if previous_content and previous_content != current_content:
                        c["content"] = f"{previous_content}\n{current_content}".strip()
                        c["merged_proposal"] = c["content"]
                    ids = list(dict.fromkeys(list(c.get("source_case_ids") or []) + list(payload.get("source_case_ids") or [])))
                    c["source_case_ids"] = ids
                    c["support"] = len(ids)
                    found = True
                    break
            if not found:
                self.new_node_candidates.append(payload)

    def reconcile_semantic_proposal_pool(
        self, merged: dict[str, list[MergedProposal]]
    ) -> None:
        """Replace lexical cache fragments with the canonical semantic clusters."""
        revisions: dict[str, list[dict[str, Any]]] = {}
        retrievals: dict[str, list[dict[str, Any]]] = {}
        for proposal in merged.get("node_revisions") or []:
            if proposal.target_node:
                revisions.setdefault(proposal.target_node, []).append(
                    {**proposal.to_dict(), "step": self.updated_step}
                )
        for proposal in merged.get("retrieval_revisions") or []:
            if proposal.target_node:
                retrievals.setdefault(proposal.target_node, []).append(
                    {**proposal.to_dict(), "step": self.updated_step}
                )
        for node_id, entry in self.nodes.items():
            entry.merged_revisions = revisions.get(node_id, [])
            entry.merged_retrieval_revisions = retrievals.get(node_id, [])
        self.new_node_candidates = [
            {**proposal.to_dict(), "step": self.updated_step}
            for proposal in (merged.get("new_nodes") or [])
        ]

    def record_gate_experience(self, experience: dict[str, Any]) -> None:
        """Persist measured Local/Big Gate experience across graph rollbacks."""
        payload = dict(experience or {})
        if not bool(payload.get("carry_to_next_epoch")):
            return
        epoch = int(payload.get("epoch") or 0)
        if epoch <= 0:
            raise ValueError("gate experience requires a positive epoch")
        self.gate_experiences = [
            item for item in self.gate_experiences
            if int(item.get("epoch") or 0) != epoch
        ]
        self.gate_experiences.append(payload)
        self.gate_experiences.sort(key=lambda item: int(item.get("epoch") or 0))

    def format_gate_experiences_for_prompt(self) -> str:
        """Return every prior measured Gate lesson; never include unseen tests."""
        reusable = [
            item for item in self.gate_experiences
            if bool(item.get("carry_to_next_epoch"))
        ]
        if not reusable:
            return ""
        return (
            "## Failed modifications from prior complete Big-Gate rollbacks\n"
            "Consult a lesson only for the same node or edge. The list contains "
            "the rejected edit and its measured failure reason; ordinary prior "
            "opinions and accepted-epoch records are excluded.\n"
            + exact_reference_json({"rollback_lessons": reusable}, indent=None)
        )

    def set_pending_harmful_hints(self, hints: list[str]) -> None:
        """Replace pending harmful hints (at most one gate-rejection batch)."""
        self.pending_harmful_hints = [h.strip() for h in hints if h and h.strip()]

    def set_pending_bad_case_prompt(self, prompt: str) -> None:
        """Replace, never append, the one prompt available to the next epoch."""
        self.pending_bad_case_prompt = str(prompt or "").strip()

    def consume_planned_proposals(self, plan: GraphEditPlan) -> None:
        """Remove evidence that has just crossed a threshold into the patch.

        Without consumption a mature cached proposal would generate the same
        edit forever, even on later steps containing no matching failure.
        """
        for item in plan.update_nodes:
            entry = self.nodes.get(str(item.get("node_id") or ""))
            if entry is None:
                continue
            if str(item.get("revision_scope") or "") == "execution_reinforcement":
                signatures = {str(x) for x in (item.get("semantic_signatures") or []) if str(x)}
                entry.execution_reinforcement_proposals = [
                    p for p in entry.execution_reinforcement_proposals
                    if str(p.get("semantic_signature") or "") not in signatures
                ]
                continue
            selected = item.get("selected_revisions") or []
            identities = {
                (
                    str(proposal.get("content") or ""),
                    str(proposal.get("operation") or "PATCH").upper(),
                    str(proposal.get("toxic_text") or ""),
                )
                for proposal in selected
                if isinstance(proposal, dict)
            }
            if not identities:
                identities = {(
                    str(item.get("merged_revision") or ""),
                    str(item.get("revision_scope") or "PATCH").upper(),
                    str(item.get("toxic_text") or ""),
                )}
            entry.merged_revisions = [
                p
                for p in entry.merged_revisions
                if (
                    str(p.get("content") or ""),
                    str(p.get("operation") or "PATCH").upper(),
                    str(p.get("toxic_text") or ""),
                ) not in identities
            ]
            if str(item.get("revision_scope") or "full") == "when_to_use":
                entry.merged_retrieval_revisions = [
                    p
                    for p in entry.merged_retrieval_revisions
                    if str(p.get("content") or "") not in {
                        identity[0] for identity in identities
                    }
                ]

        for item in plan.add_nodes:
            if str(item.get("kind") or "") != "execution_detail":
                continue
            parent = str(item.get("parent_node") or "")
            selected_ids = {
                str(value) for value in (item.get("source_case_ids") or [])
            }
            entry = self.nodes.get(parent)
            if entry is not None and selected_ids:
                kept: list[dict[str, Any]] = []
                for proposal in entry.execution_reinforcement_proposals:
                    old_ids = [
                        str(value)
                        for value in (proposal.get("source_case_ids") or [])
                    ]
                    remaining_ids = [
                        case_id for case_id in old_ids
                        if case_id not in selected_ids
                    ]
                    if not remaining_ids:
                        continue
                    if len(remaining_ids) != len(old_ids):
                        proposal = dict(proposal)
                        proposal["source_case_ids"] = remaining_ids
                        proposal["support"] = len(remaining_ids)
                        proposal["evidence_items"] = [
                            evidence
                            for evidence in (proposal.get("evidence_items") or [])
                            if str(evidence.get("case_id") or "") in remaining_ids
                        ]
                    kept.append(proposal)
                entry.execution_reinforcement_proposals = kept

        mature_new_nodes = {
            (str(item.get("content") or ""), str(item.get("parent_node") or ""))
            for item in plan.add_nodes
        }
        if mature_new_nodes:
            from graphopt.evolution.merge import _similar

            self.new_node_candidates = [
                p
                for p in self.new_node_candidates
                if not any(
                    str(p.get("parent_node") or "") == parent
                    and _similar(str(p.get("content") or ""), content)
                    for content, parent in mature_new_nodes
                )
            ]

        for item in plan.replace_edges:
            old_id = str(item.get("old_edge") or "")
            entry = self.edges.get(old_id)
            if entry is not None:
                entry.merged_corrections = []

        mature_edges = {
            (
                str(item.get("source") or ""),
                str(item.get("target") or ""),
                str(item.get("relation") or ""),
            )
            for item in plan.add_edges
        }
        if mature_edges:
            self.new_edge_candidates = [
                p
                for p in self.new_edge_candidates
                if (
                    str(p.get("source") or ""),
                    str(p.get("target") or ""),
                    str(p.get("relation") or ""),
                )
                not in mature_edges
            ]

    @staticmethod
    def _plan_for_edits(plan: GraphEditPlan, edits: list[Any]) -> GraphEditPlan:
        """Return only proposal-pool entries represented by these materialized edits."""
        node_updates = {e.node_id for e in edits if e.op == "update_node"}
        node_adds = {e.node_id for e in edits if e.op == "add_node"}
        edge_adds = {(e.src, e.dst, e.edge_type) for e in edits if e.op == "add_edge"}
        return GraphEditPlan(
            update_nodes=[item for item in plan.update_nodes if item.get("node_id") in node_updates],
            add_nodes=[item for item in plan.add_nodes if item.get("node_id") in node_adds],
            replace_edges=[item for item in plan.replace_edges if (
                str((item.get("new_edge") or item).get("source") or ""),
                str((item.get("new_edge") or item).get("target") or ""),
                str((item.get("new_edge") or item).get("relation") or "prereq"),
            ) in edge_adds],
            add_edges=[item for item in plan.add_edges if (
                str(item.get("source") or ""), str(item.get("target") or ""),
                str(item.get("relation") or "prereq"),
            ) in edge_adds],
        )

    def consume_accepted_proposals(self, plan: GraphEditPlan, edits: list[Any]) -> None:
        """Consume only proposal evidence represented by Gate-accepted edits."""
        self.consume_planned_proposals(self._plan_for_edits(plan, edits))

    def consume_rejected_proposals(self, plan: GraphEditPlan, edits: list[Any]) -> None:
        """Retire only proposal evidence represented by the rejected candidate.

        Capacity-deferred atomic groups were never behaviorally tested and must
        not be silently retired with another edit. They remain in the pool only
        when proposal persistence is enabled; formal static runs save them as
        diagnostics and require fresh later evidence to propose them again.
        """
        self.consume_planned_proposals(self._plan_for_edits(plan, edits))

    def consume_harmful_prompt(self) -> str:
        """Take only the immediately previous rejection context, then clear it."""
        from graphopt.evaluation.graph_usage import format_harmful_hints_for_prompt

        prompt = self.pending_bad_case_prompt.strip()
        if not prompt:
            prompt = format_harmful_hints_for_prompt(self.pending_harmful_hints)
        self.pending_bad_case_prompt = ""
        self.pending_harmful_hints = []
        return prompt
