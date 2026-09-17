"""Root-cause certificates for the optional causal evolution protocol.

The module is deliberately pure: it builds probe plans and decides whether an
already-generated atomic patch has enough causal evidence to reach the existing
Gate.  It never mutates a graph and legacy evolution does not import it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from graphopt.engine.grouped_schedule import GroupedBatch
from graphopt.evolution.types import EvolutionResult
from graphopt.types import GraphEdit, GraphPatch, SkillGraph
from graphopt.optimizer.skill import atomic_edit_groups


ROOT_CAUSES = frozenset({
    "INSUFFICIENT_EVIDENCE",
    "UNSTABLE_EXECUTION",
    "COVERED_BUT_INACTIVE",
    "PRECEDENCE_CONFLICT",
    "HARMFUL_RULE",
    "MISSING_SEMANTICS",
    "UNRESOLVED",
    "INSUFFICIENT_GROUP_SUPPORT",
    "INCOMPLETE_INTERVENTION",
})


@dataclass(frozen=True)
class CausalPolicy:
    shadow_mode: bool = False
    max_clusters_per_batch: int = 2
    representatives_per_cluster: int = 2
    trigger_min_groups: int = 2
    harmful_min_groups: int = 2
    content_min_groups: int = 3
    structural_min_groups: int = 3
    trace_min_groups: int = 1

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> "CausalPolicy":
        cfg = dict(cfg or {})
        return cls(
            shadow_mode=bool(cfg.get("causal_shadow_mode", False)),
            max_clusters_per_batch=max(
                1, int(cfg.get("causal_max_clusters_per_batch") or 2)
            ),
            representatives_per_cluster=max(
                1, int(cfg.get("causal_representatives_per_cluster") or 2)
            ),
            trigger_min_groups=max(
                1, int(cfg.get("causal_trigger_min_groups") or 2)
            ),
            harmful_min_groups=max(
                1, int(cfg.get("causal_harmful_min_groups") or 2)
            ),
            content_min_groups=max(
                1, int(cfg.get("causal_content_min_groups") or 3)
            ),
            structural_min_groups=max(
                1, int(cfg.get("causal_structural_min_groups") or 3)
            ),
            trace_min_groups=max(
                1, int(cfg.get("causal_trace_min_groups") or 1)
            ),
        )


@dataclass
class ProbeSpec:
    cluster_id: str
    atomic_group_ids: list[str]
    edit_scope: str
    target_nodes: list[str]
    focus_nodes: list[str]
    mask_nodes: list[str]
    source_case_ids: list[str]
    source_case_group_ids: dict[str, str]
    support_group_ids: list[str]
    representative_case_ids: list[str]
    min_groups: int
    provisional_hypothesis: str
    trace_verified_case_ids: list[str]
    trace_aligned_case_ids: list[str]
    trace_aligned_group_ids: list[str]
    trace_root_causes: list[str]
    trace_min_groups: int

    @property
    def has_support(self) -> bool:
        return len(self.support_group_ids) >= self.min_groups


    @property
    def has_trace_support(self) -> bool:
        return len(self.trace_aligned_group_ids) >= self.trace_min_groups

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "graphopt-causal-probe-plan-v1",
            "cluster_id": self.cluster_id,
            "atomic_group_ids": list(self.atomic_group_ids),
            "edit_scope": self.edit_scope,
            "target_nodes": list(self.target_nodes),
            "focus_nodes": list(self.focus_nodes),
            "mask_nodes": list(self.mask_nodes),
            "source_case_ids": list(self.source_case_ids),
            "source_case_group_ids": dict(self.source_case_group_ids),
            "support_group_ids": list(self.support_group_ids),
            "support_group_count": len(self.support_group_ids),
            "representative_case_ids": list(self.representative_case_ids),
            "minimum_support_groups": self.min_groups,
            "provisional_hypothesis": self.provisional_hypothesis,
            "support_satisfied": self.has_support,
            "trace_verified_case_ids": list(self.trace_verified_case_ids),
            "trace_aligned_case_ids": list(self.trace_aligned_case_ids),
            "trace_aligned_group_ids": list(self.trace_aligned_group_ids),
            "trace_aligned_group_count": len(self.trace_aligned_group_ids),
            "trace_root_causes": list(self.trace_root_causes),
            "trace_minimum_support_groups": self.trace_min_groups,
            "trace_support_satisfied": self.has_trace_support,
        }


@dataclass
class CausalCertificate:
    certificate_id: str
    base_graph_sha256: str
    renderer_protocol: str
    cluster_id: str
    atomic_group_ids: list[str]
    source_case_ids: list[str]
    support_group_ids: list[str]
    representative_case_ids: list[str]
    edit_scope: str
    target_nodes: list[str]
    confirmed_cause: str
    allowed_edit_scopes: list[str]
    probe_results: dict[str, dict[str, bool]]
    accepted_for_gate: bool
    trace_verified_case_ids: list[str] = field(default_factory=list)
    trace_aligned_case_ids: list[str] = field(default_factory=list)
    trace_aligned_group_ids: list[str] = field(default_factory=list)
    evidence_basis: str = "intervention_probe"
    reason: str = ""
    consumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "graphopt-causal-certificate-v1",
            "certificate_id": self.certificate_id,
            "base_graph_sha256": self.base_graph_sha256,
            "renderer_protocol": self.renderer_protocol,
            "cluster_id": self.cluster_id,
            "atomic_group_ids": list(self.atomic_group_ids),
            "source_case_ids": list(self.source_case_ids),
            "support_group_ids": list(self.support_group_ids),
            "support_group_count": len(self.support_group_ids),
            "representative_case_ids": list(self.representative_case_ids),
            "edit_scope": self.edit_scope,
            "target_nodes": list(self.target_nodes),
            "confirmed_cause": self.confirmed_cause,
            "allowed_edit_scopes": list(self.allowed_edit_scopes),
            "probe_results": {
                key: dict(value) for key, value in self.probe_results.items()
            },
            "accepted_for_gate": self.accepted_for_gate,
            "trace_verified_case_ids": list(self.trace_verified_case_ids),
            "trace_aligned_case_ids": list(self.trace_aligned_case_ids),
            "trace_aligned_group_ids": list(self.trace_aligned_group_ids),
            "evidence_basis": self.evidence_basis,
            "reason": self.reason,
            "consumed": self.consumed,
        }


def graph_sha256(graph: SkillGraph) -> str:
    payload = json.dumps(
        graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_group_ids(edits: list[GraphEdit]) -> list[str]:
    return [str(group_id) for group_id, _ in atomic_edit_groups(edits)]


def edit_scope(edits: list[GraphEdit]) -> str:
    ops = {edit.op for edit in edits}
    if ops == {"update_node"}:
        revision_kinds = {
            str(edit.edit_kind or "").strip().lower() for edit in edits
        }
        if revision_kinds and revision_kinds <= {"when_to_use"}:
            return "UPDATE_WHEN_TO_USE"
        if revision_kinds and revision_kinds <= {"rewrite"}:
            return "NARROW_EXISTING_RULE"
        if all(
            bool(edit.when_to_use) and not edit.how_to_use and edit.avoid is None
            for edit in edits
        ):
            return "UPDATE_WHEN_TO_USE"
        return "UPDATE_HOW_TO_USE"
    if "add_node" in ops:
        return "ADD_NODE"
    if "delete_node" in ops:
        return "NARROW_EXISTING_RULE"
    return "STRUCTURAL"


def edit_target_nodes(graph: SkillGraph, edits: list[GraphEdit]) -> list[str]:
    """Return stable target identifiers directly affected by an atomic group."""
    target_nodes: list[str] = []
    for edit in edits:
        if edit.node_id and (
            edit.node_id in graph.nodes or edit.op == "add_node"
        ):
            # New-node IDs are part of candidate identity even though they
            # cannot be focused in the current graph. Without this, one failed
            # ADD_NODE Gate would demote every unrelated new-node proposal.
            target_nodes.append(str(edit.node_id))
        for node_id in (edit.src, edit.dst):
            if node_id and node_id in graph.nodes:
                target_nodes.append(str(node_id))
    return list(dict.fromkeys(target_nodes))


def grouped_case_map(grouped_batch: GroupedBatch) -> dict[str, str]:
    """Return stable official-group identities for every train case.

    The same fixed group may move to a different batch position in another
    epoch. Its identity therefore comes only from member IDs, never the
    transient index inside a batch.
    """
    mapping: dict[str, str] = {}
    for group in grouped_batch.groups:
        payload = "|".join((*group.train_ids, "=>", *group.val_ids))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        group_id = f"group-{digest}"
        for case_id in group.train_ids:
            mapping[str(case_id)] = group_id
    return mapping


def _analysis_nodes(
    evolution: EvolutionResult,
    source_case_ids: set[str],
) -> tuple[list[str], list[str]]:
    focus: list[str] = []
    mask: list[str] = []
    for analysis in evolution.case_analyses:
        if str(analysis.case_id) not in source_case_ids:
            continue
        focus.extend(analysis.missed_relevant_nodes)
        focus.extend(analysis.correct_nodes)
        focus.extend(
            proposal.target_node
            for proposal in analysis.retrieval_revision_proposals
        )
        mask.extend(analysis.wrong_nodes)
        mask.extend(
            proposal.target_node
            for proposal in analysis.node_revision_proposals
            if proposal.operation == "REWRITE"
        )
    return list(dict.fromkeys(map(str, focus))), list(dict.fromkeys(map(str, mask)))


def _trace_evidence_by_case(
    edits: list[GraphEdit], evolution: EvolutionResult | None
) -> dict[str, dict[str, Any]]:
    """Collect persisted proposal evidence, then fill current-case evidence."""
    by_case: dict[str, dict[str, Any]] = {}
    for edit in edits:
        for evidence in edit.evidence_items:
            case_id = str(evidence.get("case_id") or "")
            if case_id:
                by_case[case_id] = dict(evidence)
    for analysis in (evolution.case_analyses if evolution is not None else []):
        case_id = str(analysis.case_id)
        by_case.setdefault(case_id, {
            "case_id": case_id,
            "root_cause_code": str(analysis.root_cause_code),
            "attributed_nodes": list(analysis.attributed_nodes),
            "attributed_edges": list(analysis.attributed_edges),
            "trace_evidence": dict(analysis.trace_evidence),
            "proposed_edges": [
                {
                    "target_edge": str(proposal.target_edge),
                    "source": str(proposal.source),
                    "target": str(proposal.target),
                    "relation": str(proposal.relation),
                }
                for proposal in analysis.edge_correction_proposals
            ] + [
                {
                    "target_edge": str(proposal.target_old_edge),
                    "source": str(proposal.source),
                    "target": str(proposal.target),
                    "relation": str(proposal.relation),
                }
                for proposal in analysis.new_edge_proposals
            ],
        })
    return by_case


def _structural_edit_matches_evidence(
    graph: SkillGraph, edits: list[GraphEdit], evidence: dict[str, Any]
) -> bool:
    attributed_edges = set(map(str, evidence.get("attributed_edges") or []))
    edge_map = {str(edge.id): edge for edge in graph.edges if edge.id}
    for edge_id in attributed_edges:
        edge = edge_map.get(edge_id)
        if edge is not None and any(
            edit.src == edge.src and edit.dst == edge.dst for edit in edits
        ):
            return True
    for proposed in evidence.get("proposed_edges") or []:
        if not isinstance(proposed, dict):
            continue
        source = str(proposed.get("source") or "")
        target = str(proposed.get("target") or "")
        if any(edit.src == source and edit.dst == target for edit in edits):
            return True
    return False


def _trace_alignment(
    graph: SkillGraph, edits: list[GraphEdit], evolution: EvolutionResult | None,
    source_ids: list[str], scope: str, target_nodes: list[str],
) -> tuple[list[str], list[str], list[str]]:
    evidence_by_case = _trace_evidence_by_case(edits, evolution)
    verified: list[str] = []
    aligned: list[str] = []
    causes: list[str] = []
    targets = set(target_nodes)
    for case_id in source_ids:
        evidence = evidence_by_case.get(case_id)
        if not evidence:
            continue
        trace = evidence.get("trace_evidence")
        if not isinstance(trace, dict) or not bool(trace.get("verified")):
            continue
        verified.append(case_id)
        root = str(evidence.get("root_cause_code") or "")
        attributed_nodes = set(map(str, evidence.get("attributed_nodes") or []))
        matches = False
        if scope == "NARROW_EXISTING_RULE":
            matches = root == "HARMFUL_EXISTING_RULE" and bool(
                targets & attributed_nodes
            )
        elif scope == "UPDATE_WHEN_TO_USE":
            matches = root == "MISSING_ACTIVATION_CUE" and bool(
                targets & attributed_nodes
            )
        elif scope == "UPDATE_HOW_TO_USE":
            matches = root == "MISSING_OR_INCORRECT_PROCEDURE" and bool(
                targets & attributed_nodes
            )
        elif scope == "ADD_NODE":
            matches = root == "MISSING_SKILL_FAMILY"
        elif scope == "STRUCTURAL":
            matches = root in {
                "HARMFUL_EXISTING_RULE", "MISSING_OR_INCORRECT_PROCEDURE"
            } and _structural_edit_matches_evidence(graph, edits, evidence)
        if matches:
            aligned.append(case_id)
            causes.append(root)
    return verified, aligned, list(dict.fromkeys(causes))


def trace_alignment_for_edits(
    graph: SkillGraph, edits: list[GraphEdit],
) -> tuple[list[str], list[str], list[str]]:
    """Return verified and root-cause-aligned persisted source cases."""
    source_ids = list(dict.fromkeys(
        str(case_id)
        for edit in edits
        for case_id in edit.source_case_ids
        if str(case_id)
    ))
    scope = edit_scope(edits)
    return _trace_alignment(
        graph, edits, None, source_ids, scope, edit_target_nodes(graph, edits)
    )


def build_probe_spec(
    graph: SkillGraph,
    patch: GraphPatch,
    evolution: EvolutionResult,
    grouped_batch: GroupedBatch,
    policy: CausalPolicy,
    *,
    known_case_groups: dict[str, str] | None = None,
) -> ProbeSpec:
    """Build one plan for the already bounded atomic patch.

    Group support comes from the fixed update-group mapping, never raw case count.
    """
    edits = list(patch.edits)
    atomic_ids = _atomic_group_ids(edits)
    source_ids = list(dict.fromkeys(
        str(case_id)
        for edit in edits
        for case_id in edit.source_case_ids
        if str(case_id)
    ))
    if not source_ids:
        source_ids = list(dict.fromkeys(
            str(analysis.case_id)
            for analysis in evolution.case_analyses
            if not analysis.success
        ))

    target_nodes = edit_target_nodes(graph, edits)

    analysis_focus, analysis_mask = _analysis_nodes(evolution, set(source_ids))
    scope = edit_scope(edits)
    if scope == "NARROW_EXISTING_RULE":
        mask_nodes = list(dict.fromkeys(
            node_id for node_id in (*target_nodes, *analysis_mask)
            if node_id in graph.nodes
        ))[:2]
        focus_nodes = list(dict.fromkeys(
            node_id for node_id in analysis_focus
            if node_id in graph.nodes and node_id not in mask_nodes
        ))[:3]
    else:
        focus_nodes = list(dict.fromkeys(
            node_id for node_id in (*target_nodes, *analysis_focus)
            if node_id in graph.nodes
        ))[:3]
        mask_nodes = list(dict.fromkeys(
            node_id for node_id in analysis_mask
            if node_id in graph.nodes and node_id not in focus_nodes
        ))[:2]
    if scope == "UPDATE_WHEN_TO_USE":
        min_groups = policy.trigger_min_groups
        hypothesis = "COVERED_BUT_INACTIVE"
    elif scope == "NARROW_EXISTING_RULE":
        min_groups = policy.harmful_min_groups
        hypothesis = "HARMFUL_RULE"
    elif scope in {"UPDATE_HOW_TO_USE", "ADD_NODE"}:
        min_groups = policy.content_min_groups
        hypothesis = "MISSING_SEMANTICS"
    else:
        min_groups = policy.structural_min_groups
        hypothesis = "UNRESOLVED"

    case_to_group = dict(known_case_groups or {})
    case_to_group.update(grouped_case_map(grouped_batch))
    source_case_groups = {
        case_id: case_to_group.get(case_id, f"ungrouped:{case_id}")
        for case_id in source_ids
    }
    support_groups = list(dict.fromkeys(
        source_case_groups[case_id] for case_id in source_ids
    ))
    trace_verified, trace_aligned, trace_roots = _trace_alignment(
        graph, edits, evolution, source_ids, scope, target_nodes
    )
    trace_aligned_groups = list(dict.fromkeys(
        source_case_groups[case_id] for case_id in trace_aligned
        if case_id in source_case_groups
    ))
    representatives: list[str] = []
    represented_groups: set[str] = set()
    for case_id in source_ids:
        group_id = case_to_group.get(case_id, f"ungrouped:{case_id}")
        if group_id in represented_groups:
            continue
        representatives.append(case_id)
        represented_groups.add(group_id)
        if len(representatives) >= policy.representatives_per_cluster:
            break

    key = json.dumps(
        {
            "atomic": atomic_ids,
            "scope": scope,
            "targets": target_nodes,
            "sources": source_ids,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cluster_id = "causal-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return ProbeSpec(
        cluster_id=cluster_id,
        atomic_group_ids=atomic_ids,
        edit_scope=scope,
        target_nodes=target_nodes,
        focus_nodes=focus_nodes,
        mask_nodes=mask_nodes,
        source_case_ids=source_ids,
        source_case_group_ids=source_case_groups,
        support_group_ids=support_groups,
        representative_case_ids=representatives,
        min_groups=min_groups,
        provisional_hypothesis=hypothesis,
        trace_verified_case_ids=trace_verified,
        trace_aligned_case_ids=trace_aligned,
        trace_aligned_group_ids=trace_aligned_groups,
        trace_root_causes=trace_roots,
        trace_min_groups=policy.trace_min_groups,
    )


def _all_fixed(results: dict[str, bool], representatives: list[str]) -> bool:
    return bool(representatives) and all(
        bool(results.get(case_id)) for case_id in representatives
    )


def _any_fixed(results: dict[str, bool], representatives: list[str]) -> bool:
    return any(bool(results.get(case_id)) for case_id in representatives)


def _nonregressive_improvement(
    control: dict[str, bool],
    intervention: dict[str, bool],
    representatives: list[str],
) -> bool:
    """Return true when an intervention fixes at least one case and breaks none.

    Diagnostic replays are stochastic. Requiring every representative to flip
    in one replay made a two-case probe an unnecessarily brittle hard veto.
    The downstream paired Gate remains responsible for admitting the edit.
    """
    improved = any(
        not bool(control.get(case_id)) and bool(intervention.get(case_id))
        for case_id in representatives
    )
    regressed = any(
        bool(control.get(case_id)) and not bool(intervention.get(case_id))
        for case_id in representatives
    )
    return improved and not regressed


def _unfixed_by_all(
    case_id: str,
    *results: dict[str, bool],
) -> bool:
    return not any(bool(result.get(case_id)) for result in results)


def decide_certificate(
    spec: ProbeSpec,
    *,
    base_graph_sha256: str,
    generic_results: dict[str, bool] | None = None,
    focus_results: dict[str, bool] | None = None,
    mask_results: dict[str, bool] | None = None,
) -> CausalCertificate:
    generic = dict(generic_results or {})
    focus = dict(focus_results or {})
    mask = dict(mask_results or {})
    reps = list(spec.representative_case_ids)
    cause = "UNRESOLVED"
    allowed: list[str] = []
    accepted = False
    reason = "interventions did not consistently identify a safe edit scope"

    evidence_basis = "trace_planned_intervention_probe"

    # A reasoning trace plans the intervention; it is never itself an
    # intervention. An accepted certificate may never contain empty probe
    # maps merely because the trace cited the edited node.
    if not spec.has_support and not spec.has_trace_support:
        cause = "INSUFFICIENT_GROUP_SUPPORT"
        reason = (
            f"{len(spec.support_group_ids)} independent groups; "
            f"{spec.min_groups} required for {spec.edit_scope}"
        )
    elif not reps or set(generic) != set(reps):
        cause = "INCOMPLETE_INTERVENTION"
        reason = "generic control is missing one or more representative cases"
    elif spec.edit_scope in {"UPDATE_WHEN_TO_USE", "UPDATE_HOW_TO_USE", "ADD_NODE"} and (
        not spec.focus_nodes or set(focus) != set(reps)
    ):
        cause = "INCOMPLETE_INTERVENTION"
        reason = "the required focus intervention is absent or incomplete"
    elif spec.edit_scope == "NARROW_EXISTING_RULE" and (
        not spec.mask_nodes or set(mask) != set(reps)
    ):
        cause = "INCOMPLETE_INTERVENTION"
        reason = "the required mask intervention is absent or incomplete"
    elif spec.edit_scope == "STRUCTURAL" and not (
        (spec.focus_nodes and set(focus) == set(reps))
        or (spec.mask_nodes and set(mask) == set(reps))
    ):
        cause = "INCOMPLETE_INTERVENTION"
        reason = "joint node-edge scope requires a complete focus or mask intervention"
    elif _all_fixed(generic, reps):
        cause = "UNSTABLE_EXECUTION"
        reason = "generic re-check fixed every representative without node intervention"
    elif spec.edit_scope == "UPDATE_WHEN_TO_USE":
        if (
            spec.focus_nodes
            and _nonregressive_improvement(generic, focus, reps)
        ):
            if (
                spec.mask_nodes
                and _nonregressive_improvement(generic, mask, reps)
            ):
                cause = "PRECEDENCE_CONFLICT"
            else:
                cause = "COVERED_BUT_INACTIVE"
            allowed = ["UPDATE_WHEN_TO_USE"]
            accepted = True
            reason = (
                "specific-node focus produced a non-regressive improvement "
                "over the generic control"
            )
    elif spec.edit_scope in {"UPDATE_HOW_TO_USE", "NARROW_EXISTING_RULE"}:
        unresolved_reps = [
            case_id for case_id in reps
            if _unfixed_by_all(case_id, generic, focus, mask)
        ]
        if (
            spec.mask_nodes
            and _nonregressive_improvement(generic, mask, reps)
        ):
            cause = "HARMFUL_RULE"
            allowed = ["NARROW_EXISTING_RULE", "UPDATE_HOW_TO_USE"]
            accepted = True
            reason = (
                "masking the attributed harmful node produced a "
                "non-regressive improvement"
            )
        elif unresolved_reps:
            # A focused existing rule may solve one member while another still
            # fails every intervention. That is mixed evidence, not proof that
            # all missing behavior is merely a trigger problem.
            cause = "MISSING_SEMANTICS"
            allowed = ["UPDATE_HOW_TO_USE", "ADD_NODE"]
            accepted = spec.edit_scope in allowed
            reason = (
                f"{len(unresolved_reps)}/{len(reps)} representatives remained "
                "wrong under every existing-node intervention"
            )
        elif (
            spec.focus_nodes
            and _nonregressive_improvement(generic, focus, reps)
        ):
            cause = "COVERED_BUT_INACTIVE"
            allowed = ["UPDATE_WHEN_TO_USE"]
            reason = "existing content is sufficient; a content edit is not licensed"
    elif spec.edit_scope == "ADD_NODE":
        unresolved_reps = [
            case_id for case_id in reps
            if _unfixed_by_all(case_id, generic, focus)
        ]
        if unresolved_reps:
            cause = "MISSING_SEMANTICS"
            allowed = ["ADD_NODE", "UPDATE_HOW_TO_USE"]
            accepted = True
            reason = (
                f"{len(unresolved_reps)}/{len(reps)} representatives remained "
                "wrong after focusing all attributed existing rules"
            )
    elif spec.edit_scope == "STRUCTURAL":
        if spec.mask_nodes and _nonregressive_improvement(generic, mask, reps):
            cause = "HARMFUL_RULE"
            allowed = ["STRUCTURAL"]
            accepted = True
            reason = "masking the attributed endpoint produced a non-regressive improvement"
        elif spec.focus_nodes and _nonregressive_improvement(generic, focus, reps):
            cause = "PRECEDENCE_CONFLICT"
            allowed = ["STRUCTURAL"]
            accepted = True
            reason = "focusing the attributed endpoints produced a non-regressive improvement"
        else:
            reason = "targeted endpoint interventions did not license the joint edge scope"
    else:
        reason = "unsupported causal edit scope"

    payload = json.dumps(
        {
            "graph": base_graph_sha256,
            "cluster": spec.cluster_id,
            "cause": cause,
            "scope": spec.edit_scope,
            "groups": spec.support_group_ids,
            "probes": {"generic": generic, "focus": focus, "mask": mask},
            "trace_aligned_cases": spec.trace_aligned_case_ids,
            "trace_aligned_groups": spec.trace_aligned_group_ids,
            "evidence_basis": evidence_basis,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    certificate_id = "cert-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return CausalCertificate(
        certificate_id=certificate_id,
        base_graph_sha256=base_graph_sha256,
        renderer_protocol="causal",
        cluster_id=spec.cluster_id,
        atomic_group_ids=list(spec.atomic_group_ids),
        source_case_ids=list(spec.source_case_ids),
        support_group_ids=list(spec.support_group_ids),
        representative_case_ids=reps,
        edit_scope=spec.edit_scope,
        target_nodes=list(spec.target_nodes),
        confirmed_cause=cause,
        allowed_edit_scopes=allowed,
        probe_results={"generic": generic, "focus": focus, "mask": mask},
        accepted_for_gate=accepted,
        trace_verified_case_ids=list(spec.trace_verified_case_ids),
        trace_aligned_case_ids=list(spec.trace_aligned_case_ids),
        trace_aligned_group_ids=list(spec.trace_aligned_group_ids),
        evidence_basis=evidence_basis,
        reason=reason,
    )


def validate_certificate(
    certificate: CausalCertificate,
    *,
    current_graph: SkillGraph,
    patch: GraphPatch,
) -> None:
    if certificate.confirmed_cause not in ROOT_CAUSES:
        raise ValueError(f"unknown root cause: {certificate.confirmed_cause}")
    if certificate.base_graph_sha256 != graph_sha256(current_graph):
        raise ValueError("causal certificate is stale for the current graph")
    if certificate.consumed:
        raise ValueError("causal certificate has already been consumed")
    actual_groups = set(_atomic_group_ids(list(patch.edits)))
    if actual_groups != set(certificate.atomic_group_ids):
        raise ValueError("causal certificate atomic groups do not match the patch")
    if certificate.edit_scope not in certificate.allowed_edit_scopes:
        raise ValueError("causal certificate does not license this edit scope")
    if not certificate.accepted_for_gate:
        raise ValueError("causal certificate was not accepted for Gate")
    representatives = set(certificate.representative_case_ids)
    generic = set(certificate.probe_results.get("generic") or {})
    focus = set(certificate.probe_results.get("focus") or {})
    mask = set(certificate.probe_results.get("mask") or {})
    if not representatives or generic != representatives:
        raise ValueError("accepted causal certificate requires a complete generic probe")
    if certificate.edit_scope in {
        "UPDATE_WHEN_TO_USE", "UPDATE_HOW_TO_USE", "ADD_NODE",
    } and focus != representatives:
        raise ValueError("accepted causal certificate requires a complete focus probe")
    if (
        certificate.edit_scope == "NARROW_EXISTING_RULE"
        and mask != representatives
    ):
        raise ValueError("accepted harmful-rule certificate requires a complete mask probe")
    if certificate.edit_scope == "STRUCTURAL" and (
        focus != representatives and mask != representatives
    ):
        raise ValueError("accepted structural certificate requires a targeted probe")
