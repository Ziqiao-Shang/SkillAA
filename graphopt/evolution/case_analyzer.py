"""Case-level LLM semantic attribution (template + teacher)."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib

import json
import re
import threading
from pathlib import Path
from typing import Any

from graphopt.evolution.experience_quality import diagnose_no_progress_loops, extract_actions
from graphopt.evolution.types import (
    CaseAnalysis,
    EdgeCorrectionProposal,
    NewEdgeProposal,
    NewNodeProposal,
    NodeRevisionProposal,
    RetrievalRevisionProposal,
)
from graphopt.debug.artifacts import ArtifactStore, save_llm_call, save_template_call
from graphopt.gradient.reflect import format_graph
from graphopt.types import SkillGraph, normalize_edge_type
from graphopt.evolution.trace_attribution import build_trace_evidence

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

_SEMANTIC_REASONING_TRACE_RE = re.compile(
    r"<reasoning_trace>\s*(.*?)\s*</reasoning_trace>",
    re.DOTALL | re.IGNORECASE,
)

_HARD_EPS = 1e-9
_FAILURE_TYPES = {
    "SUCCESS",
    "RETRIEVAL_MISS",
    "EXECUTION_LAPSE",
    "SKILL_DEFECT",
    "MIXED",
    "UNATTRIBUTED",
}
_BADCASE_SUMMARY_FIELDS = {
    "observed_outcome",
    "semantic_path",
    "earliest_error",
    "root_cause",
    "contrast_with_success",
    "reusable_fix",
    "evidence_provenance",
}
_SAME_SAMPLE_ANALYSIS_FIELDS = {
    "outcome_class",
    "repeat_count",
    "correct_count",
    "cross_repeat_comparison",
    "correct_repeat_pattern",
    "failed_repeat_pattern",
    "decisive_difference",
    "graph_text_diagnosis",
    "counterfactual_edit_test",
    "success_experience",
    "inconsistency_cause",
    "graph_modification_needed",
    "modification_decision_reason",
}
_TRACE_PROVENANCE = {"validated_student_trace", "generated_posthoc_trace"}
_ROOT_CAUSE_CODES = {
    "MISSING_ACTIVATION_CUE",
    "MISSING_OR_INCORRECT_PROCEDURE",
    "HARMFUL_EXISTING_RULE",
    "MISSING_SKILL_FAMILY",
    "EXECUTION_LAPSE",
    "INSUFFICIENT_EVIDENCE",
    "SUCCESS",
}

SIMPLE_BADCASE_ANALYSIS_PROTOCOL = "reasoning_trace_first_simple_v1"
CAUSAL_BADCASE_ANALYSIS_PROTOCOL = "reasoning_trace_root_cause_causal"
CASE_COMPLETE_BADCASE_ANALYSIS_PROTOCOL = "reasoning_trace_root_cause_case_complete"
ATTRIBUTION_BOUNDARY_PROTOCOL = "retrieval_execution_fixed_precedence"
_ATTRIBUTION_BOUNDARY_TYPES = {"RETRIEVAL_MISS", "EXECUTION_LAPSE"}
_ATTRIBUTION_CACHE_LOCK = threading.RLock()


def _canonical_update_protocol(update_protocol: str) -> str:
    protocol = str(update_protocol or "case_complete").strip().lower()
    return "case_complete" if protocol == "case_complete_v1" else protocol


def badcase_analysis_protocol(update_protocol: str) -> str:
    """Return the reasoning-trace attribution contract for an update route."""
    protocol = _canonical_update_protocol(update_protocol)
    if protocol == "legacy":
        return SIMPLE_BADCASE_ANALYSIS_PROTOCOL
    if protocol == "causal":
        return CAUSAL_BADCASE_ANALYSIS_PROTOCOL
    if protocol == "case_complete":
        return CASE_COMPLETE_BADCASE_ANALYSIS_PROTOCOL
    raise ValueError("update_protocol must be legacy, case_complete, or causal")


def _summary_provenance(trace: str, status: str) -> str:
    if str(trace).strip() and status in _TRACE_PROVENANCE:
        return status
    return "response_only"


def _fallback_badcase_summary(
    result: dict[str, Any], trace: str, status: str
) -> dict[str, str]:
    response = str(result.get("response") or "").strip()
    outcome = str(result.get("fail_reason") or "").strip()
    if not outcome:
        outcome = "The saved prediction did not satisfy the evaluator."
    semantic_path = str(trace or response).strip()
    if not semantic_path:
        semantic_path = "No usable semantic decision path was saved."
    stable_success = _same_sample_outcome(result) == "all_correct"
    return {
        "observed_outcome": (
            "All three saved observations satisfied the evaluator."
            if stable_success else outcome[:2000]
        ),
        "semantic_path": semantic_path[:4000],
        "earliest_error": (
            "No error was observed across the three successful observations."
            if stable_success
            else "The earliest semantic error could not be located reliably."
        ),
        "root_cause": (
            "SUCCESS: all three saved observations were correct."
            if stable_success
            else "INSUFFICIENT_EVIDENCE: no reliable reusable root cause was established."
        ),
        "contrast_with_success": (
            "The three observations consistently reached a correct answer."
            if stable_success
            else "No reliable contrast with a successful aligned sibling was established."
        ),
        "reusable_fix": (
            "Preserve the successful graph-guided procedure."
            if stable_success
            else "No graph edit is justified from the available evidence."
        ),
        "evidence_provenance": _summary_provenance(trace, status),
    }


def _fallback_same_sample_analysis(result: dict[str, Any]) -> dict[str, Any]:
    repeats = _same_sample_repeats(result)
    outcome = _same_sample_outcome(result)
    correct_count = sum(
        bool(row.get("hard")) and float(row.get("hard") or 0) >= _HARD_EPS
        for row in repeats
    )
    return {
        "outcome_class": outcome,
        "repeat_count": len(repeats),
        "correct_count": correct_count,
        "cross_repeat_comparison": (
            "The strict joint comparison could not be completed reliably."
        ),
        "correct_repeat_pattern": (
            "All observations were correct." if outcome == "all_correct"
            else "No reliable correct-repeat pattern was recovered."
        ),
        "failed_repeat_pattern": (
            "All observations were correct; no failed repeat exists."
            if outcome == "all_correct"
            else "No reliable failed-repeat pattern was recovered."
        ),
        "decisive_difference": (
            "No reliable decisive cross-repeat difference was recovered."
        ),
        "graph_text_diagnosis": (
            "No reliable mapping from the repeat difference to graph text was recovered."
        ),
        "counterfactual_edit_test": (
            "No graph edit is authorized without a supported counterfactual."
        ),
        "success_experience": (
            "All observations were correct, but no reliable reusable success "
            "explanation was recovered."
            if outcome == "all_correct"
            else "No stable reusable success experience was established."
        ),
        "inconsistency_cause": (
            "The repeat outcomes differ, but the cause is unresolved."
            if outcome == "mixed"
            else "The repeat outcomes agree."
        ),
        "graph_modification_needed": False,
        "modification_decision_reason": (
            "No edit is authorized from an incomplete joint analysis."
        ),
    }

def _environment_name(graph: SkillGraph, result: dict[str, Any]) -> str:
    explicit = str(result.get("environment") or "").strip().lower()
    aliases = {"livemath": "livemathematicianbench", "live_math": "livemathematicianbench"}
    explicit = aliases.get(explicit, explicit)
    if explicit in {"searchqa", "docvqa", "livemathematicianbench"}:
        return explicit
    node_ids = set(map(str, graph.nodes))
    if any(node.startswith("D") for node in node_ids):
        return "docvqa"
    if any(node.startswith("M") for node in node_ids):
        return "livemathematicianbench"
    return "searchqa"


def _environment_pipeline(graph: SkillGraph, result: dict[str, Any]):
    environment = _environment_name(graph, result)
    return environment, importlib.import_module(f"graphopt.runtime_envs.{environment}.pipeline")


def _same_sample_repeats(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the exact aligned observations retained for one sample."""
    repeats = result.get("same_sample_repeats")
    if not isinstance(repeats, list) or not repeats:
        return [result]
    case_id = str(result.get("id") or "")
    rows = [row for row in repeats if isinstance(row, dict)]
    if (
        len(rows) != len(repeats)
        or any(str(row.get("id") or "") != case_id for row in rows)
    ):
        raise ValueError("same-sample repeats must all belong to the same case")
    return rows


def _same_sample_outcome(result: dict[str, Any]) -> str:
    rows = _same_sample_repeats(result)
    correct = sum(
        bool(row.get("hard")) and float(row.get("hard") or 0) >= _HARD_EPS
        for row in rows
    )
    if correct == len(rows):
        return "all_correct"
    if correct == 0:
        return "all_wrong"
    return "mixed"


def _is_success(r: dict[str, Any]) -> bool:
    # Attribution success means stable success across all observations.
    # Gate still uses the majority-valued top-level hard field.
    return _same_sample_outcome(r) == "all_correct"


def _joint_trace_evidence(
    graph: SkillGraph, result: dict[str, Any]
) -> dict[str, Any]:
    """Build one lossless usage view over all observations of a sample."""
    repeats = _same_sample_repeats(result)
    if len(repeats) == 1:
        saved = result.get("trace_evidence")
        return (
            dict(saved)
            if isinstance(saved, dict)
            else build_trace_evidence(graph, result)
        )
    evidence_rows = []
    cited_nodes: list[str] = []
    cited_edges: list[str] = []
    for index, row in enumerate(repeats, start=1):
        saved = row.get("trace_evidence")
        evidence = (
            dict(saved)
            if isinstance(saved, dict)
            else build_trace_evidence(graph, row)
        )
        evidence_rows.append({
            "repeat": index,
            "hard": float(row.get("hard") or 0.0),
            "evidence": evidence,
        })
        cited_nodes.extend(map(str, evidence.get("cited_nodes") or []))
        cited_edges.extend(map(str, evidence.get("cited_edges") or []))
    all_verified = all(
        bool(row["evidence"].get("verified")) for row in evidence_rows
    )
    return {
        "status": (
            "MULTI_REPEAT_VERIFIED" if all_verified else "MULTI_REPEAT_PARTIAL"
        ),
        "verified": all_verified,
        "cited_nodes": list(dict.fromkeys(cited_nodes)),
        "cited_edges": list(dict.fromkeys(cited_edges)),
        "repeat_evidence": evidence_rows,
    }


def _normalized_action(value: str) -> str:
    """Normalize an executable action for trajectory-grounding checks."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _edge_by_id(graph: SkillGraph) -> dict[str, Any]:
    out = {}
    for e in graph.edges:
        if e.id:
            out[e.id] = e
    return out


def _nodes_from_text(graph: SkillGraph, text: str) -> list[str]:
    t = text.lower()
    hits = []
    for nid, n in graph.nodes.items():
        if nid.lower() in t or n.title.lower() in t:
            hits.append(nid)
    return sorted(set(hits))


def _reported_active_nodes(graph: SkillGraph, trajectory: str) -> list[str]:
    """Read student-reported Active nodes without scanning the full prompt text."""
    canonical = {str(node_id).casefold(): str(node_id) for node_id in graph.nodes}
    hits: list[str] = []
    for block in re.findall(
        r"active\s+nodes\s*=\s*\[([^\]]*)\]",
        str(trajectory or ""),
        flags=re.IGNORECASE,
    ):
        for token in re.findall(r"\b[a-z]+\d+\b", block, flags=re.IGNORECASE):
            node_id = canonical.get(token.casefold())
            if node_id and node_id not in hits:
                hits.append(node_id)
    return hits


def _normalize_case_usage(analysis: CaseAnalysis) -> None:
    """Enforce correct/wrong ⊆ used and no node in both correct and wrong."""
    used_n = set(analysis.used_nodes) | set(analysis.correct_nodes) | set(analysis.wrong_nodes)
    wrong_n = set(analysis.wrong_nodes)
    analysis.used_nodes = sorted(used_n)
    analysis.correct_nodes = sorted(set(analysis.correct_nodes) & used_n - wrong_n)
    analysis.wrong_nodes = sorted(wrong_n & used_n)

    used_e = set(analysis.used_edges) | set(analysis.correct_edges) | set(analysis.wrong_edges)
    wrong_e = set(analysis.wrong_edges)
    analysis.used_edges = sorted(used_e)
    analysis.correct_edges = sorted(set(analysis.correct_edges) & used_e - wrong_e)
    analysis.wrong_edges = sorted(wrong_e & used_e)


def _derive_failure_type(analysis: CaseAnalysis) -> str:
    if analysis.success:
        return "SUCCESS"
    flags = sum(
        (
            bool(analysis.missed_relevant_nodes or analysis.missed_relevant_edges),
            bool(analysis.execution_lapse_nodes or analysis.execution_lapse_edges),
            bool(
                analysis.wrong_nodes
                or analysis.wrong_edges
                or any(
                    p.target_node not in set(analysis.execution_lapse_nodes)
                    for p in analysis.node_revision_proposals
                )
                or analysis.edge_correction_proposals
                or analysis.new_node_proposals
                or analysis.new_edge_proposals
            ),
        )
    )
    if flags > 1:
        return "MIXED"
    if analysis.missed_relevant_nodes or analysis.missed_relevant_edges:
        return "RETRIEVAL_MISS"
    if analysis.execution_lapse_nodes or analysis.execution_lapse_edges:
        return "EXECUTION_LAPSE"
    if (
        analysis.wrong_nodes
        or analysis.wrong_edges
        or any(
            p.target_node not in set(analysis.execution_lapse_nodes)
            for p in analysis.node_revision_proposals
        )
        or analysis.edge_correction_proposals
        or analysis.new_node_proposals
        or analysis.new_edge_proposals
    ):
        return "SKILL_DEFECT"
    return "UNATTRIBUTED"


def _root_cause_code(analysis: CaseAnalysis) -> str:
    if analysis.success:
        return "SUCCESS"
    root = str(analysis.badcase_summary.get("root_cause") or "")
    code = root.split(":", 1)[0].strip().upper()
    return code if code in _ROOT_CAUSE_CODES else "INSUFFICIENT_EVIDENCE"


def _finalize_trace_attribution(
    graph: SkillGraph, result: dict[str, Any], analysis: CaseAnalysis
) -> None:
    """Bind attribution targets to deterministic trace evidence and root cause."""
    if not analysis.trace_evidence:
        saved = result.get("trace_evidence")
        analysis.trace_evidence = (
            dict(saved) if isinstance(saved, dict)
            else build_trace_evidence(graph, result)
        )
    analysis.root_cause_code = _root_cause_code(analysis)
    if analysis.root_cause_code == "HARMFUL_EXISTING_RULE":
        node_ids = list(analysis.wrong_nodes)
        edge_ids = list(analysis.wrong_edges)
    elif analysis.root_cause_code == "MISSING_ACTIVATION_CUE":
        node_ids = list(analysis.missed_relevant_nodes)
        edge_ids = list(analysis.missed_relevant_edges)
    elif analysis.root_cause_code == "EXECUTION_LAPSE":
        node_ids = list(analysis.execution_lapse_nodes)
        edge_ids = list(analysis.execution_lapse_edges)
    elif analysis.root_cause_code == "MISSING_OR_INCORRECT_PROCEDURE":
        node_ids = [
            proposal.target_node
            for proposal in analysis.node_revision_proposals
        ]
        edge_ids = [
            proposal.target_edge
            for proposal in analysis.edge_correction_proposals
        ]
    else:
        node_ids, edge_ids = [], []
    analysis.attributed_nodes = list(dict.fromkeys(
        node_id for node_id in node_ids if node_id in graph.nodes
    ))
    valid_edges = set(_edge_by_id(graph))
    analysis.attributed_edges = list(dict.fromkeys(
        edge_id for edge_id in edge_ids if edge_id in valid_edges
    ))


def _sanitize_case_analysis(
    graph: SkillGraph,
    result: dict[str, Any],
    analysis: CaseAnalysis,
) -> CaseAnalysis:
    """Enforce the analyzer prompt contract at the Python boundary."""
    valid_nodes = set(graph.nodes)
    valid_edges = set(_edge_by_id(graph))
    analysis.case_id = str(result.get("id") or "case")
    analysis.success = _is_success(result)
    outcome = _same_sample_outcome(result)
    repeats = _same_sample_repeats(result)
    correct_count = sum(
        bool(row.get("hard")) and float(row.get("hard") or 0) >= _HARD_EPS
        for row in repeats
    )
    if len(repeats) == 1:
        analysis.same_sample_analysis = {}
    elif not analysis.same_sample_analysis:
        analysis.same_sample_analysis = {
            "outcome_class": outcome,
            "repeat_count": len(repeats),
            "correct_count": correct_count,
            "cross_repeat_comparison": (
                "Template analysis retained all same-sample observations."
            ),
            "correct_repeat_pattern": (
                "Template mode retained the correct-repeat evidence."
            ),
            "failed_repeat_pattern": (
                "Template mode retained the failed-repeat evidence."
            ),
            "decisive_difference": (
                "Template mode does not infer a decisive causal difference."
            ),
            "graph_text_diagnosis": (
                "Template mode does not attribute variation to graph wording."
            ),
            "counterfactual_edit_test": (
                "Template mode does not authorize a counterfactual graph edit."
            ),
            "success_experience": (
                "Preserve the observed graph usage that supported all repeats."
                if outcome == "all_correct"
                else "No stable success experience was inferred in template mode."
            ),
            "inconsistency_cause": (
                "No inconsistency: all repeat outcomes agree."
                if outcome != "mixed"
                else "Mixed outcomes require teacher comparison."
            ),
            "graph_modification_needed": False,
            "modification_decision_reason": (
                "Template mode does not authorize a change from repeat variation."
            ),
        }
    _, pipeline = _environment_pipeline(graph, result)

    if not analysis.trace_evidence:
        analysis.trace_evidence = _joint_trace_evidence(graph, result)
    if bool(analysis.trace_evidence.get("verified")):
        # The teacher classifies the path; it may not rewrite the observed path.
        analysis.used_nodes = list(analysis.trace_evidence.get("cited_nodes") or [])
        analysis.used_edges = list(analysis.trace_evidence.get("cited_edges") or [])

    analysis.used_nodes = [x for x in analysis.used_nodes if x in valid_nodes]
    analysis.correct_nodes = [x for x in analysis.correct_nodes if x in valid_nodes]
    analysis.wrong_nodes = [x for x in analysis.wrong_nodes if x in valid_nodes]
    analysis.used_edges = [x for x in analysis.used_edges if x in valid_edges]
    analysis.correct_edges = [x for x in analysis.correct_edges if x in valid_edges]
    analysis.wrong_edges = [x for x in analysis.wrong_edges if x in valid_edges]
    allow_repeat_overlap = outcome == "mixed"
    analysis.missed_relevant_nodes = [
        x for x in analysis.missed_relevant_nodes
        if x in valid_nodes
        and (
            allow_repeat_overlap
            or x not in set(analysis.used_nodes)
        )
    ]
    analysis.execution_lapse_nodes = [
        x for x in analysis.execution_lapse_nodes if x in valid_nodes
    ]

    # An edge can influence a trajectory only when both endpoint skills were
    # active.  Co-occurrence is retrieval evidence, never a faulty executable
    # constraint, so it cannot be labelled wrong or receive a correction.
    edge_map = _edge_by_id(graph)
    used_node_set = set(analysis.used_nodes)
    analysis.used_edges = [
        edge_id for edge_id in analysis.used_edges
        if edge_map[edge_id].src in used_node_set
        and edge_map[edge_id].dst in used_node_set
    ]
    analysis.correct_edges = [
        edge_id for edge_id in analysis.correct_edges
        if edge_id in set(analysis.used_edges)
    ]
    analysis.wrong_edges = [
        edge_id for edge_id in analysis.wrong_edges
        if edge_id in set(analysis.used_edges)
        and normalize_edge_type(edge_map[edge_id].type) in {"prereq", "enhance"}
    ]
    available_nodes = used_node_set | set(analysis.missed_relevant_nodes)
    analysis.missed_relevant_edges = [
        edge_id
        for edge_id in analysis.missed_relevant_edges
        if edge_id in valid_edges
        and (
            allow_repeat_overlap
            or edge_id not in set(analysis.used_edges)
        )
        and edge_map[edge_id].src in available_nodes
        and edge_map[edge_id].dst in available_nodes
    ]
    analysis.execution_lapse_edges = [
        edge_id
        for edge_id in analysis.execution_lapse_edges
        if edge_id in set(analysis.correct_edges)
    ]

    if analysis.success:
        analysis.wrong_nodes = []
        analysis.wrong_edges = []
        analysis.node_revision_proposals = []
        analysis.edge_correction_proposals = []
        analysis.new_node_proposals = []
        analysis.new_edge_proposals = []
        analysis.missed_relevant_nodes = []
        analysis.missed_relevant_edges = []
        analysis.execution_lapse_nodes = []
        analysis.execution_lapse_edges = []
        analysis.retrieval_revision_proposals = []
        analysis.failure_type = "SUCCESS"
        _normalize_case_usage(analysis)
        _finalize_trace_attribution(graph, result, analysis)
        return analysis

    if (
        not bool(getattr(pipeline, "ALLOW_EXECUTION_CHILDREN", True))
        and not bool(analysis.trace_evidence.get("verified"))
    ):
        # Without a verified semantic path, a static answer does not expose the
        # boundary at which a cited rule was misapplied. A verified trace does,
        # but it still licenses diagnosis only, never an execution-child edit.
        lapse_nodes = set(analysis.execution_lapse_nodes)
        analysis.node_revision_proposals = [
            proposal for proposal in analysis.node_revision_proposals
            if proposal.target_node not in lapse_nodes
        ]
        analysis.execution_lapse_nodes = []
        analysis.execution_lapse_edges = []

    if analysis.analysis_partial:
        # A response that failed the strict contract twice is diagnostic only.
        # Never let salvaged, structurally incomplete text mutate the graph.
        analysis.node_revision_proposals = []
        analysis.retrieval_revision_proposals = []
        analysis.edge_correction_proposals = []
        analysis.new_node_proposals = []
        analysis.new_edge_proposals = []

    node_props: dict[str, NodeRevisionProposal] = {}
    allowed_patch_targets = set(analysis.correct_nodes)
    for proposal in analysis.node_revision_proposals:
        operation = str(proposal.operation or "PATCH").strip().upper()
        node = graph.nodes.get(proposal.target_node)
        searchable = "\n".join((
            node.meaning, node.when_to_use, node.how_to_use, *node.avoid
        )) if node is not None else ""
        valid_rewrite = (
            operation == "REWRITE"
            and proposal.target_node in set(analysis.wrong_nodes)
            and bool(proposal.toxic_text.strip())
            and proposal.toxic_text.strip() in searchable
        )
        valid_patch = (
            operation == "PATCH"
            and proposal.target_node in allowed_patch_targets
            and proposal.target_node not in set(analysis.execution_lapse_nodes)
        )
        if (
            (valid_patch or valid_rewrite or proposal.target_node in set(analysis.execution_lapse_nodes))
            and proposal.proposal.strip()
            and proposal.target_node not in node_props
        ):
            proposal.operation = "PATCH" if proposal.target_node in set(analysis.execution_lapse_nodes) else operation
            if proposal.operation != "REWRITE":
                proposal.toxic_text = ""
            proposal.case_id = analysis.case_id
            node_props[proposal.target_node] = proposal
    # The declared contract is one proposal per wrong node. Drop unsupported
    # wrong labels instead of allowing them to pollute stats without evidence.
    analysis.wrong_nodes = [
        x for x in analysis.wrong_nodes
        if x in node_props and node_props[x].operation == "REWRITE"
    ]
    analysis.node_revision_proposals = list(node_props.values())

    retrieval_props: dict[str, RetrievalRevisionProposal] = {}
    for proposal in analysis.retrieval_revision_proposals:
        if (
            proposal.target_node in set(analysis.missed_relevant_nodes)
            and proposal.proposed_when_to_use.strip()
            and proposal.reason.strip()
            and proposal.target_node not in retrieval_props
        ):
            proposal.case_id = analysis.case_id
            retrieval_props[proposal.target_node] = proposal
    analysis.retrieval_revision_proposals = list(retrieval_props.values())

    edge_props: dict[str, EdgeCorrectionProposal] = {}
    for proposal in analysis.edge_correction_proposals:
        relation = normalize_edge_type(proposal.relation)
        if (
            proposal.target_edge in set(analysis.wrong_edges)
            and proposal.source in valid_nodes
            and proposal.target in valid_nodes
            and proposal.source != proposal.target
            and relation in {"prereq", "enhance"}
            and proposal.reason.strip()
            and proposal.target_edge not in edge_props
        ):
            proposal.relation = relation
            proposal.case_id = analysis.case_id
            edge_props[proposal.target_edge] = proposal
    analysis.wrong_edges = [x for x in analysis.wrong_edges if x in edge_props]
    analysis.edge_correction_proposals = [edge_props[x] for x in analysis.wrong_edges]

    correct_nodes = set(analysis.correct_nodes)
    analysis.new_node_proposals = [
        p
        for p in analysis.new_node_proposals
        if p.content.strip()
        and p.parent_node in valid_nodes
        and p.parent_node in correct_nodes
        and normalize_edge_type(p.relation) == "enhance"
        and p.reason.strip()
    ]
    for proposal in analysis.new_node_proposals:
        proposal.case_id = analysis.case_id
        proposal.relation = "enhance"
    analysis.new_edge_proposals = [
        p
        for p in analysis.new_edge_proposals
        if p.source in valid_nodes
        and p.target in valid_nodes
        and p.source != p.target
        and normalize_edge_type(p.relation) in {"prereq", "enhance"}
        and p.reason.strip()
    ]
    for proposal in analysis.new_edge_proposals:
        proposal.relation = normalize_edge_type(proposal.relation)
        proposal.case_id = analysis.case_id
        if proposal.target_old_edge not in valid_edges:
            proposal.target_old_edge = ""
    _normalize_case_usage(analysis)
    analysis.execution_lapse_nodes = sorted(
        set(analysis.execution_lapse_nodes) & set(analysis.correct_nodes)
    )
    analysis.execution_lapse_edges = sorted(
        set(analysis.execution_lapse_edges) & set(analysis.correct_edges)
    )
    analysis.missed_relevant_nodes = sorted(
        set(analysis.missed_relevant_nodes)
        if allow_repeat_overlap
        else set(analysis.missed_relevant_nodes) - set(analysis.used_nodes)
    )
    analysis.missed_relevant_edges = sorted(
        set(analysis.missed_relevant_edges)
        if allow_repeat_overlap
        else set(analysis.missed_relevant_edges) - set(analysis.used_edges)
    )
    analysis.failure_type = _derive_failure_type(analysis)
    _finalize_trace_attribution(graph, result, analysis)
    return analysis


def _harmful_node_ids_from_context(gate_context: str) -> set[str]:
    import re

    ids = set(re.findall(r"\bnode\s+([A-Za-z][A-Za-z0-9_]*)\b", gate_context or "", re.I))
    ids.update(re.findall(r"\b([A-Za-z]\d{2,3})\b", gate_context or "", re.I))
    return ids


def _observed_graph_usage(
    graph: SkillGraph, result: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, Any]]:
    evidence = _joint_trace_evidence(graph, result)
    if evidence.get("verified"):
        return (
            list(evidence.get("cited_nodes") or []),
            list(evidence.get("cited_edges") or []),
            evidence,
        )
    refs = result.get("graph_refs")
    if isinstance(refs, dict) and refs.get("status") == "validated_student_usage":
        return (
            [str(value) for value in (refs.get("used_nodes") or [])],
            [str(value) for value in (refs.get("used_edges") or [])],
            evidence,
        )
    return [], [], evidence


def _template_analyze_case(
    graph: SkillGraph,
    result: dict[str, Any],
    *,
    gate_context: str = "",
    store: ArtifactStore | None = None,
    update_protocol: str = "case_complete",
) -> CaseAnalysis:
    cid = str(result.get("id") or "case")
    success = _is_success(result)
    blob = " ".join(
        str(result.get(key) or "")
        for key in (
            "fail_reason", "task_description", "question", "instruction",
            "response", "trajectory", "semantic_reasoning_trace",
        )
    ).lower()
    environment, pipeline = _environment_pipeline(graph, result)
    observed_nodes, observed_edges, trace_evidence = _observed_graph_usage(
        graph, result
    )
    reported_nodes = _reported_active_nodes(graph, str(result.get("trajectory") or ""))
    used_nodes = reported_nodes or _nodes_from_text(graph, blob)
    used_nodes = observed_nodes or used_nodes
    if success and not used_nodes:
        used_nodes = pipeline.positive_attribution_nodes(graph, blob)
    if not used_nodes and graph.nodes:
        used_nodes = [sorted(graph.nodes.keys())[0]]

    analysis = CaseAnalysis(
        case_id=cid, success=success, used_nodes=used_nodes,
        used_edges=observed_edges, trace_evidence=trace_evidence,
    )

    if success:
        analysis.correct_nodes = list(used_nodes)
        # Without a semantic model, do not invent an edge merely because the
        # episode succeeded. Explicit trajectory citations are handled only by
        # the teacher analyzer; template mode stays conservative.
        analysis.correct_edges = list(analysis.used_edges)
        _sanitize_case_analysis(graph, result, analysis)
        if store is not None:
            save_template_call(
                store,
                cid,
                stage="case_analyze",
                inputs={
                    "case_id": cid,
                    "task_description": (
                        result.get("task_description") or result.get("question")
                    ),
                    "question": (
                        result.get("question") or result.get("task_description")
                    ),
                    "hard": result.get("hard"),
                    "success": True,
                    "update_protocol": update_protocol,
                    "badcase_analysis_protocol": badcase_analysis_protocol(
                        update_protocol
                    ),
                },
                outputs=analysis.to_dict(),
            )
        return analysis

    analysis.wrong_nodes = []
    analysis.correct_nodes = []
    harmful_nodes = _harmful_node_ids_from_context(gate_context)
    environment_revision = pipeline.template_failure_revision(graph, blob)
    if environment_revision is not None:
        proposal, nodes, reason = environment_revision
        patch_targets = [
            node for node in nodes
            if node in graph.nodes and node not in harmful_nodes
        ][:2]
        analysis.used_nodes.extend(patch_targets)
        analysis.correct_nodes.extend(patch_targets)
        for node_id in patch_targets:
            analysis.node_revision_proposals.append(
                NodeRevisionProposal(
                    target_node=node_id,
                    proposal=proposal,
                    reason=reason,
                    case_id=cid,
                    operation="PATCH",
                )
            )

    if not analysis.node_revision_proposals:
        fallback = [n for n in used_nodes[:1] if n not in harmful_nodes]
        if fallback:
            analysis.correct_nodes.extend(fallback)
            analysis.node_revision_proposals.append(
                NodeRevisionProposal(
                    target_node=fallback[0],
                    proposal=f"Add missing handling for: {(result.get('fail_reason') or 'failure')[:160]}",
                    reason=str(result.get("fail_reason") or ""),
                    case_id=cid,
                    operation="PATCH",
                )
            )

    # Template mode cannot infer a semantic relation defect safely. Never pick
    # an arbitrary first edge merely because the case failed.
    analysis.used_edges = []
    analysis.wrong_edges = []
    analysis.edge_correction_proposals = []

    if "open" in blob and "leave" in blob:
        tid = f"new_node_{re.sub(r'[^a-zA-Z0-9]+', '_', cid)[:24]}"
        analysis.new_node_proposals.append(
            NewNodeProposal(
                temp_id=tid,
                content="After opening a receptacle, inspect its contents before leaving.",
                case_id=cid,
                parent_node=fallback[0] if fallback else "",
                relation="enhance",
                reason="The used correct rule is the closest observable activation context.",
            )
        )

    analysis.correct_nodes = list(dict.fromkeys(analysis.correct_nodes))
    analysis.wrong_nodes = list(dict.fromkeys(analysis.wrong_nodes))
    _sanitize_case_analysis(graph, result, analysis)
    if store is not None:
        save_template_call(
            store,
            cid,
            stage="case_analyze",
            inputs={
                "case_id": cid,
                "task_description": result.get("task_description"),
                "hard": result.get("hard"),
                "fail_reason": result.get("fail_reason"),
                "trajectory": str(result.get("trajectory") or "")[:4000],
                "semantic_reasoning_trace": str(
                    result.get("semantic_reasoning_trace") or ""
                )[:4000],
                "semantic_reasoning_trace_status": str(
                    result.get("semantic_reasoning_trace_status") or "not_requested"
                ),
                "update_protocol": update_protocol,
                "badcase_analysis_protocol": badcase_analysis_protocol(
                    update_protocol
                ),
                "gate_context": gate_context[:2000] if gate_context else "",
            },
            outputs=analysis.to_dict(),
        )
    return analysis


def _heuristic_success_analysis(
    graph: SkillGraph,
    result: dict[str, Any],
    *,
    store: ArtifactStore | None = None,
) -> CaseAnalysis:
    """Attribute successful episodes without an unnecessary teacher call."""
    cid = str(result.get("id") or "case")
    task_type = str(result.get("task_type") or "")
    environment, pipeline = _environment_pipeline(graph, result)
    trajectory = str(result.get("trajectory") or "")
    observed_nodes, observed_edges, trace_evidence = _observed_graph_usage(
        graph, result
    )
    reported_nodes = _reported_active_nodes(graph, trajectory)
    text = " ".join(
        str(result.get(key) or "")
        for key in (
            "task_description", "question", "instruction", "response",
            "fail_reason", "trajectory", "semantic_reasoning_trace",
        )
    ).casefold()
    nodes = observed_nodes or reported_nodes or pipeline.positive_attribution_nodes(graph, text)
    analysis = CaseAnalysis(
        case_id=cid,
        success=True,
        failure_type="SUCCESS",
        used_nodes=nodes,
        correct_nodes=list(nodes),
        used_edges=observed_edges,
        correct_edges=list(observed_edges),
        trace_evidence=trace_evidence,
    )
    _sanitize_case_analysis(graph, result, analysis)
    if store is not None:
        save_template_call(
            store,
            cid,
            stage="case_analyze_success_heuristic",
            inputs={
                "case_id": cid,
                "environment": environment,
                "task_type": task_type,
                "positive_attribution": "environment_task_semantic_heuristic",
            },
            outputs=analysis.to_dict(),
        )
    return analysis



def _case_analysis_correction_prompt(
    original_user: str,
    *,
    error: str,
    invalid_response: str,
) -> str:
    """Build the second teacher request after strict validation rejects attempt 1."""
    return (
        original_user
        + "\n\n## Format Correction\nYour previous answer was invalid: "
        + error
        + "\nReturn the complete strict JSON object again. Preserve the exact "
        + "input case_id and repeat-derived stable success value; include every "
        + "required usage and proposal field, with no extra keys or Markdown."
        + "\n\n## Invalid Previous Answer\n"
        + invalid_response
    )


def _cached_teacher_response(
    store: ArtifactStore | None,
    *,
    case_id: str,
    system: str,
    base_user: str,
) -> tuple[str, Any] | None:
    """Return a completed teacher response only when its exact inputs match.

    A valid response may be the first attempt, or the format-corrected second
    attempt immediately following a failed request built from ``base_user``.
    The caller always reparses the response with the current graph and case
    schema before accepting it.
    """
    if store is None:
        return None
    key = f"llm/case_analyze/{case_id}"
    payloads: list[dict[str, Any]] = []
    for entry in store.versions(key):
        file_name = entry.get("file") if isinstance(entry, dict) else None
        if not file_name:
            continue
        try:
            payload = json.loads((store.base_dir / str(file_name)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)

    for index in range(len(payloads) - 1, -1, -1):
        payload = payloads[index]
        inputs = payload.get("inputs") or {}
        outputs = payload.get("outputs") or {}
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        response = outputs.get("response")
        if not isinstance(response, str) or not isinstance(outputs.get("parsed"), dict):
            continue
        if inputs.get("system") != system:
            continue
        exact_first_attempt = inputs.get("user") == base_user
        corrected_second_attempt = False
        if index > 0 and isinstance(inputs.get("user"), str):
            previous = payloads[index - 1]
            previous_inputs = previous.get("inputs") or {}
            previous_outputs = previous.get("outputs") or {}
            corrected_second_attempt = (
                isinstance(previous_inputs, dict)
                and isinstance(previous_outputs, dict)
                and previous_inputs.get("system") == system
                and previous_inputs.get("user") == base_user
                and bool(previous_outputs.get("error"))
                and inputs["user"].startswith(base_user + "\n\n## Format Correction")
            )
        if exact_first_attempt or corrected_second_attempt:
            return response, outputs.get("usage")
    return None


def _cached_semantic_trace(
    store: ArtifactStore | None,
    *,
    case_id: str,
    system: str,
    user: str,
) -> tuple[str, str] | None:
    """Reuse only an exact-input, successfully parsed posthoc trace artifact."""
    if store is None:
        return None
    key = f"llm/semantic_trace_reconstruct/{case_id}"
    for entry in reversed(store.versions(key)):
        file_name = entry.get("file") if isinstance(entry, dict) else None
        if not file_name:
            continue
        try:
            payload = json.loads(
                (store.base_dir / str(file_name)).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        inputs = payload.get("inputs") or {}
        outputs = payload.get("outputs") or {}
        parsed = outputs.get("parsed") if isinstance(outputs, dict) else None
        if (
            isinstance(inputs, dict)
            and inputs.get("system") == system
            and inputs.get("user") == user
            and isinstance(parsed, dict)
            and parsed.get("status") == "generated_posthoc_trace"
            and str(parsed.get("trace") or "").strip()
        ):
            return str(parsed["trace"]).strip(), "generated_posthoc_trace"
    return None


def _semantic_trace_for_attribution(
    result: dict[str, Any],
    *,
    chat_fn,
    store: ArtifactStore | None,
) -> tuple[str, str]:
    """Return an observed trace or a clearly weaker teacher-generated hypothesis."""
    trace = str(result.get("semantic_reasoning_trace") or "").strip()
    status = str(result.get("semantic_reasoning_trace_status") or "not_requested")
    required = bool(result.get("semantic_reasoning_trace_required"))
    if trace or not required:
        return trace, status

    system = (PROMPTS / "semantic_trace_reconstruct.md").read_text(encoding="utf-8")
    task = {
        "case_id": result.get("id"),
        "environment": result.get("environment"),
        "task_type": result.get("task_type"),
        "question": result.get("task_description") or result.get("question"),
        "response": result.get("response"),
        "predicted_answer": result.get("predicted_answer"),
        "predicted_label": result.get("predicted_label"),
        "predicted_text": result.get("predicted_text"),
        "fail_reason": result.get("fail_reason"),
        "training_reference": result.get("training_reference"),
    }
    user = json.dumps(task, ensure_ascii=False, indent=2)
    cached = _cached_semantic_trace(
        store, case_id=str(result.get("id") or "case"), system=system, user=user
    )
    if cached is not None:
        return cached
    response = ""
    usage: Any = None
    error: str | None = None
    try:
        response, usage = chat_fn(
            system=system,
            user=user,
            max_completion_tokens=768,
            retries=3,
            stage="semantic_trace_reconstruct",
        )
        traces = list(dict.fromkeys(
            match.group(1).strip()
            for match in _SEMANTIC_REASONING_TRACE_RE.finditer(response)
            if match.group(1).strip()
        ))
        if len(traces) == 1:
            trace = traces[0]
            status = "generated_posthoc_trace"
        else:
            status = "invalid_posthoc_trace"
            error = "exactly one non-empty reasoning_trace block is required"
    except Exception as exc:
        status = "posthoc_trace_generation_failed"
        error = str(exc)
    if store is not None:
        save_llm_call(
            store,
            str(result.get("id") or "case"),
            stage="semantic_trace_reconstruct",
            system=system,
            user=user,
            response=response or None,
            usage=usage,
            parsed={"trace": trace, "status": status},
            error=error,
        )
    return trace, status


def _boundary_observable_text(task_record: dict[str, Any]) -> str:
    """Return only text that may ground an execution-lapse excerpt."""
    top_level = "\n".join(
        str(task_record.get(key) or "")
        for key in (
            "response",
            "predicted_answer",
            "predicted_label",
            "predicted_text",
            "fail_reason",
            "trajectory",
            "semantic_reasoning_trace",
        )
    )
    repeat_text = json.dumps(
        task_record.get("same_sample_repeats") or [],
        ensure_ascii=False,
        sort_keys=True,
    )
    return top_level + "\n" + repeat_text


def _validate_boundary_adjudication(
    graph: SkillGraph,
    task_record: dict[str, Any],
    verdict: Any,
) -> dict[str, Any]:
    """Validate the fixed retrieval/execution precedence decision."""
    required = {
        "case_id",
        "decision",
        "reason",
        "required_action",
        "rule_text_sufficiency",
        "reusable_correction_observable_without_training_answer",
        "sufficient_rule_node_ids",
        "sufficient_rule_edge_ids",
        "decision_node_ids",
        "decision_edge_ids",
        "correct_intermediate_excerpt",
        "later_divergence_excerpt",
        "trigger_revision",
    }
    if not isinstance(verdict, dict) or set(verdict) != required:
        raise ValueError(
            "boundary adjudication must contain exactly " + str(sorted(required))
        )
    case_id = str(task_record.get("case_id") or "case")
    if str(verdict["case_id"]) != case_id:
        raise ValueError("boundary adjudication case_id does not match the task")
    decision = str(verdict["decision"])
    if decision not in {
        "RETRIEVAL_MISS", "EXECUTION_LAPSE", "UNATTRIBUTED",
    }:
        raise ValueError("invalid boundary adjudication decision")
    if not str(verdict["reason"]).strip() or not str(
        verdict["required_action"]
    ).strip():
        raise ValueError("boundary adjudication needs a reason and required_action")
    sufficiency = str(verdict["rule_text_sufficiency"])
    if sufficiency not in {"FULL", "INSUFFICIENT", "UNCERTAIN"}:
        raise ValueError("invalid rule_text_sufficiency")
    observable = verdict[
        "reusable_correction_observable_without_training_answer"
    ]
    if not isinstance(observable, bool):
        raise ValueError(
            "reusable_correction_observable_without_training_answer must be boolean"
        )

    valid_nodes = set(graph.nodes)
    edge_map = _edge_by_id(graph)
    valid_edges = set(edge_map)

    def ids(key: str, allowed: set[str]) -> list[str]:
        value = verdict[key]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"{key} must be a string list")
        if len(value) != len(set(value)):
            raise ValueError(f"{key} contains duplicate IDs")
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"{key} contains unknown IDs: {sorted(unknown)}")
        return list(value)

    sufficient_nodes = ids("sufficient_rule_node_ids", valid_nodes)
    sufficient_edges = ids("sufficient_rule_edge_ids", valid_edges)
    decision_nodes = ids("decision_node_ids", valid_nodes)
    decision_edges = ids("decision_edge_ids", valid_edges)
    if not set(decision_nodes) <= set(sufficient_nodes) or not set(
        decision_edges
    ) <= set(sufficient_edges):
        raise ValueError("decisive IDs must be fully sufficient rule IDs")

    trace_evidence = task_record.get("trace_evidence") or {}
    used_nodes = set(trace_evidence.get("cited_nodes") or [])
    used_edges = set(trace_evidence.get("cited_edges") or [])
    trace_verified = bool(trace_evidence.get("verified"))
    failed_repeat_usage: list[tuple[set[str], set[str]]] = []
    for repeat in trace_evidence.get("repeat_evidence") or []:
        if float(repeat.get("hard") or 0.0) >= _HARD_EPS:
            continue
        repeat_evidence = repeat.get("evidence") or {}
        if not bool(repeat_evidence.get("verified")):
            continue
        failed_repeat_usage.append((
            set(map(str, repeat_evidence.get("cited_nodes") or [])),
            set(map(str, repeat_evidence.get("cited_edges") or [])),
        ))
    intermediate = str(verdict["correct_intermediate_excerpt"] or "").strip()
    divergence = str(verdict["later_divergence_excerpt"] or "").strip()
    trigger = verdict["trigger_revision"]

    if decision in _ATTRIBUTION_BOUNDARY_TYPES:
        if sufficiency != "FULL" or not observable:
            raise ValueError(
                "retrieval/execution requires FULL rule text and an observable "
                "reusable correction"
            )
        if not decision_nodes and not decision_edges:
            raise ValueError("retrieval/execution requires a decisive graph element")
    if decision == "RETRIEVAL_MISS":
        missed_in_failed_repeats = (
            bool(failed_repeat_usage)
            and all(
                not (set(decision_nodes) & repeat_nodes)
                and not (set(decision_edges) & repeat_edges)
                for repeat_nodes, repeat_edges in failed_repeat_usage
            )
        )
        if failed_repeat_usage:
            if not missed_in_failed_repeats:
                raise ValueError(
                    "retrieval-missed elements must be absent from every "
                    "verified failed repeat"
                )
        elif set(decision_nodes) & used_nodes or set(decision_edges) & used_edges:
            raise ValueError("retrieval-missed elements must be absent from used IDs")
        if intermediate or divergence:
            raise ValueError("retrieval miss must not claim execution excerpts")
        available_nodes = used_nodes | set(decision_nodes)
        for edge_id in decision_edges:
            edge = edge_map[edge_id]
            if edge.src not in available_nodes or edge.dst not in available_nodes:
                raise ValueError(
                    "retrieval-missed edge endpoints must be used or missed nodes"
                )
        if trigger is not None:
            if not isinstance(trigger, dict) or set(trigger) != {
                "target_node", "proposed_when_to_use", "reason",
            }:
                raise ValueError("trigger_revision has an invalid schema")
            if str(trigger["target_node"]) not in set(decision_nodes):
                raise ValueError("trigger_revision target must be a missed node")
            if not str(trigger["proposed_when_to_use"]).strip() or not str(
                trigger["reason"]
            ).strip():
                raise ValueError("trigger_revision text must be non-empty")
    elif decision == "EXECUTION_LAPSE":
        if not trace_verified:
            raise ValueError("execution lapse requires verified trace evidence")
        execution_present = (
            all(
                set(decision_nodes) <= repeat_nodes
                and set(decision_edges) <= repeat_edges
                for repeat_nodes, repeat_edges in failed_repeat_usage
            )
            if failed_repeat_usage
            else (
                set(decision_nodes) <= used_nodes
                and set(decision_edges) <= used_edges
            )
        )
        if not execution_present:
            raise ValueError(
                "execution-lapse elements must be present in every verified "
                "failed repeat"
            )
        if trigger is not None:
            raise ValueError("execution lapse cannot emit a trigger revision")
        if not intermediate or not divergence or intermediate == divergence:
            raise ValueError(
                "execution lapse requires distinct correct-intermediate and later-"
                "divergence excerpts"
            )
        observed = _boundary_observable_text(task_record)
        if intermediate not in observed or divergence not in observed:
            raise ValueError("execution-lapse excerpts must be verbatim evidence")
    else:
        if (
            sufficiency == "FULL"
            or observable
            or decision_nodes
            or decision_edges
            or intermediate
            or divergence
            or trigger is not None
        ):
            raise ValueError(
                "UNATTRIBUTED must not claim a fully observable sufficient rule"
            )
    return verdict


def _clear_boundary_failure_channels(analysis: CaseAnalysis) -> None:
    analysis.wrong_nodes = []
    analysis.wrong_edges = []
    analysis.missed_relevant_nodes = []
    analysis.missed_relevant_edges = []
    analysis.execution_lapse_nodes = []
    analysis.execution_lapse_edges = []
    analysis.node_revision_proposals = []
    analysis.retrieval_revision_proposals = []
    analysis.edge_correction_proposals = []
    analysis.new_node_proposals = []
    analysis.new_edge_proposals = []


def _apply_boundary_adjudication(
    graph: SkillGraph,
    result: dict[str, Any],
    analysis: CaseAnalysis,
    verdict: dict[str, Any],
) -> CaseAnalysis:
    """Apply one validated adjudication without inventing graph edits."""
    _clear_boundary_failure_channels(analysis)
    analysis.correct_nodes = list(analysis.used_nodes)
    analysis.correct_edges = list(analysis.used_edges)
    decision = str(verdict["decision"])
    reason = str(verdict["reason"]).strip()
    required_action = str(verdict["required_action"]).strip()
    if decision == "RETRIEVAL_MISS":
        analysis.missed_relevant_nodes = list(verdict["decision_node_ids"])
        analysis.missed_relevant_edges = list(verdict["decision_edge_ids"])
        trigger = verdict["trigger_revision"]
        if isinstance(trigger, dict):
            analysis.retrieval_revision_proposals = [
                RetrievalRevisionProposal(
                    target_node=str(trigger["target_node"]),
                    proposed_when_to_use=str(
                        trigger["proposed_when_to_use"]
                    ).strip(),
                    reason=str(trigger["reason"]).strip(),
                    case_id=analysis.case_id,
                )
            ]
        root = "MISSING_ACTIVATION_CUE: " + reason
    elif decision == "EXECUTION_LAPSE":
        analysis.execution_lapse_nodes = list(verdict["decision_node_ids"])
        analysis.execution_lapse_edges = list(verdict["decision_edge_ids"])
        root = "EXECUTION_LAPSE: " + reason
    else:
        root = "INSUFFICIENT_EVIDENCE: " + reason
        required_action = (
            "No graph edit is justified until the retrieval/execution boundary "
            "is observable."
        )
    analysis.badcase_summary["root_cause"] = root
    analysis.badcase_summary["reusable_fix"] = required_action
    return _sanitize_case_analysis(graph, result, analysis)


def _teacher_adjudicate_boundary(
    graph: SkillGraph,
    result: dict[str, Any],
    task_record: dict[str, Any],
    analysis: CaseAnalysis,
    *,
    chat_fn,
    store: ArtifactStore | None,
    adjudication_cache: dict[str, dict[str, Any]] | None,
) -> CaseAnalysis:
    """Resolve retrieval versus execution once from exact graph-bound evidence."""
    if analysis.failure_type not in _ATTRIBUTION_BOUNDARY_TYPES:
        return analysis
    system = (
        PROMPTS / "attribution_boundary_adjudicate.md"
    ).read_text(encoding="utf-8")
    payload = {
        "protocol": ATTRIBUTION_BOUNDARY_PROTOCOL,
        "current_graph": format_graph(graph),
        "task": task_record,
    }
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    digest = hashlib.sha256(
        (system + "\0" + user).encode("utf-8")
    ).hexdigest()
    verdict: dict[str, Any] | None = None
    cache_record: dict[str, Any] | None = None
    if adjudication_cache is not None:
        with _ATTRIBUTION_CACHE_LOCK:
            saved = adjudication_cache.get(digest)
            if isinstance(saved, dict):
                cache_record = dict(saved)
        if cache_record is not None:
            try:
                verdict = _validate_boundary_adjudication(
                    graph, task_record, cache_record.get("verdict")
                )
            except Exception:
                verdict = None
            else:
                if store is not None:
                    save_template_call(
                        store,
                        analysis.case_id,
                        stage="attribution_boundary_adjudicate_cache",
                        inputs={"input_sha256": digest},
                        outputs={
                            "protocol": ATTRIBUTION_BOUNDARY_PROTOCOL,
                            "verdict": verdict,
                            "source": "evolution_cache_exact_input",
                        },
                    )

    response = ""
    usage: Any = None
    current_user = user
    if verdict is None:
        from graphopt.json_utils import extract_json

        for attempt in range(1, 3):
            response = ""
            usage = None
            try:
                response, usage = chat_fn(
                    system=system,
                    user=current_user,
                    max_completion_tokens=2048,
                    retries=3,
                    stage="attribution_boundary_adjudicate",
                )
                verdict = _validate_boundary_adjudication(
                    graph, task_record, extract_json(response)
                )
                break
            except Exception as exc:
                if store is not None:
                    save_llm_call(
                        store,
                        analysis.case_id,
                        stage="attribution_boundary_adjudicate",
                        system=system,
                        user=current_user,
                        response=response or None,
                        usage=usage,
                        error=f"attempt {attempt}/2: {exc}",
                    )
                if attempt == 1:
                    current_user = (
                        user
                        + "\n\nYour previous adjudication was invalid: "
                        + str(exc)
                        + "\nReturn the complete strict JSON object again."
                        + "\n\nInvalid previous answer:\n"
                        + response
                    )
        if verdict is not None and adjudication_cache is not None:
            with _ATTRIBUTION_CACHE_LOCK:
                adjudication_cache[digest] = {
                    "protocol": ATTRIBUTION_BOUNDARY_PROTOCOL,
                    "input_sha256": digest,
                    "verdict": dict(verdict),
                }

    if verdict is None:
        verdict = {
            "case_id": analysis.case_id,
            "decision": "UNATTRIBUTED",
            "reason": (
                "the independent retrieval/execution adjudication did not "
                "produce a valid evidence-grounded decision"
            ),
            "required_action": "No graph edit is justified from this case.",
            "rule_text_sufficiency": "UNCERTAIN",
            "reusable_correction_observable_without_training_answer": False,
            "sufficient_rule_node_ids": [],
            "sufficient_rule_edge_ids": [],
            "decision_node_ids": [],
            "decision_edge_ids": [],
            "correct_intermediate_excerpt": "",
            "later_divergence_excerpt": "",
            "trigger_revision": None,
        }
    final = _apply_boundary_adjudication(
        graph, result, analysis, verdict
    )
    if store is not None and cache_record is None:
        save_llm_call(
            store,
            analysis.case_id,
            stage="attribution_boundary_adjudicate",
            system=system,
            user=current_user,
            response=response or None,
            usage=usage,
            parsed={
                "protocol": ATTRIBUTION_BOUNDARY_PROTOCOL,
                "input_sha256": digest,
                "verdict": verdict,
                "final_analysis": final.to_dict(),
            },
        )
    return final


def _teacher_analyze_case(
    graph: SkillGraph,
    result: dict[str, Any],
    *,
    chat_fn,
    meta_context: str = "",
    store: ArtifactStore | None = None,
    update_protocol: str = "case_complete",
    attribution_adjudication_cache: dict[str, dict[str, Any]] | None = None,
) -> CaseAnalysis | None:
    update_protocol = _canonical_update_protocol(update_protocol)
    if update_protocol not in {"legacy", "case_complete", "causal"}:
        raise ValueError("update_protocol must be legacy, case_complete, or causal")
    system = (PROMPTS / "case_analyze.md").read_text(encoding="utf-8")
    _, pipeline = _environment_pipeline(graph, result)
    addendum_fn = getattr(pipeline, "case_analyzer_prompt_addendum", None)
    if callable(addendum_fn):
        system = system.rstrip() + "\n\n" + str(addendum_fn()).strip() + "\n"
    if update_protocol == "legacy":
        simple_reasoning = (
            PROMPTS / "case_analyze_simple_reasoning.md"
        ).read_text(encoding="utf-8")
        system = system.rstrip() + "\n\n" + simple_reasoning.strip() + "\n"
    trajectory = str(result.get("trajectory") or "")
    semantic_trace, semantic_trace_status = _semantic_trace_for_attribution(
        result, chat_fn=chat_fn, store=store
    )
    trace_evidence = _joint_trace_evidence(graph, result)
    repeat_rows = _same_sample_repeats(result)
    repeat_outcome = _same_sample_outcome(result)
    repeat_correct_count = sum(
        bool(row.get("hard")) and float(row.get("hard") or 0) >= _HARD_EPS
        for row in repeat_rows
    )
    repeat_evidence_rows = list(trace_evidence.get("repeat_evidence") or [])
    repeat_comparison_table = []
    for repeat_index, repeat_row in enumerate(repeat_rows, start=1):
        evidence = (
            repeat_evidence_rows[repeat_index - 1].get("evidence") or {}
            if repeat_index <= len(repeat_evidence_rows)
            and isinstance(repeat_evidence_rows[repeat_index - 1], dict)
            else build_trace_evidence(graph, repeat_row)
        )
        repeat_comparison_table.append({
            "repeat": repeat_index,
            "outcome": (
                "correct"
                if float(repeat_row.get("hard") or 0.0) >= 1.0 - _HARD_EPS
                else "wrong"
            ),
            "predicted_answer": (
                repeat_row.get("predicted_answer")
                or repeat_row.get("predicted_text")
                or repeat_row.get("response")
            ),
            "used_nodes": list(evidence.get("cited_nodes") or []),
            "used_edges": list(evidence.get("cited_edges") or []),
            "trace_status": evidence.get("status"),
        })
    task_record = {
        "update_protocol": update_protocol,
        "badcase_analysis_protocol": badcase_analysis_protocol(update_protocol),
        "attribution_boundary_protocol": ATTRIBUTION_BOUNDARY_PROTOCOL,
        "case_id": result.get("id"),
        "same_sample_outcome": repeat_outcome,
        "same_sample_repeat_count": len(repeat_rows),
        "same_sample_correct_count": repeat_correct_count,
        "same_sample_repeats": repeat_rows,
        "same_sample_repeat_comparison": repeat_comparison_table,
        "analysis_rule": (
            "Analyze this one complete observation against the frozen graph. "
            "Use successful siblings only as contrastive protection; propose an "
            "edit only for an evidence-grounded reusable graph defect or trigger gap."
        ),
        "environment": result.get("environment") or "searchqa",
        "task_type": result.get("task_type"),
        "task_description": (
            result.get("task_description") or result.get("question")
        ),
        "question": result.get("question") or result.get("task_description"),
        "hard": result.get("hard"),
        "soft": result.get("soft"),
        "fail_reason": result.get("fail_reason"),
        "response": result.get("response"),
        "predicted_answer": result.get("predicted_answer"),
        "predicted_label": result.get("predicted_label"),
        "predicted_text": result.get("predicted_text"),
        "student_step_limit_failure": bool(
            result.get("student_step_limit_failure")
        ),
        "trajectory": trajectory,
        "semantic_reasoning_trace": semantic_trace,
        "semantic_reasoning_trace_status": semantic_trace_status,
        "semantic_reasoning_trace_required": bool(
            result.get("semantic_reasoning_trace_required")
        ),
        "trace_evidence": trace_evidence,
        "loop_diagnostics": diagnose_no_progress_loops(trajectory),
    }
    if len(repeat_rows) == 1:
        for transient_key in (
            "same_sample_outcome",
            "same_sample_repeat_count",
            "same_sample_correct_count",
            "same_sample_repeats",
            "same_sample_repeat_comparison",
        ):
            task_record.pop(transient_key, None)
    training_reference = result.get("training_reference_plan")
    if isinstance(training_reference, dict):
        task_record["training_reference_plan"] = training_reference
    generic_reference = result.get("training_reference")
    if isinstance(generic_reference, dict):
        task_record["training_reference"] = generic_reference
    same_group_rightcases = result.get("same_group_rightcases")
    if isinstance(same_group_rightcases, list):
        task_record["same_group_rightcases"] = same_group_rightcases
        task_record["same_group_rightcase_rule"] = (
            "Use only as successful sibling counterexamples from this failure's "
            "fixed triplet; do not infer support from any other group."
        )
    user = (
        (meta_context + "\n\n" if meta_context else "")
        + format_graph(graph)
        + "\n\n## Task\n"
        + json.dumps(task_record, ensure_ascii=False, indent=2)
    )
    cid = str(result.get("id") or "case")
    success = _is_success(result)
    trajectory_actions = extract_actions(trajectory)

    from graphopt.json_utils import extract_json

    required_top = {
        "case_id",
        "success",
        "failure_type",
        "badcase_summary",
        "same_sample_analysis",
        "existing_graph_usage",
        "node_revision_proposals",
        "retrieval_revision_proposals",
        "edge_correction_proposals",
        "new_node_proposals",
        "new_edge_proposals",
    }
    required_summary = set(_BADCASE_SUMMARY_FIELDS)
    required_usage = {
        "used_nodes",
        "correct_nodes",
        "wrong_nodes",
        "used_edges",
        "correct_edges",
        "wrong_edges",
        "missed_relevant_nodes",
        "missed_relevant_edges",
        "execution_lapse_nodes",
        "execution_lapse_edges",
    }

    def parse_response(response: str) -> dict[str, Any]:
        obj = extract_json(response)
        # Legacy/unit callers may still submit a genuinely single observation.
        # Keep that compatibility narrow: formal three-repeat inputs must emit
        # the complete joint-analysis schema and never receive this fill-in.
        if (
            isinstance(obj, dict)
            and len(repeat_rows) == 1
            and set(obj) == required_top - {"same_sample_analysis"}
        ):
            legacy_proposal_keys = (
                "node_revision_proposals",
                "retrieval_revision_proposals",
                "edge_correction_proposals",
                "new_node_proposals",
                "new_edge_proposals",
            )
            fallback_joint = _fallback_same_sample_analysis(result)
            fallback_joint["graph_modification_needed"] = any(
                obj.get(key) for key in legacy_proposal_keys
            )
            fallback_joint["modification_decision_reason"] = (
                "Single-observation compatibility derived the edit decision "
                "from the emitted proposal lists."
            )
            obj["same_sample_analysis"] = fallback_joint
        if not isinstance(obj, dict) or set(obj) != required_top:
            raise ValueError(
                f"case analysis must contain exactly {sorted(required_top)}"
            )
        repeat_analysis = obj["same_sample_analysis"]
        if (
            not isinstance(repeat_analysis, dict)
            or set(repeat_analysis) != _SAME_SAMPLE_ANALYSIS_FIELDS
        ):
            raise ValueError(
                "same_sample_analysis must contain exactly "
                + str(sorted(_SAME_SAMPLE_ANALYSIS_FIELDS))
            )
        if repeat_analysis["outcome_class"] != repeat_outcome:
            raise ValueError("same_sample_analysis.outcome_class is not evidence-derived")
        if repeat_analysis["repeat_count"] != len(repeat_rows):
            raise ValueError("same_sample_analysis.repeat_count is incorrect")
        if repeat_analysis["correct_count"] != repeat_correct_count:
            raise ValueError("same_sample_analysis.correct_count is incorrect")
        for key in (
            "cross_repeat_comparison",
            "correct_repeat_pattern",
            "failed_repeat_pattern",
            "decisive_difference",
            "graph_text_diagnosis",
            "counterfactual_edit_test",
            "success_experience",
            "inconsistency_cause",
            "modification_decision_reason",
        ):
            if not isinstance(repeat_analysis[key], str) or not repeat_analysis[key].strip():
                raise ValueError(f"same_sample_analysis.{key} must be non-empty")
        if not isinstance(repeat_analysis["graph_modification_needed"], bool):
            raise ValueError(
                "same_sample_analysis.graph_modification_needed must be boolean"
            )
        summary = obj["badcase_summary"]
        if not isinstance(summary, dict) or set(summary) != required_summary:
            raise ValueError(
                f"badcase_summary must contain exactly {sorted(required_summary)}"
            )
        for key in required_summary:
            if not isinstance(summary[key], str) or not summary[key].strip():
                raise ValueError(f"badcase_summary.{key} must be a non-empty string")
        expected_provenance = _summary_provenance(
            semantic_trace, semantic_trace_status
        )
        if summary["evidence_provenance"] != expected_provenance:
            raise ValueError(
                "badcase_summary.evidence_provenance must exactly match "
                f"{expected_provenance!r}"
            )
        usage = obj["existing_graph_usage"]
        if not isinstance(usage, dict) or set(usage) != required_usage:
            raise ValueError(
                f"existing_graph_usage must contain exactly {sorted(required_usage)}"
            )
        for key in required_usage:
            if not isinstance(usage[key], list) or any(
                not isinstance(item, str) for item in usage[key]
            ):
                raise ValueError(f"existing_graph_usage.{key} must be a string list")
        proposal_keys = (
            "node_revision_proposals",
            "retrieval_revision_proposals",
            "edge_correction_proposals",
            "new_node_proposals",
            "new_edge_proposals",
        )
        for key in proposal_keys:
            if not isinstance(obj[key], list) or any(
                not isinstance(item, dict) for item in obj[key]
            ):
                raise ValueError(f"{key} must be an object list")
        root_code = summary["root_cause"].split(":", 1)[0].strip().upper()
        if root_code not in _ROOT_CAUSE_CODES:
            raise ValueError(
                "badcase_summary.root_cause must begin with one allowed root-cause code"
            )
        has_graph_proposal = any(obj[key] for key in proposal_keys)
        if repeat_analysis["graph_modification_needed"] != has_graph_proposal:
            raise ValueError(
                "graph_modification_needed must exactly match whether a graph "
                "proposal is emitted"
            )
        if success and (
            root_code != "SUCCESS"
            or repeat_analysis["graph_modification_needed"]
            or obj["failure_type"] != "SUCCESS"
        ):
            raise ValueError(
                "all-correct repeats require SUCCESS and no graph modification"
            )
        if not success and root_code == "SUCCESS":
            raise ValueError("all-wrong or mixed repeats cannot use SUCCESS root cause")
        if root_code == "INSUFFICIENT_EVIDENCE" and any(
            obj[key] for key in proposal_keys
        ):
            raise ValueError(
                "INSUFFICIENT_EVIDENCE requires every proposal list to be empty"
            )
        if str(obj["case_id"]) != cid:
            raise ValueError(f"case_id must equal the input id {cid!r}")
        if not isinstance(obj["success"], bool) or obj["success"] != success:
            raise ValueError(
                "success must be true exactly for three-of-three correctness"
            )
        if obj["failure_type"] not in _FAILURE_TYPES:
            raise ValueError(f"invalid failure_type: {obj['failure_type']!r}")

        valid_nodes = set(graph.nodes)
        valid_edges = set(_edge_by_id(graph))
        for key in (
            "used_nodes", "correct_nodes", "wrong_nodes",
            "missed_relevant_nodes", "execution_lapse_nodes",
        ):
            values = usage[key]
            if len(values) != len(set(values)):
                raise ValueError(f"existing_graph_usage.{key} contains duplicate ids")
            unknown = set(values) - valid_nodes
            if unknown:
                raise ValueError(f"existing_graph_usage.{key} has unknown nodes: {sorted(unknown)}")
        for key in (
            "used_edges", "correct_edges", "wrong_edges",
            "missed_relevant_edges", "execution_lapse_edges",
        ):
            values = usage[key]
            if len(values) != len(set(values)):
                raise ValueError(f"existing_graph_usage.{key} contains duplicate ids")
            unknown = set(values) - valid_edges
            if unknown:
                raise ValueError(f"existing_graph_usage.{key} has unknown edges: {sorted(unknown)}")
        for kind in ("nodes", "edges"):
            used = set(usage[f"used_{kind}"])
            correct = set(usage[f"correct_{kind}"])
            wrong = set(usage[f"wrong_{kind}"])
            if not correct <= used or not wrong <= used or correct & wrong:
                raise ValueError(f"correct/wrong {kind} must be disjoint subsets of used")
        if (
            repeat_outcome != "mixed"
            and set(usage["missed_relevant_nodes"]) & set(usage["used_nodes"])
        ):
            raise ValueError("missed_relevant_nodes must be disjoint from used_nodes")
        if (
            repeat_outcome != "mixed"
            and set(usage["missed_relevant_edges"]) & set(usage["used_edges"])
        ):
            raise ValueError("missed_relevant_edges must be disjoint from used_edges")
        if not set(usage["execution_lapse_nodes"]) <= set(usage["correct_nodes"]):
            raise ValueError("execution_lapse_nodes must be a subset of correct_nodes")
        if not set(usage["execution_lapse_edges"]) <= set(usage["correct_edges"]):
            raise ValueError("execution_lapse_edges must be a subset of correct_edges")

        if trace_evidence.get("verified"):
            expected_nodes = set(trace_evidence.get("cited_nodes") or [])
            expected_edges = set(trace_evidence.get("cited_edges") or [])
            if set(usage["used_nodes"]) != expected_nodes:
                raise ValueError(
                    "verified trace makes used_nodes immutable; it must exactly "
                    "equal trace_evidence.cited_nodes"
                )
            if set(usage["used_edges"]) != expected_edges:
                raise ValueError(
                    "verified trace makes used_edges immutable; it must exactly "
                    "equal trace_evidence.cited_edges"
                )
            if root_code != "INSUFFICIENT_EVIDENCE":
                if (
                    set(usage["correct_nodes"]) | set(usage["wrong_nodes"])
                ) != expected_nodes:
                    raise ValueError(
                        "every node in a verified failed trace must be classified "
                        "as correct or wrong"
                    )
                if (
                    set(usage["correct_edges"]) | set(usage["wrong_edges"])
                ) != expected_edges:
                    raise ValueError(
                        "every edge in a verified failed trace must be classified "
                        "as correct or wrong"
                    )

        edge_map = _edge_by_id(graph)
        used_nodes = set(usage["used_nodes"])
        for edge_id in usage["used_edges"]:
            edge = edge_map[edge_id]
            if edge.src not in used_nodes or edge.dst not in used_nodes:
                raise ValueError(
                    f"used edge {edge_id} requires both endpoint nodes "
                    f"{edge.src} and {edge.dst} in used_nodes"
                )
        for edge_id in usage["wrong_edges"]:
            if normalize_edge_type(edge_map[edge_id].type) == "co_occur":
                raise ValueError("co_occur is a retrieval hint and cannot be a wrong edge")
        available_nodes = used_nodes | set(usage["missed_relevant_nodes"])
        for edge_id in usage["missed_relevant_edges"]:
            edge = edge_map[edge_id]
            if edge.src not in available_nodes or edge.dst not in available_nodes:
                raise ValueError(
                    f"missed edge {edge_id} requires both endpoints in used_nodes or "
                    "missed_relevant_nodes"
                )

        retrieval_targets: list[str] = []
        for proposal in obj["retrieval_revision_proposals"]:
            if set(proposal) != {"target_node", "proposed_when_to_use", "reason"}:
                raise ValueError(
                    "retrieval revision must contain exactly target_node, "
                    "proposed_when_to_use, and reason"
                )
            target = proposal["target_node"]
            if target not in set(usage["missed_relevant_nodes"]):
                raise ValueError("retrieval revision target must be a missed relevant node")
            if not str(proposal["proposed_when_to_use"]).strip() or not str(
                proposal["reason"]
            ).strip():
                raise ValueError("retrieval revision trigger and reason must be non-empty")
            retrieval_targets.append(str(target))
        if len(retrieval_targets) != len(set(retrieval_targets)):
            raise ValueError("at most one retrieval revision is allowed per missed node")
        # A retrieval miss is evidence that the skill was not activated, but it
        # does not prove that the stored trigger is defective. The analyzer may
        # therefore record a miss without proposing a trigger edit.

        probe = CaseAnalysis(
            case_id=cid,
            success=success,
            missed_relevant_nodes=list(usage["missed_relevant_nodes"]),
            missed_relevant_edges=list(usage["missed_relevant_edges"]),
            execution_lapse_nodes=list(usage["execution_lapse_nodes"]),
            execution_lapse_edges=list(usage["execution_lapse_edges"]),
            wrong_nodes=list(usage["wrong_nodes"]),
            wrong_edges=list(usage["wrong_edges"]),
        )
        lapse_targets = set(usage["execution_lapse_nodes"])
        has_non_lapse_node_proposal = any(
            str(item.get("target_node") or "") not in lapse_targets
            for item in obj["node_revision_proposals"]
        )
        if has_non_lapse_node_proposal or any(
            obj[key]
            for key in ("edge_correction_proposals", "new_node_proposals", "new_edge_proposals")
        ):
            # Presence of any graph-content/topology proposal is skill-defect
            # evidence even when it addresses a missing skill rather than a
            # currently wrong ID.
            probe.new_node_proposals = [NewNodeProposal("probe", "probe")]
        expected_failure_type = _derive_failure_type(probe)
        if obj["failure_type"] != expected_failure_type:
            raise ValueError(
                f"failure_type must be {expected_failure_type!r} for the declared attribution"
            )

        if success:
            if any(usage[key] for key in required_usage if key not in {
                "used_nodes", "correct_nodes", "used_edges", "correct_edges"
            }) or any(
                obj[key]
                for key in (
                    "node_revision_proposals", "retrieval_revision_proposals",
                    "edge_correction_proposals",
                    "new_node_proposals", "new_edge_proposals",
                )
            ):
                raise ValueError("a successful case cannot contain failure attribution or proposals")
            return obj

        node_targets: list[str] = []
        node_proposals_by_target: dict[str, dict[str, Any]] = {}
        allowed_patch_targets = set(usage["correct_nodes"])
        required_proposal_fields = {
            "target_node", "operation", "toxic_text", "proposal", "reason",
            "first_wrong_step", "observable_state", "bad_action",
            "better_action", "semantic_delta",
        }
        for proposal in obj["node_revision_proposals"]:
            if set(proposal) != required_proposal_fields:
                raise ValueError(
                    "node revision must include target/operation/text plus "
                    "first_wrong_step, observable_state, bad_action, "
                    "better_action, and semantic_delta"
                )
            target = proposal.get("target_node")
            operation = str(proposal.get("operation") or "").upper()
            toxic_text = str(proposal.get("toxic_text") or "")
            first_wrong_step = proposal.get("first_wrong_step")
            observable_state = str(proposal.get("observable_state") or "").strip()
            bad_action = str(proposal.get("bad_action") or "").strip()
            better_action = str(proposal.get("better_action") or "").strip()
            semantic_delta = str(proposal.get("semantic_delta") or "").strip()
            if (
                target not in valid_nodes
                or operation not in {"PATCH", "REWRITE"}
                or not str(proposal.get("proposal") or "").strip()
                or not str(proposal.get("reason") or "").strip()
                or not isinstance(first_wrong_step, int)
                or isinstance(first_wrong_step, bool)
                or first_wrong_step < 0
                or not observable_state
                or not bad_action
                or not better_action
                or bad_action.casefold() == better_action.casefold()
            ):
                raise ValueError(
                    "node revision needs a valid target and a concrete, distinct "
                    "bad-action -> better-action counterfactual"
                )
            if trajectory_actions and len(repeat_rows) == 1:
                if first_wrong_step >= len(trajectory_actions):
                    raise ValueError(
                        "first_wrong_step must index an action in the supplied trajectory"
                    )
                actual_action = trajectory_actions[first_wrong_step]
                if _normalized_action(bad_action) != _normalized_action(actual_action):
                    raise ValueError(
                        "bad_action must exactly match the trajectory action at "
                        "first_wrong_step"
                    )
            else:
                static_evidence = _boundary_observable_text(task_record)
                if bad_action.casefold() not in static_evidence.casefold():
                    raise ValueError(
                        "bad_action must be a verbatim excerpt of one supplied "
                        "same-sample response, trace, prediction, or evaluation"
                    )
            if target in node_proposals_by_target:
                raise ValueError("at most one node revision is allowed per target per case")
            if target in set(usage["execution_lapse_nodes"]):
                if operation != "PATCH" or toxic_text.strip() or semantic_delta:
                    raise ValueError(
                        "execution-lapse reminders must be PATCH with empty "
                        "toxic_text and semantic_delta"
                    )
            elif operation == "REWRITE":
                node = graph.nodes[str(target)]
                searchable = "\n".join(
                    (node.meaning, node.when_to_use, node.how_to_use, *node.avoid)
                )
                if (
                    target not in set(usage["wrong_nodes"])
                    or not toxic_text.strip()
                    or toxic_text not in searchable
                    or not semantic_delta
                ):
                    raise ValueError(
                        "REWRITE requires a wrong node, exact toxic_text, and "
                        "a stated semantic correction"
                    )
            elif (
                target not in allowed_patch_targets
                or toxic_text.strip()
                or not semantic_delta
            ):
                raise ValueError(
                    "PATCH requires a used correct node, empty toxic_text, and "
                    "a genuinely new semantic delta"
                )
            node_targets.append(str(target))
            node_proposals_by_target[str(target)] = proposal
        required_node_targets = set(usage["wrong_nodes"])
        if not required_node_targets <= set(node_targets):
            raise ValueError("every wrong node must have one matching REWRITE opinion")

        edge_targets: list[str] = []
        for proposal in obj["edge_correction_proposals"]:
            target_edge = proposal.get("target_edge")
            replacement = proposal.get("proposal")
            if target_edge not in valid_edges or not isinstance(replacement, dict):
                raise ValueError("edge correction has an invalid target or proposal")
            source = replacement.get("source")
            target = replacement.get("target")
            relation = normalize_edge_type(str(replacement.get("relation") or ""))
            if source not in valid_nodes or target not in valid_nodes or source == target:
                raise ValueError("edge correction endpoints must be distinct existing nodes")
            if relation not in {"prereq", "enhance"}:
                raise ValueError("edge correction relation must be prereq or enhance")
            if not str(proposal.get("reason") or "").strip():
                raise ValueError("edge correction requires a semantic reason")
            edge_targets.append(str(target_edge))
        if sorted(edge_targets) != sorted(usage["wrong_edges"]):
            raise ValueError("every wrong edge must have exactly one matching correction")

        required_new_node_fields = {
            "temp_id", "content", "parent_node", "relation", "reason",
        }
        for proposal in obj["new_node_proposals"]:
            if set(proposal) != required_new_node_fields:
                raise ValueError(
                    "new node proposal must contain exactly temp_id, content, "
                    "parent_node, relation, and reason"
                )
            if not str(proposal.get("content") or "").strip():
                raise ValueError("new node proposal content must be non-empty")
            parent_node = str(proposal.get("parent_node") or "")
            if parent_node not in set(usage["correct_nodes"]):
                raise ValueError(
                    "new node parent_node must be one trace-used correct node"
                )
            if normalize_edge_type(str(proposal.get("relation") or "")) != "enhance":
                raise ValueError("new node activation relation must be enhance")
            if not str(proposal.get("reason") or "").strip():
                raise ValueError("new node proposal requires an activation reason")
        for proposal in obj["new_edge_proposals"]:
            source = proposal.get("source")
            target = proposal.get("target")
            relation = normalize_edge_type(str(proposal.get("relation") or ""))
            if source not in valid_nodes or target not in valid_nodes or source == target:
                raise ValueError("new edge endpoints must be distinct existing nodes")
            if relation not in {"prereq", "enhance"}:
                raise ValueError("new edge relation must be prereq or enhance")
            if not str(proposal.get("reason") or "").strip():
                raise ValueError("new edge proposal requires a semantic reason")

        node_operations = {
            str(item.get("operation") or "").upper()
            for item in obj["node_revision_proposals"]
        }
        has_wrong = bool(usage["wrong_nodes"] or usage["wrong_edges"])
        has_missed = bool(
            usage["missed_relevant_nodes"] or usage["missed_relevant_edges"]
        )
        has_lapse = bool(
            usage["execution_lapse_nodes"] or usage["execution_lapse_edges"]
        )
        if root_code == "HARMFUL_EXISTING_RULE":
            if (
                not has_wrong
                or node_operations - {"REWRITE"}
                or obj["retrieval_revision_proposals"]
                or obj["new_node_proposals"]
                or obj["new_edge_proposals"]
            ):
                raise ValueError(
                    "HARMFUL_EXISTING_RULE requires a trace-used wrong element "
                    "and only its REWRITE or edge correction"
                )
        elif root_code == "MISSING_ACTIVATION_CUE":
            if (
                not has_missed or has_wrong
                or obj["node_revision_proposals"]
                or obj["edge_correction_proposals"]
                or obj["new_node_proposals"]
                or obj["new_edge_proposals"]
            ):
                raise ValueError(
                    "MISSING_ACTIVATION_CUE may only clarify when_to_use for a "
                    "missed existing element"
                )
        elif root_code == "MISSING_OR_INCORRECT_PROCEDURE":
            if (
                has_wrong or "REWRITE" in node_operations
                or obj["retrieval_revision_proposals"]
                or not (obj["node_revision_proposals"] or obj["new_edge_proposals"])
            ):
                raise ValueError(
                    "MISSING_OR_INCORRECT_PROCEDURE requires a PATCH on a "
                    "trace-used correct node or a grounded structural addition"
                )
        elif root_code == "MISSING_SKILL_FAMILY":
            if (
                not obj["new_node_proposals"]
                or obj["node_revision_proposals"]
                or obj["retrieval_revision_proposals"]
                or has_wrong
            ):
                raise ValueError(
                    "MISSING_SKILL_FAMILY requires a genuinely new node rather "
                    "than rewriting an existing traced rule"
                )
        elif root_code == "EXECUTION_LAPSE":
            if (
                not trace_evidence.get("verified") or not has_lapse
                or any(obj[key] for key in proposal_keys)
            ):
                raise ValueError(
                    "EXECUTION_LAPSE requires a verified correct-but-misapplied "
                    "element and cannot mutate the graph"
                )
        expected_failure_family = {
            "MISSING_ACTIVATION_CUE": {"RETRIEVAL_MISS", "MIXED"},
            "MISSING_OR_INCORRECT_PROCEDURE": {"SKILL_DEFECT", "MIXED"},
            "HARMFUL_EXISTING_RULE": {"SKILL_DEFECT", "MIXED"},
            "MISSING_SKILL_FAMILY": {"SKILL_DEFECT", "MIXED"},
            "EXECUTION_LAPSE": {"EXECUTION_LAPSE", "MIXED"},
            "INSUFFICIENT_EVIDENCE": {"UNATTRIBUTED"},
        }[root_code]
        if obj["failure_type"] not in expected_failure_family:
            raise ValueError(
                f"root cause {root_code} is inconsistent with failure_type "
                f"{obj['failure_type']}"
            )
        return obj

    current_user = user
    obj: dict[str, Any] | None = None
    resp = ""
    token_usage: Any = None
    reused_cached = False
    cached = _cached_teacher_response(
        store, case_id=cid, system=system, base_user=user
    )
    if cached is not None:
        cached_response, cached_usage = cached
        try:
            obj = parse_response(cached_response)
        except Exception:
            obj = None
        else:
            resp = cached_response
            token_usage = cached_usage
            reused_cached = True
            print(f"[graphopt analyzer] reused cached case={cid}", flush=True)

    for attempt in range(1, 3) if obj is None else ():
        resp = ""
        token_usage = None
        try:
            resp, token_usage = chat_fn(
                system=system,
                user=current_user,
                max_completion_tokens=4096,
                retries=3,
                stage="case_analyze",
            )
            obj = parse_response(resp)
            break
        except Exception as exc:
            error = str(exc)
            if store is not None:
                save_llm_call(
                    store,
                    cid,
                    stage="case_analyze",
                    system=system,
                    user=current_user,
                    response=resp or None,
                    usage=token_usage,
                    error=f"attempt {attempt}/2: {error}",
                )
            if attempt == 1:
                current_user = _case_analysis_correction_prompt(
                    user,
                    error=error,
                    invalid_response=str(resp),
                )
    if obj is None:
        # Field-level salvage: preserve only syntactically reliable fields from
        # the last response. Never invent proposal text or graph references.
        try:
            raw = extract_json(resp)
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            raw = {}
        raw_usage = raw.get("existing_graph_usage")
        if not isinstance(raw_usage, dict):
            raw_usage = {}
        valid_nodes = set(graph.nodes)
        valid_edges = set(_edge_by_id(graph))
        usage = {}
        for key in required_usage:
            value = raw_usage.get(key)
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                value = []
            allowed = valid_edges if key.endswith("edges") else valid_nodes
            usage[key] = list(dict.fromkeys(x for x in value if x in allowed))
        obj = {
            "case_id": cid, "success": success,
            "failure_type": "SUCCESS" if success else "UNATTRIBUTED",
            "badcase_summary": _fallback_badcase_summary(
                result, semantic_trace, semantic_trace_status
            ),
            "same_sample_analysis": _fallback_same_sample_analysis(result),
            "existing_graph_usage": usage,
        }
        for key in ("node_revision_proposals", "retrieval_revision_proposals",
                    "edge_correction_proposals", "new_node_proposals",
                    "new_edge_proposals"):
            value = raw.get(key)
            obj[key] = [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []
        result["analysis_partial"] = True

    usage_block = obj["existing_graph_usage"]

    def _lst(key: str) -> list[str]:
        v = usage_block.get(key) or obj.get(key) or []
        if not isinstance(v, (list, tuple, set)):
            return []
        return [str(x) for x in v]

    analysis = CaseAnalysis(
        case_id=cid,
        success=success,
        used_nodes=_lst("used_nodes"),
        correct_nodes=_lst("correct_nodes"),
        wrong_nodes=_lst("wrong_nodes"),
        used_edges=_lst("used_edges"),
        correct_edges=_lst("correct_edges"),
        wrong_edges=_lst("wrong_edges"),
        failure_type=str(obj["failure_type"]),
        analysis_partial=bool(result.get("analysis_partial")),
        badcase_summary={
            key: str(value).strip()
            for key, value in obj.get("badcase_summary", {}).items()
            if key in _BADCASE_SUMMARY_FIELDS
        },
        same_sample_analysis=dict(
            obj.get("same_sample_analysis")
            or _fallback_same_sample_analysis(result)
        ),
        trace_evidence=dict(trace_evidence),
        missed_relevant_nodes=_lst("missed_relevant_nodes"),
        missed_relevant_edges=_lst("missed_relevant_edges"),
        execution_lapse_nodes=_lst("execution_lapse_nodes"),
        execution_lapse_edges=_lst("execution_lapse_edges"),
    )

    if success:
        analysis.node_revision_proposals = []
        analysis.retrieval_revision_proposals = []
        analysis.edge_correction_proposals = []
        analysis.new_node_proposals = []
        analysis.new_edge_proposals = []
        _sanitize_case_analysis(graph, result, analysis)
        if store is not None and not reused_cached:
            save_llm_call(
                store,
                cid,
                stage="case_analyze",
                system=system,
                user=current_user,
                response=resp,
                usage=token_usage,
                parsed=analysis.to_dict(),
            )
        return analysis

    for p in obj.get("node_revision_proposals") or []:
        if isinstance(p, dict) and p.get("target_node"):
            analysis.node_revision_proposals.append(
                NodeRevisionProposal(
                    target_node=str(p["target_node"]),
                    proposal=str(p.get("proposal") or ""),
                    reason=str(p.get("reason") or ""),
                    case_id=cid,
                    operation=str(p.get("operation") or "PATCH").upper(),
                    toxic_text=str(p.get("toxic_text") or ""),
                    first_wrong_step=int(p.get("first_wrong_step") or 0),
                    observable_state=str(p.get("observable_state") or ""),
                    bad_action=str(p.get("bad_action") or ""),
                    better_action=str(p.get("better_action") or ""),
                    semantic_delta=str(p.get("semantic_delta") or ""),
                )
            )
    for p in obj.get("retrieval_revision_proposals") or []:
        if isinstance(p, dict) and p.get("target_node"):
            analysis.retrieval_revision_proposals.append(
                RetrievalRevisionProposal(
                    target_node=str(p["target_node"]),
                    proposed_when_to_use=str(p.get("proposed_when_to_use") or ""),
                    reason=str(p.get("reason") or ""),
                    case_id=cid,
                )
            )
    for p in obj.get("edge_correction_proposals") or []:
        if not isinstance(p, dict):
            continue
        prop = p.get("proposal") or p
        analysis.edge_correction_proposals.append(
            EdgeCorrectionProposal(
                target_edge=str(p.get("target_edge") or ""),
                source=str(prop.get("source") or ""),
                target=str(prop.get("target") or ""),
                relation=str(prop.get("relation") or "prereq"),
                reason=str(p.get("reason") or ""),
                case_id=cid,
            )
        )
    for p in obj.get("new_node_proposals") or []:
        if isinstance(p, dict) and p.get("content"):
            analysis.new_node_proposals.append(
                NewNodeProposal(
                    temp_id=str(p.get("temp_id") or f"new_{cid}"),
                    content=str(p["content"]),
                    case_id=cid,
                    parent_node=str(p.get("parent_node") or ""),
                    relation=normalize_edge_type(str(p.get("relation") or "enhance")),
                    reason=str(p.get("reason") or ""),
                )
            )
    for p in obj.get("new_edge_proposals") or []:
        if isinstance(p, dict) and p.get("source") and p.get("target"):
            analysis.new_edge_proposals.append(
                NewEdgeProposal(
                    source=str(p["source"]),
                    target=str(p.get("target") or ""),
                    relation=str(p.get("relation") or "prereq"),
                    reason=str(p.get("reason") or ""),
                    case_id=cid,
                    target_old_edge=str(p.get("target_old_edge") or ""),
                )
            )
    _sanitize_case_analysis(graph, result, analysis)
    if store is not None and not reused_cached:
        save_llm_call(
            store,
            cid,
            stage="case_analyze",
            system=system,
            user=current_user,
            response=resp,
            usage=token_usage,
            parsed=analysis.to_dict(),
        )
    return analysis


def analyze_cases(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    *,
    chat_fn=None,
    mode: str = "template",
    meta_context: str = "",
    store: ArtifactStore | None = None,
    max_workers: int = 16,
    analyze_successes_with_teacher: bool = False,
    update_protocol: str = "case_complete",
    attribution_adjudication_cache: dict[str, dict[str, Any]] | None = None,
) -> list[CaseAnalysis]:
    """Analyze cases against frozen G_t with bounded, order-preserving parallelism."""
    update_protocol = _canonical_update_protocol(update_protocol)
    if update_protocol not in {"legacy", "case_complete", "causal"}:
        raise ValueError("update_protocol must be legacy, case_complete, or causal")
    use_teacher = mode == "teacher" and chat_fn is not None
    if not use_teacher:
        return [
            _template_analyze_case(
                graph, r, gate_context=meta_context, store=store,
                update_protocol=update_protocol,
            )
            for r in results
        ]
    if not results:
        return []

    ordered: list[CaseAnalysis | None] = [None] * len(results)
    teacher_indices: list[int] = []
    for index, result in enumerate(results):
        if _is_success(result) and not analyze_successes_with_teacher:
            ordered[index] = _heuristic_success_analysis(
                graph, result, store=store
            )
        else:
            teacher_indices.append(index)

    if not teacher_indices:
        return [analysis for analysis in ordered if analysis is not None]

    workers = min(len(teacher_indices), max(1, int(max_workers)))
    print(
        f"[graphopt analyzer] teacher_cases={len(teacher_indices)} "
        f"successes_heuristic={len(results) - len(teacher_indices)} "
        f"workers={workers} mode=teacher",
        flush=True,
    )

    def analyze_one(index: int) -> tuple[int, CaseAnalysis | None]:
        return index, _teacher_analyze_case(
            graph,
            results[index],
            chat_fn=chat_fn,
            meta_context=meta_context,
            store=store,
            update_protocol=update_protocol,
            attribution_adjudication_cache=attribution_adjudication_cache,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(analyze_one, index) for index in teacher_indices]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index, analysis = future.result()
            ordered[index] = analysis
            completed += 1
            print(
                f"[graphopt analyzer] completed_teacher_cases={completed}/"
                f"{len(teacher_indices)} case={results[index].get('id')}",
                flush=True,
            )
    return [analysis for analysis in ordered if analysis is not None]
