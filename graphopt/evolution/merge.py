"""Case-complete semantic synthesis without changing usage statistics.

Every opinion retains its original task, student answer, evaluator/reference,
complete reasoning, validated usage, and companion graph proposals. Equivalent
meanings merge; distinct meanings become specialist node+edge candidates. Python
remains authoritative for case identities, support, and atomic provenance.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from graphopt.debug.artifacts import ArtifactStore, save_llm_call, save_template_call
from graphopt.evolution.exact_refs import exact_reference_json
from graphopt.evolution.types import CaseAnalysis, MergedProposal
from graphopt.types import SkillGraph, normalize_edge_type

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
OPINION_MERGE_TIMEOUT_SECONDS = 600


class OpinionMergeInfrastructureError(RuntimeError):
    """A merge request failed before a usable teacher response was available."""

    def __init__(self, *, stage: str, target: str, error: Exception) -> None:
        self.stage = stage
        self.target = target
        self.original_error = error
        super().__init__(f"{stage} target={target}: {error}")


def _is_retryable_opinion_merge_error(exc: Exception) -> bool:
    """Separate transport/time-limit failures from invalid model JSON."""
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    return (
        "timeout" in name
        or "timed out" in text
        or "timeout" in text
        or "connection error" in text
        or "connection reset" in text
        or "connection aborted" in text
        or "rate limit" in text
        or "too many requests" in text
        or "http 429" in text
        or "http 502" in text
        or "http 503" in text
        or "error code: 429" in text
        or "error code: 502" in text
        or "error code: 503" in text
    )

_STOP = frozenset(
    "a an the and or to of in on at for from with before after must do not any".split()
)
_WORD_NORMALIZATION = {
    "cabinet": "container",
    "cabinets": "container",
    "containers": "container",
    "receptacle": "container",
    "receptacles": "container",
    "opened": "open",
    "opening": "open",
    "taking": "take",
    "takes": "take",
}


def _sig(text: str) -> str:
    """Normalized signature for near-duplicate clustering."""
    t = " ".join((text or "").lower().split())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    words = [
        _WORD_NORMALIZATION.get(word, word)
        for word in t.split()
        if word and word not in _STOP
    ]
    return " ".join(words[:24])


def _tokens(text: str) -> set[str]:
    return set(_sig(text).split())


def _similar(a: str, b: str) -> bool:
    """True if two proposals are semantically close enough to merge."""
    if not a.strip() or not b.strip():
        return False
    sa, sb = _sig(a), _sig(b)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    if sa in sb or sb in sa:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union >= 0.55

def _proposal_pool_bucket(proposal: MergedProposal) -> str:
    """Return the protected bucket within which cached opinions may merge."""
    if proposal.kind == "node_revision":
        return "node|{}|{}|{}".format(
            proposal.target_node,
            str(proposal.operation or "PATCH").upper(),
            proposal.toxic_text,
        )
    if proposal.kind == "retrieval_revision":
        return f"retrieval|{proposal.target_node}"
    if proposal.kind == "new_node":
        return "new_node"
    return ""


def _merge_texts(texts: list[str]) -> str:
    """Merge semantically similar texts; drop redundant sentences."""
    if not texts:
        return ""
    ordered = sorted(texts, key=len, reverse=True)
    kept: list[str] = []
    for t in ordered:
        s = t.strip()
        if not s:
            continue
        if any(_similar(s, k) for k in kept):
            continue
        kept.append(s)
    if not kept:
        return ""
    merged = kept[0]
    for p in kept[1:]:
        if p.lower() not in merged.lower():
            merged = f"{merged} {p}".strip()
    return merged


def _cluster_text_items(items: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Group (case_id, text) by semantic similarity (transitive greedy)."""
    clusters: list[list[tuple[str, str]]] = []
    for case_id, text in items:
        placed = False
        for cluster in clusters:
            if _similar(text, cluster[0][1]):
                cluster.append((case_id, text))
                placed = True
                break
        if not placed:
            clusters.append([(case_id, text)])
    return clusters

def _execution_evidence_text(item: dict[str, Any]) -> str:
    """Lossless compact text used only for execution-lapse similarity."""
    return " | ".join(
        part for part in (
            str(item.get("observable_state") or "").strip(),
            f"bad_family={_action_family(str(item.get('bad_action') or ''))}",
            f"better_family={_action_family(str(item.get('better_action') or ''))}",
            str(item.get("proposal") or "").strip(),
            str(item.get("reason") or "").strip(),
        )
        if part and not part.endswith("=")
    )


def _action_family(action: str) -> str:
    match = re.search(r"[a-z]+", str(action or "").casefold())
    return match.group(0) if match else "unknown"


def _fallback_execution_key(group: list[dict[str, Any]]) -> str:
    representative = group[0]
    semantic_tokens = _sig(
        " ".join(
            str(representative.get(key) or "")
            for key in ("observable_state", "proposal", "reason")
        )
    ).split()[:6]
    parts = [
        _action_family(str(representative.get("bad_action") or "")),
        "to",
        _action_family(str(representative.get("better_action") or "")),
        *semantic_tokens,
    ]
    key = "_".join(re.sub(r"[^a-z0-9]+", "_", part).strip("_") for part in parts)
    key = re.sub(r"_+", "_", key).strip("_").upper()
    return (key or "EXECUTION_DETAIL")[:80]

_EXECUTION_GENERIC = frozenset({"bad", "better", "family", "unknown"})


def _execution_semantically_similar(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if (
        _action_family(str(a.get("bad_action") or ""))
        != _action_family(str(b.get("bad_action") or ""))
        or _action_family(str(a.get("better_action") or ""))
        != _action_family(str(b.get("better_action") or ""))
    ):
        return False
    left = _tokens(_execution_evidence_text(a)) - _EXECUTION_GENERIC
    right = _tokens(_execution_evidence_text(b)) - _EXECUTION_GENERIC
    if not left or not right:
        return False
    overlap = len(left & right)
    return _similar(_execution_evidence_text(a), _execution_evidence_text(b)) or (
        overlap >= 3 and overlap / len(left | right) >= 0.25
    )



def _execution_clusters_template(
    evidence: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Conservative fallback: action family must match and semantics must overlap."""
    grouped: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for item in evidence:
        bucket = (
            str(item.get("target_node") or ""),
            _action_family(str(item.get("bad_action") or "")),
            _action_family(str(item.get("better_action") or "")),
        )
        placed = False
        for cluster in grouped[bucket]:
            if _execution_semantically_similar(item, cluster[0]):
                cluster.append(item)
                placed = True
                break
        if not placed:
            grouped[bucket].append([item])
    output: list[tuple[str, list[dict[str, Any]]]] = []
    key_counts: dict[tuple[str, str], int] = {}
    for bucket in sorted(grouped):
        for group in grouped[bucket]:
            base_key = _fallback_execution_key(group)
            identity = (bucket[0], base_key)
            key_counts[identity] = key_counts.get(identity, 0) + 1
            count = key_counts[identity]
            key = (
                base_key if count == 1
                else f"{base_key[:76]}_{count}"
            )
            output.append((key, group))
    return output


def semantic_merge_execution_lapses(
    evidence: list[dict[str, Any]],
    graph: SkillGraph | None = None,
    *,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> list[MergedProposal]:
    """Cluster execution lapses by meaning while retaining every source record.

    The teacher may only partition opinion IDs and name each cluster. Python
    reconstructs the complete evidence, case support, and child input.
    """
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in evidence:
        item = {
            "target_node": str(raw.get("target_node") or ""),
            "case_id": str(raw.get("case_id") or ""),
            "proposal": str(raw.get("proposal") or "").strip(),
            "reason": str(raw.get("reason") or "").strip(),
            "observable_state": str(raw.get("observable_state") or "").strip(),
            "bad_action": str(raw.get("bad_action") or "").strip(),
            "better_action": str(raw.get("better_action") or "").strip(),
            "is_current": bool(raw.get("is_current")),
        }
        if not item["target_node"] or not item["case_id"] or not item["proposal"]:
            continue
        deduped[(item["target_node"], item["case_id"])] = item
    items = sorted(
        deduped.values(),
        key=lambda item: (
            str(item["target_node"]),
            0 if item["is_current"] else 1,
            str(item["case_id"]),
        ),
    )
    for index, item in enumerate(items, start=1):
        item["opinion_id"] = f"lapse_{index:04d}"
    if not items:
        return []

    groups: list[tuple[str, list[dict[str, Any]]]]
    use_teacher = mode == "teacher" and chat_fn is not None and len(items) > 1
    if use_teacher:
        system = (PROMPTS / "execution_lapse_merge.md").read_text(encoding="utf-8")
        relevant_nodes = {
            node_id: graph.nodes[node_id].to_dict()
            for node_id in sorted({str(item["target_node"]) for item in items})
            if graph is not None and node_id in graph.nodes
        }
        payload = {
            "parent_nodes": relevant_nodes,
            "execution_lapses": [
                {key: value for key, value in item.items() if key != "is_current"}
                for item in items
            ],
        }
        user = exact_reference_json(payload)
        from graphopt.json_utils import extract_json

        by_id = {str(item["opinion_id"]): item for item in items}
        expected_ids = set(by_id)

        def parse(response: str) -> list[tuple[str, list[dict[str, Any]]]]:
            obj = extract_json(response)
            if not isinstance(obj, dict) or set(obj) != {"clusters"}:
                raise ValueError("execution merge must contain exactly clusters")
            raw_clusters = obj["clusters"]
            if not isinstance(raw_clusters, list) or not raw_clusters:
                raise ValueError("clusters must be a non-empty list")
            seen: set[str] = set()
            semantic_keys: set[tuple[str, str]] = set()
            parsed: list[tuple[str, list[dict[str, Any]]]] = []
            for raw in raw_clusters:
                if not isinstance(raw, dict) or set(raw) != {
                    "semantic_key", "opinion_ids",
                }:
                    raise ValueError(
                        "each execution cluster needs semantic_key and opinion_ids"
                    )
                semantic_key = str(raw["semantic_key"] or "")
                if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,79}", semantic_key):
                    raise ValueError("semantic_key must be stable uppercase snake case")
                ids = raw["opinion_ids"]
                if (
                    not isinstance(ids, list) or not ids
                    or any(not isinstance(value, str) for value in ids)
                    or len(ids) != len(set(ids))
                ):
                    raise ValueError("opinion_ids must be a non-empty unique string list")
                if set(ids) - expected_ids or set(ids) & seen:
                    raise ValueError("execution cluster has unknown or repeated opinion_ids")
                group = [by_id[opinion_id] for opinion_id in ids]
                parents = {str(item["target_node"]) for item in group}
                if len(parents) != 1:
                    raise ValueError("one execution cluster cannot cross parent nodes")
                identity = (next(iter(parents)), semantic_key)
                if identity in semantic_keys:
                    raise ValueError("semantic_key must be unique within one parent")
                semantic_keys.add(identity)
                seen.update(ids)
                parsed.append((semantic_key, group))
            if seen != expected_ids:
                raise ValueError("every execution opinion_id must appear exactly once")
            return parsed

        groups = []
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
                    stage="execution_lapse_semantic_merge",
                    timeout=OPINION_MERGE_TIMEOUT_SECONDS,
                )
                groups = parse(response)
                if store is not None:
                    save_llm_call(
                        store, "all_execution_lapses",
                        stage="execution_lapse_semantic_merge",
                        system=system, user=current_user, response=response,
                        usage=usage,
                        parsed={"clusters": [
                            {"semantic_key": key, "opinion_ids": [
                                str(item["opinion_id"]) for item in group
                            ]}
                            for key, group in groups
                        ]},
                    )
                break
            except Exception as exc:
                if store is not None:
                    save_llm_call(
                        store, "all_execution_lapses",
                        stage="execution_lapse_semantic_merge",
                        system=system, user=current_user,
                        response=response or None, usage=usage,
                        error=f"attempt {attempt}/2: {exc}",
                    )
                if _is_retryable_opinion_merge_error(exc):
                    if attempt == 2:
                        raise RuntimeError(
                            "Execution-lapse opinion merge timed out again; "
                            "stopping experiment"
                        ) from exc
                    continue
                current_user = (
                    user
                    + "\n\nThe previous response was invalid. Return strict JSON; "
                    "include every opinion_id exactly once and never mix parents."
                )
        if not groups:
            groups = _execution_clusters_template(items)
    else:
        groups = _execution_clusters_template(items)
        if store is not None:
            save_template_call(
                store, "all_execution_lapses",
                stage="execution_lapse_semantic_merge",
                inputs={"execution_lapses": items},
                outputs={"clusters": [
                    {"semantic_key": key, "opinion_ids": [
                        str(item["opinion_id"]) for item in group
                    ]}
                    for key, group in groups
                ]},
            )

    proposals: list[MergedProposal] = []
    for semantic_key, group in groups:
        parent = str(group[0]["target_node"])
        ordered = sorted(
            group,
            key=lambda item: (0 if item["is_current"] else 1, str(item["case_id"])),
        )
        case_ids = [str(item["case_id"]) for item in ordered]
        evidence_items = [
            {
                key: value
                for key, value in item.items()
                if key not in {"opinion_id", "is_current"}
            }
            for item in ordered
        ]
        content = "\n".join(
            f"{index}. " + json.dumps(item, ensure_ascii=False, sort_keys=True)
            for index, item in enumerate(evidence_items, start=1)
        )
        proposals.append(MergedProposal(
            kind="execution_reinforcement",
            target_node=parent,
            content=content,
            rationale=f"execution_detail:{parent}:{semantic_key}",
            support=len(set(case_ids)),
            source_case_ids=list(dict.fromkeys(case_ids)),
            semantic_key=semantic_key,
            evidence_items=evidence_items,
        ))
    return proposals



def _new_node_units(
    proposals: list[Any],
    case_evidence: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Collapse exact new-node meanings while retaining grounded activations."""
    case_evidence = case_evidence or {}
    exact: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        content = " ".join(str(proposal.content or "").split()).strip()
        parent = str(proposal.parent_node or "")
        relation = normalize_edge_type(str(proposal.relation or ""))
        reason = str(proposal.reason or "").strip()
        case_id = str(proposal.case_id or "")
        if not content or not case_id:
            continue
        unit = exact.setdefault(
            content.casefold(),
            {"content": content, "source_case_ids": [], "activation_candidates": []},
        )
        if case_id not in unit["source_case_ids"]:
            unit["source_case_ids"].append(case_id)
        if parent and relation == "enhance" and reason:
            candidate = {
                "parent_node": parent,
                "relation": "enhance",
                "reason": reason,
                "case_id": case_id,
            }
            if candidate not in unit["activation_candidates"]:
                unit["activation_candidates"].append(candidate)
    units = list(exact.values())
    for index, unit in enumerate(units, start=1):
        unit["opinion_id"] = f"opinion_{index:04d}"
        unit["support"] = len(unit["source_case_ids"])
        unit["case_evidence"] = [
            dict(case_evidence[case_id])
            for case_id in unit["source_case_ids"]
            if case_id in case_evidence
        ]
    return units


def _node_units(
    items: list[tuple[str, str]],
    case_evidence: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Collapse exact repeats before the teacher, retaining all provenance."""
    case_evidence = case_evidence or {}
    exact: dict[str, dict[str, Any]] = {}
    for case_id, text in items:
        content = " ".join(str(text or "").split()).strip()
        if not content:
            continue
        key = content.casefold()
        unit = exact.setdefault(
            key, {"content": content, "source_case_ids": []}
        )
        if case_id not in unit["source_case_ids"]:
            unit["source_case_ids"].append(case_id)
    units = list(exact.values())
    for index, unit in enumerate(units, start=1):
        unit["opinion_id"] = f"opinion_{index:04d}"
        unit["support"] = len(unit["source_case_ids"])
        unit["case_evidence"] = [
            dict(case_evidence[case_id])
            for case_id in unit["source_case_ids"]
            if case_id in case_evidence
        ]
    return units


def _node_semantics(graph: SkillGraph | None) -> dict[str, list[str]]:
    if graph is None:
        return {}
    return {
        node_id: [
            part
            for part in (
                node.title,
                node.meaning,
                node.when_to_use,
                node.how_to_use,
                *node.avoid,
            )
            if part
        ]
        for node_id, node in graph.nodes.items()
    }


def _allocate_new_node_id(graph: SkillGraph | None, reserved: set[str]) -> str:
    existing = set(graph.nodes) if graph is not None else set()
    parsed = [
        match
        for node_id in existing
        if (match := re.fullmatch(r"([A-Za-z]+)(\d+)", node_id))
    ]
    prefix_counts: dict[str, int] = {}
    for match in parsed:
        prefix = match.group(1)
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    prefix = max(prefix_counts, key=lambda value: (prefix_counts[value], value)) if prefix_counts else "N"
    numbers = [int(match.group(2)) for match in parsed if match.group(1) == prefix]
    width = max([3, *(len(match.group(2)) for match in parsed if match.group(1) == prefix)])
    index = max(numbers, default=0) + 1
    while True:
        node_id = f"{prefix}{index:0{width}d}"
        if node_id not in existing and node_id not in reserved:
            reserved.add(node_id)
            return node_id
        index += 1


def _template_route_new_node_units(
    graph: SkillGraph | None,
    units: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]], str]]:
    """Route new-node opinions to an existing node or a genuinely new cluster."""
    semantics = _node_semantics(graph)
    routed: list[tuple[str, list[dict[str, Any]], str]] = []
    novel: list[dict[str, Any]] = []
    for unit in units:
        target = next(
            (
                node_id
                for node_id in sorted(semantics)
                if any(
                    _similar(str(unit["content"]), field)
                    for field in semantics[node_id]
                )
            ),
            "",
        )
        if target:
            routed.append((target, [unit], str(unit["content"])))
        else:
            novel.append(unit)

    if novel:
        by_id = {str(unit["opinion_id"]): unit for unit in novel}
        clusters = _cluster_text_items(
            [(str(unit["opinion_id"]), str(unit["content"])) for unit in novel]
        )
        for cluster in clusters:
            group = [by_id[opinion_id] for opinion_id, _ in cluster]
            routed.append(("NEW", group, _merge_texts([str(x["content"]) for x in group])))
    return routed


def _route_new_node_units(
    graph: SkillGraph | None,
    units: list[dict[str, Any]],
    *,
    chat_fn=None,
    mode: str = "template",
    store: ArtifactStore | None = None,
) -> list[tuple[str, list[dict[str, Any]], str]]:
    """Decide whether proposed new rules duplicate existing node semantics."""
    if mode != "teacher" or chat_fn is None or not units:
        return _template_route_new_node_units(graph, units)

    system = (PROMPTS / "new_node_route.md").read_text(encoding="utf-8")
    existing = [
        {
            "id": node_id,
            "title": node.title,
            "meaning": node.meaning,
            "when_to_use": node.when_to_use,
            "how_to_use": node.how_to_use,
            "avoid": list(node.avoid),
        }
        for node_id, node in sorted((graph.nodes if graph is not None else {}).items())
    ]
    payload = [
        {
            "opinion_id": unit["opinion_id"],
            "content": unit["content"],
            "evidence_count": unit["support"],
            "source_cases": list(unit.get("case_evidence") or []),
            "activation_candidates": list(unit.get("activation_candidates") or []),
        }
        for unit in units
    ]
    user = exact_reference_json({
        "existing_nodes": existing,
        "proposed_new_node_opinions": payload,
    })
    from graphopt.json_utils import extract_json

    by_id = {str(unit["opinion_id"]): unit for unit in units}
    expected_ids = set(by_id)
    valid_nodes = set(graph.nodes) if graph is not None else set()

    def parse_response(response: str) -> list[tuple[str, list[dict[str, Any]], str]]:
        obj = extract_json(response)
        if not isinstance(obj, dict) or set(obj) != {"clusters"}:
            raise ValueError("top-level JSON must contain exactly the key 'clusters'")
        raw_clusters = obj["clusters"]
        if not isinstance(raw_clusters, list) or not raw_clusters:
            raise ValueError("clusters must be a non-empty list")

        assigned: set[str] = set()
        routed: list[tuple[str, list[dict[str, Any]], str]] = []
        for index, raw in enumerate(raw_clusters):
            if not isinstance(raw, dict) or set(raw) != {
                "opinion_ids",
                "target_node",
            }:
                raise ValueError(
                    f"clusters[{index}] must contain exactly opinion_ids and "
                    "target_node"
                )
            opinion_ids = raw["opinion_ids"]
            if not isinstance(opinion_ids, list) or not opinion_ids:
                raise ValueError(f"clusters[{index}].opinion_ids must be non-empty")
            if any(not isinstance(opinion_id, str) for opinion_id in opinion_ids):
                raise ValueError("every opinion_id must be a string")
            if len(opinion_ids) != len(set(opinion_ids)):
                raise ValueError(f"clusters[{index}] repeats an opinion_id")
            unknown = set(opinion_ids) - expected_ids
            if unknown:
                raise ValueError(f"unknown opinion_ids: {sorted(unknown)}")
            repeated = set(opinion_ids) & assigned
            if repeated:
                raise ValueError(
                    f"opinion_ids assigned to multiple clusters: {sorted(repeated)}"
                )
            target = raw["target_node"]
            if not isinstance(target, str) or (
                target != "NEW" and target not in valid_nodes
            ):
                raise ValueError(f"invalid target_node: {target!r}")
            assigned.update(opinion_ids)
            group = [by_id[opinion_id] for opinion_id in opinion_ids]
            routed.append(
                (
                    target,
                    group,
                    _merge_texts([str(unit["content"]) for unit in group]),
                )
            )

        missing = expected_ids - assigned
        if missing:
            raise ValueError(f"missing opinion_ids: {sorted(missing)}")
        return routed

    current_user = user
    for attempt in range(1, 3):
        response = ""
        usage: Any = None
        try:
            response, usage = chat_fn(
                system=system,
                user=current_user,
                max_completion_tokens=8192,
                retries=3,
                stage="new_node_route",
                timeout=OPINION_MERGE_TIMEOUT_SECONDS,
            )
            routed = parse_response(response)
            if store is not None:
                save_llm_call(
                    store,
                    "all_new_nodes",
                    stage="new_node_route",
                    system=system,
                    user=current_user,
                    response=response,
                    usage=usage,
                    parsed=[
                        {
                            "target_node": target,
                            "opinion_ids": [unit["opinion_id"] for unit in group],
                            "merged_opinion": content,
                        }
                        for target, group, content in routed
                    ],
                )
            return routed
        except Exception as exc:
            error = str(exc)
            if store is not None:
                save_llm_call(
                    store,
                    "all_new_nodes",
                    stage="new_node_route",
                    system=system,
                    user=current_user,
                    response=response or None,
                    usage=usage,
                    error=f"attempt {attempt}/2: {error}",
                )
            if _is_retryable_opinion_merge_error(exc):
                if attempt == 2:
                    raise RuntimeError(
                        "New-node opinion routing timed out again; stopping experiment"
                    ) from exc
                continue
            if attempt == 1:
                current_user = (
                    user
                    + "\n\n## Format Correction\nYour previous answer was invalid: "
                    + error
                    + "\nReturn the complete strict JSON answer again. Include every "
                    + "input opinion_id exactly once; target only an existing node ID or "
                    + "NEW; return only opinion_ids and target_node for each cluster; "
                    + "include no extra keys or Markdown.\n\n## Invalid Previous Answer\n"
                    + str(response)
                )

    # Preserve valid case evidence when the teacher repeatedly violates the
    # routing schema. The deterministic router can only map to an existing
    # semantically similar node or a novel cluster; it never invents support.
    if store is not None:
        save_template_call(
            store,
            "all_new_nodes_fallback",
            stage="new_node_route_fallback",
            inputs={"opinions": units, "reason": "two invalid teacher responses"},
            outputs={"policy": "deterministic_semantic_fallback"},
        )
    return _template_route_new_node_units(graph, units)


def _merge_activation_edge_records(
    *groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union specialist activation edges without losing case provenance."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        for raw in group:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source") or "")
            relation = normalize_edge_type(str(raw.get("relation") or ""))
            reason = str(raw.get("reason") or "").strip()
            case_ids = sorted({
                str(value) for value in (raw.get("source_case_ids") or [])
                if str(value)
            })
            if not source or relation != "enhance" or not reason or not case_ids:
                continue
            bucket = merged.setdefault(
                (source, relation),
                {
                    "source": source, "relation": relation,
                    "reasons": [], "source_case_ids": [],
                },
            )
            if reason not in bucket["reasons"]:
                bucket["reasons"].append(reason)
            bucket["source_case_ids"] = sorted(
                set(bucket["source_case_ids"]) | set(case_ids)
            )
    return [
        {
            "source": item["source"],
            "relation": item["relation"],
            "reason": " ".join(item["reasons"])[:600],
            "source_case_ids": item["source_case_ids"],
        }
        for _, item in sorted(merged.items())
    ]


def _activation_edges_for_new_group(
    graph: SkillGraph | None, group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve every evidence-backed parent activation for one specialist."""
    valid_nodes = set(graph.nodes) if graph is not None else set()
    by_parent: dict[str, dict[str, Any]] = {}
    for unit in group:
        unit_case_ids = set(map(str, unit.get("source_case_ids") or []))
        for raw in unit.get("activation_candidates") or []:
            if not isinstance(raw, dict):
                continue
            parent = str(raw.get("parent_node") or "")
            case_id = str(raw.get("case_id") or "")
            reason = str(raw.get("reason") or "").strip()
            relation = normalize_edge_type(str(raw.get("relation") or ""))
            if (
                parent not in valid_nodes or case_id not in unit_case_ids
                or relation != "enhance" or not reason
            ):
                continue
            bucket = by_parent.setdefault(
                parent,
                {
                    "source": parent,
                    "relation": "enhance",
                    "reasons": [],
                    "source_case_ids": [],
                },
            )
            if reason not in bucket["reasons"]:
                bucket["reasons"].append(reason)
            if case_id not in bucket["source_case_ids"]:
                bucket["source_case_ids"].append(case_id)
    edges: list[dict[str, Any]] = []
    for parent in sorted(by_parent):
        bucket = by_parent[parent]
        edges.append({
            "source": parent,
            "relation": "enhance",
            "reason": " ".join(bucket.pop("reasons"))[:600],
            "source_case_ids": sorted(bucket["source_case_ids"]),
        })
    return edges


def _cluster_proposal(
    target_node: str,
    group: list[dict[str, Any]],
    *,
    kind: str,
    route: str = "TARGET",
    synthesis: str = "",
) -> MergedProposal | None:
    case_ids = list(dict.fromkeys(
        str(case_id)
        for unit in group
        for case_id in unit["source_case_ids"]
    ))
    if not case_ids:
        return None
    content = str(synthesis or "").strip() or _merge_texts(
        [str(unit["content"]) for unit in group]
    )
    evidence_by_case = {
        str(item.get("case_id") or f"anonymous:{index}"): dict(item)
        for index, item in enumerate(
            evidence
            for unit in group
            for evidence in (unit.get("case_evidence") or [])
            if isinstance(evidence, dict)
        )
    }
    if route == "NEW_SPECIALIST":
        return MergedProposal(
            kind="new_node",
            content=content,
            parent_node=target_node,
            relation="enhance",
            rationale=f"semantic_specialist_for:{target_node}",
            support=len(case_ids),
            source_case_ids=case_ids,
            evidence_items=list(evidence_by_case.values()),
        )
    return MergedProposal(
        kind=kind,
        target_node=target_node,
        content=content,
        support=len(case_ids),
        source_case_ids=case_ids,
        evidence_items=list(evidence_by_case.values()),
    )


def _template_merge_node(
    target_node: str,
    units: list[dict[str, Any]],
    *,
    kind: str = "node_revision",
) -> list[MergedProposal]:
    """Deterministically preserve one target meaning and split distinct ones."""
    if not units:
        return []
    by_id = {str(unit["opinion_id"]): unit for unit in units}
    raw_clusters = _cluster_text_items([
        (str(unit["opinion_id"]), str(unit["content"])) for unit in units
    ])
    groups = [[by_id[opinion_id] for opinion_id, _ in cluster] for cluster in raw_clusters]
    groups.sort(key=lambda group: (-sum(int(unit["support"]) for unit in group), str(group[0]["opinion_id"])))
    output: list[MergedProposal] = []
    for index, group in enumerate(groups):
        proposal = _cluster_proposal(
            target_node,
            group,
            kind=kind,
            route="TARGET" if index == 0 else "NEW_SPECIALIST",
        )
        if proposal is not None:
            output.append(proposal)
    return output


def _teacher_merge_node(
    graph: SkillGraph | None,
    target_node: str,
    units: list[dict[str, Any]],
    *,
    chat_fn,
    store: ArtifactStore | None = None,
    kind: str = "node_revision",
) -> list[MergedProposal]:
    """Synthesize all case-complete opinions and route distinct meanings."""
    if len(units) <= 1:
        proposal = _cluster_proposal(
            target_node, units, kind=kind, route="TARGET"
        ) if units else None
        return [proposal] if proposal is not None else []
    system = (PROMPTS / "evolution_merge.md").read_text(encoding="utf-8")
    node = graph.nodes.get(target_node) if graph is not None else None
    node_context = node.to_dict() if node is not None else {"id": target_node}
    payload = [
        {
            "opinion_id": unit["opinion_id"],
            "content": unit["content"],
            "evidence_count": unit["support"],
            "source_cases": list(unit.get("case_evidence") or []),
        }
        for unit in units
    ]
    user = exact_reference_json({
        "target_node": node_context,
        "opinion_kind": kind,
        "opinions_with_complete_source_cases": payload,
    })
    from graphopt.json_utils import extract_json

    by_id = {str(unit["opinion_id"]): unit for unit in units}
    expected_ids = set(by_id)

    def parse_response(response: str) -> list[MergedProposal]:
        obj = extract_json(response)
        if not isinstance(obj, dict) or set(obj) != {"clusters"}:
            raise ValueError("top-level JSON must contain exactly clusters")
        raw_clusters = obj["clusters"]
        if not isinstance(raw_clusters, list) or not raw_clusters:
            raise ValueError("clusters must be a non-empty list")
        seen: set[str] = set()
        target_routes = 0
        parsed: list[MergedProposal] = []
        for index, raw in enumerate(raw_clusters):
            if not isinstance(raw, dict) or set(raw) != {
                "opinion_ids", "route", "synthesis"
            }:
                raise ValueError(
                    f"clusters[{index}] requires opinion_ids, route, synthesis"
                )
            ids = raw["opinion_ids"]
            if (
                not isinstance(ids, list) or not ids
                or any(not isinstance(value, str) for value in ids)
                or len(ids) != len(set(ids))
            ):
                raise ValueError("opinion_ids must be a non-empty unique string list")
            if set(ids) - expected_ids or set(ids) & seen:
                raise ValueError("cluster has unknown or repeated opinion_ids")
            route = str(raw["route"] or "")
            if route not in {"TARGET", "NEW_SPECIALIST"}:
                raise ValueError("route must be TARGET or NEW_SPECIALIST")
            if route == "TARGET":
                target_routes += 1
                if target_routes > 1:
                    raise ValueError("at most one semantic cluster may update TARGET")
            synthesis = raw["synthesis"]
            if not isinstance(synthesis, str) or not synthesis.strip():
                raise ValueError("synthesis must be a non-empty grounded string")
            if len(synthesis) > 3000:
                raise ValueError("synthesis exceeds 3000 characters")
            seen.update(ids)
            proposal = _cluster_proposal(
                target_node,
                [by_id[opinion_id] for opinion_id in ids],
                kind=kind,
                route=route,
                synthesis=synthesis,
            )
            if proposal is not None:
                parsed.append(proposal)
        if seen != expected_ids:
            raise ValueError("every opinion_id must appear exactly once")
        return parsed

    current_user = user
    stage = "retrieval_proposal_merge" if kind == "retrieval_revision" else "node_proposal_merge"
    for attempt in range(1, 3):
        response = ""
        usage: Any = None
        try:
            response, usage = chat_fn(
                system=system,
                user=current_user,
                max_completion_tokens=8192,
                retries=3,
                stage=stage,
                timeout=OPINION_MERGE_TIMEOUT_SECONDS,
            )
            proposals = parse_response(response)
            if store is not None:
                save_llm_call(
                    store, target_node, stage=stage, system=system,
                    user=current_user, response=response, usage=usage,
                    parsed=[proposal.to_dict() for proposal in proposals],
                )
            return proposals
        except Exception as exc:
            if store is not None:
                save_llm_call(
                    store, target_node, stage=stage, system=system,
                    user=current_user, response=response or None, usage=usage,
                    error=f"attempt {attempt}/2: {exc}",
                )
            if _is_retryable_opinion_merge_error(exc):
                raise OpinionMergeInfrastructureError(
                    stage=stage, target=target_node, error=exc
                ) from exc
            current_user = (
                user
                + "\n\n## Format Correction\nThe previous response was invalid: "
                + str(exc)
                + "\nReturn strict JSON. Include every opinion_id exactly once. "
                  "Use at most one TARGET cluster; route every distinct reusable "
                  "meaning to NEW_SPECIALIST; provide a grounded synthesis for each. "
                  "No Markdown or extra keys.\n\n## Invalid Previous Answer\n"
                + str(response)
            )
    fallback = _template_merge_node(target_node, units, kind=kind)
    if store is not None:
        save_template_call(
            store, target_node, stage=f"{stage}_fallback",
            inputs={"opinions": units, "reason": "two invalid teacher responses"},
            outputs={"clusters": [item.to_dict() for item in fallback]},
        )
    return fallback



def _rank_and_cap_node_revisions(
    proposals: list[MergedProposal],
    top_k: int,
) -> list[MergedProposal]:
    """Keep the most repeated semantic opinions for each existing node."""
    by_node: dict[str, list[MergedProposal]] = defaultdict(list)
    for proposal in proposals:
        by_node[proposal.target_node].append(proposal)
    kept: list[MergedProposal] = []
    for target_node in sorted(by_node):
        ranked = sorted(
            by_node[target_node],
            key=lambda item: (-item.support, item.content.casefold()),
        )
        kept.extend(ranked[: max(1, int(top_k))])
    return kept


def _present_case_fields(record: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Copy available case fields without silently dropping false/zero values."""
    return {
        key: record[key]
        for key in keys
        if key in record and record[key] is not None and record[key] != ""
    }


def _complete_case_evidence(
    analyses: list[CaseAnalysis],
    case_results: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Build the lossless case packets used by every synthesis stage.

    Reasoning is evidence, not the whole case: each packet also retains the
    original task, student answer, evaluator outcome, training-only reference,
    validated graph usage, and every graph change proposed for that case.
    """
    result_by_id = {
        str(result.get("id") or result.get("case_id") or ""): result
        for result in (case_results or [])
    }
    packets: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        if analysis.success:
            continue
        case_id = str(analysis.case_id)
        result = dict(result_by_id.get(case_id) or {})
        analysis_dict = analysis.to_dict()
        graph_usage = result.get("graph_refs")
        if not isinstance(graph_usage, dict):
            graph_usage = {
                "used_nodes": list(analysis.used_nodes),
                "used_edges": list(analysis.used_edges),
                "status": "analysis_fallback",
            }
        packets[case_id] = {
            "case_id": case_id,
            "environment": str(result.get("environment") or ""),
            "task_type": str(result.get("task_type") or ""),
            "original_task": _present_case_fields(
                result,
                ("task_description", "question", "instruction", "input"),
            ),
            "student_output": _present_case_fields(
                result,
                (
                    "response", "predicted_answer", "predicted_label",
                    "predicted_text", "output", "answer",
                ),
            ),
            "evaluation": _present_case_fields(
                result,
                ("hard", "soft", "fail_reason", "reward", "score"),
            ),
            "training_only_reference": _present_case_fields(
                result,
                (
                    "training_reference_plan", "training_reference",
                    "reference_answer", "gold_answer", "gold", "label",
                ),
            ),
            "reasoning_evidence": {
                **_present_case_fields(
                    result,
                    (
                        "semantic_reasoning_trace",
                        "semantic_reasoning_trace_status",
                        "semantic_reasoning_trace_required",
                    ),
                ),
                "trace_attribution": analysis_dict["trace_attribution"],
            },
            "graph_usage": dict(graph_usage),
            "existing_graph_usage": analysis_dict["existing_graph_usage"],
            "failure_type": str(analysis.failure_type),
            "badcase_summary": dict(analysis.badcase_summary),
            "root_cause_code": str(analysis.root_cause_code),
            "attributed_nodes": list(analysis.attributed_nodes),
            "attributed_edges": list(analysis.attributed_edges),
            "trace_evidence": dict(analysis.trace_evidence),
            "proposed_edges": (
                list(analysis_dict["edge_correction_proposals"])
                + list(analysis_dict["new_edge_proposals"])
            ),
            "proposed_graph_changes": {
                key: analysis_dict[key]
                for key in (
                    "node_revision_proposals", "retrieval_revision_proposals",
                    "edge_correction_proposals", "new_node_proposals",
                    "new_edge_proposals",
                )
            },
        }
    return packets



def semantic_merge_proposals(
    analyses: list[CaseAnalysis],
    graph: SkillGraph | None = None,
    *,
    case_results: list[dict[str, Any]] | None = None,
    case_evidence_analyses: list[CaseAnalysis] | None = None,
    chat_fn=None,
    mode: str = "template",
    node_top_k: int = 4,
    store: ArtifactStore | None = None,
    failed_node_merges: set[str] | None = None,
) -> dict[str, list[MergedProposal]]:
    case_evidence = _complete_case_evidence(
        case_evidence_analyses or analyses, case_results
    )
    node_rev_raw: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for a in analyses:
        for p in a.node_revision_proposals:
            operation = str(p.operation or "PATCH").upper()
            toxic_text = p.toxic_text if operation == "REWRITE" else ""
            node_rev_raw[(p.target_node, operation, toxic_text)].append(
                (a.case_id, p.proposal)
            )

    retrieval_rev_raw: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for a in analyses:
        for p in a.retrieval_revision_proposals:
            retrieval_rev_raw[p.target_node].append(
                (a.case_id, p.proposed_when_to_use)
            )

    new_nodes_raw: list[Any] = []
    for a in analyses:
        for p in a.new_node_proposals:
            new_nodes_raw.append(p)

    edges_raw: dict[tuple[str, str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for a in analyses:
        for p in a.edge_correction_proposals:
            rel = normalize_edge_type(p.relation)
            key = (p.target_edge, p.source, p.target, rel)
            edges_raw[key].append((a.case_id, p.reason))
        for p in a.new_edge_proposals:
            rel = normalize_edge_type(p.relation)
            key = (p.target_old_edge, p.source, p.target, rel)
            edges_raw[key].append((a.case_id, p.reason))

    use_teacher = mode == "teacher" and chat_fn is not None
    merged_new_nodes: list[MergedProposal] = []
    if new_nodes_raw:
        routed = _route_new_node_units(
            graph,
            _new_node_units(new_nodes_raw, case_evidence),
            chat_fn=chat_fn,
            mode=mode,
            store=store,
        )
        for target, group, merged_content in routed:
            case_ids = list(
                dict.fromkeys(
                    case_id
                    for unit in group
                    for case_id in unit["source_case_ids"]
                )
            )
            if target == "NEW":
                activation_edges = _activation_edges_for_new_group(graph, group)
                if not activation_edges:
                    continue
                merged_new_nodes.append(
                    MergedProposal(
                        kind="new_node",
                        content=merged_content,
                        parent_node=str(activation_edges[0]["source"]),
                        relation="enhance",
                        rationale="evidence_grounded_specialist_activation",
                        activation_edges=activation_edges,
                        support=len(case_ids),
                        source_case_ids=case_ids,
                        evidence_items=[
                            dict(item)
                            for unit in group
                            for item in (unit.get("case_evidence") or [])
                            if isinstance(item, dict)
                        ],
                    )
                )
            else:
                # Convert duplicate new-node proposals directly into original
                # opinions on the matched node. Semantic merging happens once,
                # in the node-specific merge call below.
                for unit in group:
                    for case_id in unit["source_case_ids"]:
                        node_rev_raw[(target, "PATCH", "")].append(
                            (case_id, str(unit["content"]))
                        )

    merged_node_revisions: list[MergedProposal] = []
    merged_retrieval_revisions: list[MergedProposal] = []
    deferred_merges: list[dict[str, Any]] = []

    def accept_node_merge(
        proposals: list[MergedProposal], operation: str, toxic_text: str,
    ) -> None:
        for proposal in proposals:
            if proposal.kind == "new_node":
                merged_new_nodes.append(proposal)
                continue
            proposal.operation = operation
            proposal.toxic_text = toxic_text
            merged_node_revisions.append(proposal)

    def accept_retrieval_merge(proposals: list[MergedProposal]) -> None:
        for proposal in proposals:
            if proposal.kind == "new_node":
                merged_new_nodes.append(proposal)
            else:
                merged_retrieval_revisions.append(proposal)

    for target, operation, toxic_text in sorted(node_rev_raw):
        units = _node_units(node_rev_raw[(target, operation, toxic_text)], case_evidence)
        if use_teacher:
            try:
                node_merged = _teacher_merge_node(
                    graph,
                    target,
                    units,
                    chat_fn=chat_fn,
                    store=store,
                )
            except OpinionMergeInfrastructureError as exc:
                deferred_merges.append({
                    "kind": "node_revision",
                    "target": target,
                    "operation": operation,
                    "toxic_text": toxic_text,
                    "units": units,
                    "first_error": str(exc),
                })
                continue
            if not node_merged and failed_node_merges is not None:
                failed_node_merges.add(target)
        else:
            node_merged = _template_merge_node(target, units)
            if store is not None:
                save_template_call(
                    store,
                    target,
                    stage="node_proposal_merge",
                    inputs={"target_node": target, "opinions": units},
                    outputs={"clusters": [item.to_dict() for item in node_merged]},
                )
        accept_node_merge(node_merged, operation, toxic_text)

    for target in sorted(retrieval_rev_raw):
        units = _node_units(retrieval_rev_raw[target], case_evidence)
        if use_teacher:
            try:
                retrieval_merged = _teacher_merge_node(
                    graph,
                    target,
                    units,
                    chat_fn=chat_fn,
                    store=store,
                    kind="retrieval_revision",
                )
            except OpinionMergeInfrastructureError as exc:
                deferred_merges.append({
                    "kind": "retrieval_revision",
                    "target": target,
                    "operation": "",
                    "toxic_text": "",
                    "units": units,
                    "first_error": str(exc),
                })
                continue
            if not retrieval_merged and failed_node_merges is not None:
                failed_node_merges.add(target)
        else:
            retrieval_merged = _template_merge_node(
                target,
                units,
                kind="retrieval_revision",
            )
            if store is not None:
                save_template_call(
                    store,
                    target,
                    stage="retrieval_proposal_merge",
                    inputs={"target_node": target, "opinions": units},
                    outputs={
                        "clusters": [item.to_dict() for item in retrieval_merged]
                    },
                )
        accept_retrieval_merge(retrieval_merged)

    # Finish every independent first-pass merge before replaying only the
    # infrastructure failures. A repeated timeout/transport failure is fatal;
    # it must never be silently converted into a template merge decision.
    for deferred in deferred_merges:
        target = str(deferred["target"])
        kind = str(deferred["kind"])
        print(
            f"[graphopt opinion merge] deferred retry target={target} kind={kind}",
            flush=True,
        )
        try:
            proposals = _teacher_merge_node(
                graph,
                target,
                list(deferred["units"]),
                chat_fn=chat_fn,
                store=store,
                kind=kind,
            )
        except OpinionMergeInfrastructureError as exc:
            raise RuntimeError(
                "Opinion merge infrastructure failure repeated after the "
                f"deferred retry; stopping experiment: {exc}"
            ) from exc
        if not proposals and failed_node_merges is not None:
            failed_node_merges.add(target)
        if kind == "retrieval_revision":
            accept_retrieval_merge(proposals)
        else:
            accept_node_merge(
                proposals,
                str(deferred["operation"]),
                str(deferred["toxic_text"]),
            )

    merged_node_revisions = _rank_and_cap_node_revisions(
        merged_node_revisions,
        node_top_k,
    )
    merged_retrieval_revisions = _rank_and_cap_node_revisions(
        merged_retrieval_revisions,
        node_top_k,
    )

    merged_edges: list[MergedProposal] = []
    for (old_edge, source, target, relation), evidence in sorted(edges_raw.items()):
        case_ids = list(dict.fromkeys(case_id for case_id, _ in evidence))
        reasons = list(dict.fromkeys(reason.strip() for _, reason in evidence if reason.strip()))
        rationale = " ".join(reasons[:3])[:600]
        merged_edges.append(
            MergedProposal(
                kind="edge",
                source=source,
                target=target,
                relation=relation,
                target_old_edge=old_edge,
                rationale=rationale,
                content=f"{source}->{target} {relation}",
                support=len(case_ids),
                source_case_ids=case_ids,
            )
        )

    merged_groups = {
        "node_revisions": merged_node_revisions,
        "retrieval_revisions": merged_retrieval_revisions,
        "new_nodes": merged_new_nodes,
        "edges": merged_edges,
    }
    for proposals in merged_groups.values():
        for proposal in proposals:
            proposal.evidence_items = [
                dict(case_evidence[case_id])
                for case_id in proposal.source_case_ids
                if case_id in case_evidence
            ]
    return merged_groups
