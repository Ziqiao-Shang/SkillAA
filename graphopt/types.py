"""Types for the trainable rule graph and its discrete edits.

Optimizer-side experience lives in ``ExperienceLedger``, not in this document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

EdgeType = Literal["prereq", "enhance", "co_occur", "prerequisite", "enhances"]

GraphEditOp = Literal[
    "add_node",
    "delete_node",
    "update_node",
    "add_edge",
    "delete_edge",
    "change_edge_type",
]

OPTIMIZER_EDGE_TYPES = frozenset({"prereq", "enhance", "prerequisite", "enhances"})

WEIGHT_TO_FLOAT = {
    "medium": 0.55,
    "strong": 0.85,
    # Read-only aliases for normalizing pre-binary graph artifacts.
    "weak": 0.55,
    "rather_weak": 0.55,
    "rather_strong": 0.85,
}
FLOAT_TO_WEIGHT = [
    (0.70, "strong"),
    (float("-inf"), "medium"),
]

# Legacy gradient-aggregate edge tiers; the formal Evolution path is correction-driven.
EVOLUTION_EDGE_W = (1.0, 0.8, 0.6, 0.4, 0.2)


# Concrete training-case identifiers belong in structured provenance fields such
# as ``source_case_ids``. They must not leak into model-visible graph prose.
_CASE_IDENTIFIER = r"(?:[0-9]{5,}(?::[0-9]+)?|[0-9]{4,}:[0-9]+|[A-Fa-f0-9]{16,})"
_CASE_IDENTIFIER_LIST_RE = re.compile(
    rf"\bcases\s+{_CASE_IDENTIFIER}"
    rf"(?:(?:\s*,\s*(?:and\s+)?|\s+(?:and|or)\s+){_CASE_IDENTIFIER})*",
    re.IGNORECASE,
)
_SINGLE_CASE_IDENTIFIER_RE = re.compile(
    rf"\bcase(?:\s+|-){_CASE_IDENTIFIER}",
    re.IGNORECASE,
)


def sanitize_model_visible_case_references(value: Any) -> str:
    """Replace concrete case IDs in graph prose with semantic provenance text."""

    text = str(value or "")

    def _replace_list(match: re.Match[str]) -> str:
        return (
            "The supporting failures"
            if match.group(0)[:1].isupper()
            else "the supporting failures"
        )

    text = _CASE_IDENTIFIER_LIST_RE.sub(_replace_list, text)
    text = _SINGLE_CASE_IDENTIFIER_RE.sub("the supporting failure", text)
    text = re.sub(
        r"\b(?:the\s+)?failure\s+(?:in|from)\s+the supporting failure\b",
        "the documented failure",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bthe supporting failure exception\b",
        "the documented exception",
        text,
        flags=re.IGNORECASE,
    )
    return text


def normalize_edge_type(typ: str) -> str:
    if typ in ("prerequisite", "prereq"):
        return "prereq"
    if typ in ("enhances", "enhance"):
        return "enhance"
    if typ == "co_occur":
        return "co_occur"
    return typ


def weight_to_float(weight: Any) -> float:
    if isinstance(weight, (int, float)):
        return float(weight)
    return float(WEIGHT_TO_FLOAT.get(str(weight), 0.6))


def float_to_weight(value: float) -> str:
    for thr, name in FLOAT_TO_WEIGHT:
        if value >= thr:
            return name
    return "weak"


@dataclass
class SkillNode:
    id: str
    title: str
    meaning: str = ""
    when_to_use: str = ""
    how_to_use: str = ""
    avoid: list[str] = field(default_factory=list)
    category: str = "general"
    section: str = "general_principles"
    node_type: str = "rule"
    level: int = 0
    n_use: int = 0
    n_succ: int = 0
    stat_used: int = 0
    stat_correct: int = 0
    stat_wrong: int = 0
    active: bool = True
    frozen: bool = False
    graph_name: str = "stable_rule_graph"

    @property
    def name(self) -> str:
        return self.title

    @name.setter
    def name(self, value: str) -> None:
        self.title = value

    @property
    def description(self) -> str:
        when = self.when_to_use or "Always applicable."
        do = self.how_to_use
        return f"When: {when}\nDo: {do}"

    @description.setter
    def description(self, value: str) -> None:
        text = (value or "").strip()
        if text.startswith("When:") and "\nDo:" in text:
            when, do = text.split("\nDo:", 1)
            self.when_to_use = when.replace("When:", "", 1).strip()
            self.how_to_use = do.strip()
        else:
            self.how_to_use = text

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, graph_name: str = "stable_rule_graph") -> "SkillNode":
        stats = d.get("stats") or {}
        if "description" in d and "title" not in d:
            node = cls(
                id=str(d["id"]),
                title=str(d.get("name") or d["id"]),
                category=str(d.get("category") or "general"),
                level=int(d.get("level") or 0),
                n_use=int(d.get("n_use") or 0),
                n_succ=int(d.get("n_succ") or 0),
                active=bool(d.get("active", True)),
                frozen=bool(d.get("frozen", False)),
                graph_name=graph_name,
            )
            node.description = str(d.get("description") or "")
            return node
        return cls(
            id=str(d["id"]),
            title=str(d.get("title") or d.get("name") or d["id"]),
            meaning="",
            when_to_use=str(d.get("when_to_use") or ""),
            how_to_use=str(d.get("how_to_use") or ""),
            avoid=[str(x) for x in (d.get("avoid") or [])],
            category=str(d.get("category") or "general"),
            section=str(d.get("section") or "general_principles"),
            node_type=str(d.get("node_type") or "rule"),
            level=int(d.get("level") or 0),
            n_use=int(stats.get("n_use", d.get("n_use") or 0)),
            n_succ=int(stats.get("n_succ", d.get("n_succ") or 0)),
            stat_used=int(stats.get("used", d.get("stat_used") or 0)),
            stat_correct=int(stats.get("correct", d.get("stat_correct") or 0)),
            stat_wrong=int(stats.get("wrong", d.get("stat_wrong") or 0)),
            active=bool(d.get("active", True)),
            frozen=bool(d.get("frozen", False)),
            graph_name=graph_name,
        )

    def to_dict(self) -> dict[str, Any]:
        p = (self.n_succ / self.n_use) if self.n_use else 0.0
        data = {
            "id": self.id,
            "title": sanitize_model_visible_case_references(self.title),
            "section": self.section,
            "category": self.category,
            "node_type": self.node_type,
            "level": self.level,
            "when_to_use": sanitize_model_visible_case_references(self.when_to_use),
            "how_to_use": sanitize_model_visible_case_references(self.how_to_use),
            "avoid": [
                sanitize_model_visible_case_references(value) for value in self.avoid
            ],
            "stats": {
                "n_use": self.n_use,
                "n_succ": self.n_succ,
                "p_hat": round(p, 4) if self.n_use else None,
            },
            "active": self.active,
            "frozen": self.frozen,
        }
        return data


@dataclass
class Edge:
    src: str
    dst: str
    type: str = "prereq"
    w: float = 0.6
    id: str = ""
    weight: str = "medium"
    rationale: str = ""
    stat_used: int = 0
    stat_correct: int = 0
    stat_wrong: int = 0
    active: bool = True
    frozen: bool = False
    graph_name: str = "stable_rule_graph"

    def __post_init__(self) -> None:
        self.rationale = sanitize_model_visible_case_references(self.rationale)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, graph_name: str = "stable_rule_graph") -> "Edge":
        src = str(d.get("source") or d.get("src") or "")
        dst = str(d.get("target") or d.get("dst") or "")
        typ = normalize_edge_type(str(d.get("type") or "prereq"))
        if "weight" in d and not isinstance(d.get("w"), (int, float)):
            w = weight_to_float(d.get("weight"))
            weight = float_to_weight(w)
        else:
            w = float(d["w"]) if d.get("w") is not None else weight_to_float(d.get("weight", "medium"))
            weight = float_to_weight(w)
        stats = d.get("stats") or {}
        return cls(
            src=src,
            dst=dst,
            type=typ,
            w=w,
            id=str(d.get("id") or ""),
            weight=weight,
            rationale=str(d.get("rationale") or ""),
            stat_used=int(stats.get("used", d.get("stat_used") or 0)),
            stat_correct=int(stats.get("correct", d.get("stat_correct") or 0)),
            stat_wrong=int(stats.get("wrong", d.get("stat_wrong") or 0)),
            active=bool(d.get("active", True)),
            frozen=bool(d.get("frozen", False)),
            graph_name=graph_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.src,
            "target": self.dst,
            "type": normalize_edge_type(self.type),
            "weight": self.weight or float_to_weight(self.w),
            "rationale": sanitize_model_visible_case_references(self.rationale),
            "active": self.active,
            "frozen": self.frozen,
        }


@dataclass
class SkillGraph:
    """Trainable SkillGraph document (rule graph only; no experience_graph)."""

    data: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, SkillNode] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    active_max_level: int = 10**9
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SkillGraph":
        d = dict(d)
        # Drop legacy experience subgraph if present in older checkpoints.
        d.pop("experience_graph", None)

        if "stable_rule_graph" in d:
            nodes: dict[str, SkillNode] = {}
            edges: list[Edge] = []
            for n in d.get("stable_rule_graph", {}).get("nodes") or []:
                node = SkillNode.from_dict(n, graph_name="stable_rule_graph")
                nodes[node.id] = node
            for e in d.get("stable_rule_graph", {}).get("edges") or []:
                edges.append(Edge.from_dict(e, graph_name="stable_rule_graph"))
            return cls(
                data=d,
                nodes=nodes,
                edges=edges,
                active_max_level=int(d.get("active_max_level") or 10**9),
                meta=dict(d.get("meta") or {}),
            )

        raw = d.get("nodes") or []
        if isinstance(raw, dict):
            nodes = {k: SkillNode.from_dict({**v, "id": k}) for k, v in raw.items()}
        else:
            nodes = {n["id"]: SkillNode.from_dict(n) for n in raw}
        edges = [Edge.from_dict(e) for e in (d.get("edges") or [])]
        return cls(
            data=d,
            nodes=nodes,
            edges=edges,
            active_max_level=int(d["active_max_level"] if d.get("active_max_level") is not None else 10**9),
            meta=dict(d.get("meta") or {}),
        )

    def sync_data_from_views(self) -> None:
        """Write node/edge views back into the rule graph in ``data``."""
        self.data.pop("experience_graph", None)
        if "stable_rule_graph" not in self.data:
            self.data = {
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges],
                "active_max_level": self.active_max_level,
                "meta": self.meta,
            }
            return

        graph = self.data.setdefault("stable_rule_graph", {"nodes": [], "edges": []})
        graph["frozen"] = False
        graph_nodes = [n for n in self.nodes.values() if n.graph_name == "stable_rule_graph"]
        # Nodes without graph_name default to rule graph.
        if not graph_nodes:
            graph_nodes = list(self.nodes.values())
        old_ids = [n.get("id") for n in graph.get("nodes") or []]
        order = {nid: i for i, nid in enumerate(old_ids)}
        graph_nodes.sort(key=lambda n: (order.get(n.id, 10**6), n.id))
        for n in graph_nodes:
            n.frozen = False
            n.graph_name = "stable_rule_graph"
        graph["nodes"] = [n.to_dict() for n in graph_nodes]

        graph_edges = [e for e in self.edges if e.graph_name == "stable_rule_graph"]
        if not graph_edges and self.edges:
            graph_edges = list(self.edges)
        old_eids = [e.get("id") for e in graph.get("edges") or []]
        eorder = {eid: i for i, eid in enumerate(old_eids)}
        graph_edges.sort(key=lambda e: (eorder.get(e.id, 10**6), e.id or f"{e.src}->{e.dst}"))
        used: set[str] = set()
        for i, e in enumerate(graph_edges, 1):
            e.frozen = False
            e.graph_name = "stable_rule_graph"
            edge_id = re.sub(r"^SE(?=\d+\Z)", "E", str(e.id or ""))
            if not edge_id or edge_id in used:
                next_index = i
                edge_id = f"E{next_index:03d}"
                while edge_id in used:
                    next_index += 1
                    edge_id = f"E{next_index:03d}"
            e.id = edge_id
            used.add(e.id)
        graph["edges"] = [e.to_dict() for e in graph_edges]

    def to_dict(self) -> dict[str, Any]:
        self.sync_data_from_views()
        return self.data

    def copy(self) -> "SkillGraph":
        import copy

        return SkillGraph.from_dict(copy.deepcopy(self.to_dict()))


@dataclass
class GraphEdit:
    """One discrete action in the fixed Graph Edit Space."""

    op: GraphEditOp
    node_id: str = ""
    name: str = ""
    description: str = ""
    # Empty means "field not supplied" for update_node. add_node falls back
    # to the general category at the apply boundary.
    category: str = ""
    merge_ids: list[str] = field(default_factory=list)
    src: str = ""
    dst: str = ""
    edge_type: str = "prereq"
    new_edge_type: str = ""
    w: float = 0.6
    source_type: str = ""
    reasoning: str = ""
    rationale: str = ""
    meaning: str = ""
    when_to_use: str = ""
    how_to_use: str = ""
    # None means "leave unchanged"; [] means "explicitly clear the list".
    avoid: list[str] | None = None
    experience_note_id: str = ""
    group_id: str = ""
    # Optimizer-only provenance used by the bounded execution-detail probe.
    # These fields are serialized in patch artifacts but never enter the
    # stable graph node schema.
    edit_kind: str = ""
    parent_node: str = ""
    source_case_ids: list[str] = field(default_factory=list)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sanitize_model_visible_fields()

    def sanitize_model_visible_fields(self) -> None:
        # Structured provenance and raw evidence deliberately remain untouched.
        for field_name in (
            "name",
            "description",
            "rationale",
            "when_to_use",
            "how_to_use",
        ):
            setattr(
                self,
                field_name,
                sanitize_model_visible_case_references(getattr(self, field_name)),
            )
        if self.avoid is not None:
            self.avoid = [
                sanitize_model_visible_case_references(value) for value in self.avoid
            ]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphEdit":
        merge_ids = list(d.get("merge_ids") or [])
        return cls(
            op=d.get("op") or "add_node",  # type: ignore[arg-type]
            node_id=str(d.get("node_id") or d.get("id") or ""),
            name=str(d.get("name") or d.get("title") or ""),
            description=str(d.get("description") or ""),
            category=str(d.get("category") or ""),
            merge_ids=[str(x) for x in merge_ids],
            src=str(d.get("src") or d.get("source") or ""),
            dst=str(d.get("dst") or d.get("target") or ""),
            edge_type=normalize_edge_type(str(d.get("edge_type") or d.get("type") or "prereq")),
            new_edge_type=normalize_edge_type(str(d.get("new_edge_type") or "")),
            w=float(d["w"]) if d.get("w") is not None else weight_to_float(d.get("weight", "medium")),
            source_type=str(d.get("source_type") or ""),
            reasoning=str(d.get("reasoning") or ""),
            rationale=str(d.get("rationale") or ""),
            meaning="",
            when_to_use=str(d.get("when_to_use") or ""),
            how_to_use=str(d.get("how_to_use") or ""),
            avoid=(
                [str(x) for x in (d.get("avoid") or [])]
                if "avoid" in d
                else None
            ),
            experience_note_id=str(d.get("experience_note_id") or ""),
            group_id=str(d.get("group_id") or ""),
            edit_kind=str(d.get("edit_kind") or ""),
            parent_node=str(d.get("parent_node") or ""),
            source_case_ids=[str(x) for x in (d.get("source_case_ids") or [])],
            evidence_items=[
                dict(item) for item in (d.get("evidence_items") or [])
                if isinstance(item, dict)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        for k in (
            "node_id",
            "name",
            "description",
            "category",
            "src",
            "dst",
            "edge_type",
            "new_edge_type",
            "source_type",
            "reasoning",
            "rationale",
            "when_to_use",
            "how_to_use",
            "experience_note_id",
            "group_id",
            "edit_kind",
            "parent_node",
        ):
            v = getattr(self, k)
            if k in {"name", "description", "rationale", "when_to_use", "how_to_use"}:
                v = sanitize_model_visible_case_references(v)
            if v:
                d[k] = v
        if self.merge_ids:
            d["merge_ids"] = list(self.merge_ids)
        if self.avoid is not None:
            d["avoid"] = [
                sanitize_model_visible_case_references(value) for value in self.avoid
            ]
        if self.source_case_ids:
            d["source_case_ids"] = list(self.source_case_ids)
        if self.evidence_items:
            d["evidence_items"] = [dict(item) for item in self.evidence_items]
        if self.w != 0.6:
            d["w"] = self.w
        return d

    def summary(self) -> str:
        if self.op == "add_node":
            return f"ADD_NODE {self.node_id or self.name}: {(self.how_to_use or self.description)[:72]}"
        if self.op == "delete_node":
            return f"DELETE_NODE {self.node_id}"
        if self.op == "update_node":
            return f"UPDATE_NODE {self.node_id}"
        if self.op == "add_edge":
            return f"ADD_EDGE {self.src}-[{self.edge_type}]->{self.dst}"
        if self.op == "delete_edge":
            return f"DELETE_EDGE {self.src}-[{self.edge_type}]->{self.dst}"
        if self.op == "change_edge_type":
            return f"CHANGE_EDGE {self.src}->{self.dst}: {self.edge_type}->{self.new_edge_type}"
        return self.op

    @property
    def is_node_op(self) -> bool:
        return self.op in {"add_node", "delete_node", "update_node"}

    @property
    def is_edge_op(self) -> bool:
        return self.op in {"add_edge", "delete_edge", "change_edge_type"}


@dataclass
class GraphPatch:
    edits: list[GraphEdit] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphPatch":
        edits = d.get("edits") or d.get("graph_edits") or []
        return cls(
            edits=[GraphEdit.from_dict(e) if isinstance(e, dict) else e for e in edits],
            reasoning=str(d.get("reasoning") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"reasoning": self.reasoning, "edits": [e.to_dict() for e in self.edits]}
