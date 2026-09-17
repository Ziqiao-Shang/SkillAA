#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render SkillGraph JSON directly to model-readable Markdown.

Graph inputs live in the relevant method directory. Immutable graph-to-prompt
protocol and template inputs live beside each method artifact.

Usage:
    python graphopt/envs/render_skill_prompt.py graphopt/envs/searchqa/initial_skill/best_graph.json \\
        -o /tmp/rendered_skill.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ORDERING_EDGE_TYPES = {"prereq"}
ENVS_DIR = Path(__file__).resolve().parent
SECTION_ORDER = ("task_types", "general_principles", "search_control", "specialized_recovery")
SECTION_TITLES = {
    "task_types": "Task Types",
    "general_principles": "General Principles",
    "search_control": "Search Control and Recovery",
    "specialized_recovery": "Specialized Recovery Strategies",
}
FIELD_LABELS = {
    "id": "Node ID",
    "category": "Category",
    "node_type": "Knowledge type",
    "meaning": "Meaning",
    "when_to_use": "When to use",
    "how_to_use": "How to use",
    "avoid": "Avoid",
    "relation": "Relation",
    "strength": "Strength",
    "rationale": "Why this relationship applies",
}
RELATIONSHIPS = {
    "prereq": ("Prerequisite", "->"),
    "enhance": ("Enhancement", "->"),
    "co_occur": ("Co-occurrence", "<->"),
}
WEIGHT_LABELS = {
    "medium": "Medium",
    "strong": "Strong",
}
COMPACT_FULL_RENDER_ENVS = {"searchqa", "docvqa", "livemathematicianbench"}

CAUSAL_RENDER_PROTOCOL = """\
## Graph Execution Contract

- Treat every node as a conditional procedure, not as a fact or a mandatory checklist item.
- An empty `When to use` field does not mean unconditional execution: apply the rule only when its action is semantically relevant to the current question and evidence.
- A populated `When to use` field states a semantic condition. Its wording and examples are illustrative rather than an exhaustive keyword list.
- When a specific applicable node and a general node affect the same decision, follow the specific node on that decision only; retain the general node elsewhere.
- Follow `prereq` relationships before the dependent rule. Apply `enhance` relationships only when the enhancing rule is itself applicable.
- In the graph-usage sidecar, report only nodes and relationships that actually changed the answer; do not list every visible rule.
"""

DEFAULT_ORDERED_INTROS = {
    "searchqa": ("Answer only from the supplied context and return one concise answer.",),
    "docvqa": ("Inspect the supplied document image and return the smallest exact answer span.",),
    "livemathematicianbench": ("Select the single option justified by the visible mathematical question and hypotheses.",),
}


def section_order(data: dict, environment: str | None = None) -> tuple[str, ...]:
    configured = (data.get("meta") or {}).get("section_order")
    if isinstance(configured, list) and configured and all(isinstance(x, str) for x in configured):
        return tuple(configured)
    if environment == "docvqa":
        return ("general_principles", "task_types", "specialized_recovery", "search_control")
    return SECTION_ORDER


def section_titles(data: dict) -> dict[str, str]:
    configured = (data.get("meta") or {}).get("section_titles")
    titles = dict(SECTION_TITLES)
    if isinstance(configured, dict):
        titles.update({str(k): str(v) for k, v in configured.items()})
    return titles


def immutable_prompt_path(
    data: dict,
    environment: str | None = None,
    immutable_prompt_file: str | None = None,
    *,
    protocol: str = "legacy",
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"missing immutable prompt: {candidate}")
    environment = str(environment or (data.get("meta") or {}).get("environment") or "searchqa").strip()
    file_name = str(immutable_prompt_file or "immutable_prompt.md").strip()
    if Path(file_name).name != file_name or not file_name.endswith(".md"):
        raise ValueError(
            "immutable_prompt_file must be one Markdown basename"
        )
    normalized_protocol = str(protocol or "legacy").strip().lower()
    # Historical 2026-08-31 Trainer spelling. The public protocol was later
    # renamed to ``case_complete`` without changing its rendered semantics.
    if normalized_protocol == "case_complete_v1":
        normalized_protocol = "case_complete"
    if normalized_protocol in {"case_complete", "causal"}:
        method = "graphopt"
    elif file_name == "immutable_prompt_reasoning.md":
        method = "initial_skill_reasoning"
    else:
        method = "initial_skill"
    candidate = ENVS_DIR / environment / method / "immutable_prompt.md"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"missing method-local immutable prompt for environment {environment!r}: {candidate}"
    )


def ordered_intro(
    data: dict, environment: str | None = None
) -> tuple[str, ...]:
    configured = (data.get("meta") or {}).get("ordered_intro")
    if isinstance(configured, list) and configured and all(isinstance(x, str) for x in configured):
        return tuple(configured)
    name = str(environment or (data.get("meta") or {}).get("environment") or "searchqa").strip()
    if name not in DEFAULT_ORDERED_INTROS:
        raise ValueError(f"missing compact-prompt introduction for environment {name!r}")
    return DEFAULT_ORDERED_INTROS[name]


def load_graph(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(data)
    return data


def validate(data: dict) -> None:
    # Import lazily so this file remains executable as a standalone script.
    project_root = Path(__file__).resolve().parents[2]
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from graphopt.optimizer.skill import validate_graph_json_shape

    validate_graph_json_shape(data)
    graph_name = "stable_rule_graph"
    graph = data[graph_name]
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate node IDs in {graph_name}")
    id_set = set(ids)
    for e in edges:
        if e["source"] not in id_set or e["target"] not in id_set:
            raise ValueError(f"Edge references missing node in {graph_name}: {e}")
        if e["type"] not in ("prereq", "enhance", "co_occur"):
            raise ValueError(f"Unknown edge type in {graph_name}: {e['type']}")

    # Structural edges must stay acyclic.
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {nid: 0 for nid in id_set}
    for e in edges:
        if e["type"] not in ORDERING_EDGE_TYPES:
            continue
        adjacency[e["source"]].append(e["target"])
        indegree[e["target"]] += 1
    queue = [nid for nid, d in indegree.items() if d == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for nxt in adjacency[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if seen != len(id_set):
        cyclic = sorted(nid for nid, d in indegree.items() if d > 0)
        raise ValueError(f"{graph_name} has a prerequisite cycle: {cyclic}")


def node_map(graph: dict) -> Dict[str, dict]:
    return {n["id"]: n for n in graph.get("nodes") or []}


def normalize_visible(text: str) -> str:
    return text.rstrip() + "\n"


def render_node_block(node: dict, graph_name: str, labels: dict) -> List[str]:
    lines = [
        f'<!-- SG_NODE_BEGIN graph="{graph_name}" id="{node["id"]}" -->',
        f'#### {node["title"]}',
        "",
        f'- **{labels["id"]}:** `{node["id"]}`',
        f'- **{labels["category"]}:** `{node.get("category", "general")}`',
        f'- **{labels["node_type"]}:** `{node.get("node_type", "rule")}` (reusable procedural rule)',
    ]
    for key in ("when_to_use", "how_to_use"):
        value = str(node.get(key) or "").strip()
        if value:
            lines.append(f'- **{labels[key]}:** {value}')
    avoid = [str(item).strip() for item in (node.get("avoid") or []) if str(item).strip()]
    if avoid:
        lines.append(f'- **{labels["avoid"]}:**')
        lines.extend(f"  - {item}" for item in avoid)
    lines.append(f'<!-- SG_NODE_END id="{node["id"]}" -->')
    lines.append("")
    return lines


def render_edge_line(
    edge: dict,
    nodes: Dict[str, dict],
    graph_name: str,
    *,
    include_rationale: bool = False,
) -> List[str]:
    src = nodes[edge["source"]]["title"]
    dst = nodes[edge["target"]]["title"]
    rel, arrow = RELATIONSHIPS[edge["type"]]
    details = [f'**{FIELD_LABELS["relation"]}:** {rel}']
    if str(edge.get("weight") or "medium") != "medium":
        details.append(
            f'**{FIELD_LABELS["strength"]}:** {WEIGHT_LABELS[edge["weight"]]}'
        )
    rationale = str(edge.get("rationale") or "").strip()
    if include_rationale and rationale:
        details.append(f'**{FIELD_LABELS["rationale"]}:** {rationale}')
    return [
        f'<!-- SG_EDGE graph="{graph_name}" id="{edge["id"]}" source="{edge["source"]}" target="{edge["target"]}" -->',
        f'- **`{edge["id"]}`:** **{src} (`{edge["source"]}`)** {arrow} '
        f'**{dst} (`{edge["target"]}`)** | ' + ' | '.join(details),
    ]


def render_compact_node_block(node: dict, graph_name: str) -> List[str]:
    """Render only populated semantics; empty strings/lists are placeholders.

    Graph nodes have no ``meaning`` key and partition each frozen source rule
    across trigger, action, and prohibition. Those three keys always exist;
    empty string/list placeholders produce no model-visible text. A later
    update that populates a placeholder becomes visible automatically.
    """
    del graph_name
    lines = [f'### [{node["id"]}] {node["title"]}', ""]
    when_to_use = str(node.get("when_to_use") or "").strip()
    how_to_use = str(node.get("how_to_use") or "").strip()
    if when_to_use:
        lines.extend([f'**When to use:** {when_to_use}', ""])
    if how_to_use:
        lines.extend(how_to_use.splitlines())
        lines.append("")
    avoid = [str(item).strip() for item in (node.get("avoid") or []) if str(item).strip()]
    if avoid:
        lines.extend(["**Avoid:**", ""])
        lines.extend(f"- {item}" for item in avoid)
        lines.append("")
    return lines


def render_flat_atomic_node_block(node: dict, graph_name: str) -> List[str]:
    """Render LiveMath trigger, action, and prohibition semantics compactly.

    LiveMath theorem heuristics lose salience when every atomic node receives a
    full Markdown subsection. Keep each populated semantic field visible as a
    flat ID-labelled list item while leaving genuinely absent fields omitted.
    """
    del graph_name
    node_id = str(node["id"])
    lines: List[str] = []
    when_to_use = str(node.get("when_to_use") or "").strip()
    how_to_use = str(node.get("how_to_use") or "").strip()
    if when_to_use:
        lines.append(f"- [{node_id}] When to use: {when_to_use}")
    if how_to_use:
        for line in how_to_use.splitlines():
            rule = line.strip()
            if not rule:
                continue
            lines.append(f'- [{node_id}] {rule.removeprefix("- ")}')
    avoid = [str(item).strip() for item in (node.get("avoid") or []) if str(item).strip()]
    for item in avoid:
        lines.append(f"- [{node_id}] Avoid: {item}")
    lines.append("")
    return lines


def render_compact_edge_line(
    edge: dict, graph_name: str, *, include_rationale: bool = False
) -> List[str]:
    """Render a relation without inventing text for absent edge metadata."""
    del graph_name
    attributes = [str(edge["type"])]
    # ``medium`` is the neutral schema default. Its omission is deterministic;
    # changing it to a non-neutral value changes the deployed prompt.
    if str(edge.get("weight") or "medium") != "medium":
        attributes.append(f'strength={edge["weight"]}')
    lines = [
        f'- [{edge["id"]}] {edge["source"]} -['
        f'{"; ".join(attributes)}]-> {edge["target"]}'
    ]
    rationale = str(edge.get("rationale") or "").strip()
    if include_rationale and rationale:
        lines.append(
            f'  - **{FIELD_LABELS["rationale"]}:** {rationale}'
        )
    return lines


def render_stable(
    data: dict,
    *,
    environment: str | None = None,
    include_edge_rationale: bool = False,
) -> List[str]:
    stable = data["stable_rule_graph"]
    nodes = node_map(stable)
    environment = str(
        environment or (data.get("meta") or {}).get("environment") or "searchqa"
    ).strip()
    compact = environment in COMPACT_FULL_RENDER_ENVS
    flat_atomic = environment == "livemathematicianbench"
    compact_title = "Atomic Skill Rules"
    lines = [f"## {compact_title}", ""] if compact else ["<!-- SG_STABLE_BEGIN -->", "## Stable Rule Graph", ""]
    configured_order = list(section_order(data, environment))
    present_sections = [
        str(node.get("section") or "general_principles")
        for node in stable.get("nodes") or []
    ]
    configured_order.extend(
        section for section in present_sections if section not in configured_order
    )
    titles = section_titles(data)
    for section_id in configured_order:
        section_nodes = [
            n for n in stable.get("nodes") or []
            if n.get("section") == section_id and n.get("active", True)
        ]
        if not section_nodes and compact:
            continue
        if not compact:
            title = titles.get(section_id, section_id.replace("_", " ").title())
            lines.extend([f"### {title}", ""])
        # Keep file order as stored in JSON.
        for n in section_nodes:
            if flat_atomic:
                lines.extend(render_flat_atomic_node_block(n, "stable_rule_graph"))
            elif compact:
                lines.extend(render_compact_node_block(n, "stable_rule_graph"))
            else:
                lines.extend(render_node_block(n, "stable_rule_graph", FIELD_LABELS))

    common_mistakes = stable.get("common_mistakes") or []
    if common_mistakes or not compact:
        lines.extend(["### Common Mistakes to Avoid", ""])
        for item in common_mistakes:
            lines.append(f"- {item}")
        lines.append("")

    active_edges = [
        e for e in (stable.get("edges") or []) if e.get("active", True)
    ]
    if active_edges:
        lines.extend(["## Skill Relationships" if compact else "### Stable Relationship Map", ""])
    for e in active_edges:
        if compact:
            lines.extend(render_compact_edge_line(
                    e, "stable_rule_graph",
                    include_rationale=include_edge_rationale,
                ))
        else:
            lines.extend(
                render_edge_line(
                    e,
                    nodes,
                    "stable_rule_graph",
                    include_rationale=include_edge_rationale,
                )
            )
    if not compact:
        lines.append("<!-- SG_STABLE_END -->")
    lines.append("")
    return lines


def build_visible_markdown(
    data: dict,
    *,
    environment: str | None = None,
    immutable_prompt_file: str | None = None,
    protocol: str = "legacy",
    focus_nodes: Iterable[str] | None = None,
    masked_nodes: Iterable[str] | None = None,
    immutable_prompt_path_override: str | Path | None = None,
    protocol_text_path_override: str | Path | None = None,
    include_edge_rationale: bool = False,
) -> str:
    validate(data)
    normalized_protocol = str(protocol or "legacy").strip().lower()
    if normalized_protocol == "case_complete_v1":
        normalized_protocol = "case_complete"
    header = immutable_prompt_path(
        data, environment, immutable_prompt_file,
        protocol=normalized_protocol, override=immutable_prompt_path_override,
    ).read_text(encoding="utf-8").rstrip()
    parts: List[str] = [header, ""]
    if normalized_protocol not in {"legacy", "case_complete", "causal"}:
        raise ValueError(
            f"unknown graph render protocol {protocol!r}; expected legacy or case_complete"
        )
    if normalized_protocol in {"case_complete", "causal"}:
        if protocol_text_path_override is not None:
            protocol_text = Path(protocol_text_path_override).read_text(encoding="utf-8")
        else:
            local_contract = (
                ENVS_DIR / str(environment or "searchqa") / "graphopt"
                / "graph_execution_contract.md"
            )
            protocol_text = (
                local_contract.read_text(encoding="utf-8")
                if local_contract.is_file() else CAUSAL_RENDER_PROTOCOL
            )
        parts.extend([protocol_text.rstrip(), ""])
    parts.extend(
        render_stable(
            data,
            environment=environment,
            include_edge_rationale=include_edge_rationale,
        )
    )
    focus = [str(node_id) for node_id in (focus_nodes or []) if str(node_id)]
    masked = [str(node_id) for node_id in (masked_nodes or []) if str(node_id)]
    if focus or masked:
        parts.extend(["## Causal Diagnostic Replay", ""])
        if focus:
            parts.extend([
                "Explicitly check whether the following existing nodes apply: "
                + ", ".join(f"`{node_id}`" for node_id in focus)
                + ". Use them only if their existing semantics match the task; "
                "do not assume applicability merely because they are highlighted.",
                "",
            ])
        if masked:
            parts.extend([
                "The following nodes are disabled only for this diagnostic replay: "
                + ", ".join(f"`{node_id}`" for node_id in masked)
                + ". Do not reconstruct or apply their omitted instructions.",
                "",
            ])
    return normalize_visible("\n".join(parts))


def render_document(
    data: dict,
    *,
    environment: str | None = None,
    immutable_prompt_file: str | None = None,
    protocol: str = "legacy",
    focus_nodes: Iterable[str] | None = None,
    masked_nodes: Iterable[str] | None = None,
    immutable_prompt_path_override: str | Path | None = None,
    protocol_text_path_override: str | Path | None = None,
    include_edge_rationale: bool = False,
) -> str:
    """Return exactly the graph-derived skill shown to the agent."""
    return build_visible_markdown(
        data,
        environment=environment,
        immutable_prompt_file=immutable_prompt_file,
        protocol=protocol,
        focus_nodes=focus_nodes,
        masked_nodes=masked_nodes,
        immutable_prompt_path_override=immutable_prompt_path_override,
        protocol_text_path_override=protocol_text_path_override,
        include_edge_rationale=include_edge_rationale,
    )


def render_ordered(
    data: dict,
    skill_ids: Iterable[str] | None = None,
    task_description: str | None = None,
    environment: str | None = None,
) -> str:
    """Compact retrieval-style prompt for selected active skills."""
    validate(data)
    nodes = node_map(data["stable_rule_graph"])
    if skill_ids:
        selected = [sid for sid in skill_ids if sid in nodes and nodes[sid].get("active", True)]
    else:
        selected = [
            n["id"]
            for n in data["stable_rule_graph"].get("nodes") or []
            if n.get("active", True)
        ]

    # Prefer JSON order for selected ids.
    order_index = {nid: i for i, nid in enumerate(selected)}
    ordered = sorted(selected, key=lambda nid: order_index.get(nid, 10**9))

    out: List[str] = []
    for item in ordered_intro(data, environment):
        out.append(item)
    out.append("")
    if task_description:
        out.extend(["Your task is to:", task_description, ""])
    out.extend(["## Retrieved Relevant Skills", "", "### Skills", ""])
    for nid in ordered:
        n = nodes[nid]
        out.append(f"- **[{n.get('category', 'general')}] {n['title']} [{nid}]**")
        how_to_use = str(n.get("how_to_use") or "").strip()
        when_to_use = str(n.get("when_to_use") or "").strip()
        if how_to_use:
            out.append(f"  {how_to_use}")
        if when_to_use:
            out.append(f"  *When to use: {when_to_use}*")
        out.append("")
    return normalize_visible("\n".join(out))


def render_method_directory(
    method_dir: str | Path,
    *,
    output: str | Path | None = None,
    check: bool = False,
) -> Path:
    """Rebuild one frozen method prompt from files in that directory only."""
    directory = Path(method_dir)
    config_path = directory / "render_protocol.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_name = config.get("system_prompt_template")
    template = (
        (directory / template_name).read_text(encoding="utf-8")
        if template_name else None
    )
    source_type = str(config["source_type"])
    if source_type == "none":
        skill = ""
    elif source_type == "markdown":
        skill = (directory / config["skill"]).read_text(encoding="utf-8").strip()
    elif source_type == "graph":
        data = load_graph(directory / config["graph"])
        skill = render_document(
            data,
            environment=str(config["environment"]),
            protocol=str(config["graph_render_protocol"]),
            immutable_prompt_path_override=directory / config["immutable_prompt"],
            protocol_text_path_override=(
                directory / config["graph_execution_contract"]
                if config.get("graph_execution_contract") else None
            ),
            include_edge_rationale=bool(config.get("include_edge_rationale", False)),
        )
    else:
        raise ValueError(f"unknown frozen prompt source_type: {source_type!r}")
    if template is None:
        rendered = skill
    else:
        skill_section = f"## Skill\n{skill.strip()}\n\n" if skill.strip() else ""
        rendered = template.format(skill_section=skill_section)
    output_path = Path(output) if output is not None else directory / config["output"]
    if check:
        persisted = output_path.read_text(encoding="utf-8")
        if persisted != rendered:
            raise ValueError(f"frozen prompt mismatch: {output_path}")
        print(f"OK: {directory} -> {output_path.name}")
    else:
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Rendered: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render SkillGraph JSON to agent Markdown.")
    p.add_argument("input", nargs="?", default="initial.json")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--check", action="store_true")
    p.add_argument("--protocol", default="legacy", choices=["legacy", "case_complete", "causal"])
    p.add_argument("--mode", choices=["full", "ordered"], default="full")
    p.add_argument("--skills", default=None, help="Comma-separated IDs for ordered mode")
    p.add_argument("--task-description", default=None)
    p.add_argument("--environment", default=None, help="Environment-specific immutable prompt")
    p.add_argument(
        "--immutable-prompt-file", default=None,
        help="Markdown basename in the environment skills directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if input_path.is_dir():
        render_method_directory(input_path, output=args.output, check=args.check)
        return
    data = load_graph(input_path)
    environment = args.environment
    if environment is None and input_path.parent.name in {"initial_skill", "initial_skill_reasoning", "graphopt"}:
        environment = input_path.parent.parent.name
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        if stem.endswith("_zh"):
            output_path = input_path.with_name(stem.replace("initial", "initial_prompt") + ".md")
            if output_path.name == input_path.name:
                output_path = input_path.with_name("initial_prompt_zh.md")
        else:
            output_path = input_path.with_name("initial_prompt.md")

    if args.mode == "ordered":
        skill_ids = None
        if args.skills:
            skill_ids = [x.strip() for x in args.skills.split(",") if x.strip()]
        text = render_ordered(
            data, skill_ids=skill_ids, task_description=args.task_description,
            environment=environment,
        )
    else:
        text = render_document(
            data, environment=environment,
            immutable_prompt_file=args.immutable_prompt_file,
            protocol=args.protocol,
        )

    if args.check:
        if output_path.read_text(encoding="utf-8") != text:
            raise ValueError(f"rendered prompt mismatch: {output_path}")
        print(f"OK: {output_path}")
        return
    output_path.write_text(text, encoding="utf-8")
    print(f"Rendered: {output_path}")


if __name__ == "__main__":
    main()
