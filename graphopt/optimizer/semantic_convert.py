"""JSON → Agent Markdown via render_skill_prompt (immutable + rule graph only).

Each environment uses ``envs/{environment}/initial_skill/best_graph.json`` plus the selected conversion-protocol
``immutable_prompt.md``. The shared renderer lives in ``graphopt.envs`` and resolves the immutable
protocol from the explicit environment.
Optimizer experience ledger is never rendered here.
"""

from __future__ import annotations

from pathlib import Path

from graphopt.types import SkillGraph

from graphopt.envs import render_skill_prompt as rsp


def _as_document(graph: SkillGraph) -> dict:
    graph.sync_data_from_views()
    data = graph.data
    if "stable_rule_graph" not in data:
        raise ValueError(
            "SkillGraph missing stable_rule_graph. "
            "Load a formal graphopt/envs/{environment}/initial_skill/best_graph.json."
        )
    if "experience_graph" in data:
        data = dict(data)
        data.pop("experience_graph", None)
        graph.data = data
    rsp.validate(data)
    return data


def graph_to_prompt(
    graph: SkillGraph,
    *,
    shell_md: str | Path | None = None,
    include_relations: bool = True,
    environment: str | None = None,
    immutable_prompt_file: str | None = None,
    protocol: str = "legacy",
    focus_nodes: list[str] | None = None,
    masked_nodes: list[str] | None = None,
    include_edge_rationale: bool = True,
) -> str:
    """JSON → model-visible Markdown (protocol + rule graph; no experience)."""
    del shell_md, include_relations
    data = _as_document(graph)
    return rsp.render_document(
        data,
        environment=environment,
        immutable_prompt_file=immutable_prompt_file,
        protocol=protocol,
        focus_nodes=focus_nodes,
        masked_nodes=masked_nodes,
        include_edge_rationale=include_edge_rationale,
    )


def graph_to_execution_prompt(graph: SkillGraph) -> str:
    """Alias for the one canonical JSON-derived model prompt."""
    return graph_to_prompt(graph)


def json_to_prompt(
    json_path: str | Path,
    *,
    shell_md: str | Path | None = None,
    include_relations: bool = True,
) -> str:
    from graphopt.optimizer.skill import load_graph

    return graph_to_prompt(
        load_graph(json_path),
        shell_md=shell_md,
        include_relations=include_relations,
    )


def write_prompt(
    graph: SkillGraph,
    out_md: str | Path,
    *,
    shell_md: str | Path | None = None,
    include_relations: bool = True,
) -> str:
    text = graph_to_prompt(
        graph,
        shell_md=shell_md,
        include_relations=include_relations,
    )
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def prompt_node_ids(graph: SkillGraph) -> list[str]:
    data = _as_document(graph)
    stable = data["stable_rule_graph"].get("nodes") or []
    ids: list[str] = []
    for section_id in rsp.section_order(data):
        for n in stable:
            if n.get("section") == section_id and n.get("active", True):
                ids.append(n["id"])
    return ids


write_prompt_md = write_prompt


def render_skill_prompt(graph: SkillGraph, *_, **kwargs) -> str:
    return graph_to_prompt(graph, **kwargs)


def render_for_batch(graph: SkillGraph, **__) -> tuple[str, list[str]]:
    return graph_to_execution_prompt(graph), prompt_node_ids(graph)


def render_graph_prompt(graph: SkillGraph) -> str:
    return graph_to_prompt(graph)


def render_skill_body(graph: SkillGraph, *, include_relations: bool = True) -> str:
    del include_relations
    return graph_to_prompt(graph)
