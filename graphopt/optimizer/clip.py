"""Clip / rank graph edits — mirrors ``skillaa/optimizer/clip.py``.

Supports:
- total operation budget L (``max_ops`` / ``edit_budget`` / ``learning_rate``)
- optional separate node/edge caps
"""

from __future__ import annotations

from pathlib import Path

from graphopt.types import GraphPatch, SkillGraph

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def clip_budget(
    graph: SkillGraph,
    patch: GraphPatch,
    *,
    max_ops: int | None = 5,
    max_node_edits: int | None = None,
    max_edge_edits: int | None = None,
    chat_fn=None,
) -> GraphPatch:
    """Keep at most ``max_ops`` edits (SkillAA ``rank_and_select`` analogue)."""
    nodes = [e for e in patch.edits if e.is_node_op]
    edges = [e for e in patch.edits if e.is_edge_op]
    total = len(patch.edits)

    need_clip = False
    if max_ops is not None and total > max_ops:
        need_clip = True
    if max_node_edits is not None and len(nodes) > max_node_edits:
        need_clip = True
    if max_edge_edits is not None and len(edges) > max_edge_edits:
        need_clip = True
    if not need_clip:
        return patch

    if chat_fn is not None:
        ranked = _llm_rank(patch, max_ops, max_node_edits, max_edge_edits, chat_fn)
        if ranked is not None:
            return ranked

    def score(e):
        # failure-driven first, then node ops before edge polish
        pri = 0 if e.source_type == "failure" else 1
        kind = 0 if e.is_node_op else 1
        return (pri, kind)

    ordered = sorted(patch.edits, key=score)
    keep: list = []
    n_keep = e_keep = 0
    for e in ordered:
        if max_ops is not None and len(keep) >= max_ops:
            break
        if e.is_node_op:
            if max_node_edits is not None and n_keep >= max_node_edits:
                continue
            keep.append(e)
            n_keep += 1
        else:
            if max_edge_edits is not None and e_keep >= max_edge_edits:
                continue
            keep.append(e)
            e_keep += 1
    ids = {id(x) for x in keep}
    return GraphPatch(edits=[e for e in patch.edits if id(e) in ids], reasoning=patch.reasoning)


def _llm_rank(
    patch: GraphPatch,
    max_ops: int | None,
    max_n: int | None,
    max_e: int | None,
    chat_fn,
) -> GraphPatch | None:
    try:
        from graphopt.json_utils import extract_json
    except Exception:
        return None
    system = (PROMPTS / "rank.md").read_text(encoding="utf-8")
    lines = [f"[{i}] {e.summary()}" for i, e in enumerate(patch.edits)]
    user = (
        f"Budget: max_ops={max_ops}, max_node_edits={max_n}, max_edge_edits={max_e}\n\n"
        + "\n".join(lines)
        + "\n\nReturn JSON {\"selected_indices\":[...]} respecting the budget."
    )
    try:
        resp, _ = chat_fn(system=system, user=user, max_completion_tokens=4096, retries=2, stage="graph_rank")
        obj = extract_json(resp) or {}
        idxs = []
        for i in obj.get("selected_indices") or []:
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < len(patch.edits):
                idxs.append(ii)
        chosen = [patch.edits[i] for i in idxs]
        keep = []
        n_keep = e_keep = 0
        for e in chosen:
            if max_ops is not None and len(keep) >= max_ops:
                break
            if e.is_node_op:
                if max_n is not None and n_keep >= max_n:
                    continue
                keep.append(e)
                n_keep += 1
            else:
                if max_e is not None and e_keep >= max_e:
                    continue
                keep.append(e)
                e_keep += 1
        ids = {id(x) for x in keep}
        return GraphPatch(edits=[e for e in patch.edits if id(e) in ids], reasoning=patch.reasoning)
    except Exception:
        return None


# SkillAA-facing aliases
clip_patch = clip_budget
rank_and_select = clip_budget
