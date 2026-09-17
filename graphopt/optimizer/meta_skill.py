"""Epoch-level meta knowledge (SkillAA meta_skill analogue for graph edits).

Consumes improvement / regression / persistent_fail signals from longitudinal
comparisons and gate history to guide the next epoch's optimizer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from graphopt.debug.artifacts import ArtifactStore, save_json, save_llm_call, save_template_call
from graphopt.evolution.exact_refs import exact_reference_json

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def format_meta(meta_text: str) -> str:
    t = (meta_text or "").strip()
    if not t:
        return ""
    return (
        "## Optimizer Meta Knowledge (graph edits)\n"
        "Use this when proposing, merging, and ranking graph edits. Prefer it when "
        "evidence is ambiguous; do not force it if current trajectories contradict it.\n\n"
        + t
        + "\n"
    )


def summarize_comparisons(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    cats = Counter(str(p.get("category") or "unknown") for p in pairs)
    return {
        "n": len(pairs),
        "improved": cats.get("improved", 0),
        "regressed": cats.get("regressed", 0),
        "persistent_fail": cats.get("persistent_fail", 0),
        "stable_success": cats.get("stable_success", 0),
    }


def _template_meta(
    history: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    edit_ops: list[str],
    current: str = "",
) -> str:
    stats = summarize_comparisons(pairs)
    accept = sum(1 for h in history if h.get("accepted"))
    reject = len(history) - accept
    op_counts = Counter(edit_ops)
    lines = [
        "When improving the current benchmark SkillGraph:",
        "1. Prefer updating existing rule-graph nodes that match open experience-ledger opinions.",
        "2. Permanent Protocol is immutable; the rule graph after it is fully editable.",
        "3. Experience ledger is optimizer-only — never write it into the Agent skill prompt.",
        "4. co_occur edges are statistical — do not propose them.",
        "",
        f"Gate accepts={accept}, rejects={reject}.",
        (
            f"Longitudinal: improved={stats['improved']} regressed={stats['regressed']} "
            f"persistent_fail={stats['persistent_fail']} stable_success={stats['stable_success']}."
        ),
    ]
    if stats["improved"] and op_counts:
        top = ", ".join(f"{k}:{v}" for k, v in op_counts.most_common(5))
        lines.append(f"Useful edit ops seen on accepts: {top}.")
    if stats["regressed"]:
        lines.append("Avoid edits associated with regressions; prefer smaller rule-graph edits.")
    if stats["persistent_fail"]:
        lines.append(
            "Persistent failures suggest strengthening the smallest related reusable rule."
        )
    if current.strip():
        lines.extend(["", "Previously retained evidence-grounded lessons:", current.strip()])
    return "\n".join(lines)


def update_meta(
    history: list[dict[str, Any]],
    current: str = "",
    *,
    chat_fn=None,
    mode: str = "template",
    comparison_pairs: list[dict[str, Any]] | None = None,
    prev_graph_text: str = "",
    curr_graph_text: str = "",
    artifact_dir: Path | str | None = None,
    epoch: int | None = None,
    compact_json: bool = False,
) -> str:
    pairs = list(comparison_pairs or [])
    edit_ops: list[str] = []
    meta_store = ArtifactStore(artifact_dir) if artifact_dir else None
    for h in history:
        if h.get("accepted"):
            edit_ops.extend(h.get("edit_ops") or [])

    if mode == "teacher" and chat_fn is not None:
        from graphopt.json_utils import extract_json

        system = (PROMPTS / "meta.md").read_text(encoding="utf-8")
        payload = {
            "schema_version": "graphopt-meta-input-full-evidence",
            "previous_meta": current or "",
            "previous_graph_summary": prev_graph_text or "",
            "current_graph_summary": curr_graph_text or "",
            "longitudinal_pairs": pairs,
            "gate_history": history,
        }
        user = exact_reference_json(payload, indent=None if compact_json else 2)
        current_user = user
        required = {"schema_version", "bullets"}
        for attempt in range(1, 3):
            resp = ""
            token_usage: Any = None
            try:
                resp, token_usage = chat_fn(
                    system=system,
                    user=current_user,
                    max_completion_tokens=4096,
                    retries=2,
                    stage="graph_meta",
                )
                obj = extract_json(resp)
                if not isinstance(obj, dict) or set(obj) != required:
                    raise ValueError("output must contain exactly schema_version and bullets")
                if obj["schema_version"] != "graphopt-meta-v1":
                    raise ValueError("schema_version must be graphopt-meta-v1")
                bullets = obj["bullets"]
                if (
                    not isinstance(bullets, list)
                    or not 5 <= len(bullets) <= 12
                    or any(not isinstance(item, str) or not item.strip() for item in bullets)
                    or len(bullets) != len(set(item.strip() for item in bullets))
                ):
                    raise ValueError("bullets must be 5-12 unique non-empty strings")
                if any(len(item.strip()) > 500 for item in bullets):
                    raise ValueError("each bullet must contain at most 500 characters")
                text = "\n".join(f"- {item.strip()}" for item in bullets)
                if meta_store is not None:
                    name = f"epoch_{epoch:02d}" if epoch is not None else "meta"
                    save_llm_call(
                        meta_store,
                        name,
                        stage="graph_meta",
                        system=system,
                        user=current_user,
                        response=resp,
                        usage=token_usage,
                        parsed={**obj, "meta_text": text},
                    )
                return text
            except Exception as exc:
                if meta_store is not None:
                    name = f"epoch_{epoch:02d}" if epoch is not None else "meta"
                    save_llm_call(
                        meta_store,
                        name,
                        stage="graph_meta",
                        system=system,
                        user=current_user,
                        response=resp or None,
                        usage=token_usage,
                        error=f"attempt {attempt}/2: {exc}",
                    )
                if attempt == 1:
                    current_user = (
                        user
                        + "\n\nFormat Correction: your previous answer was invalid: "
                        + str(exc)
                        + "\nReturn the complete exact JSON object again."
                        + "\nInvalid Previous Answer:\n"
                        + str(resp)
                    )

        # Meta is advisory. Two invalid attempts preserve M_t exactly.
        return current

    text = _template_meta(history, pairs, edit_ops, current)
    if meta_store is not None:
        name = f"epoch_{epoch:02d}" if epoch is not None else "meta"
        save_template_call(
            meta_store,
            name,
            stage="graph_meta",
            inputs={
                "previous_meta_used": bool(current),
                "n_pairs": len(pairs),
                "n_history": len(history),
            },
            outputs={"meta_text": text},
        )
    return text
