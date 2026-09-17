"""Build case-analyzer trajectory text and validate graph-usage sidecars."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_GRAPH_USAGE_RE = re.compile(r"<graph_usage>\s*(\{.*?\})\s*</graph_usage>", re.DOTALL | re.IGNORECASE)

_SEMANTIC_REASONING_TRACE_RE = re.compile(
    r"<reasoning_trace>\s*(.*?)\s*</reasoning_trace>",
    re.DOTALL | re.IGNORECASE,
)

_READOUT_RE = re.compile(r"<readout>\s*(\{.*?\})\s*</readout>", re.DOTALL | re.IGNORECASE)


def extract_readout_trajectory(steps: list[dict[str, Any]]) -> str | None:
    """Return trajectory from final-step ``<readout>{\"trajectory\":...}</readout>`` if present."""
    for step in reversed(steps):
        for field in ("reasoning", "model_response"):
            text = str(step.get(field) or "")
            match = _READOUT_RE.search(text)
            if not match:
                continue
            try:
                obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            traj = obj.get("trajectory")
            if traj is not None and str(traj).strip():
                return str(traj).strip()
    return None


def format_trajectory_from_steps(steps: list[dict[str, Any]]) -> str:
    """Join every saved action step into the authoritative episode trajectory.

    Legacy model-emitted ``<readout>`` blocks are deliberately ignored: a model
    can omit or truncate earlier actions, whereas the harness owns the complete
    step list. Attribution is performed once over this full trajectory.
    """
    lines: list[str] = []
    for i, step in enumerate(steps):
        step_n = step.get("step", i)
        reasoning = str(step.get("reasoning") or "").strip()
        action = str(step.get("action") or "").strip()
        feedback = str(step.get("env_feedback") or "").strip()
        content = step.get("content")
        content = content.strip() if isinstance(content, str) else ""
        parts = [f"Step {step_n}"]
        if reasoning:
            parts.append(f"Reasoning: {reasoning}")
        if action:
            parts.append(f"Action: {action}")
        if feedback:
            parts.append(f"Environment feedback: {feedback}")
        if content:
            role = str(step.get("role") or step.get("type") or "").strip().lower()
            if role == "system":
                parts.append(f"Environment feedback: {content}")
            elif role == "user":
                parts.append(f"User request: {content}")
            else:
                parts.append(f"Model response: {content}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _response_blobs(pred_root: Path, row: dict[str, Any]) -> list[str]:
    """Return every saved response representation for one rollout row."""
    blobs = [str(row.get(key) or "") for key in ("response", "model_response", "raw")]
    case_id = str(row.get("id") or "")
    if case_id:
        for name in ("raw.txt", "response.txt"):
            path = pred_root / case_id / name
            if path.is_file():
                try:
                    blobs.append(path.read_text(encoding="utf-8"))
                except OSError:
                    pass
    return blobs


def attach_semantic_reasoning_traces(
    rollout_dir: str | Path,
    results: list[dict[str, Any]],
    *,
    required: bool = False,
) -> None:
    """Attach one explicit student-authored semantic trace without scoring it."""
    pred_root = Path(rollout_dir) / "predictions"
    for row in results:
        traces = [
            match.group(1).strip()
            for blob in _response_blobs(pred_root, row)
            for match in _SEMANTIC_REASONING_TRACE_RE.finditer(blob)
            if match.group(1).strip()
        ]
        traces = list(dict.fromkeys(traces))
        row["semantic_reasoning_trace_required"] = bool(required)
        if len(traces) == 1:
            row["semantic_reasoning_trace"] = traces[0]
            row["semantic_reasoning_trace_status"] = "validated_student_trace"
        elif len(traces) > 1:
            row["semantic_reasoning_trace"] = ""
            row["semantic_reasoning_trace_status"] = "invalid_multiple_student_traces"
        else:
            row["semantic_reasoning_trace"] = ""
            row["semantic_reasoning_trace_status"] = (
                "missing_required_student_trace" if required else "not_requested"
            )


def attach_trajectories(rollout_dir: str | Path, results: list[dict[str, Any]]) -> None:
    """Fill ``trajectory`` on each result from ``predictions/<id>/conversation.json``."""
    pred_root = Path(rollout_dir) / "predictions"
    if not pred_root.is_dir():
        return
    for r in results:
        tid = str(r.get("id") or "")
        if not tid:
            continue
        conv_path = pred_root / tid / "conversation.json"
        if not conv_path.is_file():
            continue
        try:
            steps = json.loads(conv_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if r.get("trajectory"):
                r["legacy_trajectory"] = r["trajectory"]
            continue
        if isinstance(steps, list) and steps:
            r["trajectory"] = format_trajectory_from_steps(steps)


def attach_graph_usage(
    rollout_dir: str | Path,
    results: list[dict[str, Any]],
    *,
    valid_nodes: set[str],
    valid_edges: set[str],
) -> None:
    """Attach validated student-reported graph refs without changing scores."""
    pred_root = Path(rollout_dir) / "predictions"
    for row in results:
        blobs = _response_blobs(pred_root, row)
        matches = [match for blob in blobs for match in _GRAPH_USAGE_RE.finditer(blob)]
        # The same response may appear in more than one result field. Deduplicate
        # identical blocks, but reject genuinely multiple sidecars.
        payloads = list(dict.fromkeys(match.group(1) for match in matches))
        if not payloads:
            row["graph_refs"] = {
                "used_nodes": [], "used_edges": [],
                "status": "missing_student_usage",
            }
            continue
        try:
            if len(payloads) != 1:
                raise ValueError("exactly one graph_usage sidecar is required")
            obj = json.loads(payloads[0])
            if not isinstance(obj, dict) or set(obj) != {"used_nodes", "used_edges"}:
                raise ValueError("strict keys required")
            if not isinstance(obj["used_nodes"], list) or not isinstance(obj["used_edges"], list):
                raise ValueError("used_nodes and used_edges must be arrays")
            if any(not isinstance(item, str) for item in obj["used_nodes"] + obj["used_edges"]):
                raise ValueError("graph usage IDs must be strings")
            nodes = list(obj["used_nodes"])
            edges = list(obj["used_edges"])
            if len(nodes) != len(set(nodes)) or len(edges) != len(set(edges)):
                raise ValueError("duplicate ids")
            bad_nodes = sorted(set(nodes) - valid_nodes)
            bad_edges = sorted(set(edges) - valid_edges)
            if bad_nodes or bad_edges:
                raise ValueError(f"unknown ids nodes={bad_nodes} edges={bad_edges}")
            row["graph_refs"] = {
                "used_nodes": nodes, "used_edges": edges,
                "status": "validated_student_usage",
            }
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            row["graph_refs"] = {
                "used_nodes": [], "used_edges": [],
                "status": "invalid_student_usage", "error": str(exc),
            }
