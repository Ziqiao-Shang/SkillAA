"""Generic task-type metrics for active benchmark reporting."""

from __future__ import annotations

from typing import Any


_DEFAULT_TASK_TYPE_BY_ENVIRONMENT = {
    "searchqa": "qa",
    "livemathematicianbench": "math_mcq",
    "livemath": "math_mcq",
    "docvqa": "docvqa",
    "spreadsheetbench": "spreadsheetbench",
    "alfworld": "alfworld",
}


def canonical_task_type(result: dict[str, Any], *, environment: str) -> str:
    """Return a stable task bucket for normal, timeout, and legacy rows."""
    env = str(environment or "").strip().lower()
    task = str(result.get("task_type") or "").strip()

    # Older SearchQA success rows omitted task_type, while timeout rows copied
    # the dataset's ``qa`` label.  Treat the former ``searchqa`` fallback and
    # legacy ``unknown`` bucket as the same native task.
    if env == "searchqa" and task.lower() in {"", "unknown", "searchqa"}:
        return "qa"
    if task and task.lower() != "unknown":
        return task
    return _DEFAULT_TASK_TYPE_BY_ENVIRONMENT.get(env, "unknown")


def compute_task_metrics(
    results: list[dict[str, Any]],
    *,
    environment: str = "searchqa",
    include_excluded: bool = False,
) -> dict[str, dict[str, float | int | str]]:
    """Return native task-type buckets and their micro-overall score."""
    buckets: dict[str, dict[str, float | int | str]] = {}
    for result in results:
        if result.get("exclude_from_metrics") and not include_excluded:
            continue
        task = canonical_task_type(result, environment=environment)
        bucket = buckets.setdefault(
            task,
            {"label": task, "n": 0, "hard_sum": 0.0, "soft_sum": 0.0},
        )
        hard = float(result.get("hard") or 0.0)
        soft = hard if result.get("soft") is None else float(result["soft"])
        bucket["n"] = int(bucket["n"]) + 1
        bucket["hard_sum"] = float(bucket["hard_sum"]) + hard
        bucket["soft_sum"] = float(bucket["soft_sum"]) + soft
    output: dict[str, dict[str, float | int | str]] = {}
    total_n = 0
    total_hard = 0.0
    total_soft = 0.0
    for task in sorted(buckets):
        bucket = buckets[task]
        n = int(bucket["n"])
        hard = float(bucket["hard_sum"]) / n if n else 0.0
        soft = float(bucket["soft_sum"]) / n if n else 0.0
        output[task] = {
            "label": task,
            "n": n,
            "hard": hard,
            "soft": soft,
            "hard_percent": 100.0 * hard,
            "soft_percent": 100.0 * soft,
        }
        total_n += n
        total_hard += float(bucket["hard_sum"])
        total_soft += float(bucket["soft_sum"])
    overall_hard = total_hard / total_n if total_n else 0.0
    overall_soft = total_soft / total_n if total_n else 0.0
    output["overall"] = {
        "label": "Overall",
        "n": total_n,
        "hard": overall_hard,
        "soft": overall_soft,
        "hard_percent": 100.0 * overall_hard,
        "soft_percent": 100.0 * overall_soft,
    }
    return output
