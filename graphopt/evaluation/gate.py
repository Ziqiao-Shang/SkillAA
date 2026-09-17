"""Validation gates for graph updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal
import re

STEP_LIMIT_EXCLUSION_REASON = "REFERENCE_PLAN_OVERLENGTH_EXCLUDED"

GateAction = Literal["accept_new_best", "accept", "reject"]


@dataclass
class GateDecision:
    accepted: bool
    action: GateAction
    current_score: float
    candidate_score: float
    best_score: float
    best_step: int
    reason: str


def normalize_results(results: Any) -> list[dict[str, Any]]:
    """Validate and normalize the rollout → scoring/gate contract.

    Every case must have one unique non-empty id and finite hard/soft scores.
    ``soft`` may be omitted, in which case it explicitly defaults to ``hard``.
    """
    if not isinstance(results, list):
        raise TypeError(f"rollout results must be a list, got {type(results).__name__}")
    if not results:
        raise ValueError("rollout results must contain at least one case")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(results):
        if not isinstance(raw, dict):
            raise TypeError(f"rollout result[{index}] must be an object")
        case_id = str(raw.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"rollout result[{index}] has no non-empty id")
        if case_id in seen:
            raise ValueError(f"duplicate rollout result id: {case_id}")
        seen.add(case_id)
        try:
            hard = float(raw.get("hard") if raw.get("hard") is not None else 0.0)
            soft_raw = raw.get("soft")
            soft = hard if soft_raw is None else float(soft_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rollout result {case_id!r} has non-numeric scores") from exc
        if not math.isfinite(hard) or not math.isfinite(soft):
            raise ValueError(f"rollout result {case_id!r} has non-finite scores")
        if not (0.0 <= hard <= 1.0) or not (0.0 <= soft <= 1.0):
            raise ValueError(f"rollout result {case_id!r} scores must be in [0, 1]")
        item = dict(raw)
        item.update({"id": case_id, "hard": hard, "soft": soft})
        normalized.append(item)
    return normalized


def score_results(
    results: list[dict[str, Any]],
    *,
    metric: str = "hard",
    mixed_weight: float = 0.8,
    skill_content: str = "",
) -> tuple[float, float, float]:
    """Return (hard, soft, gate_score)."""
    results = [r for r in results if not bool(r.get("exclude_from_metrics", False))]
    if not results:
        raise ValueError("no metric-eligible rollout results remain after exclusions")
    hard = sum(float(r.get("hard") or 0) for r in results) / len(results)
    soft = sum(
        float(r.get("hard") or 0) if r.get("soft") is None else float(r["soft"])
        for r in results
    ) / len(results)
    if metric == "soft":
        gate = soft
    elif metric == "mixed":
        w = max(0.0, min(1.0, float(mixed_weight)))
        gate = (1.0 - w) * hard + w * soft
    else:
        gate = hard

    values = (float(hard), float(soft), float(gate))
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"scoring returned non-finite values: {values}")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"scoring returned values outside [0, 1]: {values}")
    return values


def mark_step_limit_exclusions(
    results: list[dict[str, Any]], *, enabled: bool
) -> dict[str, Any]:
    """Keep step-limit failures in both the numerator decision and denominator.

    The enabled argument remains for old configs, but no longer authorizes
    dropping a case because the student or reference plan reached a step cap.
    Unrelated infrastructure failures keep their independently declared metric
    policy.
    """
    excluded: list[dict[str, str]] = []
    stale_step_reasons = {
        "OVERLENGTH_EXCLUDED",
        "REFERENCE_PLAN_OVERLENGTH_EXCLUDED",
        "STUDENT_STEP_LIMIT_COUNTED_AS_FAILURE",
    }
    for result in results:
        existing_exclusion = bool(result.get("exclude_from_metrics"))
        existing_reason = str(result.get("metric_exclusion_reason") or "").strip()
        is_step_limit = bool(
            result.get("student_step_limit_failure")
            or result.get("reference_plan_overlength")
            or existing_reason in stale_step_reasons
        )
        if is_step_limit:
            result["step_limit_excluded"] = False
            result.pop("terminal_policy_exclusion", None)
            result["exclude_from_metrics"] = False
            result["metric_exclusion_reason"] = "STUDENT_STEP_LIMIT_COUNTED_AS_FAILURE"
            if result.get("exclude_from_evolution") and not result.get(
                "infrastructure_failure"
            ):
                result["exclude_from_evolution"] = False
                result.pop("exclusion_reason", None)
        elif existing_exclusion:
            result["exclude_from_metrics"] = True
            result["metric_exclusion_reason"] = existing_reason or "EXCLUDED_FROM_METRICS"
            excluded.append(
                {"id": str(result.get("id")), "reason": result["metric_exclusion_reason"]}
            )
        else:
            result["exclude_from_metrics"] = False
    eligible = [r for r in results if not r.get("exclude_from_metrics")]
    return {
        "metric_policy": "all_task_cases_step_limit_is_failure",
        "legacy_step_limit_exclusion_flag_ignored": bool(enabled),
        "raw_n": len(results),
        "effective_n": len(eligible),
        "excluded_n": len(excluded),
        "excluded_case_ids": [item["id"] for item in excluded],
        "excluded_cases": excluded,
    }


def decide(
    current_score: float,
    candidate_score: float,
    *,
    best_score: float = 0.0,
    best_step: int = -1,
    global_step: int = 0,
    eps: float = 1e-12,
) -> GateDecision:
    """Accept iff candidate gate score strictly improves over current."""
    if candidate_score > current_score + eps:
        if candidate_score > best_score + eps:
            return GateDecision(
                True,
                "accept_new_best",
                current_score,
                candidate_score,
                candidate_score,
                global_step,
                "accept_new_best",
            )
        return GateDecision(
            True,
            "accept",
            current_score,
            candidate_score,
            best_score,
            best_step,
            "accept",
        )
    return GateDecision(
        False,
        "reject",
        current_score,
        candidate_score,
        best_score,
        best_step,
        "reject",
    )
