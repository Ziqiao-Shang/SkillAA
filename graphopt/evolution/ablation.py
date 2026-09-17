"""Alternative graph-update strategies used only by controlled ablations."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from graphopt.evolution.types import EvolutionResult, GraphEditPlan
from graphopt.debug.artifacts import save_llm_call
from graphopt.gradient.reflect import filter_edits
from graphopt.types import GraphPatch, SkillGraph

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _direct_raw_patch(
    graph: SkillGraph,
    raw_results: list[dict[str, Any]],
    *,
    chat_fn,
    mode: str,
    meta_context: str,
    strategy: str,
    store,
) -> GraphPatch:
    """Ask one teacher call to update directly from unanalysed raw rollouts."""
    if mode != "teacher" or chat_fn is None:
        return GraphPatch(reasoning=f"{strategy}_requires_teacher", edits=[])

    system = (PROMPTS / "ablation_direct_raw_update.md").read_text(encoding="utf-8")
    user = (
        (meta_context + "\n\n" if meta_context else "")
        + f"## Ablation strategy\n{strategy}\n\n"
        + "## Current complete SkillGraph JSON\n"
        + json.dumps(graph.to_dict(), ensure_ascii=False)
        + "\n\n## Raw rollout records selected for this direct update\n"
        + json.dumps(raw_results, ensure_ascii=False, indent=2)
    )
    response = ""
    error = ""
    current_user = user
    for attempt in range(1, 3):
        try:
            response, usage = chat_fn(
                system=system,
                user=current_user,
                max_completion_tokens=16384,
                retries=3,
                stage="ablation_direct_raw_update",
            )
            from graphopt.json_utils import extract_json

            obj = extract_json(response)
            if not isinstance(obj, dict) or set(obj) != {"reasoning", "edits"}:
                raise ValueError("output must contain exactly reasoning and edits")
            if not isinstance(obj["reasoning"], str) or not isinstance(obj["edits"], list):
                raise ValueError("reasoning must be a string and edits must be an array")
            patch = filter_edits(GraphPatch.from_dict(obj))
            if store is not None:
                save_llm_call(
                    store,
                    "direct_patch",
                    stage="ablation_direct_raw_update",
                    system=system,
                    user=current_user,
                    response=response,
                    usage=usage,
                    parsed=patch.to_dict(),
                )
            return patch
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            if store is not None:
                save_llm_call(
                    store,
                    "direct_patch",
                    stage="ablation_direct_raw_update",
                    system=system,
                    user=current_user,
                    response=response or None,
                    error=f"attempt {attempt}/2: {error}",
                )
            if attempt == 1:
                current_user = (
                    user
                    + "\n\n## Format Correction\nYour previous output was invalid: "
                    + error
                    + "\nReturn exactly one complete JSON object with only reasoning "
                    "and edits; do not omit the unchanged context or add Markdown."
                    + "\n\n## Invalid Previous Answer\n"
                    + response
                )
    return GraphPatch(reasoning=f"{strategy}_invalid_after_retry", edits=[])


def run_ablation_evolution(
    graph: SkillGraph,
    results: list[dict[str, Any]],
    *,
    strategy: str,
    cache,
    chat_fn,
    mode: str,
    meta_context: str,
    step: int,
    random_seed: int,
    random_sample_rate: float = 0.2,
    recorder=None,
) -> EvolutionResult:
    """Produce one direct raw-rollout patch without the evidence optimizer."""
    store = recorder.store if recorder is not None else None
    analyses: list[Any] = []

    if strategy == "raw_random_fraction":
        if not 0.0 < random_sample_rate <= 1.0:
            raise ValueError("random_sample_rate must be in (0, 1]")
        if not results:
            sampled: list[dict[str, Any]] = []
            patch = GraphPatch(reasoning="no_rollouts", edits=[])
        else:
            ordered = sorted(results, key=lambda row: str(row.get("id") or ""))
            sample_size = max(1, math.ceil(len(ordered) * random_sample_rate))
            sampled = random.Random(random_seed).sample(ordered, sample_size)
            patch = _direct_raw_patch(
                graph,
                sampled,
                chat_fn=chat_fn,
                mode=mode,
                meta_context=meta_context,
                strategy="A_random_fraction_of_raw_rollouts",
                store=store,
            )
        if store is not None:
            store.save(
                "ablation_update",
                stage="raw_random_fraction_update",
                inputs={
                    "strategy": strategy,
                    "population_size": len(results),
                    "sample_rate": random_sample_rate,
                    "sample_size": len(sampled),
                    "sampled_case_ids": [row.get("id") for row in sampled],
                    "seed": random_seed,
                    "epoch": step,
                },
                outputs=patch.to_dict(),
            )
    elif strategy == "raw_all_direct":
        if not results:
            patch = GraphPatch(reasoning="no_rollouts", edits=[])
        else:
            patch = _direct_raw_patch(
                graph,
                list(results),
                chat_fn=chat_fn,
                mode=mode,
                meta_context=meta_context,
                strategy="B_all_raw_rollouts_direct_best_patch",
                store=store,
            )
        if store is not None:
            store.save(
                "ablation_update",
                stage="raw_all_direct_update",
                inputs={
                    "strategy": strategy,
                    "population_size": len(results),
                    "case_ids": [row.get("id") for row in results],
                    "epoch": step,
                },
                outputs=patch.to_dict(),
            )
    else:
        raise ValueError(f"unknown ablation update strategy: {strategy!r}")

    return EvolutionResult(
        patch_edits=patch.edits,
        case_analyses=analyses,
        graph_statistics={},
        merged_proposals={},
        edit_plan=GraphEditPlan(),
        reasoning=patch.reasoning or strategy,
        cache=cache,
    )
