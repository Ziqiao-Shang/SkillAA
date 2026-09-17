"""Edge weight quintile assignment (co_occur vs prereq/enhance)."""

from __future__ import annotations

import random
from typing import Any

from graphopt.evolution.types import GraphEditPlan, UsageStats
from graphopt.types import SkillGraph, normalize_edge_type

# Binary relation strength: ordinary edges are omitted in prompts; only strong edges are marked.
CO_OCCUR_W = (0.55, 0.85)
STRUCTURAL_W = (0.55, 0.85)
WEIGHT_KEYS = ("medium", "strong")


def _w_to_key(value: float, *, scale: tuple[float, ...]) -> str:
    idx = min(len(WEIGHT_KEYS) - 1, max(0, len(scale) - 1))
    for i, thr in enumerate(scale):
        if value <= thr + 1e-9:
            idx = i
            break
    else:
        idx = len(WEIGHT_KEYS) - 1
    return WEIGHT_KEYS[idx]


def _quintile_assign(
    items: list[tuple[str, int]],
    *,
    scale: tuple[float, ...],
    rng: random.Random,
) -> dict[str, float]:
    """Rank by freq desc; ties shuffled randomly before quintile buckets."""
    if not items:
        return {}
    by_freq: dict[int, list[str]] = {}
    for eid, freq in items:
        by_freq.setdefault(freq, []).append(eid)
    ordered: list[tuple[str, int]] = []
    for freq in sorted(by_freq.keys(), reverse=True):
        group = list(by_freq[freq])
        rng.shuffle(group)
        ordered.extend((eid, freq) for eid in group)

    n = len(ordered)
    out: dict[str, float] = {}
    for i, (eid, _) in enumerate(ordered):
        # ``ordered`` is best evidence first, whereas ``scale`` is weak to
        # strong. Map the first quintile to the strongest bucket rather than
        # accidentally rewarding the weakest evidence with the largest weight.
        rank_bucket = min(len(scale) - 1, (i * len(scale)) // max(n, 1))
        bucket = len(scale) - 1 - rank_bucket
        out[eid] = scale[bucket]
    return out


def _structural_evidence_score(stats: UsageStats) -> int:
    """Rank relation reliability, not raw citation frequency alone."""
    # A missed-relevant relation was judged useful but failed retrieval. It is
    # positive structural evidence and should be made easier, not harder, to
    # retrieve. It is kept separate from `used` for deletion/stat reporting.
    return int(stats.correct) + int(stats.retrieval_missed) - int(stats.wrong)


def assign_edge_weights(
    graph: SkillGraph,
    plan: GraphEditPlan,
    edge_stats: dict[str, UsageStats],
    *,
    structural_min_used: int = 3,
    seed: int = 0,
) -> None:
    """Assign weight floats on plan.edge_weights keyed by edge id or new:* key."""
    del seed
    emap = {e.id: e for e in graph.edges if e.id}

    # Existing relations retain their initialized weight unless there is a
    # threshold-sized net correction. A retrieval miss moves one tier upward;
    # a wrong attribution moves one tier downward. Raw/correct usage alone is
    # never a reason to globally rerank otherwise stable relations.
    for eid, st in edge_stats.items():
        e = emap.get(eid)
        if not e:
            continue
        typ = normalize_edge_type(e.type)
        if typ not in ("co_occur", "prereq", "enhance"):
            continue
        net = int(st.retrieval_missed) - int(st.wrong)
        if abs(net) < structural_min_used:
            continue
        scale = CO_OCCUR_W if typ == "co_occur" else STRUCTURAL_W
        current_key = _w_to_key(float(e.w), scale=scale)
        index = WEIGHT_KEYS.index(current_key)
        next_index = min(len(scale) - 1, index + 1) if net > 0 else max(0, index - 1)
        new_weight = scale[next_index]
        if abs(new_weight - float(e.w)) > 1e-9:
            plan.edge_weights[eid] = new_weight

    for item in plan.add_edges + plan.replace_edges:
        ne = item.get("new_edge") or item
        src = ne.get("source")
        dst = ne.get("target")
        rel = normalize_edge_type(str(ne.get("relation") or "prereq"))
        if not src or not dst or rel not in ("prereq", "enhance"):
            continue
        key = f"new:{src}->{dst}:{rel}"
        support = int(ne.get("support") or item.get("support") or 0)
        if support < structural_min_used:
            continue
        # Binary tiers: one threshold-sized support is ordinary; two or more is strong.
        if support >= 2 * structural_min_used:
            plan.edge_weights[key] = STRUCTURAL_W[1]
        else:
            plan.edge_weights[key] = STRUCTURAL_W[0]


def weight_float_to_label(value: float, *, edge_type: str) -> str:
    typ = normalize_edge_type(edge_type)
    scale = CO_OCCUR_W if typ == "co_occur" else STRUCTURAL_W
    return _w_to_key(value, scale=scale)
