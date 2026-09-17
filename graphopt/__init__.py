"""GraphOpt: SkillAA-style optimization over a SkillGraph state.

Pipeline stages (mirrors SkillAA):
  1. Rollout   — execute episodes with rendered subgraph skill text
  2. Reflect   — analyze trajectories, generate graph-edit patches
  3. Aggregate — merge patches
  4. Budget    — clip node/edge edit counts
  5. Update    — apply edits to SkillGraph
  6. Evaluate  — validation gate accept/reject
"""

from graphopt.types import Edge, GraphEdit, GraphPatch, SkillGraph, SkillNode

__all__ = [
    "Edge",
    "GraphEdit",
    "GraphPatch",
    "SkillGraph",
    "SkillNode",
]
