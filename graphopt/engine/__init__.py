"""GraphOpt Engine — training runner.

Orchestrates rollout, reflect, aggregate, budget clip, apply, and gate.
"""

from graphopt.engine.trainer import Trainer  # noqa: F401

__all__ = ["Trainer"]
