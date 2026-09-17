"""GraphOpt Evaluation — validation gate for candidate graphs."""

from graphopt.evaluation.gate import (  # noqa: F401
    GateDecision,
    decide,
    normalize_results,
    score_results,
)

__all__ = ["GateDecision", "decide", "normalize_results", "score_results"]
