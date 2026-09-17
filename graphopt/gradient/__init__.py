"""GraphOpt Gradient — trajectory analysis and patch generation."""

from graphopt.gradient.aggregate import merge_patches  # noqa: F401
from graphopt.gradient.reflect import reflect, reflect_minibatch  # noqa: F401

__all__ = ["reflect", "reflect_minibatch", "merge_patches"]
