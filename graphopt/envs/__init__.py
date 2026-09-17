"""Frozen prompt assets plus compatibility aliases for the 12:39 trainer."""

from __future__ import annotations

import importlib
import sys


# The historical trainer imports ``graphopt.envs.<dataset>.pipeline``. Runtime
# modules were later moved out of the frozen prompt-asset directories. Keep the
# trainer byte-identical while routing those old module names to their current
# isolated locations; no executable file is added to a dataset asset folder.
for _dataset in ("searchqa", "docvqa", "livemathematicianbench"):
    _legacy_name = f"{__name__}.{_dataset}.pipeline"
    if _legacy_name not in sys.modules:
        sys.modules[_legacy_name] = importlib.import_module(
            f"graphopt.runtime_envs.{_dataset}.pipeline"
        )

del _dataset, _legacy_name
