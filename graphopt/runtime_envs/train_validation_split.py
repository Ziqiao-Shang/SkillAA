"""Loader support for the released SkillAA dataset layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MethodType
from typing import Any


UPDATE_QUADRUPLE_PROTOCOL = "skillaa_update_quadruples_v1"


def enable_train_validation_loader(loader: Any, cfg: dict[str, Any]) -> None:
    """Install the active split loader (historical name retained as an API alias)."""
    split_dir = Path(str(cfg.get("split_dir") or "")).expanduser()
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = str(manifest.get("protocol") or "")
    if protocol != UPDATE_QUADRUPLE_PROTOCOL:
        return

    def _load_active_skillaa(self) -> None:
        self._splits = {}
        for name in ("train", "test"):
            split_path = os.path.join(self.split_dir, name)
            if not os.path.isdir(split_path):
                raise ValueError(f"Missing {name!r} subdirectory in {self.split_dir}")
            items = self.load_split_items(split_path)
            if self.limit:
                items = items[: self.limit]
            self._splits[name] = items
        self._splits["val"] = []
        print(
            f"  [{type(self).__name__}] train={len(self.train_items)} "
            f"val={len(self.val_items)} test={len(self.test_items)} "
            f"protocol={protocol} (from {self.split_dir})"
        )

    loader._load_all_splits = MethodType(_load_active_skillaa, loader)
