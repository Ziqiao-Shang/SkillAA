"""Dataset batch specifications and split-backed data loading."""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BatchSpec:
    phase: str
    split: str
    seed: int
    batch_size: int
    payload: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SplitDataLoader:
    """Deterministic loader for prepared train/test benchmark directories."""

    def __init__(
        self,
        *,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        **_: Any,
    ) -> None:
        self.split_dir = os.path.abspath(split_dir) if split_dir else ""
        self.data_path = data_path
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.split_seed = int(split_seed)
        self.split_output_dir = split_output_dir
        self.seed = int(seed)
        self.limit = max(0, int(limit))
        self._splits: dict[str, list[dict]] = {}

    def setup(self, cfg: dict) -> None:
        if cfg.get("split_dir"):
            self.split_dir = os.path.abspath(str(cfg["split_dir"]))
        if not self.split_dir:
            raise ValueError(f"{type(self).__name__} requires split_dir")
        self._load_all_splits()

    def _load_all_splits(self) -> None:
        self._splits = {}
        for name in ("train", "val", "test"):
            split_path = os.path.join(self.split_dir, name)
            if not os.path.isdir(split_path):
                raise ValueError(f"Missing {name!r} subdirectory in {self.split_dir}")
            items = self.load_split_items(split_path)
            self._splits[name] = items[: self.limit] if self.limit else items

    def load_split_items(self, split_path: str) -> list[dict]:
        files = sorted(glob.glob(os.path.join(split_path, "*.json")))
        if not files:
            raise FileNotFoundError(f"No .json file found in {split_path}")
        with open(files[0], encoding="utf-8") as handle:
            items = json.load(handle)
        if not isinstance(items, list):
            raise ValueError(f"Expected JSON array in {files[0]}")
        return items

    @property
    def train_items(self) -> list[dict]:
        return self._splits.get("train", [])

    @property
    def val_items(self) -> list[dict]:
        return self._splits.get("val", [])

    @property
    def test_items(self) -> list[dict]:
        return self._splits.get("test", [])

    def get_split_items(self, split: str) -> list[dict]:
        aliases = {"validation": "val", "valid": "val", "eval": "val"}
        return list(self._splits.get(aliases.get(split, split), []))

    def get_train_size(self) -> int:
        return len(self.train_items)

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **_: Any,
    ) -> list[BatchSpec]:
        items = list(self.train_items)
        random.Random(seed + epoch * 1000).shuffle(items)
        total_batches = max(0, int(steps_per_epoch) * int(accumulation))
        batches: list[BatchSpec] = []
        cursor = 0
        for index in range(total_batches):
            batch_items = items[cursor : cursor + batch_size]
            cursor += len(batch_items)
            if not batch_items and items:
                batch_items = list(items)
                random.Random(seed + epoch * 1000 + index + 1).shuffle(batch_items)
                batch_items = batch_items[:batch_size]
            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=seed + epoch * 1000 + index + 1,
                    batch_size=len(batch_items),
                    payload=batch_items,
                )
            )
        return batches

    def build_train_batch(self, batch_size: int, seed: int, **_: Any) -> BatchSpec:
        items = list(self.train_items)
        random.Random(seed).shuffle(items)
        items = items[:batch_size]
        return BatchSpec("train", "train", seed, len(items), items)

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **_: Any,
    ) -> BatchSpec:
        items = self.get_split_items(split)
        if env_num and env_num < len(items):
            items = items[:env_num]
        return BatchSpec("eval", split, seed, len(items), items)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state

    def set_out_root(self, out_root: str) -> None:
        del out_root
