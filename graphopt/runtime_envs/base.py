"""Base interface for SkillAA benchmark adapters."""

from __future__ import annotations

from graphopt.runtime_envs.data import BatchSpec, SplitDataLoader


class EnvAdapter:
    def setup(self, cfg: dict) -> None:
        self._cfg = dict(cfg)

    def get_dataloader(self) -> SplitDataLoader | None:
        return None

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        del kwargs
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        loader = self.get_dataloader()
        if loader is None:
            raise RuntimeError("adapter has no dataloader")
        return self.build_env_from_batch(
            loader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs),
            **kwargs,
        )

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        loader = self.get_dataloader()
        if loader is None:
            raise RuntimeError("adapter has no dataloader")
        return self.build_env_from_batch(
            loader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs),
            **kwargs,
        )
