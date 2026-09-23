"""GraphSkillAA LiveMath adapter with OpenLux rollout isolation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from graphopt.runtime_envs.base import EnvAdapter
from graphopt.runtime_envs.livemathematicianbench.dataloader import (
    LiveMathematicianBenchDataLoader,
)
from graphopt.runtime_envs.livemathematicianbench.rollout import (
    run_batch as run_standard_batch,
)
from graphopt.runtime_envs.train_validation_split import enable_train_validation_loader

def _is_retryable_infrastructure_result(row: dict) -> bool:
    reason = str(row.get("fail_reason") or "").lower()
    if reason.startswith("task-timeout-"):
        return True
    if not (reason.startswith("error:") or reason.startswith("unexpected:")):
        return False
    return any(
        marker in reason
        for marker in (
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "http 429",
            "http 502",
            "http 503",
            "connection error",
            "connection reset",
            "connection aborted",
        )
    )


def _drop_retryable_infrastructure_results(out_dir: str) -> int:
    """Remove only infrastructure failures so resume reruns those case IDs."""
    path = Path(out_dir) / "results.jsonl"
    if not path.is_file():
        return 0
    kept: list[str] = []
    dropped = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            kept.append(raw_line)
            continue
        if _is_retryable_infrastructure_result(row):
            dropped += 1
        else:
            kept.append(json.dumps(row, ensure_ascii=False))
    if dropped:
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = "\n".join(kept)
        temporary.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        os.replace(temporary, path)
    return dropped


class LiveMathematicianBenchAdapter(EnvAdapter):
    """Keep LiveMath rollout policy local to this benchmark branch."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 600,
        workers: int = 64,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        shuffle_choices: bool = True,
        use_theorem: bool = False,
        use_sketch: bool = False,
        max_completion_tokens: int = 16384,
    ) -> None:
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.use_theorem = use_theorem
        self.use_sketch = use_sketch
        self.max_completion_tokens = int(max_completion_tokens)
        self.dataloader = LiveMathematicianBenchDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
            shuffle_choices=shuffle_choices,
        )

    def setup(self, cfg: dict) -> None:
        enable_train_validation_loader(self.dataloader, cfg)
        EnvAdapter.setup(self, cfg)
        self.dataloader.setup(cfg)
        target_provider = str(cfg.get("target_provider") or "openlux").lower()
        if target_provider != "openlux":
            raise ValueError("this release supports only the OpenLux gpt-5.6-sol target")
        if str(cfg.get("student_rollout_protocol") or "").strip():
            raise ValueError("custom student rollout protocols are outside this release")
        self.openlux_workers = max(1, int(cfg.get("openlux_workers") or 4))
        self.openlux_request_timeout = max(1, int(cfg.get("openlux_request_timeout") or 300))
        self.openlux_request_retries = max(1, int(cfg.get("openlux_request_retries") or 2))
        self.openlux_recovery_rounds = min(1, max(0, int(cfg.get("openlux_recovery_rounds") or 1)))
        self.openlux_recovery_backoff_seconds = max(
            0, int(cfg.get("openlux_recovery_backoff_seconds") or 30)
        )

    def get_dataloader(self):
        return self.dataloader

    def get_task_types(self) -> list[str]:
        return self.dataloader.get_task_types()

    def build_reference_text(self, item: dict) -> str:
        parts = []
        if str(item.get("theorem") or "").strip():
            parts.append(f"## Reference Theorem\n{item['theorem']}")
        if str(item.get("sketch") or "").strip():
            parts.append(f"## Reference Sketch\n{item['sketch']}")
        return "\n\n".join(parts)

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        items: list[dict] = env_manager
        if not items:
            raise RuntimeError(
                "LiveMath rollout received zero cases; check update-pool split routing"
            )
        provider_label = "openlux"
        batch_runner = run_standard_batch
        protocol_label = "standard"
        workers = min(self.workers, self.openlux_workers)
        request_timeout = self.openlux_request_timeout
        request_retries = self.openlux_request_retries
        recovery_rounds = self.openlux_recovery_rounds
        recovery_backoff = self.openlux_recovery_backoff_seconds
        task_timeout = (
            request_timeout * request_retries
            + sum(min(2 ** attempt, 30) for attempt in range(request_retries))
            + 60
        )

        retrying = _drop_retryable_infrastructure_results(out_dir)
        for recovery_round in range(recovery_rounds + 1):
            print(
                f"    [livemath {provider_label}] workers={workers} "
                f"protocol={protocol_label} "
                f"request_timeout={request_timeout}s retries={request_retries} "
                f"task_timeout={task_timeout}s recovery={recovery_round}/"
                f"{recovery_rounds} retrying_infra={retrying}",
                flush=True,
            )
            try:
                results = batch_runner(
                    items=items,
                    out_root=out_dir,
                    skill_content=skill_content,
                    max_turns=self.max_turns,
                    exec_timeout=request_timeout,
                    workers=workers,
                    max_completion_tokens=self.max_completion_tokens,
                    request_retries=request_retries,
                    use_theorem=self.use_theorem,
                    use_sketch=self.use_sketch,
                    diagnostic_mode=kwargs.get("diagnostic_mode", False),
                    diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
                    diagnostic_trace_context_by_id=kwargs.get(
                        "diagnostic_trace_context_by_id"
                    ),
                    task_timeout=task_timeout,
                )
            except Exception:
                retrying = _drop_retryable_infrastructure_results(out_dir)
                if retrying and recovery_round < recovery_rounds:
                    delay = min(
                        max(0, int(recovery_backoff)) * (2 ** recovery_round),
                        300,
                    )
                    if delay:
                        print(
                            f"    [livemath {provider_label}] infrastructure "
                            f"backoff {delay}s before recovery "
                            f"{recovery_round + 1}", flush=True,
                        )
                        time.sleep(delay)
                    continue
                raise
            retrying = _drop_retryable_infrastructure_results(out_dir)
            if not retrying:
                return results
            if recovery_round >= recovery_rounds:
                break
            delay = min(max(0, int(recovery_backoff)) * (2 ** recovery_round), 300)
            if delay:
                print(
                    f"    [livemath {provider_label}] infrastructure backoff "
                    f"{delay}s before recovery {recovery_round + 1}",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"LiveMath {provider_label} remained unavailable after "
            f"{recovery_rounds + 1} rollout attempts; "
            "infrastructure failures were removed and can be resumed safely"
        )


__all__ = ["LiveMathematicianBenchAdapter"]
