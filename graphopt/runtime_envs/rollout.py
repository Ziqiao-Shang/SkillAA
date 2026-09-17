"""Shared resume-aware parallel rollout utilities."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable


def load_system_prompt(environment: str, skill_content: str) -> str:
    path = Path(__file__).resolve().parent / environment / "rollout_system.md"
    template = path.read_text(encoding="utf-8")
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return template.format(skill_section=skill_section)


def run_parallel(
    *,
    items: list[dict],
    out_root: str,
    workers: int,
    process_one: Callable[[dict], dict],
    label: str,
) -> list[dict]:
    """Execute pending items, append durable JSONL rows, and preserve input order."""
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.jsonl"
    latest: dict[str, dict] = {}
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                latest[str(row["id"])] = row
            except Exception:
                continue

    requested = [str(item["id"]) for item in items]
    complete = {
        case_id
        for case_id in requested
        if case_id in latest and latest[case_id].get("agent_ok") is not False
    }
    pending = [item for item in items if str(item["id"]) not in complete]
    if complete:
        print(f"    [{label}] resuming: {len(complete)}/{len(items)} already done", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(pending) or 1))) as pool:
        future_to_item = {pool.submit(process_one, item): item for item in pending}
        for index, future in enumerate(as_completed(future_to_item), start=1):
            item = future_to_item[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "id": str(item["id"]),
                    "question": item.get("question", ""),
                    "hard": 0,
                    "soft": 0.0,
                    "agent_ok": False,
                    "fail_reason": f"unexpected: {exc}",
                }
            latest[str(row["id"])] = row
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"    [{label}] {len(complete) + index}/{len(items)} "
                f"id={row['id']} hard={row.get('hard', 0)}",
                flush=True,
            )

    results = [latest[case_id] for case_id in requested if case_id in latest]
    if results and all(row.get("agent_ok") is False for row in results):
        reason = str(results[0].get("fail_reason") or "unknown error")
        raise RuntimeError(
            f"{label} rollout failed for all {len(results)} items before an agent response: {reason}"
        )
    return results
