"""SearchQA single-turn rollout and evaluation."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from graphopt.model import chat_target
from graphopt.runtime_envs.rollout import load_system_prompt, run_parallel
from graphopt.runtime_envs.searchqa.evaluator import evaluate

_MAX_CONTEXT_CHARS = 6000


def _truncate_context(context: str) -> str:
    if len(context) <= _MAX_CONTEXT_CHARS:
        return context
    output = ""
    for document in context.split("[DOC]"):
        candidate = output + "[DOC]" + document if output else document
        if len(candidate) > _MAX_CONTEXT_CHARS:
            break
        output = candidate
    return output or context[:_MAX_CONTEXT_CHARS] + "\n...[truncated]"


def _build_user(
    item: dict,
    *,
    diagnostic_mode: bool,
    diagnostic_instruction: str,
    diagnostic_trace_context: str,
) -> str:
    parts = [
        f"## Context\n{_truncate_context(str(item.get('context') or ''))}",
        f"## Question\n{item['question']}",
    ]
    if diagnostic_trace_context.strip():
        parts.append(
            "## Previous Trace Snapshot\n"
            "Use this partial transcript as context for the current attempt.\n\n"
            + diagnostic_trace_context.strip()
        )
    if diagnostic_mode and diagnostic_instruction.strip():
        parts.append(f"## Training Readout\n{diagnostic_instruction.strip()}")
    return "\n\n".join(parts)


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
    exec_timeout: int = 120,
    max_completion_tokens: int = 16384,
    request_retries: int = 5,
) -> dict:
    case_id = str(item["id"])
    answers = list(item.get("answers") or [])
    result = {
        "id": case_id,
        "question": item["question"],
        "task_description": item["question"],
        "task_type": item.get("task_type") or "qa",
        "em": 0.0,
        "f1": 0.0,
        "sub_em": 0.0,
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "gold_answers": answers,
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }
    try:
        system = load_system_prompt("searchqa", skill_content)
        user = _build_user(
            item,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            diagnostic_trace_context=diagnostic_trace_context,
        )
        response = ""
        conversation: list[dict] = []
        for turn in range(max(1, int(max_turns))):
            prompt = user if turn == 0 else (
                f"Your previous answer was:\n{response}\n\n"
                "Review it against the context and question. If needed, correct it. "
                "Return the final answer inside <answer>...</answer>."
            )
            response, _ = chat_target(
                system=system,
                user=prompt,
                max_completion_tokens=max_completion_tokens,
                retries=max(1, int(request_retries)),
                stage="rollout",
                timeout=exec_timeout,
            )
            conversation.append({"type": "message", "turn": turn + 1, "content": response})
            if "<answer>" in response.lower():
                break

        score = evaluate(response, answers)
        result.update(
            {
                "em": score["em"],
                "f1": score["f1"],
                "sub_em": score["sub_em"],
                "hard": int(score["em"]),
                "soft": score["f1"],
                "predicted_answer": score["predicted_answer"],
                "response": response,
                "agent_ok": True,
                "n_turns": len(conversation),
            }
        )
        if not result["hard"]:
            result["fail_reason"] = (
                f"EM=0: predicted {score['predicted_answer']!r} but expected {answers!r}"
            )
        conversation.append(
            {
                "role": "system",
                "content": (
                    "[EVALUATION RESULT]\n"
                    f"Question: {item['question']}\n"
                    f"Predicted answer: {score['predicted_answer']!r}\n"
                    f"Gold answers: {answers!r}\n"
                    f"Exact Match: {score['em']}\nF1: {score['f1']:.4f}"
                ),
            }
        )
        pred_dir = Path(out_root) / "predictions" / case_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        (pred_dir / "target_system_prompt.txt").write_text(system, encoding="utf-8")
        (pred_dir / "target_user_prompt.txt").write_text(user, encoding="utf-8")
        (pred_dir / "conversation.json").write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        result["fail_reason"] = f"error: {exc}"
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    max_turns: int = 1,
    exec_timeout: int = 120,
    workers: int = 64,
    max_completion_tokens: int = 16384,
    request_retries: int = 5,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context_by_id: dict[str, str] | None = None,
    task_timeout: int = 600,
) -> list[dict]:
    del task_timeout
    trace_by_id = diagnostic_trace_context_by_id or {}

    def execute(item: dict) -> dict:
        return process_one(
            item,
            out_root,
            skill_content,
            max_turns=max_turns,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            diagnostic_trace_context=trace_by_id.get(str(item["id"]), ""),
            exec_timeout=exec_timeout,
            max_completion_tokens=max_completion_tokens,
            request_retries=request_retries,
        )

    return run_parallel(
        items=items,
        out_root=out_root,
        workers=workers,
        process_one=execute,
        label="searchqa",
    )
