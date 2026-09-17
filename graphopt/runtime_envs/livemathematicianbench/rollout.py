"""LiveMathematicianBench multiple-choice rollout and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from graphopt.model import chat_target
from graphopt.runtime_envs.livemathematicianbench.evaluator import evaluate
from graphopt.runtime_envs.rollout import load_system_prompt, run_parallel


def _format_choices(choices: list[dict]) -> str:
    return "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)


def _build_user(
    item: dict,
    *,
    use_theorem: bool,
    use_sketch: bool,
    diagnostic_mode: bool,
    diagnostic_instruction: str,
    diagnostic_trace_context: str,
) -> str:
    parts = [
        f"## Question\n{item['question']}",
        f"## Choices\n{_format_choices(item['choices'])}",
    ]
    if use_theorem and item.get("theorem"):
        parts.append(f"## Theorem\n{item['theorem']}")
    if use_sketch and item.get("sketch"):
        parts.append(f"## Proof Sketch\n{item['sketch']}")
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
    use_theorem: bool = False,
    use_sketch: bool = False,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
    exec_timeout: int = 300,
    max_completion_tokens: int = 16384,
    request_retries: int = 5,
) -> dict:
    case_id = str(item["id"])
    correct = item["correct_choice"]
    theorem_type = item.get("theorem_type") or []
    result = {
        "id": case_id,
        "question": item["question"],
        "task_description": item["question"],
        "task_type": theorem_type[0] if theorem_type else "math_mcq",
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "predicted_label": "",
        "predicted_text": "",
        "correct_label": correct["label"],
        "correct_text": correct["text"],
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }
    try:
        system = load_system_prompt("livemathematicianbench", skill_content)
        user = _build_user(
            item,
            use_theorem=use_theorem,
            use_sketch=use_sketch,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            diagnostic_trace_context=diagnostic_trace_context,
        )
        response = ""
        conversation: list[dict] = []
        for turn in range(max(1, int(max_turns))):
            prompt = user if turn == 0 else (
                f"Your previous answer was:\n{response}\n\n"
                "Re-evaluate the exact option wording. Output only the final "
                "choice label inside <answer>...</answer>."
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

        score = evaluate(response, correct, item["choices"])
        result.update(
            {
                "hard": int(score["em"]),
                "soft": score["f1"],
                "predicted_answer": score["predicted_answer"],
                "predicted_label": score["predicted_label"],
                "predicted_text": score["predicted_text"],
                "response": response,
                "agent_ok": True,
                "n_turns": len(conversation),
            }
        )
        if not result["hard"]:
            predicted = score["predicted_label"] or score["predicted_answer"]
            result["fail_reason"] = (
                f"MCQ=0: predicted {predicted!r} but expected {score['correct_label']!r}"
            )
        conversation.append(
            {
                "role": "system",
                "content": (
                    "[EVALUATION RESULT]\n"
                    f"Question: {item['question']}\n"
                    f"Predicted label: {score['predicted_label']!r}\n"
                    f"Predicted text: {score['predicted_text']!r}\n"
                    f"Correct label: {score['correct_label']!r}\n"
                    f"Correct text: {score['correct_text']!r}\n"
                    f"Exact Match: {score['em']}"
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
    *,
    max_turns: int = 1,
    exec_timeout: int = 300,
    workers: int = 64,
    max_completion_tokens: int = 16384,
    request_retries: int = 5,
    use_theorem: bool = False,
    use_sketch: bool = False,
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
            use_theorem=use_theorem,
            use_sketch=use_sketch,
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
        label="livemath",
    )
