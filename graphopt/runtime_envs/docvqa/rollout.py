"""DocVQA multimodal rollout and ANLS evaluation."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from graphopt.model import chat_target_messages
from graphopt.runtime_envs.docvqa.evaluator import evaluate
from graphopt.runtime_envs.rollout import load_system_prompt, run_parallel


def _image_to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_messages(
    item: dict,
    skill_content: str,
    image_detail: str,
    *,
    diagnostic_mode: bool,
    diagnostic_instruction: str,
) -> tuple[list[dict], str, str]:
    system = load_system_prompt("docvqa", skill_content)
    user = item["question"] + "\n\nReturn the final answer inside <answer>...</answer>."
    if diagnostic_mode and diagnostic_instruction.strip():
        user += f"\n\n## Training Readout\n{diagnostic_instruction.strip()}"
    image_url: dict[str, str] = {"url": _image_to_data_uri(item["image_path"])}
    if image_detail and image_detail != "auto":
        image_url["detail"] = image_detail
    return (
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": image_url},
                ],
            },
        ],
        system,
        user,
    )


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    exec_timeout: int = 120,
    image_detail: str = "auto",
    max_completion_tokens: int = 16384,
    request_retries: int = 5,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> dict:
    case_id = str(item["id"])
    answers = list(item.get("answers") or [])
    result = {
        "id": case_id,
        "question": item["question"],
        "task_type": item.get("subtask") or item.get("task_type") or "docvqa",
        "task_description": item["question"],
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
        "image_paths": item.get("image_paths") or [item.get("image_path")],
        "gold_answer": answers,
    }
    try:
        messages, system, user = _build_messages(
            item,
            skill_content,
            image_detail,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
        )
        response = ""
        conversation: list[dict] = [
            {
                "role": "user",
                "content": user + f"\n\n[image] {Path(item['image_path']).name}",
            }
        ]
        for turn in range(max(1, int(max_turns))):
            current_messages = messages if turn == 0 else [
                messages[0],
                messages[1],
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": (
                        "Review the same image carefully and answer again. "
                        "Keep the final answer inside <answer>...</answer>."
                    ),
                },
            ]
            response, _ = chat_target_messages(
                messages=current_messages,
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
                "predicted_answer": score["predicted_answer"],
                "hard": int(score["anls"] >= 0.999),
                "soft": score["anls"],
                "response": response,
                "agent_ok": True,
                "n_turns": len(conversation) - 1,
            }
        )
        if not result["hard"]:
            result["fail_reason"] = (
                f"predicted {score['predicted_answer']!r} but expected one of {answers!r}"
            )
        conversation.append(
            {
                "role": "system",
                "content": (
                    "[EVALUATION RESULT]\n"
                    f"Question: {item['question']}\n"
                    f"Predicted answer: {score['predicted_answer']!r}\n"
                    f"Gold answers: {answers!r}\nANLS: {score['anls']:.4f}"
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
    exec_timeout: int = 120,
    workers: int = 16,
    image_detail: str = "auto",
    max_completion_tokens: int = 16384,
    request_retries: int = 5,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    task_timeout: int = 600,
) -> list[dict]:
    del task_timeout

    def execute(item: dict) -> dict:
        return process_one(
            item,
            out_root,
            skill_content,
            max_turns=max_turns,
            exec_timeout=exec_timeout,
            image_detail=image_detail,
            max_completion_tokens=max_completion_tokens,
            request_retries=request_retries,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
        )

    return run_parallel(
        items=items,
        out_root=out_root,
        workers=workers,
        process_one=execute,
        label="docvqa",
    )
