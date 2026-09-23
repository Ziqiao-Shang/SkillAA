#!/usr/bin/env python3
"""GraphSkillAA training entry with SkillGraph state.

Usage
-----
    # Offline skeleton (no benchmark/API)
    python scripts/train.py --dry_run --out_root runs/dry

    # Full DocVQA experiment
    python scripts/train.py --config configs/docvqa/default.yaml

    # Override any YAML key
    python scripts/train.py --config configs/docvqa/default.yaml \\
        --cfg-options train.num_epochs=2 train.batch_size=9 optimizer.learning_rate=5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from graphopt.engine.trainer import Trainer
from graphopt.config import load_flat_config
from graphopt.runtime import (
    GRAPHOPT_ROOT,
    bounded_chat_callable,
    build_env_adapter,
    configure_models,
)


FIXED_MODEL = "gpt-5.6-sol"
FIXED_TEACHER_PROFILE = "teacher_gpt56"


def _dry_score(skill_text: str) -> float:
    t = skill_text.lower()
    s = 0.25
    if "container" in t:
        s += 0.2
    if "verification" in t or "verify" in t:
        s += 0.15
    if "open" in t and "container" in t:
        s += 0.1
    if "always open" in t or "reachable" in t:
        s += 0.1
    if "confirm goal" in t:
        s += 0.1
    if "depends on" in t:
        s += 0.05
    if "enhanced by" in t:
        s += 0.05
    return min(0.95, s)


def _dry_train_case(i: int, skill_text: str) -> dict:
    """Synthetic train case; G02 failures dominate so evolution can reach x=4."""
    st = skill_text.lower()
    bonus = 0.1 if "always open" in st or "open container" in st else 0.0
    kind = i % 5
    if kind in (0, 1, 2, 3):  # 80% share G02 failure pattern
        return {
            "id": f"train_{i:03d}",
            "hard": min(1.0, 0.0 + bonus),
            "soft": min(1.0, 0.0 + bonus),
            "task_type": "pick_and_place",
            "task_description": f"Put item {i} into fridge",
            "fail_reason": "Apple inside closed cabinet" if bonus < 0.05 else "",
            "trajectory": (
                "Step 0: Stage=locate | Active nodes=[G01, T01] | Active edges=[E001] | "
                "Obs: in kitchen | Rationale: search then deliver | Next action: go to cabinet 1\n"
                "Step 1: Stage=search | Active nodes=[G02] | Active edges=[E002] | "
                "Obs: cabinet 1 closed | Rationale: skipped open-before-take | Next action: take apple 1\n"
                "Episode Summary | Outcome: FAIL | Nodes correct: [G01] | Nodes wrong: [G02] | "
                "Edges wrong: [E002] | Decisive: took from closed cabinet"
            ),
        }
    return {
        "id": f"train_{i:03d}",
        "hard": 1.0,
        "soft": 1.0,
        "task_type": "pick_and_place",
        "task_description": f"Put egg {i} on table",
        "fail_reason": "",
        "trajectory": "find egg\ntake egg\nput egg\nSUCCESS",
    }


def dry_rollout(skill_text: str, split: str, cfg: dict):
    batch_n = max(1, int(cfg.get("batch_size") or 3))
    if split == "train":
        # Different synthetic rollout batches must still model distinct cases.
        # Validation keeps stable IDs because its split-specific seed is fixed.
        offset = int(cfg.get("seed") or 0) * batch_n
        return [_dry_train_case(offset + i, skill_text) for i in range(batch_n)]
    hard = _dry_score(skill_text)
    n = int(cfg.get("sel_env_num") or 2) or 2
    return [{"id": f"v{i}", "hard": hard, "soft": hard, "task_type": "pick_and_place"} for i in range(n)]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphSkillAA runner (fixed gpt-5.6-sol pairing)")
    p.add_argument("--config", type=str, default="", help="YAML config for one of the three supported benchmarks")
    p.add_argument("--cfg-options", nargs="+", default=[], help="section.key=value overrides")
    p.add_argument(
        "--llm",
        choices=[FIXED_MODEL],
        default=FIXED_MODEL,
        help="student model; fixed by this release",
    )
    p.add_argument(
        "--teacher_profile",
        choices=[FIXED_TEACHER_PROFILE],
        default=FIXED_TEACHER_PROFILE,
        help="teacher profile; fixed by this release",
    )
    p.add_argument(
        "--teacher_model",
        choices=[FIXED_MODEL],
        default=FIXED_MODEL,
        help="teacher model; fixed by this release",
    )
    p.add_argument(
        "--experiment_mode",
        choices=["graphopt"],
        default="graphopt",
        help="full GraphOpt training; fixed by this release",
    )
    p.add_argument(
        "--ablation_mode",
        choices=["g_full"],
        default="g_full",
        help="full method; ablations are outside this release",
    )
    p.add_argument("--dry_run", action="store_true", help="Synthetic rollouts; no benchmark/API")
    p.add_argument("--out_root", type=str, default="")
    p.add_argument("--init_graph", type=str, default="")
    p.add_argument("--reflect_mode", choices=["template", "teacher"], default="")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="dry-run only: synthetic rollout batches per epoch (real data derives ceil(N/batch_size))",
    )
    p.add_argument(
        "--accumulation",
        type=int,
        default=None,
        help="legacy compatibility option; GraphOpt updates once per complete epoch shard",
    )
    p.add_argument(
        "--max_ops",
        type=int,
        default=None,
        help="legacy alias: maximum support-ranked semantic update points per node",
    )
    p.add_argument(
        "--node_merge_top_k",
        type=int,
        default=None,
        help="maximum support-ranked semantic update points adopted for each node",
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--run_final_test", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.llm and args.llm != FIXED_MODEL:
        raise ValueError(f"this release fixes the student to {FIXED_MODEL}")
    if args.teacher_model and args.teacher_model != FIXED_MODEL:
        raise ValueError(f"this release fixes the teacher to {FIXED_MODEL}")
    if args.teacher_profile != FIXED_TEACHER_PROFILE:
        raise ValueError(f"this release fixes the teacher to {FIXED_MODEL}")
    selected_teacher_profile = FIXED_TEACHER_PROFILE
    per_node_top_k = args.node_merge_top_k if args.node_merge_top_k is not None else args.max_ops
    if per_node_top_k is not None and per_node_top_k < 1:
        raise ValueError("node_merge_top_k/--max_ops must be >= 1")

    if args.dry_run and not args.config:
        cfg = {
            "epochs": args.epochs or 1,
            "steps_per_epoch": args.steps_per_epoch or 2,
            "accumulation": args.accumulation or 1,
            "node_merge_top_k": args.node_merge_top_k or args.max_ops or 1,
            "batch_size": args.batch_size or 48,
            "sel_env_num": 2,
            "seed": 42,
            "run_final_test": args.run_final_test,
            "run_epoch_test": False,
            "epoch_test_repeats": 1,
            "final_test_repeats": 1,
            "skill_init": args.init_graph
            or str(
                GRAPHOPT_ROOT
                / "graphopt"
                / "envs"
                / "searchqa"
                / "initial_skill"
                / "best_graph.json"
            ),
            "out_root": args.out_root or str(GRAPHOPT_ROOT / "runs" / "dry"),
            "gate_metric": "hard",
            "gate_mixed_weight": 0.8,
            "delete_never_used_node": False,
            "delete_never_used_edge": False,
            "persist_proposal_pool": False,
            "execution_reinforcement_threshold": 1,
            "max_execution_child_candidates_per_epoch": 1,
            "max_active_execution_children_per_parent": 4,
            "execution_merge_evidence_cap": 64,
            "experiment_mode": args.experiment_mode or "graphopt",
            "ablation_mode": args.ablation_mode or "g_full",
            "teacher_profile": FIXED_TEACHER_PROFILE,
        }
        cfg["graph_init"] = cfg["skill_init"]
        trainer = Trainer(
            init_graph=cfg["skill_init"],
            out_root=cfg["out_root"],
            cfg=cfg,
            adapter=None,
            rollout_fn=dry_rollout,
            chat_fn=None,
            reflect_mode="template",
        )
        print(json.dumps(trainer.run(), ensure_ascii=False, indent=2))
        return

    config_path = args.config or str(GRAPHOPT_ROOT / "configs" / "searchqa" / "default.yaml")
    cfg_options = list(args.cfg_options or [])
    if args.llm:
        cfg_options.append(f"model.target={args.llm}")
    if selected_teacher_profile:
        cfg_options.append(f"model.teacher_profile={selected_teacher_profile}")
    cfg = load_flat_config(config_path, overrides=cfg_options or None)
    teacher = str(cfg.get("teacher_model_base") or "")
    student = str(cfg.get("student_model_base") or "")
    if teacher != FIXED_MODEL or student != FIXED_MODEL:
        raise ValueError(
            "GraphSkillAA requires teacher=student="
            f"{FIXED_MODEL}; resolved teacher={teacher!r}, student={student!r}"
        )

    # Default configs use ``case_complete``; the trainer retains the
    # historical spelling ``case_complete_v1`` for artifact compatibility.
    if str(cfg.get("update_protocol") or "").strip().lower() == "case_complete":
        cfg["update_protocol"] = "case_complete_v1"
    if str(os.environ.get("GRAPHOPT_UPDATE_PROTOCOL") or "").strip().lower() == "case_complete":
        os.environ["GRAPHOPT_UPDATE_PROTOCOL"] = "case_complete_v1"

    if args.out_root:
        cfg["out_root"] = os.path.abspath(args.out_root)
    if args.init_graph:
        cfg["skill_init"] = os.path.abspath(args.init_graph)
        cfg["graph_init"] = cfg["skill_init"]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
        cfg["num_epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        cfg["steps_per_epoch"] = args.steps_per_epoch
    if args.accumulation is not None:
        cfg["accumulation"] = args.accumulation
    if per_node_top_k is not None:
        cfg["node_merge_top_k"] = per_node_top_k
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.run_final_test:
        cfg["run_final_test"] = True
        cfg["eval_test"] = True
    if args.experiment_mode:
        cfg["experiment_mode"] = args.experiment_mode
    if args.ablation_mode:
        cfg["ablation_mode"] = args.ablation_mode

    reflect_mode = args.reflect_mode or ("template" if args.dry_run else "teacher")

    print(f"\n{'='*60}")
    print("  GraphSkillAA — fixed model configuration")
    print(f"{'='*60}")
    print(f"  config:         {config_path}")
    print(f"  skill_init:     {cfg.get('skill_init') or cfg.get('graph_init')}")
    print(f"  split_dir:      {cfg.get('split_dir')}")
    print(f"  epochs:         {cfg.get('epochs') or cfg.get('num_epochs')}")
    print(
        "  update input:   "
        + (
            "single complete train pool"
            if cfg.get("single_full_pool_input")
            else f"legacy grouped units (batch_size={cfg.get('batch_size')})"
        )
    )
    print(f"  accumulation:   {cfg.get('accumulation')} (legacy; grouped mode collects a frozen epoch before one synthesis)")
    print(f"  per-node top-K: {cfg.get('node_merge_top_k')} (all same-target epoch opinions are synthesized together; no patch cap)")
    print(f"  teacher model:  {cfg.get('optimizer_model')} (fixed; optimization only)")
    print(f"  teacher profile:{cfg.get('teacher_profile')}")
    print(f"  teacher timeout:{max(1, int(cfg.get('teacher_request_timeout') or 300))}s per request")
    print(f"  student model:  {cfg.get('target_model')} (rollout inference only)")
    print(f"  experiment:     {cfg.get('experiment_mode') or 'graphopt'}")
    print(f"  ablation:       {cfg.get('ablation_mode') or 'g_full'}")
    print(f"  out_root:       {cfg.get('out_root')}")
    print(f"  reflect_mode:   {reflect_mode}")
    print(f"{'='*60}\n")

    adapter = None
    chat_fn = None
    rollout_fn = None

    if args.dry_run:
        rollout_fn = dry_rollout
        reflect_mode = "template"
        # A configured formal train_size describes the real dataloader, not
        # synthetic batches. Dry-run must exercise only the requested number
        # of generated batches and must not fabricate formal shard sizes.
        cfg["train_size"] = 0
        cfg["shard_train_across_epochs"] = False
        cfg.setdefault("delete_never_used_node", False)
        cfg.setdefault("delete_never_used_edge", False)
        cfg.setdefault("steps_per_epoch", args.steps_per_epoch or 1)
        cfg.setdefault("epochs", args.epochs or cfg.get("num_epochs") or 1)
    else:
        configure_models(cfg)
        adapter = build_env_adapter(cfg)
        from graphopt.model import chat_optimizer

        chat_fn = (
            bounded_chat_callable(
                chat_optimizer,
                max(1, int(cfg.get("teacher_request_timeout") or 300)),
            )
            if reflect_mode == "teacher"
            else None
        )

    trainer = Trainer(
        init_graph=cfg.get("skill_init") or cfg["graph_init"],
        out_root=cfg["out_root"],
        cfg=cfg,
        adapter=adapter,
        rollout_fn=rollout_fn,
        chat_fn=chat_fn,
        reflect_mode=reflect_mode,
    )
    print(json.dumps(trainer.run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
