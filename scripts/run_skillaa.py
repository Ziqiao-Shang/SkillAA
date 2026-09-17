#!/usr/bin/env python3
"""Run the fixed SkillAA/GraphOpt SkillAA protocol.

Public surface: three benchmarks, seeds 42/43/44, and a single model pairing
(gpt-5.6-sol teacher + gpt-5.6-sol student).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_MODEL = "gpt-5.6-sol"
SEEDS = (42, 43, 44)
DATASETS = {
    "searchqa": {"train": 800, "test": 200},
    "docvqa": {"train": 800, "test": 200},
    "livemathematicianbench": {"train": 468, "test": 117},
}
REQUIRED_MODULES = ("omegaconf", "yaml", "openai", "json_repair")


def _load_local_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


def _read_rows(dataset: str, split: str) -> list[dict]:
    base = ROOT / "data" / dataset / split
    if dataset == "docvqa":
        path = base / "docvqa.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    path = base / "items.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path.relative_to(ROOT)}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON list")
    return rows


def _case_id(row: dict) -> str:
    return str(row.get("id") or row.get("questionId") or "").strip()


def validate_manifest(dataset: str) -> dict:
    path = ROOT / "data" / dataset / "split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = DATASETS[dataset]
    if manifest.get("protocol") != "skillaa_update_quadruples_v1":
        raise ValueError(f"{dataset}: wrong manifest protocol")
    if manifest.get("counts") != expected:
        raise ValueError(f"{dataset}: expected counts {expected}, got {manifest.get('counts')}")
    if manifest.get("test_used_for_grouping_training_or_gates") is not False:
        raise ValueError(f"{dataset}: held-out test isolation is not declared")
    policy = manifest.get("training_group_policy") or {}
    groups = list(policy.get("groups") or [])
    expected_groups = expected["train"] // 4
    if (
        policy.get("group_size") != 4
        or policy.get("update_per_group") != 4
        or policy.get("test_per_group") != 0
        or policy.get("group_count") != expected_groups
        or len(groups) != expected_groups
    ):
        raise ValueError(f"{dataset}: invalid update-quadruple policy")
    update_ids: list[str] = []
    forbidden = {"test_id", "test_ids", "question_ids", "test_question"}
    for index, group in enumerate(groups):
        leaked = forbidden.intersection(group)
        if leaked:
            raise ValueError(f"{dataset}: group {index} contains held-out fields {sorted(leaked)}")
        ids = [str(value) for value in group.get("train_ids") or []]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError(f"{dataset}: group {index} is not a distinct quadruple")
        update_ids.extend(ids)
    if len(update_ids) != len(set(update_ids)):
        raise ValueError(f"{dataset}: update IDs repeat across quadruples")
    test_ids = json.loads((ROOT / "data" / dataset / "test_ids.json").read_text(encoding="utf-8"))
    if len(test_ids) != expected["test"] or len(test_ids) != len(set(test_ids)):
        raise ValueError(f"{dataset}: invalid test ID list")
    if set(update_ids).intersection(test_ids):
        raise ValueError(f"{dataset}: update/test ID overlap")
    return {
        "manifest": manifest,
        "update_ids": update_ids,
        "test_ids": test_ids,
    }


def validate_data(dataset: str) -> dict:
    """Check only what is required to run the released framework safely."""
    declared = validate_manifest(dataset)
    train_rows = _read_rows(dataset, "train")
    test_rows = _read_rows(dataset, "test")
    train_ids = [_case_id(row) for row in train_rows]
    test_ids = [_case_id(row) for row in test_rows]
    expected = DATASETS[dataset]
    if len(train_ids) != expected["train"] or len(test_ids) != expected["test"]:
        raise ValueError(
            f"{dataset}: expected {expected}, got "
            f"train={len(train_ids)}, test={len(test_ids)}"
        )
    if any(not value for value in train_ids + test_ids):
        raise ValueError(f"{dataset}: empty case ID")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError(f"{dataset}: duplicate case ID")
    if set(train_ids) != set(declared["update_ids"]):
        raise ValueError(f"{dataset}: train IDs do not match the update quadruples")
    if set(train_ids).intersection(test_ids):
        raise ValueError(f"{dataset}: train/test overlap")
    if dataset == "docvqa":
        missing = []
        for row in train_rows + test_rows:
            raw = str(row.get("image_path") or "").strip()
            image = Path(raw)
            if not image.is_absolute():
                image = ROOT / image
            if not image.is_file():
                missing.append(raw or "<empty>")
        if missing:
            raise FileNotFoundError(
                f"docvqa: {len(missing)} images are missing; first={missing[0]!r}"
            )
    return {"dataset": dataset, "counts": {"train": len(train_ids), "test": len(test_ids)}}


def validate_dependencies() -> None:
    failures = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("dependency check failed: " + "; ".join(failures))


def build_command(dataset: str, seed: int, output_root: Path, *, resume: bool) -> list[str]:
    count = DATASETS[dataset]["train"]
    run_dir = output_root / dataset / f"seed{seed}"
    overrides = [
        f"train.seed={seed}",
        f"train.train_size={count}",
        "train.num_epochs=3",
        "train.grouped_batch_gate=true",
        "train.single_full_pool_input=true",
        "train.batch_size_scope=train",
        "train.shard_train_across_epochs=false",
        "gradient.update_protocol=case_complete",
        "gradient.node_support_threshold=1",
        "gradient.success_guard_workers=16",
        "evaluation.gate_metric=hard",
        "evaluation.max_big_gate_passes=1",
        "evaluation.use_big_gate_train_tiebreak=false",
        "evaluation.run_epoch_test=false",
        "evaluation.epoch_test_repeats=1",
        "evaluation.run_component_ablation_tests=false",
        "evaluation.final_test_repeats=1",
        "evaluation.test_repeat_seed_mode=fixed",
        "evaluation.eval_test=true",
        "evaluation.test_only=false",
        "env.no_validation_split=true",
        "env.no_test_split=false",
        "env.max_completion_tokens=16384",
        f"env.split_dir={ROOT / 'data' / dataset}",
        "model.reasoning_effort=medium",
        "model.teacher_request_timeout=300",
        "model.student_openlux_routing=nitro",
        "model.teacher_openlux_routing=nitro",
        "model.openlux_routing_style=suffix",
    ]
    if resume:
        overrides.append("train.resume=true")
    return [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config",
        str(ROOT / "configs" / dataset / "reasoning_trace.yaml"),
        "--teacher_model",
        FIXED_MODEL,
        "--llm",
        FIXED_MODEL,
        "--experiment_mode",
        "graphopt",
        "--ablation_mode",
        "g_full",
        "--out_root",
        str(run_dir),
        "--cfg-options",
        *overrides,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), required=True)
    parser.add_argument(
        "--seed",
        choices=(*(str(seed) for seed in SEEDS), "all"),
        default="all",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "skillaa")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-manifests", action="store_true", help="validate released IDs without raw benchmark payloads")
    parser.add_argument("--check-only", action="store_true", help="validate dependencies and fully materialized benchmark payloads")
    parser.add_argument("--print-command", action="store_true", help="print locked commands without running them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_local_env()
    datasets = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    seeds = SEEDS if args.seed == "all" else (int(args.seed),)

    manifest_reports = [
        {"dataset": dataset, "update": len(validate_manifest(dataset)["update_ids"]), "test": DATASETS[dataset]["test"]}
        for dataset in datasets
    ]
    if args.check_manifests:
        print(json.dumps({"ok": True, "manifests": manifest_reports}, indent=2))
        return

    if args.check_only:
        validate_dependencies()
        reports = [validate_data(dataset) for dataset in datasets]
        print(json.dumps({"ok": True, "model": FIXED_MODEL, "datasets": reports}, indent=2))
        return

    commands = [
        build_command(dataset, seed, args.output_root.resolve(), resume=args.resume)
        for dataset in datasets
        for seed in seeds
    ]
    if args.print_command:
        for command in commands:
            print(shlex.join(command))
        return

    validate_dependencies()
    for dataset in datasets:
        validate_data(dataset)
    if not (
        os.environ.get("OPENLUX_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ):
        raise RuntimeError("set OPENLUX_API_KEY in the environment or in .env")

    for command in commands:
        run_dir = Path(command[command.index("--out_root") + 1])
        if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
            raise FileExistsError(f"refusing to overwrite {run_dir}; use --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        print("[skillaa]", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
