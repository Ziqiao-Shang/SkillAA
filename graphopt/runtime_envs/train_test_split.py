"""Validation for the final GraphSkillAA train/test layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "searchqa": {"train": 800, "test": 200},
    "docvqa": {"train": 800, "test": 200},
    "livemathematicianbench": {"train": 468, "test": 117},
}


def validate_train_test_dataset(
    adapter: Any,
    env_name: str,
    split_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Fail closed unless train is complete and test is strictly held out."""
    if (
        manifest.get("protocol") != "graphskillaa_update_quadruples_v1"
        or manifest.get("split_layout") != ["train", "test"]
        or manifest.get("no_validation_split") is not True
        or manifest.get("source_train_and_validation_form_update_pool") is not True
        or manifest.get("held_out_test") is not True
        or manifest.get("test_used_for_grouping_training_or_gates") is not False
        or (split_dir / "val").exists()
    ):
        raise ValueError(f"{env_name} must use the final train/test-only layout")

    loader = adapter.get_dataloader()
    pools = {
        "train": list(getattr(loader, "train_items", []) or []),
        "val": list(getattr(loader, "val_items", []) or []),
        "test": list(getattr(loader, "test_items", []) or []),
    }
    expected = EXPECTED_COUNTS[env_name]
    loaded = {"train": len(pools["train"]), "test": len(pools["test"])}
    if loaded != expected or manifest.get("counts") != expected or pools["val"]:
        raise ValueError(
            f"{env_name} train/test counts mismatch: expected={expected}, "
            f"loaded={loaded}, validation={len(pools['val'])}"
        )

    ids = {
        name: [str(item.get("id") or "") for item in pools[name]]
        for name in ("train", "test")
    }
    if (
        any(not case_id for values in ids.values() for case_id in values)
        or len(ids["train"]) != len(set(ids["train"]))
        or len(ids["test"]) != len(set(ids["test"]))
        or set(ids["train"]).intersection(ids["test"])
    ):
        raise ValueError(f"{env_name} train/test IDs are empty, duplicated, or overlapping")

    policy = manifest.get("training_group_policy") or {}
    groups = list(policy.get("groups") or [])
    if (
        int(policy.get("group_size", -1)) != 4
        or int(policy.get("update_per_group", -1)) != 4
        or int(policy.get("validation_per_group", -1)) != 0
        or int(policy.get("test_per_group", -1)) != 0
        or int(policy.get("group_count", -1)) != expected["train"] // 4
        or len(groups) != expected["train"] // 4
    ):
        raise ValueError(f"{env_name} does not declare complete update-only quadruples")

    assigned_train: list[str] = []
    for group in groups:
        train_ids = [str(value) for value in group.get("train_ids") or []]
        if set(group).intersection({"test_id", "test_ids", "question_ids"}):
            raise ValueError(f"{env_name} update group contains held-out test metadata")
        if len(train_ids) != 4 or len(set(train_ids)) != 4:
            raise ValueError(f"{env_name} contains an invalid update quadruple")
        assigned_train.extend(train_ids)
    if (
        len(assigned_train) != len(set(assigned_train))
        or set(assigned_train) != set(ids["train"])
    ):
        raise ValueError(f"{env_name} update groups do not cover train exactly once")

    for split in ("train", "test"):
        for item in pools[split]:
            absent: list[str] = []
            if not str(item.get("id") or "").strip():
                absent.append("id")
            if not str(item.get("question") or "").strip():
                absent.append("question")
            if env_name == "searchqa":
                if not str(item.get("context") or "").strip():
                    absent.append("context")
                if not list(item.get("answers") or []):
                    absent.append("answers")
            elif env_name == "docvqa":
                if not list(item.get("answers") or []):
                    absent.append("answers")
                if not Path(str(item.get("image_path") or "")).is_file():
                    absent.append("image_path")
            else:
                if not list(item.get("choices") or []):
                    absent.append("choices")
                correct = item.get("correct_choice") or {}
                if not isinstance(correct, dict) or not str(correct.get("label") or "").strip():
                    absent.append("correct_choice.label")
            if absent:
                raise ValueError(
                    f"{env_name}/{split}/{item.get('id')}: missing {','.join(absent)}"
                )
