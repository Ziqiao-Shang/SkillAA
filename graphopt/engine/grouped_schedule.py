"""Deterministic fixed-group scheduling for GraphOpt.

Active SkillAA datasets preserve explicit update-only quadruples. The scheduler
places all four members in ``train_ids`` and keeps ``val_ids`` empty. Held-out
test examples are absent from this mapping and never enter training or either
Gate. No update case is duplicated or silently discarded.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainValidationGroup:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    group_type: str = ""
    tail: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "group_type": self.group_type,
            "tail": self.tail,
        }


@dataclass
class GroupedBatch:
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    groups: list[TrainValidationGroup] = field(default_factory=list)
    tail: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "train_size": len(self.train_ids),
            "validation_size": len(self.val_ids),
            "groups": [group.to_dict() for group in self.groups],
            "tail": self.tail,
        }


def validate_group_batch_size(batch_size: int, train_per_group: int = 3) -> int:
    value = int(batch_size)
    group_train = int(train_per_group)
    if group_train < 1 or value < group_train or value % group_train:
        raise ValueError(
            "GraphOpt grouped batch_size must be divisible by train_per_group; "
            f"got batch_size={value}, train_per_group={group_train}"
        )
    return value


def resolve_group_batch_sizes(
    batch_size: int,
    *,
    train_per_group: int = 3,
    scope: str = "train",
) -> tuple[int, int, int]:
    """Return ``(train, validation, total)`` for one grouped collection unit.

    ``scope=train`` preserves the historical API, where ``batch_size`` counted
    only training cases. ``scope=train_plus_validation`` is the final public
    protocol: the configured value counts the complete 3:1 update unit.
    """
    value = max(1, int(batch_size))
    group_train = max(1, int(train_per_group))
    normalized_scope = str(scope or "train").strip().lower()
    if normalized_scope == "train":
        train_size = validate_group_batch_size(value, group_train)
        validation_size = train_size // group_train
        return train_size, validation_size, train_size + validation_size
    if normalized_scope != "train_plus_validation":
        raise ValueError(
            "grouped batch_size_scope must be train or "
            f"train_plus_validation; got {scope!r}"
        )
    group_total = group_train + 1
    if value < group_total or value % group_total:
        raise ValueError(
            "GraphOpt total grouped batch_size must be divisible by "
            f"train_per_group + 1; got batch_size={value}, "
            f"train_per_group={group_train}"
        )
    groups = value // group_total
    return groups * group_train, groups, value


def _case_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "").strip()


def _case_type(item: dict[str, Any]) -> str:
    for key in (
        "task_type", "merged_type", "category", "question_type", "type",
        "family", "domain", "source",
    ):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return "__all__"


def _explicit_groups(split_dir: Path) -> list[TrainValidationGroup]:
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = str(manifest.get("protocol") or "")
    if protocol != "skillaa_update_quadruples_v1":
        return []
    active_policy = manifest.get("training_group_policy") or {}
    active_records = active_policy.get("groups") or []
    if active_records:
        groups: list[TrainValidationGroup] = []
        for record in active_records:
            train_ids = tuple(
                str(case_id) for case_id in record.get("train_ids") or []
            )
            if not train_ids:
                continue
            groups.append(TrainValidationGroup(
                train_ids=train_ids,
                val_ids=(),
                group_type=str(record.get("group_type") or ""),
            ))
        return groups
    return []


def build_grouped_schedule(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    *,
    split_dir: str | Path | None,
    batch_size: int | None,
    single_full_pool: bool = False,
    num_epochs: int,
    seed: int,
    train_size: int = 0,
) -> list[list[GroupedBatch]]:
    """Return epoch -> grouped batches without replacement.

    Each epoch covers the complete selected training pool. The manifest-owned
    group mapping is reshuffled per epoch and packed without splitting groups.
    Cases that cannot participate in a complete group join that epoch's final tail batch.
    """
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")

    train_by_id = {_case_id(item): item for item in train_items if _case_id(item)}
    val_by_id = {_case_id(item): item for item in val_items if _case_id(item)}
    target_train = int(train_size or len(train_by_id))
    if target_train > len(train_by_id):
        raise ValueError(
            f"configured train_size={target_train} exceeds loaded train cases={len(train_by_id)}"
        )

    explicit = _explicit_groups(Path(split_dir)) if split_dir else []
    train_per_group = len(explicit[0].train_ids) if explicit else 3
    if single_full_pool:
        effective_group_capacity = max(1, target_train // train_per_group)
    else:
        if batch_size is None:
            raise ValueError("legacy grouped scheduling requires batch_size")
        batch_size = validate_group_batch_size(batch_size, train_per_group)
        effective_group_capacity = batch_size // train_per_group
    if explicit and any(len(group.train_ids) != train_per_group for group in explicit):
        raise ValueError("explicit training groups have inconsistent sizes")
    groups: list[TrainValidationGroup] = []
    used_train: set[str] = set()
    used_val: set[str] = set()
    if explicit:
        for group in explicit:
            if not all(case_id in train_by_id for case_id in group.train_ids):
                continue
            if not all(case_id in val_by_id for case_id in group.val_ids):
                continue
            if len(used_train) + len(group.train_ids) > target_train:
                break
            groups.append(group)
            used_train.update(group.train_ids)
            used_val.update(group.val_ids)
    else:
        train_by_type: dict[str, list[str]] = {}
        val_by_type: dict[str, list[str]] = {}
        for case_id, item in train_by_id.items():
            train_by_type.setdefault(_case_type(item), []).append(case_id)
        for case_id, item in val_by_id.items():
            val_by_type.setdefault(_case_type(item), []).append(case_id)
        rng = random.Random(seed + 17041)
        for values in train_by_type.values():
            rng.shuffle(values)
        for values in val_by_type.values():
            rng.shuffle(values)
        for group_type in sorted(val_by_type):
            train_ids = train_by_type.get(group_type, [])
            val_ids = val_by_type[group_type]
            n_groups = min(len(train_ids) // train_per_group, len(val_ids))
            for index in range(n_groups):
                group_train_ids = tuple(
                    train_ids[index * train_per_group:(index + 1) * train_per_group]
                )
                if len(used_train) + train_per_group > target_train:
                    break
                group = TrainValidationGroup(
                    train_ids=group_train_ids,
                    val_ids=(val_ids[index],),
                    group_type=group_type,
                )
                groups.append(group)
                used_train.update(group_train_ids)
                used_val.add(val_ids[index])

    # If type-local matching leaves capacity, pair remaining cases globally.
    remaining_train = [
        case_id for case_id in train_by_id if case_id not in used_train
    ][: max(0, target_train - len(used_train))]
    remaining_val = [case_id for case_id in val_by_id if case_id not in used_val]
    rng = random.Random(seed + 29033)
    rng.shuffle(remaining_train)
    rng.shuffle(remaining_val)
    n_global = min(len(remaining_train) // train_per_group, len(remaining_val))
    for index in range(n_global):
        group_train_ids = tuple(
            remaining_train[index * train_per_group:(index + 1) * train_per_group]
        )
        val_id = remaining_val[index]
        groups.append(
            TrainValidationGroup(group_train_ids, (val_id,), "__nearest__")
        )
        used_train.update(group_train_ids)
        used_val.add(val_id)

    tail_train = [
        case_id for case_id in train_by_id
        if case_id not in used_train
    ][: max(0, target_train - len(used_train))]
    tail_val = [case_id for case_id in val_by_id if case_id not in used_val]

    groups_per_batch = effective_group_capacity
    schedule: list[list[GroupedBatch]] = []
    for epoch_index in range(num_epochs):
        epoch_groups = list(groups)
        random.Random(seed + 1000 + epoch_index).shuffle(epoch_groups)
        batches: list[GroupedBatch] = []
        for start in range(0, len(epoch_groups), groups_per_batch):
            selected = epoch_groups[start:start + groups_per_batch]
            batches.append(GroupedBatch(
                train_ids=[case_id for group in selected for case_id in group.train_ids],
                val_ids=[case_id for group in selected for case_id in group.val_ids],
                groups=list(selected),
            ))
        if not batches:
            batches.append(GroupedBatch())
        if tail_train or tail_val:
            tail_group = TrainValidationGroup(
                tuple(tail_train), tuple(tail_val), "__tail__", tail=True
            )
            final = batches[-1]
            final.train_ids.extend(tail_train)
            final.val_ids.extend(tail_val)
            final.groups.append(tail_group)
            final.tail = True
        schedule.append(batches)

    for epoch_index, epoch_batches in enumerate(schedule, start=1):
        scheduled_train = [
            case_id for batch in epoch_batches for case_id in batch.train_ids
        ]
        scheduled_val = [
            case_id for batch in epoch_batches for case_id in batch.val_ids
        ]
        if len(scheduled_train) != target_train or len(set(scheduled_train)) != target_train:
            raise ValueError(
                f"epoch {epoch_index} must include each selected train case exactly once"
            )
        if len(scheduled_val) != len(val_by_id) or len(set(scheduled_val)) != len(val_by_id):
            raise ValueError(
                f"epoch {epoch_index} must include each validation case exactly once"
            )
    return schedule
