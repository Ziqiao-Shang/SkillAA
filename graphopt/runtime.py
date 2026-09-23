"""GraphSkillAA runtime: configuration, model setup, and benchmark adapters."""

from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

GRAPHOPT_ROOT = Path(__file__).resolve().parents[1]


def bounded_chat_callable(chat_fn, timeout_seconds: int):
    """Attach an explicit per-request timeout to every optimizer call.

    Call sites may still choose a smaller timeout explicitly. This closes the
    previous gap where student rollouts were bounded but teacher
    analysis/synthesis inherited the OpenAI client's much longer default.
    """
    timeout = max(1, int(timeout_seconds))

    @wraps(chat_fn)
    def call(*args: Any, **kwargs: Any):
        bounded = dict(kwargs)
        bounded.setdefault("timeout", timeout)
        # The OpenAI SDK already retries transient transport failures. Avoid
        # multiplying those retries by the backend's legacy five-attempt loop.
        bounded.setdefault("retries", 1)
        return chat_fn(*args, **bounded)

    return call


def _load_project_env() -> None:
    """Load ignored project-local defaults without exposing their values."""
    path = GRAPHOPT_ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


_load_project_env()

# OpenLux — target/student rollout and optimizer/teacher calls both go through here,
# but use separate deployments.
OPENLUX_BASE_URL = "https://api.openlux.ai/v1"
OPENLUX_API_VERSION = "openai-compat"
OPENLUX_AUTH_MODE = "openai_compatible"
# This release intentionally exposes one OpenLux chat model for both roles.
OPENLUX_MODELS: tuple[str, ...] = ("gpt-5.6-sol",)
OPENLUX_MODEL_ALIASES: dict[str, str] = {}
OPENLUX_DEPLOYMENT_ALIASES: dict[str, str] = {}
OPENLUX_NON_CHAT_MODELS: frozenset[str] = frozenset()
DEFAULT_OPENLUX_MODEL = "gpt-5.6-sol"
# One fixed pairing is part of the public experiment contract.
TEACHER_PROFILES: dict[str, str] = {
    "teacher_gpt56": "gpt-5.6-sol",
}
DIRECT_TEACHER_PROFILES: dict[str, str] = {
    "gpt-5.6-sol": "teacher_gpt56",
}
DEFAULT_TEACHER_PROFILE = "teacher_gpt56"
FIXED_TEACHER_MODEL = "gpt-5.6-sol"

from graphopt.openlux import (
    DEFAULT_OPENLUX_ROUTING,
    DEFAULT_OPENLUX_ROUTING_STYLE,
    install_openlux_provider_sort_hook,
    nitro_deployed_model,
    resolve_openlux_request,
    strip_openlux_routing_suffix,
)

# Diagnostic :nitro deployment names for the supported student models.
# Formal routing is resolved per experiment and student; model.target artifacts retain the base ID.
OPENLUX_MODELS_NITRO: dict[str, str] = {
    model: nitro_deployed_model(OPENLUX_DEPLOYMENT_ALIASES.get(model, model))
    for model in OPENLUX_MODELS
}

# Model-wide routing policy.  These overrides apply to both student and
# teacher roles and intentionally take precedence over table-level routing.
OPENLUX_MODEL_ROUTING_OVERRIDES: dict[str, str] = {}


def normalize_openlux_model(model: str) -> str:
    """Strip routing suffix; map legacy IDs to OpenLux doc IDs."""
    base = strip_openlux_routing_suffix(str(model).strip())
    canonical = OPENLUX_MODEL_ALIASES.get(base, base)
    if canonical != base:
        print(f"  [model] note: {base!r} → canonical OpenLux ID {canonical!r}")
    return canonical


def apply_teacher_student_models(flat: dict[str, Any]) -> tuple[str, str]:
    """Enforce the fixed experiment contract: both roles use gpt-5.6-sol."""
    profile = str(flat.get("teacher_profile") or DEFAULT_TEACHER_PROFILE).strip()
    if profile != DEFAULT_TEACHER_PROFILE:
        raise ValueError(
            f"this release supports only teacher_profile={DEFAULT_TEACHER_PROFILE!r}"
        )
    requested_teacher = normalize_openlux_model(
        str(flat.get("optimizer_model") or FIXED_TEACHER_MODEL)
    )
    requested_student = normalize_openlux_model(
        str(
            flat.get("target_model")
            or flat.get("llm_model")
            or flat.get("model_llm")
            or flat.get("llm")
            or DEFAULT_OPENLUX_MODEL
        )
    )
    if requested_teacher != FIXED_TEACHER_MODEL or requested_student != DEFAULT_OPENLUX_MODEL:
        raise ValueError(
            "this release requires teacher=student=gpt-5.6-sol; "
            f"received teacher={requested_teacher!r}, student={requested_student!r}"
        )
    flat["teacher_profile"] = DEFAULT_TEACHER_PROFILE
    flat["optimizer_model"] = FIXED_TEACHER_MODEL
    flat["target_model"] = DEFAULT_OPENLUX_MODEL
    flat["llm_model"] = DEFAULT_OPENLUX_MODEL
    return FIXED_TEACHER_MODEL, DEFAULT_OPENLUX_MODEL


def apply_openlux_config(flat: dict[str, Any]) -> str:
    """Resolve the fixed gpt-5.6-sol teacher/student OpenLux deployments."""
    flat.setdefault("model_backend", "azure_openai")
    flat.setdefault("optimizer_backend", "openai_chat")
    flat.setdefault("target_backend", "openai_chat")
    flat.setdefault("azure_openai_endpoint", OPENLUX_BASE_URL)
    flat.setdefault("azure_openai_api_version", OPENLUX_API_VERSION)
    flat.setdefault("azure_openai_auth_mode", OPENLUX_AUTH_MODE)
    flat.setdefault("openlux_routing", DEFAULT_OPENLUX_ROUTING)
    flat.setdefault("openlux_routing_style", DEFAULT_OPENLUX_ROUTING_STYLE)
    flat.setdefault("student_openlux_routing", flat["openlux_routing"])
    flat.setdefault("teacher_openlux_routing", flat["openlux_routing"])

    if not flat.get("azure_openai_api_key"):
        flat["azure_openai_api_key"] = (
            os.environ.get("OPENLUX_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )

    teacher_base, student_base = apply_teacher_student_models(flat)
    style = flat.get("openlux_routing_style", DEFAULT_OPENLUX_ROUTING_STYLE)
    student_routing = flat.get("student_openlux_routing", flat["openlux_routing"])
    teacher_routing = flat.get("teacher_openlux_routing", flat["openlux_routing"])
    teacher_deployed, teacher_provider_sort = resolve_openlux_request(
        teacher_base, routing=teacher_routing, style=style
    )
    student_deployed, student_provider_sort = resolve_openlux_request(
        student_base, routing=student_routing, style=style
    )
    if teacher_provider_sort != student_provider_sort and (
        teacher_provider_sort is not None or student_provider_sort is not None
    ):
        raise ValueError(
            "role-specific OpenLux routing requires suffix style; one global "
            "provider-sort hook cannot route teacher and student differently"
        )

    flat["optimizer_model"] = teacher_deployed
    flat["target_model"] = student_deployed
    flat["llm_model"] = student_deployed
    flat["teacher_model_base"] = teacher_base
    flat["student_model_base"] = student_base
    flat["openlux_model_base"] = student_base
    flat["openlux_routing"] = student_routing
    flat["student_openlux_routing"] = student_routing
    flat["teacher_openlux_routing"] = teacher_routing
    flat["openlux_provider_sort"] = student_provider_sort
    flat["target_provider"] = "openlux"
    flat["target_api_style"] = "chat_completions"
    flat["target_enable_thinking"] = False
    return student_deployed


def resolve_path(path: str | Path, *, bases: list[Path] | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    for base in bases or [GRAPHOPT_ROOT, Path.cwd()]:
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    # Prefer graphopt-relative even if missing (for clearer errors later)
    return (GRAPHOPT_ROOT / p).resolve()


def load_flat_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load YAML and flatten it to the trainer configuration dictionary."""
    from graphopt.config_loader import flatten_config, is_structured, load_config

    cfg = load_config(str(config_path), overrides=overrides)
    flat = flatten_config(cfg) if is_structured(cfg) else dict(cfg)

    # The compact flatten map omits evolution knobs, so copy them through.
    if is_structured(cfg):
        for section in ("gradient", "optimizer", "train", "evaluation"):
            block = cfg.get(section)
            if isinstance(block, dict):
                for key, val in block.items():
                    flat.setdefault(key, val)
        model_block = cfg.get("model")
        if isinstance(model_block, dict):
            if model_block.get("teacher_profile"):
                flat["teacher_profile"] = model_block["teacher_profile"]
            if model_block.get("llm"):
                flat["llm_model"] = model_block["llm"]
                # Backward compatibility: model.llm now selects only the student.
                flat["target_model"] = model_block["llm"]
            if model_block.get("openlux_routing") is not None:
                flat["openlux_routing"] = model_block["openlux_routing"]
            if model_block.get("openlux_routing_style") is not None:
                flat["openlux_routing_style"] = model_block["openlux_routing_style"]
            if model_block.get("student_openlux_routing") is not None:
                flat["student_openlux_routing"] = model_block["student_openlux_routing"]
            if model_block.get("teacher_openlux_routing") is not None:
                flat["teacher_openlux_routing"] = model_block["teacher_openlux_routing"]
            if model_block.get("teacher_request_timeout") is not None:
                flat["teacher_request_timeout"] = model_block["teacher_request_timeout"]

    # Resolve data / graph paths
    if flat.get("split_dir"):
        flat["split_dir"] = str(
            resolve_path(flat["split_dir"], bases=[GRAPHOPT_ROOT, Path.cwd()])
        )
    if flat.get("data_path"):
        flat["data_path"] = str(
            resolve_path(flat["data_path"], bases=[GRAPHOPT_ROOT, Path.cwd()])
        )
    if flat.get("data_root"):
        flat["data_root"] = str(
            resolve_path(flat["data_root"], bases=[GRAPHOPT_ROOT, Path.cwd()])
        )
    for list_key in ("data_dirs", "docs_dirs"):
        raw_dirs = flat.get(list_key)
        if isinstance(raw_dirs, str):
            raw_dirs = [raw_dirs]
        if isinstance(raw_dirs, list):
            flat[list_key] = [
                str(resolve_path(value, bases=[GRAPHOPT_ROOT, Path.cwd()]))
                for value in raw_dirs
            ]
    if flat.get("skill_markdown_path"):
        flat["skill_markdown_path"] = str(
            resolve_path(
                flat["skill_markdown_path"],
                bases=[GRAPHOPT_ROOT, Path.cwd()],
            )
        )
    if flat.get("frozen_g0_rollout_dir"):
        flat["frozen_g0_rollout_dir"] = str(
            resolve_path(
                flat["frozen_g0_rollout_dir"],
                bases=[GRAPHOPT_ROOT, Path.cwd()],
            )
        )
    # env.skill_init points at the trainable SkillGraph JSON artifact.
    skill_init = (
        flat.get("skill_init")
        or flat.get("graph_init")
        or flat.get("init_graph")
        or ""
    )
    if not skill_init:
        skill_init = str(
            GRAPHOPT_ROOT / "graphopt" / "envs" / "searchqa" / "initial_skill" / "best_graph.json"
        )
    skill_init = str(resolve_path(skill_init, bases=[GRAPHOPT_ROOT, Path.cwd()]))
    flat["skill_init"] = skill_init
    flat["graph_init"] = skill_init  # trainer / CLI alias

    # Normalize configuration aliases used by the trainer.
    if "num_epochs" in flat and "epochs" not in flat:
        flat["epochs"] = flat["num_epochs"]
    # learning_rate/edit_budget is not a patch-wide edit cap. Per-node update
    # scope is controlled by node_merge_top_k.
    if flat.get("eval_test") and "run_final_test" not in flat:
        flat["run_final_test"] = bool(flat["eval_test"])

    apply_openlux_config(flat)

    if not flat.get("out_root"):
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model = str(flat.get("target_model") or flat.get("llm_model") or "student").replace("/", "-")
        environment = str(flat.get("env_name") or flat.get("env") or "searchqa")
        flat["out_root"] = str(
            GRAPHOPT_ROOT / "outputs" / f"graphopt_{environment}_{model}_{ts}"
        )
    flat["out_root"] = os.path.abspath(flat["out_root"])
    return flat


def _preflight_official_manifest(env_name: str, cfg: dict[str, Any]) -> None:
    """Reject missing/legacy static splits before adapter setup or model setup."""
    if not bool(cfg.get("require_official_full_dataset", False)):
        return
    split_dir = Path(str(cfg.get("split_dir") or "")).expanduser()
    manifest_path = split_dir / "split_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{env_name} requires an official-full split manifest at {manifest_path}; "
            "adapter/model setup was not started. Run "
            "scripts/prepare_data.py --dataset all first."
        ) from exc
    if manifest.get("protocol") == "graphskillaa_update_quadruples_v1":
        if (
            manifest.get("split_layout") != ["train", "test"]
            or manifest.get("no_validation_split") is not True
            or manifest.get("held_out_test") is not True
            or (split_dir / "val").exists()
        ):
            raise ValueError(
                f"{env_name} final dataset must contain train and held-out test only"
            )
        return
    if manifest.get("protocol") == "graphskillaa_train_validation":
        if (
            manifest.get("split_layout") != ["train", "val"]
            or manifest.get("no_test_split") is not True
            or (split_dir / "test").exists()
        ):
            raise ValueError(f"{env_name} active dataset must not contain a test split")
        return
    if manifest.get("source_scope") != "official_full":
        raise ValueError(
            f"{env_name} split is not official_full: {manifest_path}. "
            "Old subset/ID manifests are intentionally rejected before adapter/model setup."
        )


def _static_distribution(
    env_name: str, items: list[dict[str, Any]], *, month: bool = False, primary: bool = False
) -> dict[str, int]:
    if env_name in {"livemathematicianbench"} and not month and not primary:
        # Keep preflight on the exact same auditable signature function used by
        # the official split materializer; drift must fail before paid calls.
        from graphopt.grouping import static_split_type
        values = [static_split_type(env_name, item) for item in items]
    elif env_name == "livemathematicianbench" and month:
        values = [str(item.get("month") or "unknown") for item in items]
    elif env_name == "livemathematicianbench" and primary:
        values = [
            next(
                (str(value).strip() for value in (item.get("theorem_type") or []) if str(value).strip()),
                "unknown",
            )
            for item in items
        ]
    else:
        raise ValueError(f"unsupported static distribution environment: {env_name}")
    return dict(sorted(Counter(values).items()))


def _distribution_tvd(left: dict[str, int], right: dict[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise ValueError("cannot validate empty static benchmark distributions")
    return 0.5 * sum(
        abs(left.get(label, 0) / left_total - right.get(label, 0) / right_total)
        for label in set(left) | set(right)
    )


def _validate_train_validation_dataset(
    adapter, env_name: str, split_dir: Path, manifest: dict[str, Any]
) -> None:
    if (
        manifest.get("split_layout") != ["train", "val"]
        or manifest.get("no_test_split") is not True
        or manifest.get("test_merged_into_train") is not True
        or (split_dir / "test").exists()
    ):
        raise ValueError(f"{env_name} must use a train/validation-only layout")
    loader = adapter.get_dataloader()
    pools = {
        "train": list(getattr(loader, "train_items", []) or []),
        "val": list(getattr(loader, "val_items", []) or []),
        "test": list(getattr(loader, "test_items", []) or []),
    }
    expected = {
        "searchqa": {"train": 800, "val": 200},
        "docvqa": {"train": 800, "val": 200},
        "livemathematicianbench": {"train": 468, "val": 117},
    }[env_name]
    loaded = {"train": len(pools["train"]), "val": len(pools["val"])}
    if loaded != expected or manifest.get("counts") != expected or pools["test"]:
        raise ValueError(
            f"{env_name} train/validation counts mismatch: "
            f"expected={expected}, loaded={loaded}, test={len(pools['test'])}"
        )
    ids = {
        name: [str(item.get("id") or "") for item in pools[name]]
        for name in ("train", "val")
    }
    if (
        any(not case_id for values in ids.values() for case_id in values)
        or len(ids["train"]) != len(set(ids["train"]))
        or len(ids["val"]) != len(set(ids["val"]))
        or set(ids["train"]).intersection(ids["val"])
    ):
        raise ValueError(f"{env_name} train/validation IDs are empty, duplicated, or overlapping")
    policy = manifest.get("training_group_policy") or {}
    groups = list(policy.get("groups") or [])
    if (
        int(policy.get("group_size", -1)) != 5
        or int(policy.get("train_per_group", -1)) != 4
        or int(policy.get("validation_per_group", -1)) != 1
        or int(policy.get("group_count", -1)) != expected["val"]
        or len(groups) != expected["val"]
    ):
        raise ValueError(f"{env_name} does not declare complete 4:1 training groups")
    assigned_train: set[str] = set()
    assigned_val: set[str] = set()
    for group in groups:
        train_ids = [str(value) for value in group.get("train_ids") or []]
        val_id = str(group.get("val_id") or "")
        if len(train_ids) != 4 or len(set(train_ids + [val_id])) != 5:
            raise ValueError(f"{env_name} contains an invalid 4:1 group")
        assigned_train.update(train_ids)
        assigned_val.add(val_id)
    if assigned_train != set(ids["train"]) or assigned_val != set(ids["val"]):
        raise ValueError(f"{env_name} 4:1 groups do not cover train/validation exactly")
    for item in pools["train"] + pools["val"]:
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
        elif env_name == "livemathematicianbench":
            if not list(item.get("choices") or []):
                absent.append("choices")
            correct = item.get("correct_choice") or {}
            if not isinstance(correct, dict) or not str(correct.get("label") or "").strip():
                absent.append("correct_choice.label")
        if absent:
            raise ValueError(
                f"{env_name}/{item.get('id')}: missing {','.join(absent)}"
            )


def _validate_static_benchmark_data(adapter, env_name: str, cfg: dict[str, Any]) -> None:
    """Fail before paid calls when an ID-only manifest or required asset is used."""
    if not bool(cfg.get("validate_dataset_assets", False)):
        return
    official_manifest: dict[str, Any] | None = None
    loader = adapter.get_dataloader()
    pools = [
        list(getattr(loader, "train_items", []) or []),
        list(getattr(loader, "val_items", []) or []),
        list(getattr(loader, "test_items", []) or []),
    ]
    items = [item for pool in pools for item in pool]
    if not items:
        raise ValueError(f"{env_name} has no materialized train/validation items")
    split_dir = Path(str(cfg.get("split_dir") or "")).expanduser()
    manifest_path = split_dir / "split_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {}
    )
    if manifest.get("protocol") == "graphskillaa_update_quadruples_v1":
        from graphopt.runtime_envs.train_test_split import validate_train_test_dataset
        validate_train_test_dataset(adapter, env_name, split_dir, manifest)
        return
    if manifest.get("protocol") == "graphskillaa_train_validation":
        _validate_train_validation_dataset(adapter, env_name, split_dir, manifest)
        return
    if env_name == "docvqa":
        from graphopt.runtime_envs.docvqa.pipeline import validate_materialized_split
        validate_materialized_split(adapter, cfg)
        return

    if bool(cfg.get("require_official_full_dataset", False)):
        split_dir = Path(str(cfg.get("split_dir") or "")).expanduser()
        manifest_path = split_dir / "split_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            official_manifest = manifest
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{env_name} requires an official-full split manifest at {manifest_path}; "
                "paid rollout was not started. Run scripts/prepare_data.py --dataset all."
            ) from exc
        if manifest.get("source_scope") != "official_full":
            raise ValueError(
                f"{env_name} split is not official_full: {manifest_path}. "
                "Old subset/ID manifests are intentionally rejected before paid rollout."
            )
        discard_remainders = env_name == "livemathematicianbench"
        searchqa_skill_semantic_v3 = (
            env_name == "searchqa"
            and manifest.get("schema_version")
            == "graphopt-searchqa-g0-skill-semantic-split-v3"
        )
        required_ratio = {
            "searchqa": "grouped_3:1:1_g0_skill_semantic_medoid",
            "livemathematicianbench": "grouped_3:1:1_discard_per_type_remainder",
        }[env_name]
        if str(manifest.get("split_ratio") or "") != required_ratio:
            raise ValueError(
                f"{env_name} experiment requires split_ratio={required_ratio}, "
                "got={}".format(manifest.get("split_ratio"))
            )
        counts = manifest.get("counts") or {}
        loaded_counts = {
            "train": len(pools[0]),
            "val": len(pools[1]),
            "test": len(pools[2]),
        }
        for split_name, loaded_count in loaded_counts.items():
            if int(counts.get(split_name, -1)) != loaded_count:
                raise ValueError(
                    f"{env_name} {split_name} count disagrees with manifest: "
                    f"loaded={loaded_count}, manifest={counts.get(split_name)}"
                )
        required_counts = {
            "searchqa": {"train": 600, "val": 200, "test": 200},
            "livemathematicianbench": {"train": 351, "val": 117, "test": 117},
        }[env_name]
        if loaded_counts != required_counts:
            raise ValueError(
                f"{env_name} grouped-split counts mismatch: "
                f"expected={required_counts}, got={loaded_counts}"
            )
        if loaded_counts["train"] != 3 * loaded_counts["val"]:
            raise ValueError(f"{env_name} complete groups must contribute train:val=3:1")
        if loaded_counts["test"] < loaded_counts["val"]:
            raise ValueError(f"{env_name} test must contain at least one case per group")
        if loaded_counts["test"] != loaded_counts["val"]:
            raise ValueError(f"{env_name} complete-group-only test must equal validation size")

        policy = manifest.get("grouped_five_policy") or {}
        if (
            policy.get("allocation_per_complete_group")
            != {"train": 3, "val": 1, "test": 1}
            or policy.get("remainder_policy")
            != (
                (
                    "no remainder; all 1,000 fixed-pool cases are repartitioned"
                    if searchqa_skill_semantic_v3
                    else "no remainder; exactly 200 selected non-long-tail groups"
                )
                if env_name == "searchqa"
                else "discard all 0--4 leftovers of each fine type"
            )
            or policy.get("same_case_duplicated_across_splits") is not False
            or policy.get("all_train_cases_have_matched_val_and_test") is not True
            or policy.get("all_validation_cases_have_matched_train_and_test") is not True
            or (
                not searchqa_skill_semantic_v3
                and int(policy.get("approximate_groups", -1)) != 0
            )
        ):
            raise ValueError(f"{env_name} manifest does not declare the strict grouped policy")

        loaded_id_sets = {
            name: {str(item.get("id") or "") for item in pool}
            for name, pool in zip(("train", "val", "test"), pools)
        }
        assigned = {"train": set(), "val": set(), "test": set()}
        record_type_counts: Counter[str] = Counter()
        records = policy.get("complete_group_assignments") or []
        if len(records) != loaded_counts["val"]:
            raise ValueError(f"{env_name} must have exactly one complete group per validation case")
        for record in records:
            label = str(
                record.get("grouping_protocol")
                if searchqa_skill_semantic_v3
                else record.get("merged_type") or ""
            )
            raw_types = [str(value) for value in record.get("raw_types") or []]
            train_ids = [str(value) for value in record.get("train_ids") or []]
            val_ids = [str(record.get("val_id") or "")]
            test_ids = [str(record.get("test_id") or "")]
            if (
                not label
                or len(train_ids) != 3
                or len(set(train_ids + val_ids + test_ids)) != 5
                or any(not value for value in train_ids + val_ids + test_ids)
            ):
                raise ValueError(f"{env_name} complete group is not a valid 3/1/1 group")
            if searchqa_skill_semantic_v3:
                if (
                    label
                    != "exact_learned_skill_signature_plus_g0_similarity_and_bge_semantics"
                    or record.get("test_selection")
                    != "highest_mean_combined_similarity_medoid"
                    or record.get("test_is_unrestricted_group_medoid") is not True
                    or record.get("test_has_exact_skill_signature_train_peer") is not True
                    or record.get("test_has_exact_skill_signature_validation_peer") is not True
                    or not (record.get("test_skill_signature") or {}).get("used_nodes")
                ):
                    raise ValueError(
                        "SearchQA group does not prove the train/validation/test skill-use closure"
                    )
            elif record.get("exact_raw_type") is not True or raw_types != [label] * 5:
                raise ValueError(
                    f"{env_name} complete group is not an exact fine-type 3/1/1 group"
                )
            record_type_counts[label] += 1
            for name, values in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
                if any(value not in loaded_id_sets[name] for value in values):
                    raise ValueError(f"{env_name} group assignment points to the wrong split")
                if assigned[name].intersection(values):
                    raise ValueError(f"{env_name} case appears in multiple complete groups")
                assigned[name].update(values)
        if env_name == "searchqa":
            audit = policy.get("similarity_audit") or {}
            source_audit = manifest.get(
                "source_grouping_baseline_recomputed_with_v2_metrics"
            ) or {}
            lineage = manifest.get("lineage") or {}
            learned_closure = manifest.get("learned_skill_usage_closure") or {}
            learned_usage = learned_closure.get("changed_node_usage_by_role") or {}
            covered_changed_nodes = list(
                learned_closure.get("changed_nodes_covered_by_train_validation_test")
                or []
            )
            if (
                not searchqa_skill_semantic_v3
                or manifest.get("test_protocol")
                != "common_g0_and_learned_skill_constrained_semantic_group_medoid"
                or manifest.get("experiment_scope")
                != "repartitioned_fixed_1000_from_official_full_common_g0_and_learned_graph"
                or int(policy.get("same_signature_train_validation_test_core_groups", -1))
                != 200
                or int(audit.get("test_cases_with_exact_skill_train_peer", -1)) != 200
                or int(audit.get("test_cases_with_exact_skill_validation_peer", -1)) != 200
                or int(audit.get("test_cases_that_are_unrestricted_group_medoid", -1)) != 200
                or float(audit.get("mean_test_to_train_semantic_gain_over_source_grouping", 0.0))
                <= 0.0
                or float((audit.get("test_to_train_semantic_cosine") or {}).get("minimum", 0.0))
                <= float(
                    (source_audit.get("test_to_train_semantic_cosine") or {}).get(
                        "minimum", 0.0
                    )
                )
                or lineage.get("pool_membership_unchanged") is not True
                or lineage.get("selection_does_not_use_scores_predictions_or_gold_answers")
                is not True
                or len(lineage.get("common_g0_usage_artifacts") or []) != 3
                or len(lineage.get("learned_graph_usage_artifacts") or []) != 3
                or not (lineage.get("learned_graph") or {}).get("sha256")
                or not covered_changed_nodes
                or any(
                    int((learned_usage.get(role) or {}).get(node_id, 0)) <= 0
                    for node_id in covered_changed_nodes
                    for role in ("train", "val", "test")
                )
            ):
                raise ValueError(
                    "SearchQA manifest does not prove skill-use closure and semantic improvement"
                )
        if discard_remainders and (policy.get("remainder_test_ids_by_type") or {}):
            raise ValueError(f"{env_name} discarded remainders must not appear in test")
        for values in (policy.get("remainder_test_ids_by_type") or {}).values():
            remainder_ids = [str(value) for value in values]
            if len(remainder_ids) >= 5:
                raise ValueError(f"{env_name} fine-type remainder must contain fewer than five cases")
            if any(value not in loaded_id_sets["test"] for value in remainder_ids):
                raise ValueError(f"{env_name} remainder assignment points outside test")
            if assigned["test"].intersection(remainder_ids):
                raise ValueError(f"{env_name} matched and long-tail test assignments overlap")
            assigned["test"].update(remainder_ids)
        if assigned != loaded_id_sets:
            raise ValueError(f"{env_name} group assignments do not cover each split exactly")
        if int(policy.get("matched_test_case_count", -1)) != loaded_counts["val"]:
            raise ValueError(f"{env_name} matched-test accounting disagrees with validation")
        if int(policy.get("long_tail_test_case_count", -1)) != loaded_counts["test"] - loaded_counts["val"]:
            raise ValueError(f"{env_name} long-tail test accounting is inconsistent")

        all_loaded_ids = [str(item.get("id") or "") for pool in pools for item in pool]
        if any(not case_id for case_id in all_loaded_ids) or len(all_loaded_ids) != len(set(all_loaded_ids)):
            raise ValueError(f"{env_name} grouped splits contain empty or overlapping IDs")
        selected_hash = hashlib.sha256(
            "\n".join(sorted(all_loaded_ids)).encode("utf-8")
        ).hexdigest()
        if selected_hash != str(manifest.get("selected_id_sha256") or ""):
            raise ValueError(f"{env_name} selected ID hash disagrees with manifest")
        test_ids = [str(item.get("id") or "") for item in pools[2]]
        test_hash = hashlib.sha256("\n".join(sorted(test_ids)).encode("utf-8")).hexdigest()
        test_order_hash = hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest()
        if (
            test_hash != str(manifest.get("test_id_sha256") or "")
            or test_order_hash != str(manifest.get("test_order_sha256") or "")
        ):
            raise ValueError(f"{env_name} test IDs/order disagree with manifest")

        if env_name == "searchqa":
            if (
                not searchqa_skill_semantic_v3
                or manifest.get("result_scope")
                != "held_out_role_within_transductively_grouped_fixed_pool"
            ):
                raise ValueError(
                    "SearchQA manifest does not declare the G0 skill-semantic protocol"
                )
        if env_name in {"livemathematicianbench"}:
            actual_distributions = {
                name: _static_distribution(env_name, pool)
                for name, pool in zip(("train", "val", "test"), pools)
            }
            if manifest.get("distributions") != actual_distributions:
                raise ValueError(
                    f"{env_name} type distributions disagree with the materialized items"
                )
            actual_type_tvd = {
                "train_vs_test": _distribution_tvd(
                    actual_distributions["train"], actual_distributions["test"]
                ),
                "val_vs_test": _distribution_tvd(
                    actual_distributions["val"], actual_distributions["test"]
                ),
            }
            recorded_type_tvd = manifest.get("type_distribution_tvd") or {}
            for key, actual in actual_type_tvd.items():
                if abs(float(recorded_type_tvd.get(key, -1)) - actual) > 1e-12:
                    raise ValueError(f"{env_name} {key} TVD disagrees with manifest")
            if env_name == "livemathematicianbench":
                actual_primary = {
                    name: _static_distribution(env_name, pool, primary=True)
                    for name, pool in zip(("train", "val", "test"), pools)
                }
                if manifest.get("primary_type_distributions") != actual_primary:
                    raise ValueError(
                        "LiveMath primary-type distributions disagree with materialized items"
                    )
                actual_months = {
                    name: _static_distribution(env_name, pool, month=True)
                    for name, pool in zip(("train", "val", "test"), pools)
                }
                if manifest.get("month_distributions") != actual_months:
                    raise ValueError("LiveMath month distributions disagree with materialized items")
                actual_month_tvd = {
                    "train_vs_test": _distribution_tvd(
                        actual_months["train"], actual_months["test"]
                    ),
                    "val_vs_test": _distribution_tvd(
                        actual_months["val"], actual_months["test"]
                    ),
                }
                recorded_month_tvd = manifest.get("month_distribution_tvd") or {}
                for key, actual in actual_month_tvd.items():
                    if abs(float(recorded_month_tvd.get(key, -1)) - actual) > 1e-12:
                        raise ValueError(f"LiveMath {key} month TVD disagrees with manifest")
        expected = {
            "searchqa": (
                "exact", 134364, "c1a979068ba118d85467179b704031d113d689cc"
            ),
            "livemathematicianbench": (
                "minimum", 608, "6f53c5ff7227633ea954b2847cd590314d582047"
            ),
        }[env_name]
        official_count = int(counts.get("official_all", -1))
        runnable_count = int(counts.get("runnable_all", official_count))
        invalid_count = int(counts.get("invalid_official", official_count - runnable_count))
        partition_count = sum(loaded_counts.values())
        selected_count = int(counts.get("selected_experiment", -1))
        unused_count = int(counts.get("unused_runnable", -1))
        if selected_count != partition_count or unused_count != runnable_count - partition_count:
            raise ValueError(
                f"{env_name} selected/unused accounting is inconsistent: "
                f"selected={selected_count}, partition={partition_count}, "
                f"unused={unused_count}, runnable_all={runnable_count}"
            )
        invalid_items = manifest.get("invalid_official_items") or []
        if official_count - runnable_count != invalid_count or len(invalid_items) != invalid_count:
            raise ValueError(
                f"{env_name} invalid-official accounting is inconsistent: "
                f"official_all={official_count}, runnable_all={runnable_count}, "
                f"count={invalid_count}, listed={len(invalid_items)}"
            )
        for entry in invalid_items:
            if not str(entry.get("id") or "").strip() or not list(
                entry.get("missing_fields") or []
            ):
                raise ValueError(
                    f"{env_name} invalid_official_items must record an ID and missing fields"
                )
        rule, baseline, revision = expected
        if str(manifest.get("source_revision") or "") != revision:
            raise ValueError(
                f"{env_name} source revision is not the pinned official snapshot: "
                f"expected={revision}, got={manifest.get('source_revision')}"
            )
        if (rule == "exact" and official_count != baseline) or (
            rule == "minimum" and official_count < baseline
        ):
            raise ValueError(
                f"{env_name} official-full count check failed: rule={rule} "
                f"baseline={baseline}, got={official_count}"
            )

    missing: list[str] = []
    if env_name == "searchqa":
        for item in items:
            absent = [
                key for key in ("id", "question", "context")
                if not str(item.get(key) or "").strip()
            ]
            if not list(item.get("answers") or []):
                absent.append("answers")
            if absent:
                missing.append(f"{item.get('id')}: missing {','.join(absent)}")
    elif env_name == "livemathematicianbench":
        for item in items:
            absent = []
            if not str(item.get("id") or "").strip():
                absent.append("id")
            if not str(item.get("question") or "").strip():
                absent.append("question")
            if not list(item.get("choices") or []):
                absent.append("choices")
            correct = item.get("correct_choice") or {}
            if not isinstance(correct, dict) or not str(correct.get("label") or "").strip():
                absent.append("correct_choice.label")
            if absent:
                missing.append(f"{item.get('id')}: missing {','.join(absent)}")
    if missing:
        preview = "; ".join(missing[:8])
        raise ValueError(
            f"{env_name} dataset is not fully materialized; paid rollout was not started. "
            f"{preview}. See data/README.md and scripts/prepare_data.py."
        )


def build_env_adapter(cfg: dict[str, Any]):
    """Instantiate the configured benchmark adapter.

    Active benchmarks reuse GraphSkillAA's authoritative rollout and evaluator
    implementations so hard/soft scores do not drift.
    """
    env_name = str(cfg.get("env_name") or cfg.get("env") or "searchqa").strip().lower()
    aliases = {
        "livemath": "livemathematicianbench",
        "live_math": "livemathematicianbench",
    }
    env_name = aliases.get(env_name, env_name)
    _preflight_official_manifest(env_name, cfg)
    if env_name == "docvqa":
        from graphopt.runtime_envs.docvqa.adapter import DocVQAAdapter as Adapter
    elif env_name == "searchqa":
        from graphopt.runtime_envs.searchqa.adapter import SearchQAAdapter as Adapter
    elif env_name == "livemathematicianbench":
        from graphopt.runtime_envs.livemathematicianbench.adapter import (
            LiveMathematicianBenchAdapter as Adapter,
        )
    else:
        raise ValueError(
            f"unsupported env.name {env_name!r}; expected "
            "docvqa, searchqa, or livemathematicianbench"
        )

    sig = inspect.signature(Adapter.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: cfg[k] for k in accepted if k in cfg}
    adapter = Adapter(**kwargs)
    adapter.setup(cfg)
    _validate_static_benchmark_data(adapter, env_name, cfg)
    setattr(adapter, "graphopt_environment_name", env_name)
    return adapter

def configure_models(cfg: dict[str, Any]) -> None:
    """Configure the fixed OpenLux teacher/student pair."""
    apply_openlux_config(cfg)
    if (
        cfg.get("teacher_model_base") != FIXED_TEACHER_MODEL
        or cfg.get("student_model_base") != DEFAULT_OPENLUX_MODEL
    ):
        raise ValueError("this release requires teacher=student=gpt-5.6-sol")
    if cfg.get("optimizer_backend") != "openai_chat" or cfg.get("target_backend") != "openai_chat":
        raise ValueError("this release requires the OpenLux chat backend for both roles")

    from graphopt.model import (
        configure_openlux,
        set_optimizer_deployment,
        set_reasoning_effort,
        set_target_deployment,
    )

    endpoint = cfg.get("azure_openai_endpoint") or cfg.get("azure_endpoint") or None
    api_key = cfg.get("azure_openai_api_key") or cfg.get("azure_api_key") or None
    configure_openlux(
        endpoint=endpoint,
        api_key=api_key,
        optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or endpoint,
        optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or api_key,
        target_endpoint=cfg.get("target_azure_openai_endpoint") or endpoint,
        target_api_key=cfg.get("target_azure_openai_api_key") or api_key,
    )
    teacher = cfg["optimizer_model"]
    student = cfg["target_model"]
    set_optimizer_deployment(teacher)
    set_target_deployment(student)
    install_openlux_provider_sort_hook(cfg.get("openlux_provider_sort"))
    set_reasoning_effort(cfg.get("reasoning_effort", "") or None)
    print(
        f"  [model] teacher={teacher}  student={student}  provider=openlux  "
        f"api=chat_completions  reasoning_effort={cfg.get('reasoning_effort', 'off')}"
    )


def resolve_steps(cfg: dict[str, Any], dataloader) -> tuple[int, int, int]:
    """Return ``(train_size, rollout_batches_per_epoch, total_batches)``.

    Grouped GraphOpt uses train batches only as frozen-graph collection chunks,
    synthesizes once from the complete epoch, runs proposal-dependent affected-scope
    joint Gates, and then one full-validation Big Gate per epoch.  The
    value historically named ``steps_per_epoch`` remains only the dry-run
    fallback for how many rollout batches to synthesize.
    """
    batch_size = max(1, int(cfg.get("batch_size") or 96))
    grouped = bool(cfg.get("grouped_batch_gate", False))
    mode = str(cfg.get("experiment_mode") or "graphopt").strip().lower()
    effective_train_batch_size = batch_size
    single_full_pool = bool(cfg.get("single_full_pool_input", False))
    if mode == "graphopt" and grouped and not single_full_pool:
        train_per_group = max(1, int(cfg.get("train_per_validation") or 3))
        scope = str(cfg.get("batch_size_scope") or "train").strip().lower()
        if scope == "train_plus_validation":
            total_per_group = train_per_group + 1
            if batch_size < total_per_group or batch_size % total_per_group:
                raise ValueError(
                    "total grouped GraphOpt batch_size must be divisible by "
                    f"train_per_validation + 1; got {batch_size}"
                )
            effective_train_batch_size = (
                batch_size // total_per_group * train_per_group
            )
        elif scope == "train":
            if batch_size < train_per_group or batch_size % train_per_group:
                raise ValueError(
                    "grouped GraphOpt batch_size must be divisible by "
                    f"train_per_validation; got {batch_size}"
                )
        else:
            raise ValueError(f"unknown batch_size_scope: {scope!r}")
    epochs = max(1, int(cfg.get("epochs") or cfg.get("num_epochs") or 1))

    configured = int(cfg.get("train_size") or 0)
    inferred = None
    if dataloader is not None:
        getter = getattr(dataloader, "get_train_size", None)
        if callable(getter):
            try:
                inferred = int(getter())
            except Exception:
                inferred = None
    train_size = configured if configured > 0 else (inferred or 0)
    if train_size <= 0:
        # dry-run / no data: fall back to explicit steps
        batches = max(1, int(cfg.get("steps_per_epoch") or 2))
        cfg["rollout_batches_per_epoch"] = batches
        return 0, batches, epochs * batches

    if single_full_pool:
        batches = 1
        total_batches = epochs
    elif bool(cfg.get("shard_train_across_epochs", True)):
        base, remainder = divmod(train_size, epochs)
        shard_sizes = [base] * (epochs - remainder) + [base + 1] * remainder
        if not shard_sizes or min(shard_sizes) <= 0:
            raise ValueError(
                f"cannot shard train_size={train_size} across epochs={epochs} without empty shards"
            )
        shard_batches = [
            int(math.ceil(size / effective_train_batch_size)) for size in shard_sizes
        ]
        batches = max(shard_batches)
        total_batches = sum(shard_batches)
        cfg["train_shard_sizes"] = shard_sizes
    else:
        batches = max(1, int(math.ceil(train_size / effective_train_batch_size)))
        total_batches = epochs * batches
    cfg["train_size"] = train_size
    cfg["steps_per_epoch"] = 1
    cfg["rollout_batches_per_epoch"] = batches
    return train_size, batches, total_batches
