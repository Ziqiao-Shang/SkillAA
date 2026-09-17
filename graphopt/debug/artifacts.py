"""Persist versioned intermediate artifacts (inputs + outputs) for pipeline debugging."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path | str, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


class ArtifactStore:
    """Write ``{key}_v1.json``, ``{key}_v2.json``, … keeping every revision."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "artifact_index.json"
        self._index: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        if self.index_path.exists():
            try:
                obj = json.loads(self.index_path.read_text(encoding="utf-8"))
                self._index = dict(obj.get("artifacts") or {})
            except json.JSONDecodeError:
                self._index = {}

    def _path_for(self, key: str, version: int) -> Path:
        parts = key.split("/")
        name = parts[-1]
        if len(parts) == 1:
            return self.base_dir / f"{name}_v{version}.json"
        sub = self.base_dir.joinpath(*parts[:-1])
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{name}_v{version}.json"

    def _latest_flat_path(self, key: str) -> Path:
        """Backward-compat flat name (``case_analysis.json``) = latest outputs."""
        return self.base_dir / f"{key.split('/')[-1]}.json"

    def next_version(self, key: str) -> int:
        return len(self._index.get(key, [])) + 1

    def save(
        self,
        key: str,
        *,
        stage: str,
        inputs: Any = None,
        outputs: Any = None,
        data: Any = None,
        **extra: Any,
    ) -> tuple[Path, int]:
        with self._lock:
            version = self.next_version(key)
            path = self._path_for(key, version)
            payload: dict[str, Any] = {
                "artifact_key": key,
                "version": version,
                "stage": stage,
                "ts": _utc_now(),
            }
            if inputs is not None:
                payload["inputs"] = inputs
            if outputs is not None:
                payload["outputs"] = outputs
            if data is not None:
                payload["data"] = data
            for k, v in extra.items():
                if v is not None:
                    payload[k] = v
            save_json(path, payload)
    
            latest_body = outputs if outputs is not None else (data if data is not None else payload)
            save_json(self._latest_flat_path(key), latest_body)
    
            self._index.setdefault(key, []).append(
                {
                    "version": version,
                    "file": str(path.relative_to(self.base_dir)),
                    "stage": stage,
                    "ts": payload["ts"],
                }
            )
            save_json(self.index_path, {"artifacts": self._index, "updated_at": _utc_now()})
            return path, version

    def versions(self, key: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._index.get(key, []))


class StepRecorder:
    """Append-only stage log; ``step_manifest`` is also versioned via ``ArtifactStore``."""

    def __init__(self, step_dir: Path | str):
        self.step_dir = Path(step_dir)
        self.step_dir.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.step_dir)
        self.entries: list[dict[str, Any]] = []
        self.llm_dir = self.step_dir / "llm"
        self.debug_dir = self.step_dir / "debug"
        self.llm_dir.mkdir(exist_ok=True)
        self.debug_dir.mkdir(exist_ok=True)

    def record(self, stage: str, *, files: list[str] | None = None, **meta: Any) -> None:
        rel_files = []
        for f in files or []:
            p = Path(f)
            try:
                rel_files.append(str(p.relative_to(self.step_dir)))
            except ValueError:
                rel_files.append(str(p))
        self.entries.append(
            {
                "ts": _utc_now(),
                "stage": stage,
                "files": rel_files,
                **{k: v for k, v in meta.items() if v is not None},
            }
        )

    def flush(self) -> Path:
        _, v = self.store.save(
            "step_manifest",
            stage="step_manifest",
            inputs={"step_dir": str(self.step_dir)},
            outputs={"entries": self.entries},
        )
        flat = self.step_dir / "step_manifest.json"
        save_json(flat, {"step_dir": str(self.step_dir), "version": v, "entries": self.entries})
        return flat


def save_llm_call(
    store: ArtifactStore,
    key: str,
    *,
    stage: str,
    system: str,
    user: str,
    response: str | None = None,
    usage: Any = None,
    parsed: Any = None,
    error: str | None = None,
) -> tuple[Path, int]:
    inputs = {"system": system, "user": user}
    outputs: dict[str, Any] = {"mode": "teacher"}
    if response is not None:
        outputs["response"] = response
    if usage is not None:
        outputs["usage"] = usage
    if parsed is not None:
        outputs["parsed"] = parsed
    if error is not None:
        outputs["error"] = error
    return store.save(f"llm/{stage}/{key}", stage=stage, inputs=inputs, outputs=outputs)


def save_template_call(
    store: ArtifactStore,
    key: str,
    *,
    stage: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    return store.save(
        f"debug/{stage}/{key}",
        stage=stage,
        inputs=inputs,
        outputs=outputs or {},
        mode="template",
    )


def archive_epoch_artifacts(
    out_root: Path | str,
    *,
    epoch: int,
    step: int,
    decision: dict[str, Any] | None = None,
) -> Path:
    """Write the one canonical diagnostic snapshot for epoch N.

    vN always means epoch N. Repeated non-LLM writes keep only their final
    value. If a teacher call is format-corrected, its final answer is _vN
    and preceding invalid answers are _vN_pre1, _vN_pre2, and so on.
    Re-creating the same epoch atomically replaces its prior snapshot.
    """
    root = Path(out_root)
    archive_root = root / "epoch_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    base_name = f"epoch_v{epoch}"
    archive_dir = archive_root / base_name
    temp_dir = archive_root / f".{base_name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    sources: list[Path] = [
        root / "steps" / f"step_{step:04d}",
        root / "epochs" / f"epoch_{epoch:02d}",
        root / "epoch_tests" / f"epoch_{epoch:02d}",
        root / "meta",
        root / "graphs" / "graph_step0000.json",
        root / "graphs" / f"graph_step{step:04d}.json",
        root / "config.json",
        root / "evolution_cache.json",
        root / "gate_history.jsonl",
        root / "epoch_versions.md",
        root / "meta.txt",
        root / f"meta_epoch_{epoch:02d}.txt",
        root / f"comparison_pairs_epoch_{epoch:02d}.json",
        root / "best_graph.json",
        root / "best_skill.md",
    ]
    rollout_root = root / "rollouts"
    if rollout_root.exists():
        sources.extend(sorted(rollout_root.glob(f"epoch_{epoch:02d}_*")))
        sources.extend(sorted(rollout_root.glob(f"step_{step:04d}_*")))
        if epoch == 1:
            sources.extend(sorted(rollout_root.glob("baseline_selection_r*")))
            sources.append(root / "baseline.json")

    source_files: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        if not source.exists():
            continue
        if source.is_file():
            candidates = [source]
        else:
            candidates = [child for child in sorted(source.rglob("*")) if child.is_file()]
        for child in candidates:
            resolved = child.resolve()
            if resolved not in seen:
                seen.add(resolved)
                source_files.append(child)

    # ArtifactStore's working directory contains both flat latest aliases and
    # private attempt counters. Convert those into the public epoch
    # convention while constructing the canonical archive.
    indexed_sources: set[Path] = set()
    flat_aliases: set[Path] = set()
    index_sources: set[Path] = set()
    artifact_copies: list[tuple[Path, Path]] = []
    normalized_indexes: list[tuple[Path, dict[str, Any]]] = []

    for index_path in [p for p in source_files if p.name == "artifact_index.json"]:
        try:
            raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        artifacts = raw_index.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        index_sources.add(index_path.resolve())
        base = index_path.parent
        archive_base = base.relative_to(root)
        normalized: dict[str, list[dict[str, Any]]] = {}

        for key, raw_entries in artifacts.items():
            if not isinstance(key, str) or not isinstance(raw_entries, list):
                continue
            valid_entries: list[tuple[dict[str, Any], Path]] = []
            for entry in raw_entries:
                if not isinstance(entry, dict) or not entry.get("file"):
                    continue
                source_path = base / str(entry["file"])
                indexed_sources.add(source_path.resolve())
                if source_path.is_file():
                    valid_entries.append((entry, source_path))

            flat_aliases.add((base / f"{key.split('/')[-1]}.json").resolve())
            if not valid_entries:
                continue

            # The run-level Meta store accumulates one distinct key per epoch.
            # A canonical epoch snapshot must contain only its own Meta call,
            # not redundant copies of every earlier epoch's teacher exchange.
            if (
                base.resolve() == (root / "meta").resolve()
                and key.split("/")[-1] != f"epoch_{epoch:02d}"
            ):
                continue

            key_parts = key.split("/")
            key_parent = Path(*key_parts[:-1]) if len(key_parts) > 1 else Path()
            key_name = key_parts[-1]
            final_entry, final_source = valid_entries[-1]
            final_relative = archive_base / key_parent / f"{key_name}_v{epoch}.json"
            artifact_copies.append((final_source, final_relative))

            normalized_entries: list[dict[str, Any]] = []
            if key.startswith("llm/"):
                # pre1 is the answer immediately before the final answer.
                for pre_number, (entry, source_path) in enumerate(
                    reversed(valid_entries[:-1]), start=1
                ):
                    pre_relative = (
                        archive_base
                        / key_parent
                        / f"{key_name}_v{epoch}_pre{pre_number}.json"
                    )
                    artifact_copies.append((source_path, pre_relative))
                    normalized_entries.insert(
                        0,
                        {
                            **entry,
                            "epoch": epoch,
                            "attempt": f"pre{pre_number}",
                            "file": str(pre_relative.relative_to(archive_base)),
                        },
                    )
            normalized_entries.append(
                {
                    **final_entry,
                    "epoch": epoch,
                    "attempt": "final",
                    "file": str(final_relative.relative_to(archive_base)),
                }
            )
            normalized[key] = normalized_entries

        normalized_indexes.append(
            (
                archive_base / "artifact_index.json",
                {
                    "schema_version": "graphopt-artifact-index-epoch-v1",
                    "epoch": epoch,
                    "artifacts": normalized,
                    "updated_at": _utc_now(),
                },
            )
        )

    copied: list[dict[str, Any]] = []

    def copy_to_archive(source: Path, relative: Path) -> None:
        destination = temp_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source.relative_to(root)),
                "archive": str(relative),
                "bytes": destination.stat().st_size,
            }
        )

    for source in source_files:
        resolved = source.resolve()
        if (
            resolved in indexed_sources
            or resolved in flat_aliases
            or resolved in index_sources
        ):
            continue
        copy_to_archive(source, source.relative_to(root))

    for source, relative in artifact_copies:
        copy_to_archive(source, relative)

    for relative, payload in normalized_indexes:
        destination = temp_dir / relative
        save_json(destination, payload)
        copied.append(
            {
                "source": str(relative),
                "archive": str(relative),
                "bytes": destination.stat().st_size,
                "normalized": True,
            }
        )

    accepted = bool((decision or {}).get("accepted"))
    action = str((decision or {}).get("action") or "")
    if accepted:
        version_status = "COMMIT_PARTIAL" if action == "accept_partial" else "COMMIT"
    elif action.startswith("reject"):
        version_status = "ROLLBACK"
    else:
        version_status = "NO_UPDATE"
    formal_graph_path = root / "graphs" / f"graph_step{step:04d}.json"
    previous_graph_path = root / "graphs" / f"graph_step{max(0, step - 1):04d}.json"
    formal_graph_sha256 = _sha256(formal_graph_path)
    previous_graph_sha256 = _sha256(previous_graph_path)
    same_as_previous = bool(
        formal_graph_sha256
        and previous_graph_sha256
        and formal_graph_sha256 == previous_graph_sha256
    )

    manifest = {
        "schema_version": "graphopt-epoch-archive-v1",
        "epoch": epoch,
        "step": step,
        "archive": base_name,
        "version_status": version_status,
        "gate_action": action or None,
        "formal_graph": str(formal_graph_path.relative_to(root)),
        "formal_graph_sha256": formal_graph_sha256,
        "previous_graph_sha256": previous_graph_sha256,
        "formal_graph_same_as_previous": same_as_previous,
        "formal_graph_relation": (
            f"same as epoch_v{max(0, epoch - 1)}"
            if same_as_previous
            else "updated formal graph"
            if accepted
            else "no graph update"
        ),
        "created_at": _utc_now(),
        "n_files": len(copied),
        "files": copied,
        "version_semantics": {
            "epoch_vN": "the final canonical snapshot for epoch N",
            "file_vN": "the final value produced in epoch N",
            "file_vN_preM": "the M-th preceding invalid teacher answer; pre1 is nearest to final",
        },
    }
    save_json(temp_dir / "epoch_archive_manifest.json", manifest)
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    temp_dir.rename(archive_dir)
    save_json(archive_root / f"epoch_archive_manifest_{base_name}.json", manifest)
    return archive_dir


def archive_evaluation_artifacts(out_root: Path | str) -> Path:
    """Create the canonical ``evaluation_v1`` snapshot for frozen baselines.

    Evaluation-only methods have no optimization epoch, but their inputs,
    rollout, scores, and rendered skill still need the same stable version
    semantics as training runs. Re-running the same output directory replaces
    this one snapshot atomically; formal experiment runners prevent accidental
    reuse of a non-empty directory.
    """
    root = Path(out_root)
    archive_root = root / "evaluation_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_root / "evaluation_v1"
    temp_dir = archive_root / ".evaluation_v1.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    sources = [
        root / "config.json",
        root / "summary.json",
        root / "final_test.json",
        root / "best_graph.json",
        root / "best_skill.md",
    ]
    sources.extend(sorted(root.glob("final_test_v*.json")))
    sources.extend(sorted((root / "rollouts").glob("final_test_v*")))
    copied: list[dict[str, Any]] = []
    for source in sources:
        if not source.exists():
            continue
        candidates = [source] if source.is_file() else [
            child for child in sorted(source.rglob("*")) if child.is_file()
        ]
        for child in candidates:
            relative = child.relative_to(root)
            destination = temp_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)
            copied.append(
                {
                    "source": str(relative),
                    "archive": str(relative),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )

    manifest = {
        "schema_version": "graphopt-evaluation-archive-v1",
        "version": "v1",
        "archive": "evaluation_v1",
        "version_status": "FROZEN_EVALUATION",
        "created_at": _utc_now(),
        "n_files": len(copied),
        "files": copied,
        "version_semantics": {
            "evaluation_v1": "the complete immutable snapshot of one evaluation-only run",
            "v1": "evaluation version 1; evaluation-only runs have no optimizer epochs",
        },
    }
    save_json(temp_dir / "evaluation_archive_manifest.json", manifest)
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    temp_dir.rename(archive_dir)
    save_json(archive_root / "evaluation_archive_manifest_v1.json", manifest)
    return archive_dir


def write_epoch_versions_doc(
    out_root: Path | str,
    history: list[dict[str, Any]],
) -> Path:
    """Write the run-specific epoch/rollback ledger consumed by human reviewers."""
    root = Path(out_root)
    lines = [
        "# Epoch Version Ledger",
        "",
        "`vN` always means epoch N. A rejected candidate still produces `epoch_vN`,",
        "but its canonical graph equals the previous epoch and its status is `ROLLBACK`.",
        "",
        "| Version | Status | Gate action | Canonical graph relation | SHA-256 | Candidate score | Final score |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for record in history:
        epoch = int(record.get("epoch") or record.get("step") or 0)
        action = str(record.get("action") or "")
        accepted = bool(record.get("accepted"))
        if accepted:
            status = "COMMIT_PARTIAL" if action == "accept_partial" else "COMMIT"
            relation = f"new canonical graph from epoch_v{epoch}"
        elif action.startswith("reject"):
            status = "ROLLBACK"
            relation = f"same canonical graph as epoch_v{max(0, epoch - 1)}"
        else:
            status = "NO_UPDATE"
            relation = f"same canonical graph as epoch_v{max(0, epoch - 1)}"
        graph_path = root / "graphs" / f"graph_step{epoch:04d}.json"
        graph_hash = _sha256(graph_path)
        short_hash = graph_hash[:12] if graph_hash else "—"
        candidate = record.get("candidate_val_score")
        final = record.get("val_score", record.get("current_score"))
        candidate_text = "—" if candidate is None else str(candidate)
        final_text = "—" if final is None else str(final)
        lines.append(
            f"| epoch_v{epoch} | {status} | {action or '—'} | {relation} | "
            f"{short_hash} | {candidate_text} | {final_text} |"
        )
    lines.extend(
        [
            "",
            "Each epoch archive retains only the final value of ordinary artifacts.",
            "After teacher format correction, the final answer uses `<name>_vN.json` and",
            "the immediately preceding invalid answer uses `<name>_vN_pre1.json`.",
            "",
        ]
    )
    path = root / "epoch_versions.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_rollout_artifact(
    rollout_dir: Path | str,
    *,
    stage: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[Path, int]:
    store = ArtifactStore(rollout_dir)
    return store.save("rollout", stage=stage, inputs=inputs, outputs=outputs)
