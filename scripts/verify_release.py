#!/usr/bin/env python3
"""Offline protocol and source-release checks for GraphSkillAA."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_FILE_BYTES = 100 * 1024 * 1024

FORBIDDEN_DIR_NAMES = {
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "outputs",
}
FORBIDDEN_MODEL_SUFFIXES = {
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _check_protocol() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_graphskillaa.py"),
            "--dataset",
            "all",
            "--seed",
            "all",
            "--check-manifests",
        ],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    for manifest_path in sorted(ROOT.glob("data/*/split_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        serialized = json.dumps(payload)
        if '"test_id"' in serialized or '"test_ids"' in serialized:
            raise SystemExit(
                f"held-out ID leaked into update groups: {_relative(manifest_path)}"
            )


def _check_release_tree() -> None:
    forbidden_paths = [
        ROOT / ".env",
        ROOT / "data" / ".cache",
        ROOT / "data" / "docvqa" / "images",
        ROOT / "graphopt" / "envs" / "alfworld",
    ]
    forbidden_paths.extend(ROOT.glob("data/*/train"))
    forbidden_paths.extend(ROOT.glob("data/*/test"))
    forbidden_paths.extend(ROOT.glob("graphopt/envs/*/no_skill"))
    forbidden_paths.extend(ROOT.glob("graphopt/envs/*/graphskillaa_md"))
    present = sorted({_relative(path) for path in forbidden_paths if path.exists()})

    for path in ROOT.rglob("*"):
        relative_parts = path.relative_to(ROOT).parts
        if ".git" in relative_parts:
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
            present.append(_relative(path))
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES:
            present.append(_relative(path))
        if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
            present.append(f"{_relative(path)} (>100 MiB)")

    if present:
        raise SystemExit(
            "forbidden generated/private/large release paths: "
            + ", ".join(sorted(set(present)))
        )


def _check_secrets_and_private_paths() -> None:
    secret_patterns = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        re.compile(
            r"(?im)^\s*(?:OPENLUX_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN)\s*=\s*"
            r"(?!$|your_|<|\$\{)[^\s#]+"
        ),
    )
    private_markers = (
        "".join(("/", "sata", "/")),
        "".join(("/", "data", "/", "shang", "zq")),
        "".join(("/", "root", "/", "autodl")),
    )
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(ROOT).parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in secret_patterns):
            findings.append(f"possible credential in {_relative(path)}")
        if any(marker in text for marker in private_markers):
            findings.append(f"private absolute path in {_relative(path)}")
    if findings:
        raise SystemExit("; ".join(sorted(set(findings))))


def main() -> None:
    _check_protocol()
    _check_release_tree()
    _check_secrets_and_private_paths()
    print(json.dumps({"ok": True, "checks": ["protocol", "tree", "secrets"]}, indent=2))


if __name__ == "__main__":
    main()
