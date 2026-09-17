#!/usr/bin/env python3
"""Offline integrity checks for the SkillAA project."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_skillaa.py"), "--dataset", "all", "--seed", "all", "--check-manifests"],
        cwd=ROOT,
        check=True,
    )
    forbidden_paths = [
        ROOT / ".env",
        ROOT / "graphopt" / "envs" / "alfworld",
    ]
    forbidden_paths.extend(ROOT.glob("graphopt/envs/*/no_skill"))
    forbidden_paths.extend(ROOT.glob("graphopt/envs/*/skillaa_md"))
    present = [str(path.relative_to(ROOT)) for path in forbidden_paths if path.exists()]
    if present:
        raise SystemExit("forbidden private/out-of-scope paths: " + ", ".join(present))
    for manifest_path in sorted(ROOT.glob("data/*/split_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        serialized = json.dumps(payload)
        if '"test_id"' in serialized or '"test_ids"' in serialized:
            raise SystemExit(f"held-out ID leaked into update groups: {manifest_path}")
    print(json.dumps({"ok": True, "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
