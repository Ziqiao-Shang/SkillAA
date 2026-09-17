#!/usr/bin/env python3
"""Aggregate the three independent SkillAA seeds as mean ± half-range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ("searchqa", "livemathematicianbench", "docvqa")
SEEDS = (42, 43, 44)


def percent(value: float) -> float:
    return value * 100.0 if value <= 1.0 else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/skillaa"))
    args = parser.parse_args()
    print("| Benchmark | seed 42 | seed 43 | seed 44 | Mean ± half-range |")
    print("|---|---:|---:|---:|---:|")
    for dataset in DATASETS:
        values = []
        for seed in SEEDS:
            path = args.output_root / dataset / f"seed{seed}" / "final_test.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("test_repeats") or len(payload.get("repeat_results") or [])) != 1:
                raise ValueError(f"{path}: expected exactly one held-out evaluation")
            values.append(percent(float(payload["hard_mean"])))
        mean = sum(values) / len(values)
        half_range = (max(values) - min(values)) / 2.0
        cells = " | ".join(f"{value:.1f}" for value in values)
        print(f"| {dataset} | {cells} | {mean:.1f} ± {half_range:.1f} |")


if __name__ == "__main__":
    main()
