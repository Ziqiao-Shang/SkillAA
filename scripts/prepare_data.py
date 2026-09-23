#!/usr/bin/env python3
"""Download pinned benchmark snapshots and materialize the configured GraphSkillAA IDs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_graphskillaa import DATASETS, ROOT, validate_data

REVISIONS = {
    "searchqa": "c1a979068ba118d85467179b704031d113d689cc",
    "docvqa": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
    "livemathematicianbench": "6f53c5ff7227633ea954b2847cd590314d582047",
}
REPOS = {
    "searchqa": "lucadiliello/searchqa",
    "docvqa": "lmms-lab/DocVQA",
    "livemathematicianbench": "LiveMathematicianBench/LiveMathematicianBench",
}
SEARCHQA_FILES = (
    "data/train-00000-of-00001-55e7116bea868a35.parquet",
    "data/validation-00000-of-00001-2092e81367c2ca98.parquet",
)
LIVE_FILES = tuple(
    f"data/{month}/qa_{month}_final.json"
    for month in ("202511", "202512", "202601", "202602", "202603", "202604", "202605", "202606")
)
DOCVQA_FILES = tuple(
    f"DocVQA/validation-{index:05d}-of-00006.parquet" for index in range(6)
)
DOCVQA_FIELDS = (
    "id", "questionId", "docId", "question", "answer", "ground_truth",
    "image_path", "ucsf_document_id", "ucsf_document_page_no", "topic",
    "category", "source_dataset", "source_config", "source_split",
)


def _download(dataset: str, filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPOS[dataset],
            repo_type="dataset",
            filename=filename,
            revision=REVISIONS[dataset],
            cache_dir=str(cache_dir),
            token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
        )
    )


def _ids(dataset: str) -> tuple[list[str], list[str]]:
    base = ROOT / "data" / dataset
    manifest = json.loads((base / "split_manifest.json").read_text())
    groups = (manifest.get("training_group_policy") or {}).get("groups") or []
    train = [str(value) for group in groups for value in (group.get("train_ids") or [])]
    test = [str(value) for value in json.loads((base / "test_ids.json").read_text())]
    expected = DATASETS[dataset]
    if len(train) != expected["train"] or len(test) != expected["test"]:
        raise ValueError(f"{dataset}: released ID counts do not match {expected}")
    if len(train) != len(set(train)) or len(test) != len(set(test)) or set(train) & set(test):
        raise ValueError(f"{dataset}: released train/test IDs are invalid")
    return train, test


def _prepare_output(dataset: str, overwrite: bool) -> None:
    base = ROOT / "data" / dataset
    targets = [base / "train", base / "test"]
    if dataset == "docvqa":
        targets.append(base / "images")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path.relative_to(ROOT)) for path in existing)
        raise FileExistsError(f"{names} already exist; pass --overwrite to replace them")
    if overwrite:
        for path in existing:
            shutil.rmtree(path)


def _write_json_rows(dataset: str, by_id: dict[str, dict[str, Any]]) -> None:
    train_ids, test_ids = _ids(dataset)
    missing = [case_id for case_id in train_ids + test_ids if case_id not in by_id]
    if missing:
        raise KeyError(f"{dataset}: pinned source is missing selected IDs; first={missing[:8]}")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        target = ROOT / "data" / dataset / split
        target.mkdir(parents=True, exist_ok=False)
        (target / "items.json").write_text(
            json.dumps([by_id[case_id] for case_id in ids], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _normalize_searchqa(row: dict[str, Any], source_split: str) -> dict[str, Any]:
    answers = row.get("answers")
    if isinstance(answers, str):
        answers = [answers]
    case_id = str(row.get("key") or row.get("id") or "").strip()
    return {
        "id": case_id,
        "key": case_id,
        "question": str(row.get("question") or "").strip(),
        "context": str(row.get("context") or "").strip(),
        "answers": [str(value).strip() for value in (answers or []) if str(value).strip()],
        "task_type": "qa",
        "source_split": source_split,
    }


def prepare_searchqa(cache_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for filename in SEARCHQA_FILES:
        source_split = Path(filename).name.split("-", 1)[0]
        path = _download("searchqa", filename, cache_dir)
        rows.extend(
            _normalize_searchqa(dict(row), source_split)
            for row in pq.read_table(path).to_pylist()
        )
    by_id = {row["id"]: row for row in rows if row["id"]}
    if len(by_id) != len(rows):
        raise ValueError("searchqa: pinned source contains empty or duplicate IDs")
    _write_json_rows("searchqa", by_id)


def prepare_livemath(cache_dir: Path) -> None:
    from graphopt.runtime_envs.livemathematicianbench.dataloader import load_items

    rows: list[dict[str, Any]] = []
    for filename in LIVE_FILES:
        path = _download("livemathematicianbench", filename, cache_dir)
        for row in load_items(str(path)):
            normalized = dict(row)
            normalized["source_path"] = filename
            rows.append(normalized)
    by_id = {str(row.get("id") or ""): row for row in rows if str(row.get("id") or "")}
    if len(by_id) != len(rows):
        raise ValueError("livemathematicianbench: pinned source contains duplicate IDs")
    _write_json_rows("livemathematicianbench", by_id)


def _image_suffix(path_hint: str, payload: bytes) -> str:
    suffix = Path(path_hint).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix
    if payload.startswith(b"\x89PNG"):
        return ".png"
    if payload.startswith(b"\xff\xd8"):
        return ".jpg"
    return ".bin"


def _docvqa_csv_row(row: dict[str, Any], image_path: str) -> dict[str, str]:
    answers = [str(value).strip() for value in (row.get("answers") or []) if str(value).strip()]
    question_types = [
        str(value).strip()
        for value in (row.get("question_types") or [])
        if str(value).strip()
    ]
    topic = "|".join(question_types) or "docvqa"
    question_id = str(row.get("questionId") or "")
    return {
        "id": question_id,
        "questionId": question_id,
        "docId": str(row.get("docId") or ""),
        "question": " ".join(str(row.get("question") or "").split()),
        "answer": repr(answers),
        "ground_truth": repr(answers),
        "image_path": image_path,
        "ucsf_document_id": str(row.get("ucsf_document_id") or ""),
        "ucsf_document_page_no": str(row.get("ucsf_document_page_no") or ""),
        "topic": topic,
        "category": topic,
        "source_dataset": REPOS["docvqa"],
        "source_config": "DocVQA",
        "source_split": "validation",
    }


def prepare_docvqa(cache_dir: Path) -> None:
    train_ids, test_ids = _ids("docvqa")
    selected = set(train_ids) | set(test_ids)
    image_root = ROOT / "data" / "docvqa" / "images"
    image_root.mkdir(parents=True, exist_ok=False)
    rows: dict[str, dict[str, str]] = {}

    for filename in DOCVQA_FILES:
        path = _download("docvqa", filename, cache_dir)
        parquet = pq.ParquetFile(path)
        columns = [
            "questionId", "docId", "question", "answers", "question_types",
            "ucsf_document_id", "ucsf_document_page_no", "image",
        ]
        for batch in parquet.iter_batches(columns=columns, batch_size=64):
            for raw in batch.to_pylist():
                question_id = str(raw.get("questionId") or "")
                if question_id not in selected or question_id in rows:
                    continue
                image = raw.get("image") or {}
                payload = image.get("bytes") or b""
                if not payload:
                    raise ValueError(f"docvqa: question {question_id} has no image bytes")
                suffix = _image_suffix(str(image.get("path") or ""), payload)
                relative = Path("data") / "docvqa" / "images" / (
                    f"q{question_id}_d{raw.get('docId')}{suffix}"
                )
                (ROOT / relative).write_bytes(payload)
                rows[question_id] = _docvqa_csv_row(dict(raw), relative.as_posix())

    missing = selected - rows.keys()
    if missing:
        raise KeyError(f"docvqa: pinned source is missing selected IDs; first={sorted(missing)[:8]}")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        target = ROOT / "data" / "docvqa" / split
        target.mkdir(parents=True, exist_ok=False)
        with (target / "docvqa.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DOCVQA_FIELDS)
            writer.writeheader()
            writer.writerows(rows[case_id] for case_id in ids)


def prepare(dataset: str, cache_dir: Path, overwrite: bool) -> None:
    _prepare_output(dataset, overwrite)
    if dataset == "searchqa":
        prepare_searchqa(cache_dir)
    elif dataset == "docvqa":
        prepare_docvqa(cache_dir)
    else:
        prepare_livemath(cache_dir)
    print(json.dumps(validate_data(dataset), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the runnable reference split from pinned official snapshots"
    )
    parser.add_argument("--dataset", required=True, choices=[*DATASETS, "all"])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / ".cache" / "huggingface",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only materialized train/test payloads (and DocVQA images)",
    )
    args = parser.parse_args()
    selected = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        print(f"[prepare] {dataset} revision={REVISIONS[dataset]}")
        prepare(dataset, args.cache_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()
