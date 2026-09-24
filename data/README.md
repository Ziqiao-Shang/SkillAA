# Data preparation and fixed splits

GraphSkillAA does not redistribute raw benchmark payloads. This directory
contains only fixed update-group manifests and held-out test IDs. The preparation
script downloads pinned source snapshots from Hugging Face and materializes only
the released IDs.

## Sources

| Benchmark | Hugging Face dataset | Pinned revision |
|---|---|---|
| SearchQA | `lucadiliello/searchqa` | `c1a979068ba118d85467179b704031d113d689cc` |
| DocVQA | `lmms-lab/DocVQA` | `539088ef8a8ada01ac8e2e6d4e372586748a265e` |
| LiveMathematicianBench | `LiveMathematicianBench/LiveMathematicianBench` | `6f53c5ff7227633ea954b2847cd590314d582047` |

Obtain and use each dataset under its own license and terms. If Hugging Face
requires authentication, set `HF_TOKEN` or `HUGGINGFACE_TOKEN` in the shell.

## Released identifiers

Each benchmark directory contains:

- `split_manifest.json`: ordered groups of four update IDs;
- `test_ids.json`: the disjoint held-out evaluation IDs.

The expected update/test counts are SearchQA 800/200,
LiveMathematicianBench 468/117, and DocVQA 800/200. Held-out IDs never occur in
the update groups and must not be used for graph editing, attribution, Local Gate
decisions, or Big Gate decisions.

Validate the identifier protocol without downloading payloads:

```bash
python scripts/run_graphskillaa.py --dataset all --seed all --check-manifests
```

## Download and materialize

Install the project dependencies, then run:

```bash
python scripts/prepare_data.py --dataset searchqa
python scripts/prepare_data.py --dataset livemathematicianbench
python scripts/prepare_data.py --dataset docvqa
```

Use `--dataset all` to prepare all three. Downloads are cached under
`data/.cache/huggingface/` by default; `--cache-dir PATH` selects another cache.
Existing materialized payloads are never replaced unless `--overwrite` is
passed.

The resulting layout is:

```text
data/
├── searchqa/
│   ├── split_manifest.json
│   ├── test_ids.json
│   ├── train/items.json
│   └── test/items.json
├── livemathematicianbench/
│   ├── split_manifest.json
│   ├── test_ids.json
│   ├── train/items.json
│   └── test/items.json
└── docvqa/
    ├── split_manifest.json
    ├── test_ids.json
    ├── train/docvqa.csv
    ├── test/docvqa.csv
    └── images/
```

The materialized payloads, images, and download cache are ignored by Git. Do not
commit them to either the public repository or an anonymous review archive.

## Validate prepared data

Before making model API calls, verify that every released ID is present and that
the update and held-out sets remain disjoint:

```bash
python scripts/run_graphskillaa.py --dataset all --seed all --check-only
```

`--check-only` requires the materialized payloads. `--check-manifests` checks only
the committed identifier metadata.
