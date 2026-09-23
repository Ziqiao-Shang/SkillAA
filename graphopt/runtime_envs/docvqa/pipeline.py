"""DocVQA-specific update and rollback pipeline policy.

This module is the environment-owned boundary used by the formal runner.  It
must not import SearchQA or LiveMath policy modules.
"""

ENVIRONMENT = "docvqa"
ALLOW_EXECUTION_CHILDREN = False
MAX_ATOMIC_GROUPS_PER_SMALL_GATE = 1
REQUIRE_NONNEGATIVE_COMBINED_ON_VALIDATION_GAIN = True
REUSE_PREVIOUS_COMMITTED_TRAIN_AS_NEXT_EPOCH_COLLECTION = True
PROTECTED_ROOT_NODE_IDS: frozenset[str] = frozenset()

import json
from pathlib import Path
from typing import Any


def case_analyzer_prompt_addendum() -> str:
    return """\
## DocVQA-specific conservative update policy

DocVQA supplies one answer response rather than an observable sequence of
graph-guided visual actions. When semantic_reasoning_trace_status is
validated_student_trace, treat semantic_reasoning_trace as the primary observable
semantic decision path: locate its earliest unsupported evidence selection,
disambiguation, or span-boundary decision, then cross-check that diagnosis against
the response, evaluator evidence, and successful siblings. The trace is a student
self-report, not document ground truth. When status is generated_posthoc_trace, use
it only as a weaker root-cause hypothesis and never treat unquoted document content as
observed evidence. Do not infer any hidden step absent from these fields.
Use badcase_summary to isolate the earliest visual evidence-selection or span-
boundary divergence before considering an edit.

When trace_evidence.status is VERIFIED or MULTI_REPEAT_VERIFIED, used_nodes and used_edges are
immutable. A cited correct node is an execution lapse only when the trace shows a
correct intermediate visual result followed by a later stated divergence; record
the lapse but emit no graph proposal. A PATCH is legal only when semantic_delta adds a new reusable visual-anchor, layout-
disambiguation, table-coordinate, or exact-span decision that is absent from the
graph; otherwise emit no proposal.
"""


def accept_small_candidate(validation_audit: dict, train_audit: dict, *, use_train_tiebreak: bool) -> bool:
    """Accept a Local-Gate candidate only on positive affected-train gain."""
    train_net = int(train_audit.get("hard_net_case_gain") or 0)
    train_observed = int(train_audit.get("n_eligible") or 0) > 0
    del validation_audit, use_train_tiebreak
    return bool(train_observed and train_net > 0)


def validate_materialized_split(adapter: Any, cfg: dict[str, Any]) -> None:
    """Validate the supported update-only or train/validation DocVQA payload."""
    split_dir = Path(str(cfg.get("split_dir") or "")).expanduser()
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    protocol = str(manifest.get("protocol") or "")
    loader = adapter.get_dataloader()
    pools = {
        "train": list(getattr(loader, "train_items", []) or []),
        "val": list(getattr(loader, "val_items", []) or []),
        "test": list(getattr(loader, "test_items", []) or []),
    }

    if protocol == "graphskillaa_update_quadruples_v1":
        from graphopt.runtime_envs.train_test_split import validate_train_test_dataset

        validate_train_test_dataset(adapter, "docvqa", split_dir, manifest)
        return

    if protocol == "graphskillaa_train_validation":
        expected = {"train": 800, "val": 200}
        loaded = {"train": len(pools["train"]), "val": len(pools["val"])}
        if loaded != expected or manifest.get("counts") != expected:
            raise ValueError(
                f"DocVQA active split counts mismatch: expected={expected}, loaded={loaded}"
            )
        if pools["test"] or (split_dir / "test").exists():
            raise ValueError("DocVQA active dataset must not contain a test split")
        ids = {
            split: [str(item.get("id") or "") for item in pools[split]]
            for split in ("train", "val")
        }
        all_ids = ids["train"] + ids["val"]
        if any(not case_id for case_id in all_ids) or len(all_ids) != len(set(all_ids)):
            raise ValueError("DocVQA active split IDs are empty, duplicated, or overlapping")
        for split in ("train", "val"):
            for item in pools[split]:
                absent = [
                    key for key in ("id", "question")
                    if not str(item.get(key) or "").strip()
                ]
                if not list(item.get("answers") or []):
                    absent.append("answers")
                if not Path(str(item.get("image_path") or "")).is_file():
                    absent.append("image_path")
                if absent:
                    raise ValueError(
                        f"DocVQA {split}/{item.get('id')}: missing {','.join(absent)}"
                    )
        return

    raise ValueError(f"DocVQA unsupported split protocol: {protocol!r}")


def accept_complete_candidate(validation_audit: dict, train_audit: dict, *, allow_validation_tie: bool, use_train_tiebreak: bool) -> bool:
    """Retain only a strict full train+validation hard net gain."""
    validation_net = int(validation_audit.get("hard_net_case_gain") or 0)
    validation_observed = int(validation_audit.get("n_eligible") or 0) > 0
    train_net = int(train_audit.get("hard_net_case_gain") or 0)
    train_observed = int(train_audit.get("n_eligible") or 0) > 0
    del allow_validation_tie
    if not use_train_tiebreak:
        return bool(validation_observed and validation_net > 0)
    return bool(
        validation_observed and train_observed
        and validation_net + train_net > 0
    )

def positive_attribution_nodes(graph, text: str) -> list[str]:
    """Infer conservative success usage only from DocVQA semantics."""
    text = text.casefold()
    nodes = [node for node in ("D001", "D004") if node in graph.nodes]
    families = (
        (("table", "row", "column", "header", "cell"), "D005"),
        (("form", "receipt", "field", "party", "role"), "D006"),
        (("contents", "indexed", "numbered", "bulleted", "list"), "D007"),
        (("handwritten", "margin", "adjacent"), "D008"),
        (("name", "number", "date", "quoted", "punctuation"), "D003"),
        (("nearby", "similar", "layout", "label"), "D002"),
    )
    nodes.extend(node for keys, node in families if node in graph.nodes and any(k in text for k in keys))
    return list(dict.fromkeys(nodes))


def template_failure_revision(graph, text: str):
    """Return one DocVQA-specific conservative PATCH target."""
    text = text.casefold()
    rules = (
        (("table", "row", "column", "header", "cell"),
         "Recheck the named row and requested column before copying the cell.", ("D005",),
         "The response used the wrong table coordinate."),
        (("form", "receipt", "field", "party", "role"),
         "Match the requested label to its adjacent filled value.", ("D006",),
         "The response crossed a form-field boundary."),
        (("contents", "indexed", "numbered", "bulleted", "list"),
         "Follow the requested list item to its associated value.", ("D007",),
         "The response used a nearby list item."),
        (("handwritten", "margin", "adjacent"),
         "Anchor on the requested term before reading the adjacent handwritten span.", ("D008",),
         "The handwriting lookup was not anchored."),
        (("nearby", "similar", "layout", "label"),
         "Use surrounding labels and layout to distinguish nearby candidates.", ("D002",),
         "The response chose the wrong nearby candidate."),
        (("name", "number", "date", "quoted", "punctuation", "exact"),
         "Copy the requested visible surface form exactly.", ("D003",),
         "The answer surface form drifted from the document."),
    )
    for keys, proposal, targets, reason in rules:
        if any(key in text for key in keys):
            return proposal, [node for node in targets if node in graph.nodes], reason
    return None
