"""LiveMath-owned update, rollback, and attribution policy."""

import hashlib
import json
from pathlib import Path
from typing import Any

ENVIRONMENT = "livemathematicianbench"

ALLOW_EXECUTION_CHILDREN = False
MAX_ATOMIC_GROUPS_PER_SMALL_GATE = 1
REQUIRE_NONNEGATIVE_COMBINED_ON_VALIDATION_GAIN = True
REUSE_PREVIOUS_COMMITTED_TRAIN_AS_NEXT_EPOCH_COLLECTION = True
# Large LiveMath components can exceed the complete per-edit response contract.
# Use the engine deterministic fallback only for this dataset: preserve every
# grounded edit for paired Local Gate measurement instead of dropping updates.
MAX_JOINT_SEMANTIC_COMPONENT_EDITS = 32

_META_OMIT = object()
_META_RAW_EVIDENCE_KEYS = frozenset({
    "before",
    "after",
    "paired_cases",
    "changed_pairs",
    "question",
    "task_description",
    "response",
    "trajectory",
    "training_reference",
    "conversation",
    "target_system_prompt",
    "target_user_prompt",
    "system",
    "user",
    "attempts",
})
_META_MAX_CHARS = 3_000_000


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _MetaEvidenceCatalog:
    """Deduplicate exact task inputs and student outputs by content hash."""

    def __init__(self) -> None:
        self.inputs: dict[str, dict[str, Any]] = {}
        self.outputs: dict[str, dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.analysis_items: dict[str, Any] = {}

    @staticmethod
    def _insert(table: dict[str, dict[str, Any]], ref: str, value: dict[str, Any]) -> None:
        if ref in table and table[ref] != value:
            raise ValueError(f"Meta evidence hash collision: {ref}")
        table[ref] = value

    def add_state(self, value: Any, *, fallback_case_id: str = "") -> str | None:
        if not isinstance(value, dict):
            return None
        case_id = str(value.get("id") or fallback_case_id or "")
        question = value.get("question")
        task_description = value.get("task_description")
        input_payload: dict[str, Any] = {"case_id": case_id}
        if question not in (None, ""):
            input_payload["question"] = question
        if task_description not in (None, "", question):
            input_payload["task_description"] = task_description
        input_digest = _value_sha256(input_payload)
        input_ref = f"input:{input_digest[:24]}"
        self._insert(self.inputs, input_ref, {**input_payload, "content_sha256": input_digest})

        response = str(value.get("response") or "")
        trace = str(value.get("semantic_reasoning_trace") or "")
        if trace and trace not in response:
            raise ValueError(f"LiveMath state {case_id!r} has a trace outside response")
        output_payload = {"response": response}
        output_digest = _value_sha256(output_payload)
        output_ref = f"output:{output_digest[:24]}"
        self._insert(self.outputs, output_ref, {**output_payload, "content_sha256": output_digest})

        state_payload = {
            "case_id": case_id,
            "input_ref": input_ref,
            "output_ref": output_ref,
            "hard": value.get("hard"),
            "soft": value.get("soft"),
            "predicted_text": value.get("predicted_text"),
            "correct_text": value.get("correct_text"),
            "fail_reason": value.get("fail_reason"),
            "semantic_reasoning_trace_status": value.get("semantic_reasoning_trace_status"),
            "graph_refs": value.get("graph_refs"),
            "raw_state_sha256": _value_sha256(value),
        }
        state_digest = _value_sha256(state_payload)
        state_ref = f"state:{state_digest[:24]}"
        self._insert(self.states, state_ref, state_payload)
        return state_ref

    def pair_refs(self, value: Any) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        if not isinstance(value, list):
            return refs
        for pair in value:
            if not isinstance(pair, dict):
                continue
            case_id = str(pair.get("case_id") or "")
            row = {
                key: pair[key]
                for key in ("case_id", "transition", "category", "hard_before", "hard_after", "soft_before", "soft_after")
                if pair.get(key) not in (None, "")
            }
            before_ref = self.add_state(pair.get("before"), fallback_case_id=case_id)
            after_ref = self.add_state(pair.get("after"), fallback_case_id=case_id)
            if before_ref:
                row["before_state_ref"] = before_ref
            if after_ref:
                row["after_state_ref"] = after_ref
            row["raw_pair_sha256"] = _value_sha256(pair)
            refs.append(row)
        return refs

    def analysis_refs(self, value: Any) -> list[str]:
        refs: list[str] = []
        if not isinstance(value, list):
            return refs
        for item in value:
            digest = _value_sha256(item)
            ref = f"analysis:{digest[:24]}"
            if ref in self.analysis_items and self.analysis_items[ref] != item:
                raise ValueError(f"Meta analysis evidence hash collision: {ref}")
            self.analysis_items[ref] = item
            refs.append(ref)
        return refs

    def render(self) -> dict[str, Any]:
        return {
            "schema_version": "livemath-meta-evidence-catalog-v1",
            "input_contract": "exact_task_input_stored_once_by_content_hash",
            "output_contract": "exact_student_response_stored_once_by_content_hash",
            "trace_contract": "semantic_reasoning_trace_is_verbatim_inside_exact_response",
            "inputs": dict(sorted(self.inputs.items())),
            "outputs": dict(sorted(self.outputs.items())),
            "states": dict(sorted(self.states.items())),
            "analysis_items": dict(sorted(self.analysis_items.items())),
        }


def _compact_meta_value(value: Any, *, key: str = "") -> Any:
    """Remove raw rollout bodies while retaining measured decisions and edit semantics."""
    if key in _META_RAW_EVIDENCE_KEYS:
        return _META_OMIT
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for child_key, child_value in value.items():
            rendered = _compact_meta_value(child_value, key=str(child_key))
            if rendered is not _META_OMIT:
                compact[str(child_key)] = rendered
        return compact
    if isinstance(value, list):
        compact_list = []
        for child in value:
            rendered = _compact_meta_value(child)
            if rendered is not _META_OMIT:
                compact_list.append(rendered)
        return compact_list
    return value


def _semantic_with_evidence_refs(
    value: Any, catalog: _MetaEvidenceCatalog, *, key: str = ""
) -> Any:
    """Preserve semantic fields verbatim while replacing state bodies by refs."""
    if key in {"paired_cases", "changed_pairs"}:
        return catalog.pair_refs(value)
    if key == "evidence_items":
        return catalog.analysis_refs(value)
    if key in _META_RAW_EVIDENCE_KEYS:
        return _META_OMIT
    if isinstance(value, dict):
        rendered: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child = _semantic_with_evidence_refs(
                child_value, catalog, key=str(child_key)
            )
            if child is not _META_OMIT:
                rendered[str(child_key)] = child
        return rendered
    if isinstance(value, list):
        rendered_list = []
        for item in value:
            child = _semantic_with_evidence_refs(item, catalog)
            if child is not _META_OMIT:
                rendered_list.append(child)
        return rendered_list
    return value


def _meta_transition_summary(
    value: Any, catalog: _MetaEvidenceCatalog
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    transitions = value.get("transitions")
    transitions = transitions if isinstance(transitions, dict) else {}
    return {
        "n_cases": len(value.get("case_ids") or []),
        "n_improved": int(value.get("n_improved") or 0),
        "n_regressed": int(value.get("n_regressed") or 0),
        "net": int(value.get("net") or 0),
        "improved_case_ids": list(transitions.get("0->1") or []),
        "regressed_case_ids": list(transitions.get("1->0") or []),
        "changed_pairs": catalog.pair_refs(value.get("changed_pairs")),
    }


def _meta_edit_summary(
    value: Any, catalog: _MetaEvidenceCatalog
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    edit = value.get("edit") if isinstance(value.get("edit"), dict) else value
    summary: dict[str, Any] = {}
    for key in (
        "edit_index", "op", "node_id", "edge_id", "source", "target",
        "relation", "group_id", "joint_group_id", "decision",
    ):
        candidate = value.get(key) if key in value else edit.get(key)
        if candidate not in (None, "", [], {}):
            summary[key] = candidate
    source_case_ids = edit.get("source_case_ids")
    if isinstance(source_case_ids, list):
        summary["source_case_ids"] = source_case_ids
    for key in ("reason", "reasoning", "semantic_delta"):
        candidate = value.get(key) if key in value else edit.get(key)
        if candidate not in (None, ""):
            summary[key] = _semantic_with_evidence_refs(candidate, catalog)
    return summary


def _meta_attribution_summary(
    value: Any, catalog: _MetaEvidenceCatalog
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    summary = {
        key: value[key]
        for key in (
            "atomic_group_id", "edit_indices", "touch_keys",
            "direct_source_train_case_ids", "protected_train_regression_case_ids",
            "protected_validation_regression_case_ids", "script_keep",
            "script_decision", "usage_complete", "scope_coverage_complete",
        )
        if value.get(key) not in (None, "", [], {})
    }
    summary["train"] = _meta_transition_summary(value.get("train"), catalog)
    summary["validation"] = _meta_transition_summary(value.get("validation"), catalog)
    summary["edits"] = [
        _meta_edit_summary(edit, catalog) for edit in value.get("edits") or []
    ]
    return summary


def _meta_small_gate_summary(
    value: Any, catalog: _MetaEvidenceCatalog
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    summary = {
        key: value[key]
        for key in (
            "epoch", "joint_candidate_index", "attempt", "accepted", "action",
            "patch_sha256", "n_edits", "n_evidence_cases", "train_size",
            "validation_size", "train_case_ids", "validation_case_ids",
        )
        if value.get(key) not in (None, "", [], {})
    }
    local = value.get("local_gate")
    if isinstance(local, dict):
        local_summary = {
            key: local[key]
            for key in (
                "policy", "accepted", "n_effective", "n_ineffective",
                "effective_case_ids", "harmful_case_ids", "ineffective_case_ids",
                "unresolved_source_case_ids", "source_case_ids", "decision_reason",
            )
            if local.get(key) not in (None, "", [], {})
        }
        explanation = local.get("semantic_explanation")
        if isinstance(explanation, dict):
            local_summary["semantic_explanation"] = {
                key: _semantic_with_evidence_refs(explanation[key], catalog, key=key)
                for key in ("status", "joint_summary", "edits")
                if explanation.get(key) not in (None, "", [], {})
            }
        elif explanation not in (None, ""):
            local_summary["semantic_explanation"] = _semantic_with_evidence_refs(
                explanation, catalog
            )
        local_summary["paired_cases"] = catalog.pair_refs(local.get("paired_cases"))
        summary["local_gate"] = local_summary
    patch = value.get("joint_patch")
    if isinstance(patch, dict):
        summary["joint_patch"] = {
            "reasoning": _semantic_with_evidence_refs(patch.get("reasoning") or "", catalog),
            "edits": [_meta_edit_summary(edit, catalog) for edit in patch.get("edits") or []],
        }
    return summary


def _coverage_manifest(raw: dict[str, Any], compact: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_record_sha256": _value_sha256(raw),
        "epoch": raw.get("epoch"),
        "small_gate_count": len(raw.get("small_gates") or []),
        "attribution_group_count": len(raw.get("edit_attribution") or []),
        "root_attribution_edit_indices": [
            list(row.get("edit_indices") or [])
            for row in raw.get("edit_attribution") or []
        ],
        "small_gate_explanation_edit_indices": [
            [
                edit.get("edit_index")
                for edit in (
                    ((gate.get("local_gate") or {}).get("semantic_explanation") or {}).get("edits")
                    or []
                )
                if isinstance(edit, dict)
            ]
            for gate in raw.get("small_gates") or []
        ],
        "kept_edit_indices": list(raw.get("kept_edit_indices") or []),
        "rolled_back_edit_indices": list(raw.get("rolled_back_edit_indices") or []),
        "compact_sha256_before_manifest": _value_sha256(compact),
    }


def validate_compact_meta_gate_history(
    raw_history: list[dict[str, Any]], compact_history: list[dict[str, Any]]
) -> None:
    """Fail closed unless the compact ledger covers every LiveMath Gate decision."""
    if len(raw_history) != len(compact_history):
        raise ValueError("LiveMath Meta record count changed during compaction")
    for raw, compact in zip(raw_history, compact_history):
        manifest = compact.get("coverage_manifest") or {}
        if manifest.get("raw_record_sha256") != _value_sha256(raw):
            raise ValueError("LiveMath Meta raw Gate hash mismatch")
        if raw.get("epoch") != compact.get("epoch"):
            raise ValueError("LiveMath Meta epoch changed during compaction")
        raw_gates = raw.get("small_gates") or []
        compact_gates = compact.get("small_gates") or []
        if len(raw_gates) != len(compact_gates):
            raise ValueError("LiveMath Meta small-Gate coverage mismatch")
        raw_attr = raw.get("edit_attribution") or []
        compact_attr = compact.get("edit_attribution") or []
        if len(raw_attr) != len(compact_attr):
            raise ValueError("LiveMath Meta attribution coverage mismatch")
        for raw_gate, compact_gate in zip(raw_gates, compact_gates):
            for key in ("accepted", "action", "joint_candidate_index", "attempt"):
                if raw_gate.get(key) != compact_gate.get(key):
                    raise ValueError(f"LiveMath Meta small-Gate field changed: {key}")
            raw_local = raw_gate.get("local_gate") or {}
            compact_local = compact_gate.get("local_gate") or {}
            for key in (
                "accepted", "effective_case_ids", "harmful_case_ids",
                "ineffective_case_ids", "unresolved_source_case_ids",
            ):
                if key == "accepted":
                    equal = raw_local.get(key) == compact_local.get(key)
                else:
                    equal = list(raw_local.get(key) or []) == list(
                        compact_local.get(key) or []
                    )
                if not equal:
                    raise ValueError(f"LiveMath Meta Local-Gate field changed: {key}")
            raw_edits = (raw_local.get("semantic_explanation") or {}).get("edits") or []
            compact_edits = (compact_local.get("semantic_explanation") or {}).get("edits") or []
            if [row.get("edit_index") for row in raw_edits] != [
                row.get("edit_index") for row in compact_edits
            ]:
                raise ValueError("LiveMath Meta semantic edit coverage mismatch")
            raw_pairs = (raw_local.get("paired_cases") or [])
            compact_pairs = compact_local.get("paired_cases") or []
            if [_value_sha256(row) for row in raw_pairs] != [
                row.get("raw_pair_sha256") for row in compact_pairs
            ]:
                raise ValueError("LiveMath Meta Local-Gate evidence pair mismatch")
        for raw_row, compact_row in zip(raw_attr, compact_attr):
            if list(raw_row.get("edit_indices") or []) != list(compact_row.get("edit_indices") or []):
                raise ValueError("LiveMath Meta attribution edit indices changed")
            for split in ("train", "validation"):
                transitions = (raw_row.get(split) or {}).get("transitions") or {}
                rendered = compact_row.get(split) or {}
                if (
                    list(transitions.get("0->1") or [])
                    != list(rendered.get("improved_case_ids") or [])
                    or list(transitions.get("1->0") or [])
                    != list(rendered.get("regressed_case_ids") or [])
                ):
                    raise ValueError(f"LiveMath Meta {split} transition IDs changed")
                if [
                    _value_sha256(row)
                    for row in (raw_row.get(split) or {}).get("changed_pairs") or []
                ] != [
                    row.get("raw_pair_sha256")
                    for row in rendered.get("changed_pairs") or []
                ]:
                    raise ValueError(f"LiveMath Meta {split} evidence pair mismatch")
        catalog = compact.get("evidence_catalog") or {}
        states = catalog.get("states") or {}
        inputs = catalog.get("inputs") or {}
        outputs = catalog.get("outputs") or {}
        analysis_items = catalog.get("analysis_items") or {}
        for ref, item in inputs.items():
            payload = {key: value for key, value in item.items() if key != "content_sha256"}
            digest = _value_sha256(payload)
            if ref != f"input:{digest[:24]}" or item.get("content_sha256") != digest:
                raise ValueError("invalid LiveMath Meta input content hash")
        for ref, item in outputs.items():
            payload = {key: value for key, value in item.items() if key != "content_sha256"}
            digest = _value_sha256(payload)
            if ref != f"output:{digest[:24]}" or item.get("content_sha256") != digest:
                raise ValueError("invalid LiveMath Meta output content hash")
        for ref, item in analysis_items.items():
            digest = _value_sha256(item)
            if ref != f"analysis:{digest[:24]}":
                raise ValueError("invalid LiveMath Meta analysis content hash")
        for state_ref, state in states.items():
            if not str(state_ref).startswith("state:"):
                raise ValueError("invalid LiveMath Meta state ref")
            if state.get("input_ref") not in inputs or state.get("output_ref") not in outputs:
                raise ValueError("unresolved LiveMath Meta input/output ref")
        def collect_analysis_refs(value: Any) -> set[str]:
            if isinstance(value, dict):
                found: set[str] = set()
                for child in value.values():
                    found.update(collect_analysis_refs(child))
                return found
            if isinstance(value, list):
                found = set()
                for child in value:
                    found.update(collect_analysis_refs(child))
                return found
            if isinstance(value, str) and value.startswith("analysis:"):
                return {value}
            return set()
        if not collect_analysis_refs(compact).issubset(set(analysis_items)):
            raise ValueError("unresolved LiveMath Meta analysis ref")
        rendered_chars = len(_canonical_json(compact))
        if rendered_chars > _META_MAX_CHARS:
            raise ValueError(
                f"LiveMath Meta compact record is still too large: "
                f"{rendered_chars} > {_META_MAX_CHARS} chars"
            )


def compact_gate_experience(record: dict[str, Any]) -> dict[str, Any]:
    """Keep prior LiveMath Gate decisions without repeating raw rollout bodies."""
    from graphopt.optimizer.meta_compaction import compact_meta_decision_ledger

    return compact_meta_decision_ledger(
        [record], environment=ENVIRONMENT
    )[0]


def compact_meta_gate_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate exact LiveMath I/O while retaining Gate decision semantics."""
    compact_history: list[dict[str, Any]] = []
    for record in history:
        catalog = _MetaEvidenceCatalog()
        compact = _compact_meta_value(record)
        if isinstance(compact, dict):
            compact["small_gates"] = [
                _meta_small_gate_summary(row, catalog)
                for row in record.get("small_gates") or []
            ]
            compact["edit_attribution"] = [
                _meta_attribution_summary(row, catalog)
                for row in record.get("edit_attribution") or []
            ]
            compact["meta_history_view"] = "livemath_exact_io_refs_v2"
            compact["raw_gate_artifact"] = {
                "path": "gate_history.jsonl",
                "epoch": record.get("epoch"),
                "record_sha256": _value_sha256(record),
                "contract": "complete_raw_record_unchanged_on_disk",
            }
            compact["evidence_catalog"] = catalog.render()
            compact["coverage_manifest"] = _coverage_manifest(record, compact)
            compact_history.append(compact)
    validate_compact_meta_gate_history(history, compact_history)
    return compact_history


def _initial_node_ids() -> frozenset[str]:
    """Protect exactly this environment's current semantic G0 roots."""
    path = (
        Path(__file__).resolve().parents[2]
        / "envs"
        / "livemathematicianbench"
        / "initial_skill"
        / "best_graph.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(
        str(node["id"])
        for node in data["stable_rule_graph"]["nodes"]
    )


# Case-specific theorem recipes must become genuinely new narrow nodes rather
# than rewriting the reusable semantic roots established by initialization.
PROTECTED_ROOT_NODE_IDS: frozenset[str] = _initial_node_ids()


def case_analyzer_prompt_addendum() -> str:
    return """\
## LiveMath-specific conservative update policy

LiveMath supplies one answer response rather than graph-guided reasoning
actions. When semantic_reasoning_trace_status is validated_student_trace, treat
the trace as the primary observable semantic path through implication,
condition, scope, boundary, quantitative check, and option comparison.
Cross-check it against the saved response, evaluator evidence, and same-group
successful sibling traces. A generated_posthoc_trace is a weaker root-cause
hypothesis and cannot establish an unstated mathematical step. Do not infer
hidden steps. When trace_evidence.status is VERIFIED or MULTI_REPEAT_VERIFIED, used_nodes and used_edges
are immutable. A cited correct node is an execution lapse only when the trace
shows it produced the correct mathematical intermediate result and a later stated
step diverged; record that lapse but emit no graph proposal. Use
badcase_summary to isolate the earliest mathematical divergence before
considering an edit. In particular, do not restate or strengthen the protected LiveMath
G0 semantic roots. Their option-comparison, theorem-precision, condition,
scope, quantitative, and final-answer semantics are already represented. Propose a
normal PATCH or genuinely new narrow node only when semantic_delta identifies
new reusable mathematical decision semantics not entailed by the existing
graph; otherwise emit no proposal.
"""


def accept_small_candidate(validation_audit: dict, train_audit: dict, *, use_train_tiebreak: bool) -> bool:
    """Accept a Local-Gate candidate only on positive affected-train gain."""
    train_net = int(train_audit.get("hard_net_case_gain") or 0)
    train_observed = int(train_audit.get("n_eligible") or 0) > 0
    del validation_audit, use_train_tiebreak
    return bool(train_observed and train_net > 0)


def accept_complete_candidate(
    validation_audit: dict,
    train_audit: dict,
    *,
    allow_validation_tie: bool,
    use_train_tiebreak: bool,
) -> bool:
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
    """Infer conservative success usage only from LiveMath semantics."""
    text = text.casefold()
    nodes = [node for node in ("M004", "M014") if node in graph.nodes]
    families = (
        (("stronger", "meta-option"), "M001"),
        (("nested conclusion", "rank by strength"), "M003"),
        (("quantifier", "for every", "there exists", "unique"), "M005"),
        (("converse", "classification"), "M006"),
        (("characterization", "equality clause"), "M007"),
        (("overstatement",), "M008"),
        (("equivalent", "equivalence", "if and only if"), "M009"),
        (("congruence", "broader condition", "modulus"), "M017"),
        (("threshold", "parameter sign"), "M010"),
        (("endpoint", "strict", "non-strict"), "M018"),
        (("equality regime", "positive regime", "negative regime"), "M019"),
        (("scope", "quantifier strength"), "M011"),
        (("hypothesis", "domain", "condition"), "M012"),
        (("equality", "extremal", "family"), "M013"),
        (("global", "localized", "local"), "M015"),
        (("estimate", "constant", "exponent", "derivative", "regularity"), "M016"),
        (("logarithmic", "additive term", "exceptional-set", "rate"), "M020"),
        (("pointwise", "uniform convergence", "one-sided", "two-sided"), "M021"),
    )
    nodes.extend(node for keys, node in families if node in graph.nodes and any(k in text for k in keys))
    return list(dict.fromkeys(nodes))


def template_failure_revision(graph, text: str):
    """Return one LiveMath-specific conservative PATCH target."""
    text = text.casefold()
    rules = (
        (("equivalent", "equivalence", "if and only if", "converse"),
         "Check both directions and the exact admitted cases before selecting an equivalence or classification claim.", ("M009", "M006", "M017"),
         "The response did not verify both logical directions."),
        (("threshold", "endpoint", "strict", "equality"),
         "Verify the threshold sign, endpoint, and equality regime exactly.", ("M010", "M018", "M019", "M013"),
         "The quantitative boundary was not checked exactly."),
        (("hypothesis", "domain", "condition"),
         "Verify every stated hypothesis and domain restriction.", ("M012",),
         "A required hypothesis or domain restriction was missed."),
        (("stronger", "weaker", "nested", "meta-option"),
         "Rank all justified conclusions by strength before selecting the option.", ("M001", "M002", "M003", "M004"),
         "The selected option was not the strongest justified conclusion."),
        (("global", "local", "localized", "scope", "quantifier"),
         "Compare scope and quantifier strength without promoting a local claim to a global one.", ("M005", "M011", "M015"),
         "The conclusion's scope or quantifier was overstated."),
        (("estimate", "constant", "exponent", "derivative", "logarithmic", "convergence"),
         "Compare every quantitative estimate detail and convergence mode exactly.", ("M016", "M020", "M021"),
         "A quantitative estimate detail or convergence mode was mismatched."),
        (("option", "label", "format"),
         "Return exactly one justified option label after completing the comparison.", ("M014",),
         "The final answer format was invalid."),
    )
    for keys, proposal, targets, reason in rules:
        if any(key in text for key in keys):
            return proposal, [node for node in targets if node in graph.nodes], reason
    return None
