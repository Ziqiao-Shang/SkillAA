"""SearchQA-owned update, rollback, and attribution policy."""

import re

ENVIRONMENT = "searchqa"

# SearchQA remains a one-response task, so verified trace lapses are diagnostic
# only. They may identify the exact correct-but-misapplied node, but may not
# synthesize an execution-child skill.
ALLOW_EXECUTION_CHILDREN = False
MAX_ATOMIC_GROUPS_PER_SMALL_GATE = 1
REQUIRE_NONNEGATIVE_COMBINED_ON_VALIDATION_GAIN = True
REUSE_PREVIOUS_COMMITTED_TRAIN_AS_NEXT_EPOCH_COLLECTION = True
PROTECTED_ROOT_NODE_IDS: frozenset[str] = frozenset()


def _contains_any(text: str, signals: tuple[str, ...]) -> bool:
    """Match atomic word signals without letting ``pun`` match punctuation."""
    for signal in signals:
        if re.fullmatch(r"[a-z0-9]+", signal):
            if re.search(rf"(?<![a-z0-9_]){re.escape(signal)}(?![a-z0-9_])", text):
                return True
        elif signal in text:
            return True
    return False


def case_analyzer_prompt_addendum() -> str:
    return """\
## SearchQA root-cause and solution policy

SearchQA supplies one final response rather than graph-guided actions. When
semantic_reasoning_trace_status is validated_student_trace, treat the trace as
the primary observable path from requested answer type through evidence,
relation/constraint, and surface form. Cross-check every step against the saved
response, evaluator evidence, and all same-group successful sibling traces. A
generated_posthoc_trace is a weaker root-cause hypothesis and cannot establish
unstated context evidence. Do not infer hidden steps. When trace_evidence.status
is VERIFIED or MULTI_REPEAT_VERIFIED, used_nodes and used_edges are immutable. Mark a cited correct node
as an execution lapse only when the trace shows that it produced the right
intermediate result and a later stated step diverged; emit no graph proposal for
that lapse. Otherwise distinguish a harmful stored rule from a missing procedure.

Before proposing an edit, use badcase_summary to identify the smallest
observable semantic divergence between the failed path and successful siblings.

Choose the edit channel from that diagnosis, rather than defaulting to a
trigger rewrite:

- retrieval_revision_proposals is only for a demonstrably absent activation
  cue in the original when_to_use;
- node_revision_proposals is for missing reusable evidence selection,
  relation direction, answer-type, constraint, or surface-form procedure;
- new_node_proposals is for a coherent procedure not represented by any
  existing node.

A high-confidence reusable gap should produce a concrete opinion even when it
first appears in one fixed update group; epoch-wide joint synthesis and Gates decide
whether it generalizes. Each opinion's reason and semantic_delta must state
the observable condition, the wrong decision family, the replacement action,
and why successful siblings remain valid. If distinct evidence supports two
different causes, emit separate non-duplicate opinions for their appropriate
targets. Never copy an answer, named instance, or hindsight-only condition.
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
    """Infer conservative success usage only from SearchQA semantics."""
    text = text.casefold()
    nodes = [node for node in ("Q001", "Q008") if node in graph.nodes]
    families = (
        (("spelling", "punctuation", "surface form", "exact name"), "Q004"),
        (("person", "surname", "saint"), "Q005"),
        (("plural", "singular", "grammatical base"), "Q006"),
        (("replaced", "substitute", "definition", "headword"), "Q007"),
        (("known as", "called", "defined as", "canonical"), "Q010"),
        (("hard modifier", "largest", "date constraint"), "Q012"),
        (("author", "director", "actor", "played", "wrote"), "Q013"),
        (("placeholder", "fill-in"), "Q014"),
        (("shared category", "shared class", "shared synonym"), "Q015"),
        (("crossword", "letter count", "dual-definition"), "Q016"),
        (("pictured", "seen here", "image"), "Q017"),
        (("corroborating", "multiple snippets"), "Q018"),
        (("right:", "category |", "jeopardy", "answer field"), "Q019"),
        (("wife", "husband", "named after", "inverse", "relation direction"), "Q020"),
        (("include", "such as", "parent class"), "Q021"),
        (("quote", "quotation", "song", "lyric", "poem", "continuation"), "Q022"),
        (("abbreviation", "acronym", "first name"), "Q023"),
        (("wordplay", "pun"), "Q024"),
        (("quoted title", "slogan", "associated entity"), "Q025"),
    )
    nodes.extend(node for keys, node in families if node in graph.nodes and _contains_any(text, keys))
    return list(dict.fromkeys(nodes))


def template_failure_revision(graph, text: str):
    """Return one SearchQA-specific conservative PATCH target."""
    text = text.casefold()
    rules = (
        (("wife", "husband", "named after", "inverse", "relation direction"),
         "Preserve relation direction and return the requested endpoint.", ("Q020",),
         "The answer selected the wrong endpoint of the stated relation."),
        (("include", "such as", "parent class"),
         "Return the parent class requested by the clue rather than one named example.", ("Q021",),
         "The response returned an example instead of the requested parent class."),
        (("quote", "quotation", "lyric", "continuation"),
         "Distinguish a requested quotation continuation from its associated work or creator.", ("Q022", "Q025"),
         "The response confused quotation text with an associated entity."),
        (("crossword", "letter count"),
         "Apply the clue's requested form and length constraints before answering.", ("Q016",),
         "The response did not satisfy the clue form."),
        (("placeholder", "fill-in"),
         "Substitute the candidate into the clue and preserve the requested standalone form.", ("Q014",),
         "The response did not satisfy the fill-in form."),
        (("shared category", "shared class", "shared synonym"),
         "Infer the category shared by the listed examples.", ("Q015",),
         "The response did not identify the requested shared category."),
        (("author", "director", "actor", "played", "wrote", "type"),
         "Validate the requested answer type and every hard modifier.", ("Q011", "Q012", "Q013"),
         "The response did not match the requested entity type."),
        (("spelling", "punctuation", "exact", "surface"),
         "Preserve the strongest evidence's exact supported surface form.", ("Q004",),
         "The answer surface form drifted from the evidence."),
        (("evidence", "snippet", "context", "title"),
         "Prefer evidence that jointly matches the question's distinctive terms.", ("Q008", "Q009", "Q010"),
         "The selected evidence was insufficiently specific."),
    )
    for keys, proposal, targets, reason in rules:
        if _contains_any(text, keys):
            return proposal, [node for node in targets if node in graph.nodes], reason
    return None
