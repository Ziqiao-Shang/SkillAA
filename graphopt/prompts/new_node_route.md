# GraphOpt New-Node Semantic Router

The user JSON uses `graphopt-exact-evidence-references`: resolve every `__graphopt_exact_ref__` through `evidence_catalog` verbatim before reasoning. This is lossless deduplication, not a summary.

Decide whether each proposed new-node opinion truly requires a new node.

Before routing, inspect every `source_cases` record as a whole: original task,
student output/answer, evaluator and training-only reference, complete reasoning trace,
and validated graph usage. The trace is supporting evidence, not the sole authority.

Compare every opinion with all four semantic fields of every Existing Node: title,
when_to_use, how_to_use, and avoid. Do not rely on keyword matching. If an existing
node's fields already cover the rule as a whole, treat the opinion as a duplicate.

- If an existing node expresses the same rule, route the opinion to that node and emit its
  real ID in target_node.
- Emit target_node: "NEW" only when no existing node covers the rule.
- Identify and preserve every distinct reusable error family before clustering; do not let a frequent family erase a lower-support family. Put semantically equivalent opinions in one cluster only after their source questions, answers, references, and failure causes agree. Separate opinions that solve
  different problems or conflict.
- Every opinion_id must appear exactly once. Never invent an existing node ID.
- Preserve composite opinions that coordinate several nodes or edges; their shared case provenance makes the downstream edits atomic.
- Inspect activation_candidates together with the full source cases. Never cluster semantically incompatible specialists merely because their wording overlaps. Python preserves every evidence-backed parent activation and materializes the new node plus all of those enhance edges as one atomic group.
- Do not rewrite or summarize any opinion. Python preserves and combines the exact original
  text after you return only the grouping and route.
- Do not output support or case IDs. Python preserves original evidence and assigns final
  IDs to accepted new nodes.

Output JSON only:

~~~json
{
  "clusters": [
    {"opinion_ids": ["opinion_0001", "opinion_0002"], "target_node": "G02"},
    {"opinion_ids": ["opinion_0003"], "target_node": "NEW"}
  ]
}
~~~
