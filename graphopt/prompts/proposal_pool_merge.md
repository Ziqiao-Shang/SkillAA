You compare the current batch summary against the cumulative modification opinions from earlier batches. The first batch has no earlier opinions; each later call produces the new cumulative version, so no final extra summary is run.

The user JSON uses `graphopt-exact-evidence-references`: resolve every `__graphopt_exact_ref__` through `evidence_catalog` verbatim before reasoning. This is lossless deduplication, not a summary.

Return strict JSON only:

{"clusters":[{"opinion_ids":["pool_001_0001","pool_001_0002"],"synthesis":"one new cumulative modification opinion"}]}

Rules:

- This rolling comparison receives modification-opinion text only; `source_cases` is intentionally empty so earlier raw evidence is never resent. Compare observable conditions, actions, exceptions, and output constraints in the opinions themselves.
- Every input opinion_id must appear exactly once.
- Never put opinions with different `bucket` values in one cluster. All new specialists share the new-node bucket: merge them only when the modification-opinion semantics are genuinely equivalent. Different parent IDs alone do not force a split; `activation_edges` are evidence-backed triggers that Python preserves and unions for an equivalent specialist.
- Merge opinions when they express the same observable condition, wrong-decision
  family, and replacement action or output constraint, even if wording, entities,
  cell ranges, or examples differ.
- Keep opinions separate when conditions, required actions, output types,
  exception behavior, or safety constraints differ.
- For every cluster, write one `synthesis` that becomes the new cumulative modification opinion. Preserve every supported observable condition, action, exception, and output constraint; do not invent rules. For a singleton, preserve its meaning without unnecessary rewriting.
- Return only `opinion_ids` and `synthesis` for each cluster.
