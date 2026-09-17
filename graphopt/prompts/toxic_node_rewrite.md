# Complete-case evidence rule

Every source bad case must be checked using its original task, student answer, evaluator/training reference, complete reasoning evidence, graph usage, and companion node/edge proposals together. The reasoning trace is evidence, not ground truth. Preserve source-coupled composite changes.

# GraphOpt Toxic Span Rewriter

An exact span in an otherwise useful node has been proven harmful. Replace only that span.

Return exactly `{"replacement":""}` or `{"replacement":null}`.

- The replacement must correct the supplied bad cases and remain compatible with every
  protected success.
- Preserve the original scope and all valid exceptions; do not add unrelated experience.
- Return one concise replacement of at most 80 words. No Markdown or extra keys.
- Return null if the evidence does not justify a safe replacement, if the replacement would
  alter valid behavior outside the exact toxic span, or if protected successes conflict.
