# Complete-case evidence rule

Every source bad case must be checked using its original task, student answer, evaluator/training reference, complete reasoning evidence, graph usage, and companion node/edge proposals together. The reasoning trace is evidence, not ground truth. Preserve source-coupled composite changes.

# Execution Reinforcement

Lightly emphasize the repeated execution reminder in `how_to_use` only.

Return exactly one JSON object: `{"emphasis":""}` or `{"emphasis":null}`. The program appends this short
emphasis to the unchanged original procedure.

- Preserve the original meaning, trigger, procedure, exceptions, and scope.
- Do not add a new rule, condition, prohibition, threshold, or graph relation.
- Do not make the instruction unconditional or stronger than its original semantics.
- Prefer a small wording emphasis or short reminder over rewriting the procedure.
- Treat every Successful Use as counter-evidence: remain compatible with every path.
- Verify in every Execution-Lapse Case that the original instruction was active, already
  stated the requested behavior, and was not followed. Otherwise return null.
- Return null if the reminder introduces any new semantics or cannot be expressed as a
  faithful emphasis of text already present in the original procedure.
- Output no field other than `emphasis` and do not use Markdown.
