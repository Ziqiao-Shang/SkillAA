# GraphOpt New-Node Synthesizer

The proposed rule has already been judged novel against every existing node and has reached
the required distinct-case support. `source_bad_cases` contains every involved complete case: original question, student answer, evaluator/reference, full reasoning evidence, graph usage, and all companion graph proposals. Convert the merged evidence into one complete reusable
SkillGraph node.

Return strict JSON with exactly these four fields:

```json
{
  "title": "Short stable rule name",
  "when_to_use": "When the agent should activate it",
  "how_to_use": "Concrete ordered actions",
  "avoid": ["Specific behavior to avoid"]
}
```

`avoid` must be a JSON array even when there is only one sentence; never return a string.

Do not output an ID, statistics, category, section, Markdown, or unsupported rules. Preserve
all useful evidence in the merged opinion while removing repetition. The title must be concise;
the other fields must be operational and self-contained.

Inspect every complete source case and reconcile the rule against the original question and answer; never treat the reasoning trace as the sole authority. Preserve a composite rule when the same cases require coordinated node/edge behavior.

Use the narrowest reusable rule supported by every source statement. `when_to_use` must name
only observable pre-action cues. `how_to_use` must specify the state check, next action, and
state update or stopping condition. Put genuine exceptions in the procedure instead of turning
one failure pattern into an unconditional always/never rule. Do not add plausible domain advice
that is absent from the evidence. If source statements conflict, preserve the conflict as a
conditioned branch only when the observable distinction is explicit; otherwise return invalid
output so materialization is abandoned.
