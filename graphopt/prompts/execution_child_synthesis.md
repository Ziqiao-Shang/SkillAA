# GraphOpt Execution-Detail Child Synthesizer

The parent skill is correct and was retrieved, but the student cited it without carrying it
into the next consequential decision. Create one narrow child rule that operationalizes the parent at action, tool-query, code-generation, calculation, or option-selection time. The parent will remain unchanged.

Return strict JSON with exactly these four fields and no Markdown: title,
when_to_use, how_to_use, and avoid. If the evidence cannot justify a safe, reusable execution
detail, return the same four keys with all four values set to null.

The non-null JSON schema is exact:

{"title":"non-empty string","when_to_use":"non-empty string","how_to_use":"non-empty string","avoid":["zero or more strings"]}

`avoid` must always be a JSON array, even when it contains only one sentence. Do not return a
single string in `avoid`.

Hard requirements:

- Do not rewrite, summarize, weaken, or contradict any parent field.
- Do not invent a new task policy. Translate the parent into a small pre-action check.
- The input semantic_cluster_evidence contains every original case record selected for this
  cluster. Derive the shared executable invariant from all records; do not copy case IDs,
  instance numbers, or layout-specific facts into the child.
- Preserve meaningful within-cluster distinctions as observable alternatives or exceptions.
  For example, different empty receptacle types are surface variants, while a state-changing
  revisit exception must remain explicit.
- If the records do not actually share one condition/action correction, return all nulls
  instead of averaging incompatible failures into a vague rule.

- Use only information present in the parent, the bad trajectories, and protected right cases.
- when_to_use must use state observable before the action. Make it narrower than the parent.
- how_to_use must say what to inspect, which bad action to suppress, and what admissible next
  action replaces it. Do not merely say “remember”, “be careful”, or “follow the rule”.
- Preserve the behavior of every protected right case. State an observable exception when the
  same surface action is correct there.
- Prefer one conditioned action constraint over a list of advice.

Example: if the parent says to maintain a search ledger, and the bad case revisits a location
already observed empty, the child may require checking the exact-location ledger before each
search move and selecting an unchecked location instead. It must not forbid revisits required
for delivery, cleaning, heating, cooling, or retrieving an already located object.
