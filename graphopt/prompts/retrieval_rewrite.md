# Complete-case evidence rule

Every source bad case must be checked using its original task, student answer, evaluator/training reference, complete reasoning evidence, graph usage, and companion node/edge proposals together. The reasoning trace is evidence, not ground truth. Preserve source-coupled composite changes.

# GraphOpt Retrieval Trigger Rewriter

The skill itself is correct, but the Agent failed to retrieve it in relevant cases. Clarify only
its activation condition.

Return exactly one strict JSON object:

```json
{"additional_when":""}
```

Rules:

- Return only the missing trigger clause; the code appends it to the original trigger.
- Treat every listed Successful Use as counter-evidence. The revised trigger must still
  retrieve the node in every successful task/state represented there.
- Broaden or clarify the trigger minimally. Never narrow away a successful use merely
  because the failure proposal describes a more specific situation.
- Integrate the repeated retrieval evidence as concrete, recognizable task/state cues.
- Re-read every Retrieval-Miss Case and use only observations available immediately before
  the node should have been activated. If the cases do not share such a cue, return null.
- Do not output or modify `title`, `meaning`, `how_to_use`, `avoid`, IDs, category, or topology.
- Do not turn a single task instance, object instance, or trajectory into a general trigger.
- Return `{"additional_when":null}` if the original trigger is already clear, the miss is
  better explained by execution/routing noise, or no safe additive cue is supported.
- Otherwise `additional_when` must be a non-empty English string. No Markdown or extra keys.
