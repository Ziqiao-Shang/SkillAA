# SearchQA SkillGraph

<!-- SG_IMMUTABLE_BEGIN -->
## Permanent Protocol

Answer only from the supplied context.

Before the final answer, write exactly one plain-text graph execution record inside `<reasoning_trace>...</reasoning_trace>`. It must expose the semantic rule path, not merely summarize the clue:

- For every applied node, cite its exact Stable Rule Graph ID in brackets, state the evidence or intermediate state entering that node, state what the node did, and state the new intermediate result it produced.
- For every followed edge, cite its exact edge ID in brackets, name the source and target node IDs, explain why the edge trigger or dependency applied, and state what the transition enabled or changed.
- State the requested answer type, decisive grounded context evidence, relation direction or constraint, rejected plausible endpoint when relevant, and final surface-form decision.
- End the trace by stating the resulting conclusion and the exact answer surface form.

Do not cite a node or edge that was not actually applied. Do not include hidden token-level reasoning, confidence theater, graph notes, tool logs, case IDs, or the reference answer.

Each node below is one reusable part of the bundled SkillAA skill. First interpret the requested answer type, then match grounded evidence, check relation traps and constraints, and finally normalize the supported surface form. `prereq` edges give required order; `enhance` edges are optional checks whose trigger must hold. Edge strength is priority, not evidence.

Example: if the clue says “His third wife was Jiang Qing,” interpret the requested endpoint before extraction; the answer is the husband supported by context, not the already named wife.

Output exactly these three blocks in this order and no other text:

`<reasoning_trace>Applied [Q...] to the stated evidence; this produced .... Followed [E...] from [Q...] to [Q...] because its trigger held; this enabled .... Applied [Q...] to that intermediate result; this produced .... Therefore the final answer surface form is ....</reasoning_trace>`
`<graph_usage>{"used_nodes":["Q..."],"used_edges":["E..."]}</graph_usage>`
`<answer>...</answer>`

The `graph_usage` arrays must be the exact deduplicated set of node and edge IDs explicitly cited as applied or followed in `reasoning_trace`: no unmentioned IDs may appear, and no cited applied ID may be omitted. Use only IDs visible in the Stable Rule Graph. If no edge was followed, cite no edge in the trace and use an empty `used_edges` list.

Only the Stable Rule Graph may evolve. Learned rules must remain context-grounded and reusable; they may not memorize a case ID or gold answer.
<!-- SG_IMMUTABLE_END -->
