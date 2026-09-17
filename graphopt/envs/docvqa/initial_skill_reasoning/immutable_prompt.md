# DocVQA SkillGraph

<!-- SG_IMMUTABLE_BEGIN -->
## Permanent Protocol

Inspect the supplied document image and answer the question using visible document evidence.

Before the final answer, write exactly one plain-text graph execution record inside `<reasoning_trace>...</reasoning_trace>`. It must expose the semantic rule path, not merely summarize the document:

- For every applied node, cite its exact Stable Rule Graph ID in brackets, state the visible evidence or intermediate state entering that node, state what the node did, and state the new intermediate result it produced.
- For every followed edge, cite its exact edge ID in brackets, name the source and target node IDs, explain why the edge trigger or dependency applied, and state what the transition enabled or changed.
- State what the question asks for, the decisive visible document evidence, how nearby or semantically similar alternatives were excluded when relevant, and the final text-span normalization decision.
- End the trace by stating the resulting conclusion and the exact answer text.

Do not cite a node or edge that was not actually applied. Do not include hidden token-level reasoning, confidence theater, graph notes, ungrounded image coordinates, tool logs, question IDs, or the reference answer.

Each node below is one reusable part of the bundled SkillAA skill. `prereq` edges give required order; `enhance` edges are optional checks whose trigger must hold. Edge strength is priority, not evidence.

Output exactly these three blocks in this order and no other text:

`<reasoning_trace>Applied [D...] to the visible evidence; this produced .... Followed [E...] from [D...] to [D...] because its trigger held; this enabled .... Applied [D...] to that intermediate result; this produced .... Therefore the final answer text is ....</reasoning_trace>`
`<graph_usage>{"used_nodes":["D..."],"used_edges":["E..."]}</graph_usage>`
`<answer>...</answer>`

The `graph_usage` arrays must be the exact deduplicated set of node and edge IDs explicitly cited as applied or followed in `reasoning_trace`: no unmentioned IDs may appear, and no cited applied ID may be omitted. Use only IDs visible in the Stable Rule Graph. If no edge was followed, cite no edge in the trace and use an empty `used_edges` list.

Only the Stable Rule Graph may evolve. Learned rules must remain grounded in visible document evidence and reusable; they may not memorize a question ID or gold answer.
<!-- SG_IMMUTABLE_END -->
