# LiveMath SkillGraph

Use only the visible question, answer choices, and hypotheses. Each `[ID]` marks one atomic rule; IDs are audit handles, not mathematical evidence. Do not infer an answer from hidden labels, theorem metadata, proof sketches, paper identity, or case IDs.

Before the final answer, write exactly one plain-text graph execution record inside `<reasoning_trace>...</reasoning_trace>`. It must expose the semantic rule path, not merely summarize the mathematical topic:

- For every applied node, cite its exact Stable Rule Graph ID in brackets, state the mathematical fact or intermediate result entering that node, state what implication, comparison, boundary, scope, or quantitative check the node applied, and state the new intermediate result it produced.
- For every followed edge, cite its exact edge ID in brackets, name the source and target node IDs, explain why the dependency or enhancement trigger applied, and state what the transition enabled or changed.
- Compare the relevant competing options and explain which distinction eliminates them.
- End the trace by stating the resulting conclusion and exact selected choice label.

Do not cite a node or edge that was not actually applied. Do not include hidden token-level reasoning, confidence theater, graph notes, tool logs, case IDs, or the reference choice.

Output exactly these three blocks in this order and no other text:

`<reasoning_trace>Applied [M...] to the stated hypothesis; this produced .... Followed [E...] from [M...] to [M...] because its trigger held; this enabled .... Applied [M...] to that intermediate result; this produced .... Therefore the selected choice is LABEL.</reasoning_trace>`
`<graph_usage>{"used_nodes":["M..."],"used_edges":["E..."]}</graph_usage>`
`<answer>LABEL</answer>`

The `graph_usage` arrays must be the exact deduplicated set of node and edge IDs explicitly cited as applied or followed in `reasoning_trace`: no unmentioned IDs may appear, and no cited applied ID may be omitted. Use only IDs visible in the Stable Rule Graph. If no edge was followed, cite no edge in the trace and use an empty `used_edges` list.
