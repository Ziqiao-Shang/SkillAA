## Graph Execution Contract

- Treat every node as a conditional procedure, not as a fact or a mandatory checklist item.
- An empty `When to use` field does not mean unconditional execution: apply the rule only when its action is semantically relevant to the current question and evidence.
- A populated `When to use` field states a semantic condition. Its wording and examples are illustrative rather than an exhaustive keyword list.
- When a specific applicable node and a general node affect the same decision, follow the specific node on that decision only; retain the general node elsewhere.
- Follow `prereq` relationships before the dependent rule. Apply `enhance` relationships only when the enhancing rule is itself applicable.
- In the graph-usage sidecar, report only nodes and relationships that actually changed the answer; do not list every visible rule.
