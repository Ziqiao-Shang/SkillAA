# DocVQA SkillGraph

<!-- SG_IMMUTABLE_BEGIN -->
## Permanent Protocol

Inspect the supplied document image using visible document evidence.

Each node below is one reusable part of the bundled SkillAA skill. Use its ID only for internal reasoning. `prereq` edges give required order; `enhance` edges are optional checks whose trigger must hold. Edge strength is priority, not evidence. Do not output node IDs, graph notes, or multiple answers.

Append exactly one machine-readable sidecar `<graph_usage>{"used_nodes":["node IDs that actually affected the response"],"used_edges":["edge IDs actually followed"]}</graph_usage>`. Use only IDs visible in the Stable Rule Graph; use empty lists when none affected the response. The sidecar is audit metadata.
Only the Stable Rule Graph may evolve. Learned rules must remain grounded in visible document evidence and reusable; they may not memorize a question ID or gold answer.
<!-- SG_IMMUTABLE_END -->
