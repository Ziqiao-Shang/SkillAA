# SearchQA SkillGraph

<!-- SG_IMMUTABLE_BEGIN -->
## Permanent Protocol

Use only the supplied context.

Each node below is one reusable part of the bundled SkillAA skill. Use its ID only for internal reasoning: first interpret the requested answer type, then match grounded evidence, check relation traps and constraints, and finally normalize the supported surface form. `prereq` edges give required order; `enhance` edges are optional checks whose trigger must hold. Edge strength is priority, not evidence. Do not output node IDs, graph notes, or multiple answers.

Example: if the clue says “His third wife was Jiang Qing,” interpret the requested endpoint before extraction; the answer is the husband supported by context, not the already named wife.


Append exactly one machine-readable sidecar `<graph_usage>{\"used_nodes\":[\"node IDs that actually affected the response\"],\"used_edges\":[\"edge IDs actually followed\"]}</graph_usage>`. Use only IDs visible in the Stable Rule Graph; use empty lists when none affected the response. The sidecar is audit metadata.
Only the Stable Rule Graph may evolve. Learned rules must remain context-grounded and reusable; they may not memorize a case ID or gold answer.
<!-- SG_IMMUTABLE_END -->
