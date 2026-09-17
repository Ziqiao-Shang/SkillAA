## GraphOpt-Simple reasoning-trace analysis contract

This is the Simple update protocol, but it is not an old answer-only analysis.
The student used the current reasoning output protocol. Treat the supplied
validated_student_trace as the primary observable semantic path and use the
response/final answer only to confirm its terminal outcome.

Analyze the bad case in this order:

1. Read the trace as an ordered sequence of explicitly cited nodes and edges.
   For every step, state what semantic input it consumed and what intermediate
   conclusion it produced.
2. Find the last supported intermediate conclusion and the first unsupported
   transition or result. Attribute the error to the cited node/edge used at that
   transition, a relevant graph element that was omitted, or a genuinely absent
   reusable procedure. Do not attribute only from the final wrong answer.
3. Cross-check that attribution against exact graph_usage, evaluator feedback,
   and same-group successful sibling traces. A successful sibling is a boundary
   check for generalization, not an additional Gate vote.
4. Propose the smallest reusable graph change that would replace the first wrong
   decision while preserving the successful sibling paths. Never copy a gold
   answer, option label, document value, named entity, or case-specific constant.
5. If the trace and usage are unavailable, inconsistent, or insufficient to
   locate a reusable graph defect, return INSUFFICIENT_EVIDENCE with no proposal.

Downstream Simple admission deliberately skips causal intervention certificates,
and generic/focus/mask probes. That simplification must not weaken this badcase
analysis: still return the complete badcase_summary, exact root-cause code,
attributed graph usage, and grounded proposal required by the shared schema.
