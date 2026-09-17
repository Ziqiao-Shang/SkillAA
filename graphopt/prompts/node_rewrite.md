# GraphOpt Node Rewriter

Synthesize the complete epoch evidence in the Merged Revision Proposal into one precise conditional lesson. Every Bad Case is a complete source packet, not a trace-only summary. The program will append it to the Original Node; you are not allowed to rewrite the node.

Return exactly one JSON object. Use a concise English string when a safe lesson exists,
or null when no non-destructive addition can be justified:

~~~json
{"addition":""}
~~~

## Rules

- Produce only a small additive procedure that fixes the repeated failure. The code keeps
  every byte of the old meaning, trigger, procedure, and avoid list.
- For every Bad Case, jointly inspect `original_task`, `student_output`, `evaluation`, `training_only_reference`, complete `reasoning_evidence`, `graph_usage`, and all `proposed_graph_changes`. Never trust or merge on the reasoning trace alone.
- Cover every compatible opinion in the merged epoch evidence. When opinions recommend opposite actions, identify an observable separator and write both branches as if/then behavior; never choose a side merely because it has more support.
- When one source case proposes changes to several nodes or edges, preserve the composite dependency in this node addition; Python will validate and commit all source-coupled edits together.
- If no observable condition separates contradictory opinions, return null instead of producing a broad compromise.
- Do not summarize the whole node or restate knowledge already present in the Original Node.
- Make the addition actionable and conditional: state the observable situation, the next
  action/state update, and any narrow exception needed to protect valid behavior.
- Treat every listed Successful Use as local counter-evidence against over-correction. A record whose task_type is `aggregate_success_guard` summarizes every related successful use of this node and must never be ignored. Other records are representative raw cases. Check whether any new trigger, priority, prohibition, threshold, formula, retrieval rule, or state transition would have blocked that successful behavior.
- Treat the Bad Cases and Successful Uses as a paired counterfactual set. Identify the
  smallest observable state that separates the failure from protected behavior. A patch
  is useful only if it supplies a different next decision in the bad state while leaving
  the successful decision path available.
- The revision must remain compatible with every Successful Use. If the proposal conflicts
  with one, preserve the original behavior and express the fix as the narrowest observable
  exception that addresses the failures without blocking success.
- Do not turn a failure-only pattern into an unconditional always/never rule. Add a hard
  filter only when it is compatible with every supplied successful use.
- Do not mistake stronger wording for broader coverage. Condition the addition on the smallest observable state that separates failures from successes: clue/evidence/relation context in SearchQA, visual anchor/layout/exact-span context in DocVQA, or exact logical wording in LiveMath.
- Prefer one to four concise sentences. For contradictory evidence, explicitly state the main branch and its narrow exception. Do not produce a replacement meaning, trigger,
  full procedure, avoid list, title, or node summary.
- Keep the addition under 80 words. Return `{"addition":null}` if the proposal is already
  covered, belongs to another node, conflicts with any Successful Use, lacks an observable
  condition and action, or would require weakening/replacing old knowledge.
- Never modify the Permanent Protocol.
- Do not output id, title, stats, category, section, or any key except `addition`.

## Example

Proposal: Before take, open closed cabinets and confirm that the object is visible.

The output must preserve the original search procedure and integrate open-before-take,
instead of collapsing the node to one sentence.

~~~json
{"addition":"When the selected source is a closed container, open it and confirm the exact target is visible before taking; otherwise continue the existing search order."}
~~~
