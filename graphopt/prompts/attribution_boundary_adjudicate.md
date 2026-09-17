# GraphOpt Retrieval/Execution Boundary Adjudicator

Independently adjudicate one failed case whose first-pass attribution lies on the
boundary between RETRIEVAL_MISS and EXECUTION_LAPSE. Ignore any prior label and
apply this fixed order:

1. State the reusable corrective action supported by the observable case evidence.
   The training answer may reveal that the prediction is wrong, but a label-only or
   answer-only difference cannot establish a reusable correction.
2. Search the complete current graph for the narrowest existing node or edge whose
   stored text fully prescribes that corrective action. A generic rule that merely
   concerns the same broad topic is not sufficient.
3. If no existing graph element fully prescribes the correction, or the correction
   cannot be justified without copying the training answer, return UNATTRIBUTED.
   This boundary stage must not invent a new skill or rewrite an existing one.
4. If a fully sufficient element exists but is absent from the verified used set,
   return RETRIEVAL_MISS. This takes precedence over blaming a broader used rule.
5. Return EXECUTION_LAPSE only when the fully sufficient element is present in the
   verified used set and the observable trace contains both: (a) a correct
   intermediate result produced under that rule, and (b) a distinct later step that
   diverges from it. Merely producing a wrong final answer after citing a generic
   rule is not an execution lapse.

For RETRIEVAL_MISS, emit a trigger_revision only when the selected missed node's
original when_to_use is demonstrably absent or unclear. Otherwise use null: a miss
does not by itself prove that the stored trigger is defective.

Return exactly one JSON object with these keys and no Markdown:

```json
{
  "case_id": "",
  "decision": "RETRIEVAL_MISS|EXECUTION_LAPSE|UNATTRIBUTED",
  "reason": "",
  "required_action": "",
  "rule_text_sufficiency": "FULL|INSUFFICIENT|UNCERTAIN",
  "reusable_correction_observable_without_training_answer": true,
  "sufficient_rule_node_ids": [],
  "sufficient_rule_edge_ids": [],
  "decision_node_ids": [],
  "decision_edge_ids": [],
  "correct_intermediate_excerpt": "",
  "later_divergence_excerpt": "",
  "trigger_revision": null
}
```

When trigger_revision is not null, it must contain exactly target_node,
proposed_when_to_use, and reason. Copy only IDs from the supplied graph and copy
observable excerpts verbatim from the supplied case trajectory, response, semantic
trace, or evaluator text.
