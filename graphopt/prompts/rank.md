# GraphOpt Edit Ranker

Legacy clip utility only. The main GraphOpt evolution path does not call this prompt and
has no patch-wide edit cap.

Select edit indices from the candidates under the supplied budget.

## Output

Return JSON only:

~~~json
{"selected_indices":[0,2],"reasoning":""}
~~~

## Rules

- Prefer failure-sourced update_node, then add_node, then add_edge, then deletion.
- Indices are zero-based and their count must not exceed max_ops.

## Example

~~~json
{"selected_indices":[0,2],"reasoning":"Budget 2: G02 update plus its prerequisite edge have the highest failure impact."}
~~~
