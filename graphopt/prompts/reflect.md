# GraphOpt Reflect

Legacy prompt. The main evolution path uses the Case Analyzer.

Propose a GraphEdit patch from a failed rollout batch.

## Output

Return JSON only:

~~~json
{"reasoning":"","edits":[]}
~~~

## Rules

- Allowed operations: add/delete/update node and add/delete/change prereq or enhance edge.
- Never physically merge nodes. Only semantically merge revision opinions that target the
  same node.
- Never propose co_occur or modify the Permanent Protocol.
- Prefer updates to relevant nodes, then edges, then new nodes. Prefer source_type=failure.

## Example

~~~json
{"reasoning":"Closed-cabinet take failures","edits":[{"op":"update_node","node_id":"G02","how_to_use":"Open closed containers before take.","source_type":"failure"}]}
~~~
