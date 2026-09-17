# GraphOpt Proposal Merge

Legacy Reflect path. Main Evolution uses programmatic merging.

Semantically merge multiple patches or proposals. support is the number of distinct
cases and must never decrease during merging.

## Output

Return JSON only:

~~~json
{"reasoning":"","edits":[]}
~~~

## Rules

- Prioritize failure-sourced evidence.
- Merge revisions that target the same node.
- Merge edges by (source, target, relation), with relation limited to prereq or enhance.
- Never propose co_occur or modify the Permanent Protocol.

## Example

~~~json
{"reasoning":"Three cases support the same G02 open-before-take revision.","edits":[{"op":"update_node","node_id":"G02","how_to_use":"Open closed containers before take.","source_type":"failure"}]}
~~~
