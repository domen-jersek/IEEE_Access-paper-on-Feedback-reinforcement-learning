# Data protocol specification — revision

## Implemented contract (schema 1)

The data contract is independent of lift/routing/generator settings. JSON and YAML
loaders reject duplicate, unknown, missing and unsupported settings. Relative
input paths are resolved against an explicit project root, never the shell or
configuration directory. Every input is pinned by full SHA-256.

`familiar_procedure_possible` permits related procedures across query partitions.
`held_out_groups` requires a separate pinned mapping and no shared group between
any pair of train/dev/eval partitions. Neither name claims temporal evaluation.
Both use train-only history. Splits must have unique IDs, nonempty partitions,
no ID overlap, and exact dataset coverage.

`text_only` exposes only query ID, title and description. `posthoc_diagnostic`
explicitly exposes the stored resolving team and reply-derived intent class; it
is not a serving-time availability claim. Predicted/submission-time metadata
are not implemented and are rejected rather than simulated by stored labels.

Allowed feedback rows have both endpoints in train, no self-pairs, finite scores
in [0, 1], and the requested protocol. Unknown IDs and duplicate selected pairs
fail. For train LOO an explicit target is mandatory and is removed in BOTH roles.
The returned frame must feed both aggregation and prior estimation. It can be
empty; the future feedback experiment preflight must decide whether an empty
source is invalid rather than silently changing the requested method.

`filtered_existing` describes reuse of an incomplete historical pair set. It does
not assert complete top-k judging. Other evidence policies require explicit
implementation before acceptance by the configuration parser.

## Group artifact interface

A separate JSON artifact has exactly these fields:

```json
{
  "schema_version": 1,
  "rule_id": "versioned_rule_name",
  "semantic_status": "structural_proxy",
  "groups": {"ticket_id": "group_id"}
}
```

The mapping must cover every dataset ID with a nonempty group ID. A declaration
of `domain_validated` is also supported, but the resolver checks the mapping,
not the truth of domain validation: the study must retain the actual validation
evidence. Canonical ticket IDs and legacy group columns are not overwritten.

## Decisions not yet frozen

- Final structural grouping rules and any domain-confirmed procedure links.
- Final group-aware split and the scope of any semantic procedure-disjoint claim.
- Serving-time routing metadata source/model.
- Whether the main panel uses filtered evidence or a newly completed pair plan.

These are research choices, not defaults to optimize using eval outcomes.
