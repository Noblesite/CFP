# ComfyUI workflow schema notes

These observations come from the immutable `CFP-02.json` fixture produced by
ComfyUI workflow schema version `0.4`.

## Fixture summary

```text
Top-level nodes: 11
Top-level links: 11
Groups: 7
Subgraph definitions: 3
last_node_id: 227
last_link_id: 11
```

The fixture contains sequential FLUX.1 Kontext stages for camera 000°, 180°,
and 090°.

## Top-level graph

Top-level links are arrays:

```json
[link_id, origin_node_id, origin_slot, target_node_id, target_slot, type]
```

Node inputs store one `link` ID. Node outputs store `links`, which can be an
array, an empty array, or `null`. Editing therefore requires updating all three
representations:

1. the top-level link record;
2. the target input's `link`;
3. the origin output's `links`.

Preview Image nodes in CFP-02 act as pass-through sources for later stages.
The builder supports this because references identify an arbitrary node and
output slot rather than assuming generation nodes are always the source.

## Subgraphs

Top-level subgraph instances use the subgraph definition UUID as their node
`type`. The matching object lives in `definitions.subgraphs`.

Internal subgraph links are objects rather than top-level arrays:

```json
{
  "id": 237,
  "origin_id": 6,
  "origin_slot": 0,
  "target_id": 135,
  "target_slot": 0,
  "type": "CONDITIONING"
}
```

The fixture uses input and output pseudo-nodes with IDs `-10` and `-20`.
Internal node and link IDs only need to be unique within their containing
subgraph.

Cloned subgraphs in the fixture reuse several input/output UUIDs. CFP does not
currently treat those port UUIDs as globally unique because the producing
ComfyUI workflow does not. Definition IDs themselves must be unique.

The internal CLIP node retains the original template's swan prompt, while the
top-level subgraph instance carries the effective camera prompt in
`widgets_values`. Builders must preserve the exposed widget contract.

## Opaque fields

Fields such as these are extension or frontend owned and must not be discarded:

```text
pos
size
flags
order
mode
properties
widgets_values
extra
config
previewExposures
proxyWidgetErrorQuarantine
ue_properties
```

The loader therefore keeps the workflow as a generic dictionary. Typed domain
objects describe CFP edit intent; they do not normalize the whole ComfyUI
schema.

## Optional definitions

ComfyUI 0.4 workflows without subgraphs may omit `definitions` entirely. The
downloaded TRELLIS2 examples use that form. Loading, inspection, and validation
must not add an empty `definitions` object as a side effect.

Builders that append a subgraph create `definitions.subgraphs` explicitly
through a mutating operation.

## Current validation boundary

The first validator checks:

- required graph containers and counters;
- duplicate top-level node and link IDs;
- link endpoint and slot existence;
- node input/output link correspondence;
- `last_node_id` and `last_link_id`;
- missing or duplicate subgraph definitions;
- duplicate internal node/link IDs;
- internal endpoints and real-node slots.

It does not attempt to validate widget semantics, model availability, ComfyUI
execution order, visual group containment, or whether a prompt actually
produces the requested camera angle.
