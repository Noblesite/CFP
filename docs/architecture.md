# CFP architecture

## Product objective

CFP is intended to produce a deliberately structured manufacturing assembly,
not merely an attractive fused mesh. The preferred flow is:

```text
approved concept
→ consistent camera views
→ physical component definitions in 2D
→ isolated per-component image sets
→ independent component reconstruction
→ assembly validation
→ engineered clearances and connectors
→ printable export
```

Early versions may combine one structural body/core mesh with independently
reconstructed armor shells, plates, inserts, and accessories.

## Current software boundary

This milestone owns ComfyUI workflow JSON only:

```text
Workflow
  ├── load/save without schema-destructive normalization
  ├── inspect
  ├── validate
  └── append explicit stages
```

It intentionally does not own model execution, image review, segmentation,
mesh repair, Blender automation, project persistence, or orchestration.

## Transformation rules

- Source artifacts are immutable by default.
- Transformations operate on an in-memory deep copy when callers request one.
- Unknown fields remain opaque and are preserved.
- Findings are structured as severity, code, message, and optional location.
- Validators report corruption but do not repair it.
- Editing commands validate before and after mutation.
- A generated file is refused when the edit introduces errors.
- Every change returns a `ChangeReport`.

## Builder strategy

The initial `append_kontext_stage` builder discovers a known-good,
two-reference subgraph from the fixture instead of relying on a hardcoded node
ID. It clones that definition behind a narrow abstraction, assigns a
deterministic UUID, creates fresh graph IDs, wires both sources, and adds
preview/save outputs.

This is deliberately transitional. A later canonical Kontext template can
replace fixture cloning without changing the part-isolation API.

Long prompts live in packaged text templates rather than Python source so the
manufacturing and camera contracts remain readable and editable.

## Human review

ComfyUI graphs remain acyclic. A rejected artifact is revised by reloading an
approved checkpoint, modifying stage inputs, and executing that stage again.
Approval, rejection, branching, and promotion are explicit human operations,
not hidden feedback loops.

