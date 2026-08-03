# Near-term roadmap

## Completed foundation

- Preserve the available CFP-02 production fixture.
- Load and save ComfyUI workflow JSON.
- Validate graph and subgraph integrity.
- Inspect workflow counts and stage instances.
- Append a reusable two-reference Kontext stage.
- Wrap the builder for part isolation.
- Emit a change report.
- Provide `validate`, `inspect`, and `append-part-isolation`.
- Accept workflows that omit `definitions` when they contain no subgraphs.
- Preserve downloaded TRELLIS2 workflows as external structural references.

## Next engineering slices

1. Add the original CFP-03 fixture when it is supplied and test it unchanged.
2. Compare the generated faceplate workflow against the original CFP-03
   baseline and document intentional differences.
3. Add a canonical Kontext subgraph template independent of CFP-02.
4. Add a camera-stage wrapper using `CFP_TURNTABLE_V1`.
5. Define the first reconstruction adapter against the documented,
   model-agnostic contract.
6. Add machine-readable human-review contracts without automating approval.
7. Add a small project manifest after documenting JSON, TOML, and YAML
   tradeoffs.

## Native TRELLIS.2 MLX track

Completed reliability gate:

- identical queued generation and conditioning requests always execute;
- native BiRefNet matting completes before TRELLIS registration;
- engine phase output and Python startup/stall watchdogs prevent indefinite jobs;
- two consecutive real matting-enabled MLX generations are verified.

Completed modular generation stages:

- native DINOv3 conditioning exports a reviewable safetensors artifact;
- standalone sparse-structure generation consumes only an explicit promoted
  conditioning artifact and stops after 32-cubed occupancy decoding;
- any workflow that sanitizes topology previews the exact untouched incoming
  mesh before mutation.

Completed integrated workbench:

- the single-image manufacturing workflow combines RMBG, input-mask gating,
  geometry-only TRELLIS reconstruction, immutable raw review/export, floater
  filtering, sanitation, background-geometry protection, guarded 768-cell voxel
  refinement, conservative polish, print scaling, final review, and export;
- the voxel-resolution A/B experiment remains separate so the production lane
  has one deterministic refinement path.

Requested workflow set:

1. Character turnaround-sheet generation — existing CFP image workbench,
   consolidation still pending.
2. Part isolation from an approved turnaround — existing CFP/Flux MLX work,
   consolidation remains outside the TRELLIS node track.
3. Single image to refined mesh — integrated workflow implemented.
4. Multiple aligned images to refined mesh — next integrated workflow after
   the single-image graph is validated in ComfyUI.

Pinned next slice:

1. Run and inspect the integrated single-image workbench in ComfyUI, preserving
   the raw and final artifacts for comparison.
2. Once validated, mirror the same refinement contract around the native
   multi-view generation node without merging image-turnaround responsibilities
   into the reconstruction workflow.

Deferred native modular slice:

- consume the promoted `TRELLIS2_MLX_SPARSE_STRUCTURE` artifact in a standalone
  shape-SLat generation node, stopping before shape decoding or mesh creation.

## Later pipeline stages

```text
CFP-00 Project Definition
CFP-01 Concept Workbench
CFP-02 Turnaround Workbench
CFP-03 Part Definition and Isolation
CFP-04 Part Turnaround Workbench
CFP-05 Reconstruction
CFP-06 Mesh Evaluation
CFP-07 Assembly
CFP-08 Manufacturing Engineering
CFP-09 Blender Handoff
CFP-10 Export
```

No later stage should weaken deterministic transformation, human inspection, or
source-artifact preservation.
