# Downloaded TRELLIS2 workflow findings

The two workflows in `examples/downloaded_examples` are external reference
material. Their JSON is also copied byte-for-byte into
`tests/fixtures/external` to exercise valid ComfyUI workflows without
subgraphs.

## High Quality GGUF

The graph exposes a useful staged geometry pipeline:

```text
image
→ preprocess
→ voxel generation
→ quad reconstruction
→ simplify
→ fill holes
→ convert to trimesh
→ export white mesh
→ refine against source image
→ reconstruct / simplify / fill again
→ export refined mesh
→ texture
→ smooth normals
→ export textured mesh
→ preview
```

Useful CFP patterns:

- one preprocessed source image fans out to generation, refinement, and
  texturing;
- one model-loader output is shared across inference stages;
- white, refined, and textured meshes are distinct artifacts;
- exports act as durable checkpoints;
- `Trellis2Continue` nodes create execution barriers after exports;
- fixed seeds improve reproducibility;
- geometry and texturing remain separate concerns.

The continuation nodes are not approval gates. CFP still requires a human to
promote an exported artifact before the next stage is considered approved.

## Multiview GGUF

The reconstruction node accepts explicit roles:

```text
front_image
back_image
left_image
right_image
```

Left and right are optional in the downloaded graph. Each connected image is
preprocessed independently. Reconstruction produces both a voxel mesh and BVH,
which are consumed together by post-processing.

This supports keeping CFP camera roles explicit rather than passing an
unlabeled image batch.

The combined `Trellis2PostProcessAndUnWrapAndRasterizer` node is convenient but
too opaque for CFP's review-oriented architecture. CFP should keep repair,
simplification, unwrap, rasterization, and export independently inspectable
where the selected reconstruction backend permits it.

## Compatibility requirements

The workflows reference:

```text
visualbruno/ComfyUI-Trellis2
Aero-Ex/ComfyUI-Trellis2-GGUF
```

They are configured for:

```text
TRELLIS.2-4B
GGUF Q4_K_M
CUDA
flash_attn
```

Those exact nodes are not installed in the current Noblesite ComfyUI checkout.
The active workstation is an M3 Max using MPS, so the workflows are structural
references rather than runnable local baselines.

The High Quality workflow also contains node metadata from several different
VisualBruno commit hashes. Before using it on a CUDA host, its node versions
should be normalized against one tested node-pack revision.

Additional portability concerns:

- the input filenames are author-specific;
- one Preview3D widget retains an absolute Windows path;
- the two-million-face simplification target is an intermediate-quality
  choice, not an assumed CFP manufacturing default;
- neither workflow implements human approval checkpoints.
