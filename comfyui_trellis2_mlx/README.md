# TRELLIS.2 MLX for ComfyUI

This custom node runs the native Swift/MLX TRELLIS.2 engine as a local
subprocess on Apple Silicon. Model weights never enter ComfyUI's Python or
PyTorch process.

The current adapter is verified against
`xocialize/mlx-trellis2-swift@005a372`. The installer applies CFP's small
memory-budget and deterministic-seed CLI patch before building.

## Current contract

- macOS on Apple Silicon;
- one input image;
- 512 reconstruction tier;
- deterministic seed;
- optional ComfyUI mask converted to PNG alpha;
- genuine multi-view conditioning through separate 000°, 090°, 180°, and
  optional 270° images;
- `textured` and `geometry_only` output modes;
- GLB output compatible with ComfyUI's `Preview 3D` and `Save 3D Model` nodes.

The node unloads cached ComfyUI models before starting Swift because both
runtimes share unified memory.

Generation and conditioning nodes deliberately opt out of ComfyUI result
caching. Queueing an unchanged workflow therefore launches a fresh native
engine process and materializes a new artifact, even with a fixed seed. The
wrapper also enforces startup and phase-stall timeouts; Swift emits named phase
changes and informational heartbeats so a wedged subprocess is terminated
instead of leaving the ComfyUI job active indefinitely.

Native BiRefNet matting runs as a separate pre-TRELLIS phase. The matted RGBA
views are produced first, BiRefNet is evicted, and only then is TRELLIS
registered. This avoids a re-entrant MLX engine call and permits consecutive
`matting=on` generations in the same ComfyUI session.

`TRELLIS.2 MLX Image Conditioning` is the first modular native-engine stage.
It accepts approved 000°, 090°, 180°, and 270° views, preprocesses and encodes
each view independently with DINOv3, concatenates the ordered token sequences,
and saves `cond_512` plus `neg_cond_512` in MLX-compatible safetensors. It then
stops: sparse structure, shape, texture, mesh extraction, and GLB export do not
run. The typed `TRELLIS2_MLX_CONDITIONING` output is reserved for the next
standalone sparse-generation workflow; automatic promotion is disabled.

`TRELLIS.2 MLX Load Conditioning Artifact` begins the next single-purpose
workflow from an explicit, human-promoted artifact path. It validates the CFP
conditioning schema and content hash before producing the typed handoff.
`TRELLIS.2 MLX Sparse Structure` then runs only the sparse-structure sampler
and decoder and writes the occupied `(batch,x,y,z)` coordinates on TRELLIS's
32³ grid. Shape SLat, texture, mesh extraction, topology operations, and GLB
export remain out of scope. Because this artifact is not yet a surface mesh,
this stage has a report rather than a 3D mesh preview.

`TRELLIS.2 MLX Single Image Manufacturing` is the integrated single-image
workbench. It combines RMBG-2.0, the input-mask quality gate, geometry-only
TRELLIS reconstruction, immutable raw preview/report/export, floater removal,
pre-sanitizer preview and diagnostics, conservative sanitation, the background
geometry gate, guarded 1024-cell watertight voxel refinement, post-voxel polish,
non-destructive surface-shading preview, and a 250 mm print-scale feature gate.
Every transformation writes a distinct
artifact; the raw TRELLIS GLB is never overwritten or silently promoted.

`TRELLIS.2 MLX Surface Shading` is the single-purpose display workflow. It
compares the original geometry against smooth, angle-limited, or unchanged
normals and saves a separate GLB without moving vertices.

## Install

Run the installer with the target ComfyUI checkout:

```bash
./scripts/install_trellis2_mlx_node.sh /path/to/ComfyUI
```

Restart ComfyUI. Add:

```text
TRELLIS.2 MLX Model
        |
        v
TRELLIS.2 MLX Image to 3D
        +--> Preview 3D & Animation
        +--> Save 3D Model
        +--> TRELLIS.2 MLX Mesh Report
```

For an approved model sheet, use `TRELLIS.2 MLX Multi-View to 3D`. The node
requires the 000° front view and accepts optional 090°, 180°, and 270° views.
Each image is preprocessed and encoded independently by DINOv3; the ordered
token sets are concatenated into one TRELLIS.2 conditioning context. Supply
separate aligned images rather than a flattened contact sheet.

The mask-gated multi-view workflow places a separate Input Mask Quality Gate
after RMBG on the populated 000°, 090°, and 180° branches. Each gate derives
its own correctly polarized mask from that view's RGBA alpha. A failure on any
camera branch stops the shared reconstruction before the native engine starts;
approved views are never averaged into one mask. The 270° image and mask inputs
remain optional and disconnected for the next camera experiment.

The four-view mask-gated workbench preserves that three-view workflow as an A/B
baseline and adds the complete 270° right-side lane: `Load Image -> RMBG ->
Input Mask Quality Gate -> image_270 + mask_270`. All four cardinal views are
therefore populated, independently checked, and passed to TRELLIS in the fixed
000°, 090°, 180°, 270° order.

`TRELLIS.2 MLX Model-Sheet Consistency Gate` then compares the four derived
foreground masks as one camera set. It requires matching canvas dimensions and
checks normalized subject height, horizontal centerline, vertical center, and
ground baseline. Width is compared only between opposing cameras—000° versus
180° and 090° versus 270°—so valid side profiles are not rejected for being
narrower than front views. The node is read-only and blocks reconstruction by
default when the sheet drifts outside its thresholds.

`TRELLIS.2 MLX Model-Sheet Alignment Review` is the visual debugger for that
gate. It creates a read-only 2x2 contact sheet with each camera label and
normalized measurements. Cyan marks the canvas center, green the foreground
bounds, magenta the subject centerline, and yellow the baseline. The review
branch reads directly from the four per-view mask gates, so it never resizes,
repositions, replaces, or promotes the images sent to reconstruction.

`TRELLIS.2 MLX Model-Sheet Alignment Candidate` creates separate transparent
copies with uniform scaling and translation. `median` uses the median
foreground height of the four cameras; `explicit_fraction` uses a normalized
canvas fraction. Both modes share the median baseline and 50% canvas
centerline. `design_target_height_mm` is carried only as manufacturing metadata
and does not claim that the source pixels have physical scale. The included
workbench renders an AFTER contact sheet but deliberately leaves every
candidate output disconnected from TRELLIS until a human promotes it.

Set `output_mode` to `geometry_only` to skip the texture SLat flow and texture
decoder, UV unwrap, rasterization, and texture-atlas bake. This produces a
texture-free GLB containing positions, faces, and normals while retaining
TRELLIS mesh extraction, cleanup, simplification, and export. The dedicated
geometry-only workflow makes that selection explicit and is intended for
reconstruction benchmarking and manufacturing work where learned color is
unnecessary.

`TRELLIS.2 MLX Mesh Report` is a read-only post-generation checkpoint. It
reports vertex and triangle counts, connected components, boundary and
non-manifold edges, watertightness, bounding-box dimensions, artifact path,
and SHA-256. Its `PASS`, `REVIEW`, or `FAIL` status is intentionally advisory:
the GLB is never repaired or modified.

`TRELLIS.2 MLX Remove Floaters` is a conservative geometry-only cleanup pass.
It always preserves the largest connected component, then retains additional
components only when they meet both the absolute face-count floor and the
relative-to-largest floor. Its defaults (`100` faces and `0.001`) are intended
to remove sparse O-Voxel debris without silently discarding meaningful armor
shells. The node reports before/after topology and does not claim to repair
overlapping shells, self-intersections, boundaries, or non-manifold edges.
Because the current implementation exports geometry and normals, use it on the
geometry-only lane rather than on a textured production GLB.

`TRELLIS.2 MLX O-Voxel Topology Diagnostics` is a read-only classification
checkpoint. It confirms duplicate and degenerate faces, coincident unwelded
vertices, open boundaries, and edges shared by more than two faces. It also
runs analysis-only probes showing how many overloaded edges remain after
ignoring duplicate or degenerate faces. Component bounding-box overlaps and
residual O-Voxel shell junctions are labeled as candidates rather than proof of
triangle-level self-intersection; exact intersection testing is deliberately
reported as not run until a robust backend is selected.

`TRELLIS.2 MLX Topology Sanitizer` is the first deterministic topology mutation
checkpoint. It welds vertices within a conservative bounding-box-relative
tolerance, removes zero-area and duplicate faces, and compacts unused vertices.
It runs diagnostics before and after and returns `CHANGED_REVIEW` when removing
stacked duplicate sheets reveals additional boundaries. It does not remesh,
fill holes, or resolve O-Voxel shell junctions. Use it only on geometry-focused
GLBs and review its output before promotion.

Every bundled workflow that invokes the sanitizer now includes a
`BEFORE SANITIZER — Untouched Incoming Mesh` 3D preview. That preview and the
sanitizer are wired to the exact same source output, so the raw baseline is
visible before any topology mutation occurs.

`TRELLIS.2 MLX Watertight Voxel Remesh Candidate` creates a separate filled
voxel grid and marching-cubes surface at an adjustable resolution. It reports
topology before and after, dimensional change, and deterministic bidirectional
nearest-vertex deviation as a surface-detail proxy. The source is never
overwritten. Thin details near the voxel pitch can be lost or fused, so the
included workflow presents source and candidate previews side by side and saves
only the explicitly reviewed candidate branch.

`TRELLIS.2 MLX Voxel Candidate Comparison` accepts the structured reports from
three fixed voxel candidates. The included A/B workbench runs 128, 192, and 256
from the same sanitized source, provides three independent previews and GLB
exports, and compares topology, triangle count, dimensional drift, and p95
nearest-vertex deviation. Its recommendations are decision aids only; the node
never auto-promotes a mesh.

`TRELLIS.2 MLX Post-Voxel Topology Polish` removes only exact coincident,
opposite-winding face pairs left as zero-thickness internal sheets by marching
cubes. It does not remesh or move vertices. A proposed deletion is accepted only
when overloaded edges decrease while boundaries and connected components do not
increase. The dedicated workflow keeps raw and polished 256-resolution previews
and diagnostics side by side and saves only the polished branch.

`TRELLIS.2 MLX Print Scale & Feature Gate` uniformly scales an approved GLB to
a target height in millimeters while writing standard meter-based glTF units.
It reports the effective source voxel pitch and estimated feature floor against
the chosen nozzle and layer height. The node does not claim to measure local
wall thickness, clearance, overhangs, or slicer toolpaths. The character
workflow defaults to the Z height axis so background remnants or base geometry
cannot make automatic longest-axis selection choose the wrong orientation.

`TRELLIS.2 MLX Background Geometry Guard` runs before voxel filling. In the
default `character_z_up` profile, it requires Z height to dominate the lateral
bounds and stops the graph when backdrop or base geometry makes the dimensions
implausibly isotropic. Large planar components are reported as evidence but are
never deleted automatically: TRELLIS O-Voxel character surfaces legitimately
contain many thin disconnected patches. Use `generic` for cubic props, or set
acknowledgement to `yes` only after inspecting the upstream preview.

`TRELLIS.2 MLX Input Mask Quality Gate` runs immediately after background
removal and before the expensive native inference step. It checks foreground
coverage, border contact, likely inverted or empty alpha, and disconnected mask
noise. The RMBG workflow deliberately feeds the node's RGBA `IMAGE`, not
RMBG's opposite-polarity `MASK`; the gate derives a standard ComfyUI
transparency mask from alpha and forwards the image without changing pixels.
Suspicious inputs stop the graph unless the human acknowledgement is set to
`yes`.

For a standard ComfyUI alpha mask, connect both `IMAGE` and `MASK` and leave
`matting` set to `off`. The included RMBG workflow instead connects RMBG's
RGBA `IMAGE` output and intentionally leaves its foreground-mask output
disconnected because RMBG uses the opposite mask polarity.

The model node accepts `auto` for both paths when installed from the CFP
checkout. Explicit paths remain available for custom layouts.

A ready-to-run workflow is included at:

```text
workflows/trellis2_mlx_image_to_3d.json
```

The CFP development install also places it under:

```text
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Image_to_3D_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Background_Clean_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_MultiView_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Image_Conditioning_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Sparse_Structure_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Single_Image_Manufacturing_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Geometry_Only_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Remove_Floaters_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Topology_Diagnostics_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Topology_Sanitizer_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Voxel_Remesh_Candidate_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Voxel_Resolution_AB_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Post_Voxel_Polish_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Print_Scale_Gate_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Background_Geometry_Guard_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Input_Mask_Quality_Gate_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_MultiView_Mask_Gated_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Four_View_Mask_Gated_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Four_View_Consistency_Gated_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Four_View_Alignment_Review_v001.json
ComfyUI/user/default/workflows/CFP/CFP_TRELLIS2_MLX_Four_View_Alignment_Candidate_v001.json
```

## RMBG-2.0 model metadata

The background-clean workflow embeds the four required RMBG-2.0 files on its
`RMBG` node using ComfyUI's `properties.models` metadata. ComfyUI can therefore
show the dependency in its missing-model/download panel and place it under:

```text
ComfyUI/models/RMBG/RMBG-2.0/
```

The links point to the official gated
[`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) repository. Accept
BRIA's access terms and authenticate with Hugging Face before downloading.
RMBG-2.0 is licensed for non-commercial use under its published terms;
commercial use requires a separate agreement with BRIA. CFP does not bundle or
mirror the weights.
