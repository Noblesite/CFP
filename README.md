# The Foundry Character Fabrication Pipeline

The Foundry CFP is a local-first, manufacturing-oriented pipeline for turning
approved character concepts into editable, separable, 3D-printable assemblies.
This repository begins with a deliberately small tool: safe engineering of
ComfyUI workflow JSON.

The first milestone can:

- load and inspect ComfyUI workflow files;
- report graph-integrity errors without silently repairing them;
- preserve unknown and extension-owned JSON fields;
- append a two-reference FLUX.1 Kontext stage;
- wrap that stage as a component-isolation operation;
- save a new, reviewable workflow without overwriting its source.

The repository also contains an experimental Apple-Silicon reconstruction
adapter in `comfyui_trellis2_mlx/`. It wraps the native Swift/MLX TRELLIS.2
engine and exposes a ComfyUI `FILE_3D_GLB` output for single-image and
multi-view 512-resolution lanes. Textured and geometry-only generation are
selectable; the geometry-only path skips learned texture inference, UV unwrap,
rasterization, and atlas baking, then emits a texture-free GLB for faster
manufacturing-focused iteration.

The first independent mesh operation, `TRELLIS.2 MLX Remove Floaters`, removes
only small disconnected components and pairs the result with before/after mesh
reports. It deliberately does not conflate O-Voxel non-manifold topology with
removable debris.

The companion read-only topology diagnostic separates confirmed defects
(duplicates, degenerates, coincident vertices, boundaries, and overloaded
edges) from broad-phase overlap and O-Voxel shell-intersection candidates.

The topology sanitizer can then weld nearly coincident vertices and remove
duplicate or degenerate faces without remeshing. Before/after diagnostics keep
boundary regressions visible instead of silently declaring the mesh repaired.

The watertight voxel-remesh stage emits a separate candidate plus quantitative
topology, dimensional-change, and nearest-vertex deviation reports. Source and
candidate remain side by side until a human promotes the result.

The voxel-resolution A/B workbench fans the same sanitized source into fixed
128, 192, and 256 resolution candidates. It saves each candidate separately and
reports detail-, dimension-, lightweight-, and balanced-priority suggestions;
promotion always remains a human decision.

The post-voxel polish then targets a narrow marching-cubes artifact: exact
coincident face pairs with opposite winding. It deletes the complete
zero-thickness sheet only when overloaded edges decrease without increasing
boundaries or connected components; otherwise, it returns the source unchanged.

The print-scale gate uniformly converts an approved GLB to a target physical
height using meter-based glTF units, reports dimensions in millimeters, and
compares the effective source voxel pitch with the selected nozzle and layer
height. It is deliberately an advisory sampling check, not wall-thickness or
clearance analysis.

The pre-voxel background geometry guard protects character workflows from
isotropic bounds caused by backdrop or base geometry. It reports large planar
components but never auto-deletes them because TRELLIS O-Voxel surfaces are
themselves composed of thin disconnected patches. Suspicious character bounds
stop execution unless a human explicitly acknowledges them.

The upstream input-mask quality gate inspects RMBG alpha before inference for
empty or inverted masks, excessive foreground coverage, border contact, and
disconnected noise. It preserves the RGBA image, derives standard ComfyUI mask
polarity, and blocks expensive TRELLIS generation until suspicious input is
corrected or explicitly acknowledged.

The model-sheet workbench applies that gate independently to the populated
000°, 090°, and 180° camera branches. All three images and masks remain
separate through TRELLIS conditioning, and any rejected view stops the shared
native inference run.

The first modular generation-stage workflow ends after native DINOv3 image
conditioning. It writes a reusable MLX safetensors artifact containing the
ordered 512-resolution positive and negative conditioning tensors; it does not
run sparse, shape, texture, or mesh generation. This establishes the typed
handoff required by the future standalone sparse-generation workflow.

Native generation nodes are repeat-safe: unchanged queued requests bypass
ComfyUI artifact caching, the CLI reports explicit execution phases, and the
Python wrapper terminates startup or phase stalls. BiRefNet now completes and
evicts before TRELLIS starts, avoiding the nested engine call that previously
could leave a second run waiting forever.

The four-view workbench extends that baseline with the complete 270° right-side
camera branch. This keeps the three-view workflow available for comparison
while providing a fixed 000°, 090°, 180°, 270° cardinal model sheet.

The model-sheet consistency gate adds a read-only set-level checkpoint after
the four independent mask gates. It blocks canvas, scale, centerline, vertical
alignment, baseline, and opposing-view width drift before native inference.

The alignment-review branch renders those measurements as an annotated 2x2
contact sheet without changing reconstruction inputs. It exposes canvas center,
foreground bounds, subject center, and baseline for human diagnosis.

The alignment-candidate branch can then create normalized copies using either
the shared median foreground height or an explicit canvas fraction. Desired
millimeter height remains report-only metadata until the post-reconstruction
print-scale gate. Candidates never auto-promote into TRELLIS.

It does not execute ComfyUI, segment images, reconstruct meshes, control
Blender, or run autonomous retry loops.

## Setup

Python 3.12 or newer is required.

```bash
uv sync
```

Run the CLI without installing it globally:

```bash
uv run cfp inspect tests/fixtures/CFP-02.json
uv run cfp validate tests/fixtures/CFP-02.json
```

Append the faceplate-isolation experiment:

```bash
uv run cfp append-part-isolation \
  --workflow tests/fixtures/CFP-02.json \
  --part-id faceplate \
  --part-name "Helmet Faceplate" \
  --source-1-node 191 \
  --source-2-node 227 \
  --output-prefix CFP-03/faceplate_candidate \
  --output examples/CFP-03_Part_Isolation_Faceplate_generated.json
```

The source file is never overwritten unless `--in-place` is explicitly used.
An edit is not saved when it introduces validation errors.

Repair legacy FLUX Kontext subgraphs whose hidden CLIP widget or quarantined
proxy value still contains the template prompt:

```bash
uv run cfp repair-kontext-prompts \
  --workflow path/to/workflow.json \
  --in-place
```

The repair treats the visible outer-node prompt as authoritative, synchronizes
the nested `CLIPTextEncode` fallback, and removes only the obsolete quarantined
`text` proxy entry. If ComfyUI has already replaced the visible prompt with the
legacy template value, restore it from a known-good workflow:

```bash
uv run cfp repair-kontext-prompts \
  --workflow path/to/workflow.json \
  --prompt-source tests/fixtures/CFP-02.json \
  --in-place
```

## Repository map

```text
src/cfp/                 Python package and CLI
src/cfp/builders/        Reusable graph transformations
src/cfp/prompts/         Editable prompt templates packaged with the CLI
tests/fixtures/          Immutable source workflow fixtures
tests/                   Automated graph and round-trip tests
examples/                Generated outputs and downloaded references
docs/                    Architecture and schema notes
comfyui_trellis2_mlx/    Native macOS Swift/MLX ComfyUI custom node
```

The currently available fixture is preserved byte-for-byte:

```text
CFP-02.json
SHA-256: 854a49ca4a3bff5d22de7794803fe0098baf9fdefc76df26d779bf02278a23d7
```

The project orders also name
`CFP-03_Part_Isolation_Faceplate_v001.json` as an existing baseline. That file
was not present in the accessible ComfyUI, shared-data, attachment, or CFP
directories during repository initialization. It has not been fabricated or
misrepresented as a supplied fixture. The programmatically generated example
in `examples/` is a new artifact.

Two downloaded TRELLIS2 workflows are preserved as external reference fixtures:

```text
examples/downloaded_examples/High_Quality_GGUF.json
examples/downloaded_examples/Trellis2Multiviews_GGUF.json
```

They are not CFP baselines and are not expected to run in the current ComfyUI
installation. Their structural lessons and compatibility requirements are
documented in
[downloaded_trellis2_examples.md](docs/downloaded_trellis2_examples.md).

## Design rule

The recurring camera-generation contract is:

> Rotate only the camera. The object has not changed.

See [camera_convention.md](docs/camera_convention.md) for the coordinate
contract used by filenames, prompts, metadata, and future tests.

See [reconstruction_contract.md](docs/reconstruction_contract.md) for the
model-agnostic boundary between approved camera artifacts and future 3D
reconstruction adapters.
