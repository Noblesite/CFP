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
engine and exposes a ComfyUI `FILE_3D_GLB` output for the proven 512-resolution
single-image lane.

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
