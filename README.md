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

## Repository map

```text
src/cfp/                 Python package and CLI
src/cfp/builders/        Reusable graph transformations
src/cfp/prompts/         Editable prompt templates packaged with the CLI
tests/fixtures/          Immutable source workflow fixtures
tests/                   Automated graph and round-trip tests
examples/                Generated, reviewable workflow outputs
docs/                    Architecture and schema notes
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

## Design rule

The recurring camera-generation contract is:

> Rotate only the camera. The object has not changed.

See [camera_convention.md](docs/camera_convention.md) for the coordinate
contract used by filenames, prompts, metadata, and future tests.

