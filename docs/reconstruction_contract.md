# Reconstruction-stage contract

This contract separates CFP's approved image artifacts from any particular
3D-reconstruction backend. TRELLIS2 is one possible adapter, not part of the
contract itself.

## Required inputs

A reconstruction request identifies:

```text
project_id
component_id
camera_convention
approved camera artifacts
reconstruction settings
output location
```

Every camera artifact has an explicit role:

```text
camera_000
camera_090
camera_180
camera_270
```

Adapters may accept fewer views, but they must declare which roles they
consumed. Images are never inferred from list position alone.

Each artifact should eventually carry:

```text
path
content hash
approval revision
azimuth
elevation
roll
pixel dimensions
background policy
```

## Backend request

Backend-specific controls remain inside an adapter-owned settings object:

```text
backend_id
model_id
model_revision
seed
quality preset
backend parameters
```

CFP preserves those settings as metadata without making the broader pipeline
depend on TRELLIS-specific types such as `MESHWITHVOXEL` or `BVH`.

## Required outputs

A reconstruction adapter returns an artifact set:

```text
raw geometry
optional acceleration structure
optional textures
execution metadata
diagnostics
```

Raw geometry is exported before repair or simplification. Later stages produce
new artifacts rather than overwriting it:

```text
mesh_raw
mesh_repaired
mesh_refined
mesh_textured
```

## Review boundary

Successful backend execution means only that an artifact was produced. It does
not mean the artifact is approved.

Before promotion, human or future diagnostic review evaluates:

```text
camera-set consistency
component identity
silhouette agreement
missing geometry
floating components
gross topology problems
scale plausibility
component-fit plausibility
```

The next stage consumes a promoted artifact reference, not an implicit
in-memory continuation from reconstruction.
