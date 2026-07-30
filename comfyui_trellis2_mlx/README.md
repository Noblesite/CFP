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
- textured GLB output compatible with ComfyUI's `Preview 3D` and
  `Save 3D Model` nodes.

The node unloads cached ComfyUI models before starting Swift because both
runtimes share unified memory.

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
        |
        +--> Preview 3D & Animation
        |
        +--> Save 3D Model
```

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
