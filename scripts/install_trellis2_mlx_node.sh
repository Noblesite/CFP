#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(dirname -- "$script_dir")
engine_project="$project_root/mlx-trellis2-swift"
engine_binary="$engine_project/.build/arm64-apple-macosx/release/trellis2-run-engine"
metal_library="$engine_project/.build/arm64-apple-macosx/release/mlx.metallib"
engine_patch="$project_root/patches/mlx-trellis2-swift-comfy-engine.patch"
node_source="$project_root/comfyui_trellis2_mlx"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/ComfyUI" >&2
    exit 2
fi

comfyui_dir=$1
if [ ! -f "$comfyui_dir/main.py" ] || [ ! -d "$comfyui_dir/custom_nodes" ]; then
    echo "Not a ComfyUI checkout: $comfyui_dir" >&2
    exit 2
fi

if [ ! -d "$engine_project/.git" ]; then
    echo "Missing mlx-trellis2-swift checkout: $engine_project" >&2
    exit 1
fi

if git -C "$engine_project" apply --unidiff-zero --reverse --check "$engine_patch" >/dev/null 2>&1; then
    echo "CFP engine patch is already applied."
elif git -C "$engine_project" apply --unidiff-zero --check "$engine_patch" >/dev/null 2>&1; then
    echo "Applying CFP Xcode, memory-budget, deterministic-seed, and multi-view patch..."
    git -C "$engine_project" apply --unidiff-zero "$engine_patch"
else
    echo "The CFP engine patch does not match the mlx-trellis2-swift checkout." >&2
    echo "Expected upstream commit: 005a372" >&2
    exit 1
fi

echo "Building native TRELLIS.2 MLX engine..."
(cd "$engine_project" && swift build -c release --product trellis2-run-engine)

if [ ! -x "$engine_binary" ]; then
    echo "Release engine was not produced: $engine_binary" >&2
    exit 1
fi

if [ -L "$metal_library" ] && [ ! -e "$metal_library" ]; then
    unlink "$metal_library"
fi

if [ ! -f "$metal_library" ]; then
    derived_data_root="$HOME/Library/Developer/Xcode/DerivedData"
    mlx_metal=$(find "$derived_data_root" -type f \
        -path '*/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib' \
        -print -quit 2>/dev/null || true)
    if [ -z "$mlx_metal" ]; then
        echo "MLX default.metallib was not found in Xcode DerivedData." >&2
        echo "Build mlx-trellis2-swift once in Xcode, then rerun this installer." >&2
        exit 1
    fi
    cp "$mlx_metal" "$metal_library"
fi

node_target="$comfyui_dir/custom_nodes/comfyui_trellis2_mlx"
if [ -L "$node_target" ]; then
    current_target=$(readlink "$node_target")
    if [ "$current_target" != "$node_source" ]; then
        echo "Existing symlink points somewhere else: $node_target -> $current_target" >&2
        exit 1
    fi
elif [ -e "$node_target" ]; then
    echo "Custom-node target already exists and is not a symlink: $node_target" >&2
    exit 1
else
    ln -s "$node_source" "$node_target"
fi

workflow_dir="$comfyui_dir/user/default/workflows/CFP"
mkdir -p "$workflow_dir"
cp "$node_source/workflows/trellis2_mlx_image_to_3d.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Image_to_3D_v001.json"
cp "$node_source/workflows/trellis2_mlx_background_clean.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Background_Clean_v001.json"
cp "$node_source/workflows/trellis2_mlx_multiview.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_MultiView_v001.json"
cp "$node_source/workflows/trellis2_mlx_image_conditioning.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Image_Conditioning_v001.json"
cp "$node_source/workflows/trellis2_mlx_geometry_only.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Geometry_Only_v001.json"
cp "$node_source/workflows/trellis2_mlx_remove_floaters.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Remove_Floaters_v001.json"
cp "$node_source/workflows/trellis2_mlx_topology_diagnostics.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Topology_Diagnostics_v001.json"
cp "$node_source/workflows/trellis2_mlx_topology_sanitizer.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Topology_Sanitizer_v001.json"
cp "$node_source/workflows/trellis2_mlx_voxel_remesh_candidate.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Voxel_Remesh_Candidate_v001.json"
cp "$node_source/workflows/trellis2_mlx_voxel_resolution_ab.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Voxel_Resolution_AB_v001.json"
cp "$node_source/workflows/trellis2_mlx_post_voxel_polish.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Post_Voxel_Polish_v001.json"
cp "$node_source/workflows/trellis2_mlx_print_scale_gate.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Print_Scale_Gate_v001.json"
cp "$node_source/workflows/trellis2_mlx_background_geometry_guard.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Background_Geometry_Guard_v001.json"
cp "$node_source/workflows/trellis2_mlx_input_mask_quality_gate.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Input_Mask_Quality_Gate_v001.json"
cp "$node_source/workflows/trellis2_mlx_multiview_mask_gated.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_MultiView_Mask_Gated_v001.json"
cp "$node_source/workflows/trellis2_mlx_four_view_mask_gated.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Four_View_Mask_Gated_v001.json"
cp "$node_source/workflows/trellis2_mlx_four_view_consistency_gated.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Four_View_Consistency_Gated_v001.json"
cp "$node_source/workflows/trellis2_mlx_four_view_alignment_review.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Four_View_Alignment_Review_v001.json"
cp "$node_source/workflows/trellis2_mlx_four_view_alignment_candidate.json" \
    "$workflow_dir/CFP_TRELLIS2_MLX_Four_View_Alignment_Candidate_v001.json"

echo "TRELLIS.2 MLX node and workflows installed."
echo "Restart ComfyUI, then open Workflows -> CFP."
