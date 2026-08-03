import json
import os
import time
from pathlib import Path

import pytest

from comfyui_trellis2_mlx.runner import (
    Trellis2MLXConfig,
    build_environment,
    build_sparse_environment,
    default_paths,
    inspect_safetensors,
    run_engine,
)


REQUIRED_WEIGHTS = {
    "dino.safetensors",
    "normalization.json",
    "shape_dec.safetensors",
    "shape_flow_512.safetensors",
    "struct_dec.safetensors",
    "struct_flow.safetensors",
    "tex_dec.safetensors",
    "tex_flow_512.safetensors",
}


def make_fake_config(tmp_path: Path, script: str) -> Trellis2MLXConfig:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    engine = engine_dir / "trellis2-run-engine"
    engine.write_text(script, encoding="utf-8")
    engine.chmod(0o755)
    (engine_dir / "mlx.metallib").write_bytes(b"metal")
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in REQUIRED_WEIGHTS:
        (weights / name).write_bytes(b"fixture")
    return Trellis2MLXConfig(engine, weights, 0.95)


def test_default_paths_are_relative_to_cfp_checkout():
    node_directory = Path("/workspace/CFP/comfyui_trellis2_mlx")

    engine, weights = default_paths(node_directory)

    assert engine == Path(
        "/workspace/CFP/mlx-trellis2-swift/.build/arm64-apple-macosx/release/trellis2-run-engine"
    )
    assert weights == Path("/workspace/CFP/trellis2-mlx")


def test_background_workflow_embeds_official_rmbg_model_metadata():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_background_clean.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    rmbg_node = next(node for node in workflow["nodes"] if node["type"] == "RMBG")
    models = rmbg_node["properties"]["models"]

    assert {model["name"] for model in models} == {
        "BiRefNet_config.py",
        "birefnet.py",
        "config.json",
        "model.safetensors",
    }
    assert all(model["directory"] == "RMBG/RMBG-2.0" for model in models)
    assert all(
        model["url"].startswith("https://huggingface.co/briaai/RMBG-2.0/resolve/main/")
        for model in models
    )

    note = next(node for node in workflow["nodes"] if node["type"] == "MarkdownNote")
    markdown = note["widgets_values"][0]
    assert "does not turn the backdrop into geometry" in markdown
    assert "https://huggingface.co/briaai/RMBG-2.0" in markdown
    assert "ComfyUI/models/RMBG/RMBG-2.0/" in markdown
    assert "Commercial use requires a separate agreement with BRIA" in markdown


def test_input_mask_quality_workflow_gates_rmbg_alpha_before_inference():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_input_mask_quality_gate.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    rmbg = next(node for node in workflow["nodes"] if node["type"] == "RMBG")
    gate = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXInputMaskQualityGate"
    )
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXImageTo3D"
    )

    assert gate["widgets_values"] == [0.5, 0.05, 0.85, 0.1, 0.02, "no"]
    assert rmbg["outputs"][0]["links"] == [3]
    assert rmbg["outputs"][1]["links"] is None
    assert gate["inputs"][0]["link"] == 3
    assert gate["inputs"][7]["link"] is None
    assert generator["inputs"][1]["link"] == 7
    assert generator["inputs"][6]["link"] == 8
    assert gate["outputs"][0]["links"] == [7]
    assert gate["outputs"][1]["links"] == [8]


def test_single_image_manufacturing_workflow_wraps_complete_refinement_lane():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_single_image_manufacturing.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    def node(node_type, title=None):
        return next(
            candidate
            for candidate in workflow["nodes"]
            if candidate["type"] == node_type
            and (title is None or candidate.get("title") == title)
        )

    def source_for(target_node, input_name):
        target_input = next(
            item for item in target_node["inputs"] if item["name"] == input_name
        )
        link = links[target_input["link"]]
        return nodes[link[1]], link[2], link[5]

    rmbg = node("RMBG")
    mask_gate = node("Trellis2MLXInputMaskQualityGate")
    generator = node("Trellis2MLXImageTo3D")
    raw_preview = node("Preview3D", "RAW TRELLIS — Untouched Generated Mesh")
    raw_save = node("SaveGLB", "Save RAW TRELLIS Artifact")
    remove_floaters = node("Trellis2MLXRemoveFloaters")
    before_sanitizer = node(
        "Preview3D", "BEFORE SANITIZER — Untouched Incoming Mesh"
    )
    sanitizer = node("Trellis2MLXTopologySanitizer")
    background_guard = node("Trellis2MLXBackgroundGeometryGuard")
    voxel = node("Trellis2MLXVoxelRemeshCandidate")
    polish = node("Trellis2MLXPostVoxelTopologyPolish")
    scale = node("Trellis2MLXPrintScaleFeatureGate")
    final_preview = node("Preview3D", "FINAL 250 mm — Print Preview")
    final_save = node("SaveGLB", "Save FINAL 250 mm Candidate")
    contract = node("MarkdownNote", "Single-Image Manufacturing Flow Contract")

    assert len(rmbg["properties"]["models"]) == 4
    assert source_for(mask_gate, "image")[0] is rmbg
    assert source_for(generator, "image")[0] is mask_gate
    assert source_for(generator, "mask")[0] is mask_gate
    assert generator["widgets_values"] == [0, "fixed", 12, "geometry_only", "off"]

    assert source_for(raw_preview, "model_file")[0] is generator
    assert source_for(raw_save, "mesh")[0] is generator
    assert raw_save["widgets_values"][0] == "CFP/TRELLIS2_MLX_SINGLE_IMAGE_RAW"

    assert source_for(remove_floaters, "model_3d")[0] is generator
    assert source_for(before_sanitizer, "model_file")[0] is remove_floaters
    assert source_for(sanitizer, "model_3d")[0] is remove_floaters
    assert source_for(background_guard, "model_3d")[0] is sanitizer
    assert source_for(voxel, "model_3d")[0] is background_guard
    assert source_for(polish, "candidate_3d")[0] is voxel
    assert source_for(scale, "model_3d")[0] is polish
    assert source_for(final_preview, "model_file")[0] is scale
    assert source_for(final_save, "mesh")[0] is scale

    assert voxel["widgets_values"] == [1024]
    assert scale["widgets_values"] == [250.0, "z", 1024, 0.4, 0.2]
    assert final_save["widgets_values"][0] == (
        "CFP/TRELLIS2_MLX_SINGLE_IMAGE_FINAL_250MM"
    )
    assert "No stage overwrites the raw TRELLIS artifact" in contract["widgets_values"][0]
    assert not any(
        candidate["type"] == "Trellis2MLXMultiViewTo3D"
        for candidate in workflow["nodes"]
    )


@pytest.mark.parametrize(
    "workflow_name",
    [
        "trellis2_mlx_image_to_3d.json",
        "trellis2_mlx_background_clean.json",
        "trellis2_mlx_geometry_only.json",
    ],
)
def test_trellis_workflows_include_mesh_report_checkpoint(workflow_name):
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / workflow_name
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXImageTo3D"
    )
    report_node = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMeshReport"
    )

    report_link = next(
        link for link in workflow["links"] if link[3] == report_node["id"]
    )
    assert report_link[1:3] == [generator["id"], 0]
    assert report_link[4:] == [0, "FILE_3D_GLB"]
    assert report_link[0] in generator["outputs"][0]["links"]


def test_multiview_workflow_uses_three_separate_ordered_views():
    root = Path(__file__).parents[1]
    workflow_path = (
        root
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_multiview.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMultiViewTo3D"
    )
    rmbg_nodes = [node for node in workflow["nodes"] if node["type"] == "RMBG"]

    assert len(rmbg_nodes) == 3
    assert all(
        len(node["properties"]["models"]) == 4
        for node in rmbg_nodes
    )
    assert [generator["inputs"][index]["link"] for index in (1, 2, 3, 4)] == [
        7,
        8,
        9,
        None,
    ]
    assert generator["inputs"][1]["name"] == "image_000"
    assert generator["inputs"][2]["name"] == "image_090"
    assert generator["inputs"][3]["name"] == "image_180"
    assert generator["inputs"][4]["name"] == "image_270"

    patch_text = (
        root / "patches" / "mlx-trellis2-swift-comfy-engine.patch"
    ).read_text(encoding="utf-8")
    assert "ADDITIONAL_VIEWS_MANIFEST" in patch_text
    assert "additionalViews: additionalViews.isEmpty ? nil : additionalViews" in patch_text
    assert 'switch env["TEXTURE"]?.lowercased()' in patch_text
    assert "texture: textureEnabled" in patch_text
    assert 'rec["output_mode"] = configuration.texture ? "textured" : "geometry_only"' in patch_text
    assert "MeshBake.runGeometry(" in patch_text
    assert 'metrics.backend = texture ? unwrapBackend.rawValue : "none"' in patch_text
    assert 'rec["uv_enabled"] = generated.uvs != nil' in patch_text
    assert 'rec["texture_embedded"] = generated.texRGBA != nil' in patch_text
    assert "baseColorRGBA: baseColorRGBA" in patch_text
    assert "Trellis2ConditioningRequest" in patch_text
    assert "MLX.saveToData(arrays: arrays" in patch_text
    assert 'engineStage == "conditioning"' in patch_text
    assert "Trellis2ImagePreparation.mattedIfNeeded" in patch_text
    assert 'progress.update("matting_evict")' in patch_text
    assert "engine.trimCaches()" in patch_text


def test_generation_nodes_always_rerun_external_engine_artifacts():
    node_source = (
        Path(__file__).parents[1] / "comfyui_trellis2_mlx" / "nodes.py"
    ).read_text(encoding="utf-8")

    assert node_source.count("def fingerprint_inputs(cls, **kwargs):") == 4
    assert node_source.count('return float("nan")') == 4


def test_conditioning_workflow_stops_before_sparse_or_mesh_generation():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_image_conditioning.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    conditioning = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXImageConditioning"
    )
    load_nodes = [node for node in workflow["nodes"] if node["type"] == "LoadImage"]

    assert len(load_nodes) == 4
    assert conditioning["widgets_values"] == ["off"]
    assert [conditioning["inputs"][index]["link"] for index in (1, 2, 3, 4)] == [2, 4, 6, 8]
    assert [conditioning["inputs"][index]["link"] for index in (6, 7, 8, 9)] == [3, 5, 7, 9]
    assert conditioning["outputs"][0]["type"] == "TRELLIS2_MLX_CONDITIONING"
    assert not any(
        node["type"] in {"Trellis2MLXImageTo3D", "Trellis2MLXMultiViewTo3D", "SaveGLB"}
        for node in workflow["nodes"]
    )


def test_sparse_workflow_consumes_explicit_conditioning_and_stops_before_shape():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_sparse_structure.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    loader = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXLoadConditioningArtifact"
    )
    sparse = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXSparseStructure"
    )
    note = next(node for node in workflow["nodes"] if node["type"] == "MarkdownNote")

    assert loader["widgets_values"] == ["/path/to/promoted/trellis2_mlx_conditioning.safetensors"]
    assert sparse["widgets_values"] == [0, "fixed", 12]
    assert sparse["inputs"][1]["link"] == 2
    assert sparse["outputs"][0]["type"] == "TRELLIS2_MLX_SPARSE_STRUCTURE"
    assert "explicitly promoted" in note["widgets_values"][0]
    assert "Shape, texture, mesh extraction" in note["widgets_values"][0]
    assert not any(
        node["type"]
        in {
            "LoadImage",
            "Trellis2MLXImageConditioning",
            "Trellis2MLXImageTo3D",
            "Trellis2MLXMultiViewTo3D",
            "Preview3D",
            "SaveGLB",
        }
        for node in workflow["nodes"]
    )


def test_every_sanitizing_workflow_previews_the_exact_untouched_input():
    workflow_directory = (
        Path(__file__).parents[1] / "comfyui_trellis2_mlx" / "workflows"
    )
    sanitizer_workflows = []

    for workflow_path in workflow_directory.glob("*.json"):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        sanitizer = next(
            (
                node
                for node in workflow["nodes"]
                if node["type"] == "Trellis2MLXTopologySanitizer"
            ),
            None,
        )
        if sanitizer is None:
            continue
        sanitizer_workflows.append(workflow_path.name)
        preview = next(
            node
            for node in workflow["nodes"]
            if node.get("title") == "BEFORE SANITIZER — Untouched Incoming Mesh"
        )
        sanitizer_input_link = next(
            link
            for link in workflow["links"]
            if link[0] == sanitizer["inputs"][0]["link"]
        )
        preview_input_link = next(
            link for link in workflow["links"] if link[0] == preview["inputs"][0]["link"]
        )

        assert sanitizer_input_link[1:3] == preview_input_link[1:3]
        assert sanitizer_input_link[5] == preview_input_link[5] == "FILE_3D_GLB"

    assert set(sanitizer_workflows) == {
        "trellis2_mlx_background_geometry_guard.json",
        "trellis2_mlx_post_voxel_polish.json",
        "trellis2_mlx_print_scale_gate.json",
        "trellis2_mlx_single_image_manufacturing.json",
        "trellis2_mlx_topology_sanitizer.json",
        "trellis2_mlx_voxel_remesh_candidate.json",
        "trellis2_mlx_voxel_resolution_ab.json",
    }


def test_multiview_mask_gated_workflow_checks_each_populated_camera_branch():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_multiview_mask_gated.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    gates = sorted(
        (
            node
            for node in workflow["nodes"]
            if node["type"] == "Trellis2MLXInputMaskQualityGate"
        ),
        key=lambda node: node["id"],
    )
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMultiViewTo3D"
    )
    rmbg_nodes = [node for node in workflow["nodes"] if node["type"] == "RMBG"]

    assert [node["title"] for node in gates] == [
        "Mask Gate 000° — Front",
        "Mask Gate 090° — Left",
        "Mask Gate 180° — Rear",
    ]
    assert all(node["widgets_values"] == [0.5, 0.05, 0.85, 0.1, 0.02, "no"] for node in gates)
    assert all(node["inputs"][7]["link"] is None for node in gates)
    assert all(node["outputs"][0]["links"] for node in gates)
    assert all(node["outputs"][1]["links"] for node in gates)
    assert all(node["outputs"][1]["links"] is None for node in rmbg_nodes)
    assert [generator["inputs"][index]["link"] for index in (1, 2, 3)] == [13, 14, 15]
    assert generator["inputs"][4]["link"] is None
    assert [generator["inputs"][index]["link"] for index in (9, 10, 11)] == [16, 17, 18]
    assert generator["inputs"][12]["link"] is None


def test_four_view_mask_gated_workflow_populates_right_camera_and_mask():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_four_view_mask_gated.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMultiViewTo3D"
    )
    load_nodes = [node for node in workflow["nodes"] if node["type"] == "LoadImage"]
    gates = [
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXInputMaskQualityGate"
    ]

    assert {node["title"] for node in load_nodes} == {
        "Load Camera 000° — Front",
        "Load Camera 090° — Left",
        "Load Camera 180° — Rear",
        "Load Camera 270° — Right",
    }
    assert {node["title"] for node in gates} == {
        "Mask Gate 000° — Front",
        "Mask Gate 090° — Left",
        "Mask Gate 180° — Rear",
        "Mask Gate 270° — Right",
    }
    assert [generator["inputs"][index]["link"] for index in (1, 2, 3, 4)] == [
        13,
        14,
        15,
        21,
    ]
    assert [generator["inputs"][index]["link"] for index in (9, 10, 11, 12)] == [
        16,
        17,
        18,
        22,
    ]
    assert next(
        node for node in workflow["nodes"] if node["type"] == "SaveGLB"
    )["widgets_values"][0] == "CFP/TRELLIS2_MLX_FOUR_VIEW_MASK_GATED"


def test_four_view_consistency_workflow_routes_all_views_through_set_gate():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_four_view_consistency_gated.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    gate = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXModelSheetConsistencyGate"
    )
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMultiViewTo3D"
    )

    assert gate["widgets_values"] == [0.08, 0.05, 0.05, 0.03, 0.15, "no"]
    assert [node_input["link"] for node_input in gate["inputs"][:8]] == [
        13,
        14,
        15,
        21,
        16,
        17,
        18,
        22,
    ]
    assert [output["links"] for output in gate["outputs"][:8]] == [
        [23],
        [24],
        [25],
        [26],
        [27],
        [28],
        [29],
        [30],
    ]
    assert [generator["inputs"][index]["link"] for index in (1, 2, 3, 4)] == [
        23,
        24,
        25,
        26,
    ]
    assert [generator["inputs"][index]["link"] for index in (9, 10, 11, 12)] == [
        27,
        28,
        29,
        30,
    ]


def test_alignment_review_workflow_branches_without_mutating_generation_inputs():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_four_view_alignment_review.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    review = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXModelSheetAlignmentReview"
    )
    consistency = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXModelSheetConsistencyGate"
    )
    preview = next(
        node
        for node in workflow["nodes"]
        if node.get("title") == "Alignment Contact Sheet — Human Review"
    )

    assert review["widgets_values"] == [384]
    assert [node_input["link"] for node_input in review["inputs"][:8]] == [
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
    ]
    assert review["outputs"][0]["links"] == [39, 40]
    assert preview["inputs"][0]["link"] == 39
    assert [node_input["link"] for node_input in consistency["inputs"][:8]] == [
        13,
        14,
        15,
        21,
        16,
        17,
        18,
        22,
    ]
    assert next(
        node
        for node in workflow["nodes"]
        if node.get("title") == "Save Alignment Review"
    )["widgets_values"][0] == "CFP/TRELLIS2_MLX_ALIGNMENT_REVIEW"


def test_alignment_candidate_workflow_keeps_candidate_disconnected_from_trellis():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_four_view_alignment_candidate.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    candidate = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXModelSheetAlignmentCandidate"
    )
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMultiViewTo3D"
    )
    after_review = next(
        node
        for node in workflow["nodes"]
        if node.get("title") == "AFTER — Candidate Alignment Review"
    )

    assert candidate["widgets_values"] == ["median", 0.82, 250.0]
    assert [node_input["link"] for node_input in candidate["inputs"][:8]] == [
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
    ]
    assert [node_input["link"] for node_input in after_review["inputs"][:8]] == [
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
    ]
    assert [generator["inputs"][index]["link"] for index in (1, 2, 3, 4)] == [
        23,
        24,
        25,
        26,
    ]
    candidate_node_id = candidate["id"]
    assert not any(
        link[1] == candidate_node_id and link[3] == generator["id"]
        for link in workflow["links"]
    )
    assert next(
        node
        for node in workflow["nodes"]
        if node.get("title") == "Save AFTER Alignment Review"
    )["widgets_values"][0] == "CFP/TRELLIS2_MLX_ALIGNMENT_CANDIDATE_REVIEW"


def test_geometry_only_workflow_selects_geometry_mode_and_preserves_review_gate():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_geometry_only.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    generator = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXImageTo3D"
    )
    input_names = [item["name"] for item in generator["inputs"]]
    output_mode_index = input_names.index("output_mode")

    assert generator["widgets_values"][3] == "geometry_only"
    assert generator["title"] == "TRELLIS.2 MLX Geometry Only"
    assert generator["inputs"][output_mode_index]["link"] is None
    assert next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMeshReport"
    )
    assert next(
        node for node in workflow["nodes"] if node["type"] == "SaveGLB"
    )["widgets_values"][0] == "CFP/TRELLIS2_MLX_GEOMETRY_ONLY"

    node_source = (
        Path(__file__).parents[1] / "comfyui_trellis2_mlx" / "nodes.py"
    ).read_text(encoding="utf-8")
    assert "texture-free GLB with no UVs or atlas" in node_source


def test_remove_floaters_workflow_has_before_after_review_gates():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_remove_floaters.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    cleanup = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXRemoveFloaters"
    )
    reports = [
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXMeshReport"
    ]

    assert cleanup["widgets_values"] == [100, 0.001]
    assert cleanup["outputs"][0]["links"] == [6, 7, 8]
    assert {node["title"] for node in reports} == {
        "BEFORE — Raw Geometry Report",
        "AFTER — Cleaned Geometry Report",
    }
    assert next(
        node for node in workflow["nodes"] if node["type"] == "SaveGLB"
    )["widgets_values"][0] == "CFP/TRELLIS2_MLX_FLOATERS_REMOVED"


def test_topology_diagnostics_workflow_analyzes_cleaned_geometry():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_topology_diagnostics.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    cleanup = next(
        node for node in workflow["nodes"] if node["type"] == "Trellis2MLXRemoveFloaters"
    )
    diagnostics = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXTopologyDiagnostics"
    )
    diagnostic_link = next(link for link in workflow["links"] if link[0] == 9)

    assert cleanup["outputs"][0]["links"] == [6, 7, 8, 9]
    assert diagnostics["widgets_values"] == [0.000001]
    assert diagnostic_link[1:5] == [4, 0, 9, 0]


def test_topology_sanitizer_workflow_has_before_after_diagnostics():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_topology_sanitizer.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    sanitizer = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXTopologySanitizer"
    )
    diagnostics = [
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXTopologyDiagnostics"
    ]

    assert sanitizer["widgets_values"] == [1e-8]
    assert sanitizer["outputs"][0]["links"] == [8, 9, 10, 11]
    assert {node["title"] for node in diagnostics} == {
        "BEFORE SANITIZER — O-Voxel Diagnostics",
        "AFTER SANITIZER — O-Voxel Diagnostics",
    }
    assert all(node["widgets_values"] == [1e-8] for node in diagnostics)
    assert next(
        node for node in workflow["nodes"] if node["type"] == "SaveGLB"
    )["widgets_values"][0] == "CFP/TRELLIS2_MLX_TOPOLOGY_SANITIZED"


def test_voxel_remesh_workflow_preserves_source_and_reviews_candidate():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_voxel_remesh_candidate.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    remesh = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXVoxelRemeshCandidate"
    )
    previews = [node for node in workflow["nodes"] if node["type"] == "Preview3D"]
    diagnostics = [
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXTopologyDiagnostics"
    ]

    assert remesh["widgets_values"] == [192]
    assert remesh["outputs"][0]["links"] == [13, 14, 15, 16]
    assert {node["title"] for node in previews} >= {
        "SOURCE — Sanitized Geometry",
        "CANDIDATE — Watertight Voxel Remesh",
    }
    assert {node["title"] for node in diagnostics} >= {
        "BEFORE VOXEL — Sanitized Diagnostics",
        "AFTER VOXEL — Candidate Diagnostics",
    }
    assert next(
        node for node in workflow["nodes"] if node["type"] == "SaveGLB"
    )["widgets_values"][0] == "CFP/TRELLIS2_MLX_VOXEL_REMESH_CANDIDATE"


def test_voxel_resolution_ab_workflow_fans_out_and_requires_human_review():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_voxel_resolution_ab.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    remesh_nodes = [
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXVoxelRemeshCandidate"
    ]
    previews = [
        node["title"]
        for node in workflow["nodes"]
        if node["type"] == "Preview3D" and "Candidate Preview" in node.get("title", "")
    ]
    save_prefixes = [
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "SaveGLB"
    ]
    comparison = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXVoxelCandidateComparison"
    )

    assert sorted(node["widgets_values"][0] for node in remesh_nodes) == [
        128,
        192,
        256,
    ]
    assert set(previews) == {
        "128 — Candidate Preview",
        "192 — Candidate Preview",
        "256 — Candidate Preview",
    }
    assert set(save_prefixes) == {
        "CFP/TRELLIS2_MLX_VOXEL_128",
        "CFP/TRELLIS2_MLX_VOXEL_192",
        "CFP/TRELLIS2_MLX_VOXEL_256",
    }
    assert [node_input["link"] for node_input in comparison["inputs"]] == [
        23,
        24,
        25,
    ]
    assert comparison["title"] == "128 / 192 / 256 — Human Review Comparison"


def test_post_voxel_polish_workflow_preserves_raw_and_saves_polished_candidate():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_post_voxel_polish.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    remesh = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXVoxelRemeshCandidate"
    )
    polish = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXPostVoxelTopologyPolish"
    )
    previews = {
        node["title"]
        for node in workflow["nodes"]
        if node["type"] == "Preview3D"
    }
    diagnostics = {
        node["title"]
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXTopologyDiagnostics"
    }
    save = next(node for node in workflow["nodes"] if node["type"] == "SaveGLB")

    assert remesh["widgets_values"] == [256]
    assert remesh["outputs"][0]["links"] == [13, 16, 17]
    assert polish["inputs"][0]["link"] == 17
    assert polish["outputs"][0]["links"] == [14, 15, 18, 19]
    assert {
        "RAW 256 — Before Polish",
        "POLISHED 256 — After Conservative Repair",
    } <= previews
    assert {
        "RAW 256 — Diagnostics",
        "POLISHED 256 — Diagnostics",
    } <= diagnostics
    assert save["widgets_values"][0] == "CFP/TRELLIS2_MLX_VOXEL_256_POLISHED"


def test_print_scale_workflow_uses_polished_256_and_saves_250mm_candidate():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_print_scale_gate.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    polish = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXPostVoxelTopologyPolish"
    )
    scale = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXPrintScaleFeatureGate"
    )
    save = next(node for node in workflow["nodes"] if node["type"] == "SaveGLB")

    assert polish["outputs"][0]["links"] == [20]
    assert scale["inputs"][0]["link"] == 20
    assert scale["widgets_values"] == [250.0, "z", 256, 0.4, 0.2]
    assert scale["outputs"][0]["links"] == [14, 15, 18, 19]
    assert save["widgets_values"][0] == "CFP/TRELLIS2_MLX_PRINT_250MM"
    assert next(
        node
        for node in workflow["nodes"]
        if node.get("title") == "SCALED 250 mm — Print Preview"
    )


def test_background_guard_workflow_blocks_before_voxel_by_default():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_background_geometry_guard.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    sanitizer = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXTopologySanitizer"
    )
    guard = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXBackgroundGeometryGuard"
    )
    remesh = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "Trellis2MLXVoxelRemeshCandidate"
    )

    assert sanitizer["outputs"][0]["links"] == [8, 11, 21]
    assert guard["inputs"][0]["link"] == 21
    assert guard["widgets_values"] == [
        "character_z_up",
        1.25,
        0.02,
        0.4,
        "no",
    ]
    assert guard["outputs"][0]["links"] == [12]
    assert remesh["inputs"][0]["link"] == 12


def test_build_environment_preserves_parent_and_sets_proven_contract(tmp_path):
    config = Trellis2MLXConfig(
        engine_binary=Path("/engine"),
        weights_directory=Path("/weights"),
        memory_fraction=0.95,
    )

    environment = build_environment(
        {"PATH": "/usr/bin"},
        config=config,
        image_path=tmp_path / "input.png",
        output_path=tmp_path / "output.glb",
        metrics_path=tmp_path / "metrics.json",
        seed=42,
        steps=12,
        use_matting=False,
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["HR_RES"] == "512"
    assert environment["SEED"] == "42"
    assert environment["STEPS"] == "12"
    assert environment["MATTING"] == "off"
    assert environment["TEXTURE"] == "on"
    assert environment["ENGINE_MEMORY_FRACTION"] == "0.95"
    assert "ADDITIONAL_VIEWS_MANIFEST" not in environment


def test_build_environment_passes_ordered_additional_views_manifest(tmp_path):
    config = Trellis2MLXConfig(
        engine_binary=Path("/engine"),
        weights_directory=Path("/weights"),
        memory_fraction=0.95,
    )
    manifest = tmp_path / "views.json"

    environment = build_environment(
        {"ADDITIONAL_VIEWS_MANIFEST": "/stale/views.json"},
        config=config,
        image_path=tmp_path / "000.png",
        output_path=tmp_path / "output.glb",
        metrics_path=tmp_path / "metrics.json",
        seed=42,
        steps=12,
        use_matting=False,
        additional_views_manifest=manifest,
    )

    assert environment["IMG"].endswith("000.png")
    assert environment["ADDITIONAL_VIEWS_MANIFEST"] == str(manifest)


def test_build_environment_disables_texture_for_geometry_only(tmp_path):
    config = Trellis2MLXConfig(Path("/engine"), Path("/weights"), 0.95)

    environment = build_environment(
        {},
        config=config,
        image_path=tmp_path / "input.png",
        output_path=tmp_path / "output.glb",
        metrics_path=tmp_path / "metrics.json",
        seed=0,
        steps=12,
        use_matting=False,
        output_mode="geometry_only",
    )

    assert environment["TEXTURE"] == "off"


def test_build_environment_configures_conditioning_stage_and_camera_order(tmp_path):
    config = Trellis2MLXConfig(Path("/engine"), Path("/weights"), 0.95)

    environment = build_environment(
        {},
        config=config,
        image_path=tmp_path / "000.png",
        output_path=tmp_path / "conditioning.safetensors",
        metrics_path=tmp_path / "report.json",
        seed=0,
        steps=12,
        use_matting=False,
        output_mode="geometry_only",
        engine_stage="conditioning",
        view_angles=(0, 90, 180, 270),
    )

    assert environment["ENGINE_STAGE"] == "conditioning"
    assert environment["VIEW_ANGLES"] == "0,90,180,270"
    assert environment["OUT_CONDITIONING"].endswith("conditioning.safetensors")


def test_build_sparse_environment_uses_only_promoted_conditioning(tmp_path):
    config = Trellis2MLXConfig(Path("/engine"), Path("/weights"), 0.95)

    environment = build_sparse_environment(
        {
            "IMG": "/stale/input.png",
            "OUT_GLB": "/stale/output.glb",
            "VIEW_ANGLES": "0,90",
            "ADDITIONAL_VIEWS_MANIFEST": "/stale/views.json",
        },
        config=config,
        conditioning_path=tmp_path / "conditioning.safetensors",
        output_path=tmp_path / "sparse.safetensors",
        metrics_path=tmp_path / "sparse.json",
        seed=93,
        steps=12,
    )

    assert environment["ENGINE_STAGE"] == "sparse"
    assert environment["IN_CONDITIONING"].endswith("conditioning.safetensors")
    assert environment["OUT_SPARSE"].endswith("sparse.safetensors")
    assert environment["SEED"] == "93"
    assert environment["STEPS"] == "12"
    assert environment["MATTING"] == "off"
    assert environment["TEXTURE"] == "off"
    assert not {
        "IMG",
        "OUT_GLB",
        "OUT_CONDITIONING",
        "VIEW_ANGLES",
        "ADDITIONAL_VIEWS_MANIFEST",
    } & environment.keys()


def test_build_environment_rejects_unknown_output_mode(tmp_path):
    config = Trellis2MLXConfig(Path("/engine"), Path("/weights"), 0.95)

    with pytest.raises(ValueError, match="output_mode"):
        build_environment(
            {},
            config=config,
            image_path=tmp_path / "input.png",
            output_path=tmp_path / "output.glb",
            metrics_path=tmp_path / "metrics.json",
            seed=0,
            steps=12,
            use_matting=False,
            output_mode="surprise",
        )


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_build_environment_rejects_invalid_seed(tmp_path, seed):
    config = Trellis2MLXConfig(Path("/engine"), Path("/weights"), 0.95)

    with pytest.raises(ValueError, match="unsigned 64-bit"):
        build_environment(
            {},
            config=config,
            image_path=tmp_path / "input.png",
            output_path=tmp_path / "output.glb",
            metrics_path=tmp_path / "metrics.json",
            seed=seed,
            steps=12,
            use_matting=False,
        )


def test_run_engine_returns_metrics_and_artifact_hash(tmp_path):
    config = make_fake_config(
        tmp_path,
        """#!/bin/sh
printf 'glTF00000000' > "$OUT_GLB"
printf '{"seed": 42}' > "$METRICS_JSON"
printf 'fake engine complete\n'
""",
    )
    output_path = tmp_path / "output.glb"
    metrics_path = tmp_path / "metrics.json"
    environment = build_environment(
        os.environ,
        config=config,
        image_path=tmp_path / "input.png",
        output_path=output_path,
        metrics_path=metrics_path,
        seed=42,
        steps=12,
        use_matting=False,
    )

    metrics, log_text = run_engine(
        config=config,
        environment=environment,
        output_path=output_path,
        metrics_path=metrics_path,
        on_log=lambda _: None,
        check_cancelled=lambda: None,
    )

    assert metrics["seed"] == 42
    assert metrics["artifact_bytes"] == 12
    assert len(metrics["artifact_sha256"]) == 64
    assert log_text == "fake engine complete"


def test_run_engine_accepts_conditioning_safetensors(tmp_path):
    config = make_fake_config(
        tmp_path,
        """#!/usr/bin/env python3
import json
import os

header = {
    "cond_512": {"dtype": "F32", "shape": [1, 2, 3], "data_offsets": [0, 24]},
    "neg_cond_512": {"dtype": "F32", "shape": [1, 2, 3], "data_offsets": [24, 48]},
}
encoded = json.dumps(header).encode("utf-8")
with open(os.environ["OUT_CONDITIONING"], "wb") as artifact:
    artifact.write(len(encoded).to_bytes(8, "little"))
    artifact.write(encoded)
    artifact.write(bytes(48))
with open(os.environ["METRICS_JSON"], "w", encoding="utf-8") as report:
    json.dump({"status": "CONDITIONING_READY", "view_count": 4}, report)
print("fake conditioning complete")
""",
    )
    output_path = tmp_path / "conditioning.safetensors"
    metrics_path = tmp_path / "conditioning.json"
    environment = build_environment(
        os.environ,
        config=config,
        image_path=tmp_path / "000.png",
        output_path=output_path,
        metrics_path=metrics_path,
        seed=0,
        steps=12,
        use_matting=False,
        output_mode="geometry_only",
        engine_stage="conditioning",
        view_angles=(0, 90, 180, 270),
    )

    metrics, log_text = run_engine(
        config=config,
        environment=environment,
        output_path=output_path,
        metrics_path=metrics_path,
        on_log=lambda _: None,
        check_cancelled=lambda: None,
        artifact_kind="safetensors",
    )

    assert metrics["status"] == "CONDITIONING_READY"
    assert metrics["view_count"] == 4
    assert metrics["artifact_kind"] == "safetensors"
    assert len(metrics["artifact_sha256"]) == 64
    assert log_text == "fake conditioning complete"


def test_run_engine_accepts_sparse_structure_safetensors(tmp_path):
    config = make_fake_config(
        tmp_path,
        """#!/usr/bin/env python3
import json
import os

header = {
    "__metadata__": {"schema": "cfp.trellis2-mlx-sparse-structure.v1"},
    "cond_512": {"dtype": "F32", "shape": [1, 2, 3], "data_offsets": [0, 24]},
    "neg_cond_512": {"dtype": "F32", "shape": [1, 2, 3], "data_offsets": [24, 48]},
    "coords_32": {"dtype": "I32", "shape": [2, 4], "data_offsets": [48, 80]},
}
encoded = json.dumps(header).encode("utf-8")
with open(os.environ["OUT_SPARSE"], "wb") as artifact:
    artifact.write(len(encoded).to_bytes(8, "little"))
    artifact.write(encoded)
    artifact.write(bytes(80))
with open(os.environ["METRICS_JSON"], "w", encoding="utf-8") as report:
    json.dump({"status": "SPARSE_STRUCTURE_READY", "voxel_count": 2}, report)
print("fake sparse complete")
""",
    )
    output_path = tmp_path / "sparse.safetensors"
    metrics_path = tmp_path / "sparse.json"
    environment = build_sparse_environment(
        os.environ,
        config=config,
        conditioning_path=tmp_path / "conditioning.safetensors",
        output_path=output_path,
        metrics_path=metrics_path,
        seed=93,
        steps=12,
    )

    metrics, log_text = run_engine(
        config=config,
        environment=environment,
        output_path=output_path,
        metrics_path=metrics_path,
        on_log=lambda _: None,
        check_cancelled=lambda: None,
        artifact_kind="sparse_safetensors",
    )
    header, artifact_sha256 = inspect_safetensors(output_path)

    assert metrics["status"] == "SPARSE_STRUCTURE_READY"
    assert metrics["voxel_count"] == 2
    assert metrics["artifact_kind"] == "sparse_safetensors"
    assert header["__metadata__"]["schema"] == "cfp.trellis2-mlx-sparse-structure.v1"
    assert "coords_32" in header
    assert artifact_sha256 == metrics["artifact_sha256"]
    assert log_text == "fake sparse complete"


def test_run_engine_terminates_subprocess_when_cancelled(tmp_path):
    config = make_fake_config(
        tmp_path,
        """#!/bin/sh
sleep 10
""",
    )
    output_path = tmp_path / "output.glb"
    metrics_path = tmp_path / "metrics.json"
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise InterruptedError("cancelled")

    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancelled"):
        run_engine(
            config=config,
            environment=os.environ.copy(),
            output_path=output_path,
            metrics_path=metrics_path,
            on_log=lambda _: None,
            check_cancelled=cancel,
        )

    assert time.monotonic() - started < 3


def test_run_engine_terminates_when_startup_produces_no_output(tmp_path):
    base = make_fake_config(tmp_path, "#!/bin/sh\nsleep 10\n")
    config = Trellis2MLXConfig(
        base.engine_binary,
        base.weights_directory,
        base.memory_fraction,
        startup_timeout_seconds=0.05,
        stall_timeout_seconds=1.0,
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="no startup output"):
        run_engine(
            config=config,
            environment=os.environ.copy(),
            output_path=tmp_path / "output.glb",
            metrics_path=tmp_path / "metrics.json",
            on_log=lambda _: None,
            check_cancelled=lambda: None,
        )

    assert time.monotonic() - started < 3


def test_run_engine_terminates_when_phase_stalls(tmp_path):
    base = make_fake_config(
        tmp_path,
        "#!/bin/sh\nprintf '[engine] phase=trellis_run\\n'\nsleep 10\n",
    )
    config = Trellis2MLXConfig(
        base.engine_binary,
        base.weights_directory,
        base.memory_fraction,
        startup_timeout_seconds=1.0,
        stall_timeout_seconds=0.05,
    )

    with pytest.raises(RuntimeError, match="no phase progress"):
        run_engine(
            config=config,
            environment=os.environ.copy(),
            output_path=tmp_path / "output.glb",
            metrics_path=tmp_path / "metrics.json",
            on_log=lambda _: None,
            check_cancelled=lambda: None,
        )


def test_engine_heartbeat_does_not_hide_a_stalled_phase(tmp_path):
    base = make_fake_config(
        tmp_path,
        """#!/bin/sh
printf '[engine] phase=trellis_run\n'
while true; do
  printf '[engine] heartbeat phase=trellis_run elapsed=15s\n'
  sleep 0.02
done
""",
    )
    config = Trellis2MLXConfig(
        base.engine_binary,
        base.weights_directory,
        base.memory_fraction,
        startup_timeout_seconds=1.0,
        stall_timeout_seconds=0.08,
    )

    with pytest.raises(RuntimeError, match="no phase progress"):
        run_engine(
            config=config,
            environment=os.environ.copy(),
            output_path=tmp_path / "output.glb",
            metrics_path=tmp_path / "metrics.json",
            on_log=lambda _: None,
            check_cancelled=lambda: None,
        )
