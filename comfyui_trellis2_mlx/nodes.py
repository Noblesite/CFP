from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import IO, Types

from .mesh_cleanup import (
    cleanup_report_json,
    format_cleanup_report,
    remove_small_components_glb,
)
from .input_mask_quality import (
    format_input_mask_report,
    input_mask_report_json,
    inspect_input_mask,
)
from .model_sheet_consistency import (
    format_model_sheet_report,
    inspect_model_sheet_consistency,
    model_sheet_report_json,
)
from .model_sheet_alignment_review import (
    alignment_review_report_json,
    build_alignment_review,
    format_alignment_review_report,
)
from .model_sheet_alignment_candidate import (
    align_model_sheet_candidate,
    alignment_candidate_report_json,
    format_alignment_candidate_report,
)
from .background_geometry_guard import (
    background_geometry_guard_report_json,
    format_background_geometry_guard_report,
    inspect_background_geometry,
)
from .mesh_report import analyze_glb, format_mesh_report, mesh_report_json
from .runner import (
    Trellis2MLXConditioningArtifact,
    Trellis2MLXConfig,
    build_environment,
    default_paths,
    run_engine,
)
from .topology_diagnostics import (
    diagnose_ovoxel_topology,
    format_topology_diagnostics,
    topology_diagnostics_json,
)
from .topology_sanitizer import (
    format_sanitizer_report,
    sanitize_topology_glb,
    sanitizer_report_json,
)
from .post_voxel_polish import (
    format_post_voxel_polish_report,
    polish_post_voxel_glb,
    post_voxel_polish_report_json,
)
from .print_scale import (
    format_print_scale_report,
    print_scale_report_json,
    scale_glb_for_print,
)
from .voxel_remesh import (
    format_voxel_remesh_report,
    voxel_remesh_candidate_glb,
    voxel_remesh_report_json,
)
from .voxel_comparison import (
    compare_voxel_candidates,
    format_voxel_comparison,
    voxel_comparison_json,
)


log = logging.getLogger("trellis2_mlx")
NODE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ENGINE, DEFAULT_WEIGHTS = default_paths(NODE_DIRECTORY)


class Trellis2MLXInputMaskQualityGate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXInputMaskQualityGate",
            display_name="TRELLIS.2 MLX Input Mask Quality Gate",
            category="The Foundry/TRELLIS.2 MLX/Input",
            description=(
                "Inspect foreground coverage, border contact, polarity, and disconnected mask "
                "noise before running TRELLIS.2. RGBA alpha is preferred; otherwise the optional "
                "input uses standard ComfyUI mask semantics where white is transparent."
            ),
            inputs=[
                IO.Image.Input("image"),
                IO.Float.Input("alpha_threshold", default=0.5, min=0.01, max=0.99, step=0.01),
                IO.Float.Input("min_foreground_fraction", default=0.05, min=0.0, max=0.99, step=0.01),
                IO.Float.Input("max_foreground_fraction", default=0.85, min=0.01, max=1.0, step=0.01),
                IO.Float.Input("max_border_contact_fraction", default=0.10, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("max_minor_component_fraction", default=0.02, min=0.0, max=1.0, step=0.01),
                IO.Combo.Input(
                    "acknowledge_suspicious_mask",
                    options=["no", "yes"],
                    default="no",
                    tooltip="Human override. The diagnostic remains recorded and no pixels are changed.",
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Used only when IMAGE has no alpha channel. White means transparent.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
                IO.String.Output(display_name="mask_report"),
                IO.String.Output(display_name="mask_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        alpha_threshold: float,
        min_foreground_fraction: float,
        max_foreground_fraction: float,
        max_border_contact_fraction: float,
        max_minor_component_fraction: float,
        acknowledge_suspicious_mask: str,
        mask=None,
    ):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Input Mask Quality Gate accepts one image, got shape {tuple(image.shape)}")
        if image.shape[-1] == 4:
            foreground_alpha = image[0, ..., 3]
        elif mask is not None:
            foreground_alpha = 1.0 - mask
            while foreground_alpha.ndim > 2:
                foreground_alpha = foreground_alpha[0]
        else:
            raise ValueError("Input Mask Quality Gate requires an RGBA image or a connected MASK")

        report = inspect_input_mask(
            foreground_alpha.detach().cpu().float().numpy(),
            alpha_threshold=alpha_threshold,
            min_foreground_fraction=min_foreground_fraction,
            max_foreground_fraction=max_foreground_fraction,
            max_border_contact_fraction=max_border_contact_fraction,
            max_minor_component_fraction=max_minor_component_fraction,
            acknowledge_suspicious_mask=acknowledge_suspicious_mask == "yes",
        )
        report_text = format_input_mask_report(report)
        if not report["decision"]["proceed_allowed"]:
            raise ValueError(report_text)

        comfy_mask = (1.0 - foreground_alpha).clip(0.0, 1.0).unsqueeze(0)
        return IO.NodeOutput(
            image,
            comfy_mask,
            report_text,
            input_mask_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXModelSheetConsistencyGate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXModelSheetConsistencyGate",
            display_name="TRELLIS.2 MLX Model-Sheet Consistency Gate",
            category="The Foundry/TRELLIS.2 MLX/Input",
            description=(
                "Compare four cardinal foreground masks for canvas size, subject height, "
                "centerline, baseline, vertical alignment, and opposing-view width before "
                "multi-view inference. Images and masks pass through unchanged."
            ),
            inputs=[
                IO.Image.Input("image_000"),
                IO.Image.Input("image_090"),
                IO.Image.Input("image_180"),
                IO.Image.Input("image_270"),
                IO.Mask.Input("mask_000"),
                IO.Mask.Input("mask_090"),
                IO.Mask.Input("mask_180"),
                IO.Mask.Input("mask_270"),
                IO.Float.Input("max_height_spread", default=0.08, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("max_center_x_spread", default=0.05, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("max_vertical_center_spread", default=0.05, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("max_baseline_spread", default=0.03, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("max_opposing_width_delta", default=0.15, min=0.0, max=1.0, step=0.01),
                IO.Combo.Input(
                    "acknowledge_inconsistent_sheet",
                    options=["no", "yes"],
                    default="no",
                    tooltip="Human override. All images and masks remain unchanged.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image_000"),
                IO.Image.Output(display_name="image_090"),
                IO.Image.Output(display_name="image_180"),
                IO.Image.Output(display_name="image_270"),
                IO.Mask.Output(display_name="mask_000"),
                IO.Mask.Output(display_name="mask_090"),
                IO.Mask.Output(display_name="mask_180"),
                IO.Mask.Output(display_name="mask_270"),
                IO.String.Output(display_name="sheet_report"),
                IO.String.Output(display_name="sheet_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_000,
        image_090,
        image_180,
        image_270,
        mask_000,
        mask_090,
        mask_180,
        mask_270,
        max_height_spread: float,
        max_center_x_spread: float,
        max_vertical_center_spread: float,
        max_baseline_spread: float,
        max_opposing_width_delta: float,
        acknowledge_inconsistent_sheet: str,
    ):
        images = {0: image_000, 90: image_090, 180: image_180, 270: image_270}
        masks = {0: mask_000, 90: mask_090, 180: mask_180, 270: mask_270}
        foreground_masks = {}
        for angle in (0, 90, 180, 270):
            image = images[angle]
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError(f"Camera {angle:03d} must contain one image, got {tuple(image.shape)}")
            foreground = 1.0 - masks[angle]
            while foreground.ndim > 2:
                foreground = foreground[0]
            if tuple(foreground.shape) != tuple(image.shape[1:3]):
                raise ValueError(
                    f"Camera {angle:03d} image and mask dimensions differ: "
                    f"{tuple(image.shape[1:3])} vs {tuple(foreground.shape)}"
                )
            foreground_masks[angle] = foreground.detach().cpu().float().numpy()

        report = inspect_model_sheet_consistency(
            foreground_masks,
            max_height_spread=max_height_spread,
            max_center_x_spread=max_center_x_spread,
            max_vertical_center_spread=max_vertical_center_spread,
            max_baseline_spread=max_baseline_spread,
            max_opposing_width_delta=max_opposing_width_delta,
            acknowledge_inconsistent_sheet=acknowledge_inconsistent_sheet == "yes",
        )
        report_text = format_model_sheet_report(report)
        if not report["decision"]["proceed_allowed"]:
            raise ValueError(report_text)

        return IO.NodeOutput(
            image_000,
            image_090,
            image_180,
            image_270,
            mask_000,
            mask_090,
            mask_180,
            mask_270,
            report_text,
            model_sheet_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXModelSheetAlignmentReview(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXModelSheetAlignmentReview",
            display_name="TRELLIS.2 MLX Model-Sheet Alignment Review",
            category="The Foundry/TRELLIS.2 MLX/Input",
            description=(
                "Create a read-only 2x2 annotated contact sheet showing foreground bounds, "
                "canvas and subject centerlines, baselines, and measured camera-set drift."
            ),
            is_output_node=True,
            inputs=[
                IO.Image.Input("image_000"),
                IO.Image.Input("image_090"),
                IO.Image.Input("image_180"),
                IO.Image.Input("image_270"),
                IO.Mask.Input("mask_000"),
                IO.Mask.Input("mask_090"),
                IO.Mask.Input("mask_180"),
                IO.Mask.Input("mask_270"),
                IO.Int.Input("cell_size", default=384, min=192, max=1024, step=64),
            ],
            outputs=[
                IO.Image.Output(display_name="alignment_contact_sheet"),
                IO.String.Output(display_name="alignment_report"),
                IO.String.Output(display_name="alignment_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_000,
        image_090,
        image_180,
        image_270,
        mask_000,
        mask_090,
        mask_180,
        mask_270,
        cell_size: int,
    ):
        image_tensors = {0: image_000, 90: image_090, 180: image_180, 270: image_270}
        mask_tensors = {0: mask_000, 90: mask_090, 180: mask_180, 270: mask_270}
        images = {}
        foreground_masks = {}
        for angle in (0, 90, 180, 270):
            image = image_tensors[angle]
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError(f"Camera {angle:03d} must contain one image, got {tuple(image.shape)}")
            images[angle] = image[0].detach().cpu().float().numpy()
            foreground = 1.0 - mask_tensors[angle]
            while foreground.ndim > 2:
                foreground = foreground[0]
            foreground_masks[angle] = foreground.detach().cpu().float().numpy()

        contact_sheet, report = build_alignment_review(
            images,
            foreground_masks,
            cell_size=cell_size,
        )
        report_text = format_alignment_review_report(report)
        output_image = image_000.new_tensor(contact_sheet / 255.0).unsqueeze(0)
        return IO.NodeOutput(
            output_image,
            report_text,
            alignment_review_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXModelSheetAlignmentCandidate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXModelSheetAlignmentCandidate",
            display_name="TRELLIS.2 MLX Model-Sheet Alignment Candidate",
            category="The Foundry/TRELLIS.2 MLX/Input",
            description=(
                "Create non-destructive aligned copies of four cardinal views. Median mode "
                "uses the shared median foreground height; explicit mode uses a normalized "
                "canvas fraction. Millimeter height is report-only manufacturing metadata."
            ),
            inputs=[
                IO.Image.Input("image_000"),
                IO.Image.Input("image_090"),
                IO.Image.Input("image_180"),
                IO.Image.Input("image_270"),
                IO.Mask.Input("mask_000"),
                IO.Mask.Input("mask_090"),
                IO.Mask.Input("mask_180"),
                IO.Mask.Input("mask_270"),
                IO.Combo.Input(
                    "alignment_mode",
                    options=["median", "explicit_fraction"],
                    default="median",
                ),
                IO.Float.Input(
                    "target_subject_height_fraction",
                    default=0.82,
                    min=0.10,
                    max=0.98,
                    step=0.01,
                    tooltip="Used only when alignment_mode is explicit_fraction.",
                ),
                IO.Float.Input(
                    "design_target_height_mm",
                    default=250.0,
                    min=1.0,
                    max=5000.0,
                    step=1.0,
                    tooltip="Report-only metadata. Physical scaling occurs after reconstruction.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="candidate_image_000"),
                IO.Image.Output(display_name="candidate_image_090"),
                IO.Image.Output(display_name="candidate_image_180"),
                IO.Image.Output(display_name="candidate_image_270"),
                IO.Mask.Output(display_name="candidate_mask_000"),
                IO.Mask.Output(display_name="candidate_mask_090"),
                IO.Mask.Output(display_name="candidate_mask_180"),
                IO.Mask.Output(display_name="candidate_mask_270"),
                IO.String.Output(display_name="candidate_report"),
                IO.String.Output(display_name="candidate_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_000,
        image_090,
        image_180,
        image_270,
        mask_000,
        mask_090,
        mask_180,
        mask_270,
        alignment_mode: str,
        target_subject_height_fraction: float,
        design_target_height_mm: float,
    ):
        image_tensors = {0: image_000, 90: image_090, 180: image_180, 270: image_270}
        mask_tensors = {0: mask_000, 90: mask_090, 180: mask_180, 270: mask_270}
        images = {}
        foreground_masks = {}
        for angle in (0, 90, 180, 270):
            image = image_tensors[angle]
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError(f"Camera {angle:03d} must contain one image, got {tuple(image.shape)}")
            images[angle] = image[0].detach().cpu().float().numpy()
            foreground = 1.0 - mask_tensors[angle]
            while foreground.ndim > 2:
                foreground = foreground[0]
            foreground_masks[angle] = foreground.detach().cpu().float().numpy()

        candidate_images, candidate_foregrounds, report = align_model_sheet_candidate(
            images,
            foreground_masks,
            alignment_mode=alignment_mode,
            target_subject_height_fraction=target_subject_height_fraction,
            design_target_height_mm=design_target_height_mm,
        )
        output_images = [
            image_000.new_tensor(candidate_images[angle] / 255.0).unsqueeze(0)
            for angle in (0, 90, 180, 270)
        ]
        output_masks = [
            mask_000.new_tensor(1.0 - candidate_foregrounds[angle]).unsqueeze(0)
            for angle in (0, 90, 180, 270)
        ]
        report_text = format_alignment_candidate_report(report)
        return IO.NodeOutput(
            *output_images,
            *output_masks,
            report_text,
            alignment_candidate_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


def _save_input_image(image, mask, path: Path) -> None:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"TRELLIS.2 MLX currently accepts one image, got shape {tuple(image.shape)}")

    pixels = (image[0].detach().cpu().float().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    if pixels.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB or RGBA ComfyUI image, got shape {tuple(pixels.shape)}")

    if mask is None:
        Image.fromarray(pixels).save(path)
        return

    mask_array = mask.detach().cpu().float().numpy()
    while mask_array.ndim > 2:
        mask_array = mask_array[0]
    alpha = ((1.0 - mask_array.clip(0, 1)) * 255).round().astype(np.uint8)
    if alpha.shape != pixels.shape[:2]:
        alpha = np.asarray(
            Image.fromarray(alpha, mode="L").resize(
                (pixels.shape[1], pixels.shape[0]),
                Image.Resampling.LANCZOS,
            )
        )
    rgba = np.dstack((pixels[..., :3], alpha))
    Image.fromarray(rgba, mode="RGBA").save(path)


def _run_generation(
    *,
    mlx_config: Trellis2MLXConfig,
    views: list[tuple[int, object, object | None]],
    seed: int,
    steps: int,
    matting: str,
    output_mode: str,
):
    temp_directory = Path(folder_paths.get_temp_directory())
    temp_directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output_path = temp_directory / f"trellis2_mlx_{run_id}.glb"
    metrics_path = temp_directory / f"trellis2_mlx_{run_id}_metrics.json"
    manifest_path = temp_directory / f"trellis2_mlx_{run_id}_views.json"
    input_paths: list[Path] = []

    try:
        for angle, image, mask in views:
            input_path = temp_directory / f"trellis2_mlx_{run_id}_{angle:03d}.png"
            _save_input_image(image, mask, input_path)
            input_paths.append(input_path)

        additional_manifest = None
        if len(input_paths) > 1:
            manifest_path.write_text(
                json.dumps([str(path) for path in input_paths[1:]]),
                encoding="utf-8",
            )
            additional_manifest = manifest_path

        environment = build_environment(
            dict(os.environ),
            config=mlx_config,
            image_path=input_paths[0],
            output_path=output_path,
            metrics_path=metrics_path,
            seed=seed,
            steps=steps,
            use_matting=matting == "on",
            output_mode=output_mode,
            additional_views_manifest=additional_manifest,
        )

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(force=True)
        progress = comfy.utils.ProgressBar(100)

        def on_log(line: str) -> None:
            log.info("[engine] %s", line)
            if "prepare OK" in line:
                progress.update_absolute(25)
            elif "[engine] tier:" in line:
                progress.update_absolute(30)
            elif "[engine] run OK" in line:
                progress.update_absolute(95)
            elif "[engine] wrote GLB" in line:
                progress.update_absolute(100)

        metrics, engine_log = run_engine(
            config=mlx_config,
            environment=environment,
            output_path=output_path,
            metrics_path=metrics_path,
            on_log=on_log,
            check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted,
        )
        metrics["input_view_count"] = len(views)
        metrics["input_view_angles"] = [angle for angle, _, _ in views]
        metrics["output_mode"] = output_mode
        metrics["texture_enabled"] = output_mode == "textured"
    finally:
        for input_path in input_paths:
            input_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    return IO.NodeOutput(
        Types.File3D(str(output_path), file_format="glb"),
        str(output_path),
        json.dumps(metrics, indent=2, sort_keys=True),
        engine_log,
        metrics["artifact_sha256"],
    )


def _run_conditioning(
    *,
    mlx_config: Trellis2MLXConfig,
    views: list[tuple[int, object, object | None]],
    matting: str,
):
    temp_directory = Path(folder_paths.get_temp_directory())
    temp_directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output_path = temp_directory / f"trellis2_mlx_conditioning_{run_id}.safetensors"
    metrics_path = temp_directory / f"trellis2_mlx_conditioning_{run_id}_report.json"
    manifest_path = temp_directory / f"trellis2_mlx_conditioning_{run_id}_views.json"
    input_paths: list[Path] = []

    try:
        for angle, image, mask in views:
            input_path = temp_directory / f"trellis2_mlx_conditioning_{run_id}_{angle:03d}.png"
            _save_input_image(image, mask, input_path)
            input_paths.append(input_path)

        additional_manifest = None
        if len(input_paths) > 1:
            manifest_path.write_text(
                json.dumps([str(path) for path in input_paths[1:]]),
                encoding="utf-8",
            )
            additional_manifest = manifest_path

        view_angles = tuple(angle for angle, _, _ in views)
        environment = build_environment(
            dict(os.environ),
            config=mlx_config,
            image_path=input_paths[0],
            output_path=output_path,
            metrics_path=metrics_path,
            seed=0,
            steps=12,
            use_matting=matting == "on",
            output_mode="geometry_only",
            additional_views_manifest=additional_manifest,
            engine_stage="conditioning",
            view_angles=view_angles,
        )

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(force=True)
        progress = comfy.utils.ProgressBar(100)

        def on_log(line: str) -> None:
            log.info("[conditioning-engine] %s", line)
            if "prepare OK" in line:
                progress.update_absolute(40)
            elif "conditioning OK" in line:
                progress.update_absolute(95)
            elif "wrote conditioning" in line:
                progress.update_absolute(100)

        metrics, engine_log = run_engine(
            config=mlx_config,
            environment=environment,
            output_path=output_path,
            metrics_path=metrics_path,
            on_log=on_log,
            check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted,
            artifact_kind="safetensors",
        )
        metrics["input_view_angles"] = list(view_angles)
        metrics["automatic_promotion"] = False
    finally:
        for input_path in input_paths:
            input_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    report_json = json.dumps(metrics, indent=2, sort_keys=True)
    report_text = (
        "TRELLIS.2 MLX Conditioning\n"
        f"Status: {metrics.get('status', 'CONDITIONING_READY')}\n"
        f"Views: {', '.join(f'{angle:03d}°' for angle in view_angles)}\n"
        f"cond_512: {metrics.get('cond_512_shape', 'unknown')}\n"
        f"Artifact: {output_path}\n"
        "Automatic promotion: No"
    )
    artifact = Trellis2MLXConditioningArtifact(
        path=output_path,
        view_angles=view_angles,
        tier="res512",
        artifact_sha256=metrics["artifact_sha256"],
    )
    return IO.NodeOutput(
        artifact,
        str(output_path),
        report_text,
        report_json,
        engine_log,
        metrics["artifact_sha256"],
        metrics.get("status", "CONDITIONING_READY"),
        ui={"text": [report_text]},
    )


class Trellis2MLXModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXModel",
            display_name="TRELLIS.2 MLX Model",
            category="The Foundry/TRELLIS.2 MLX",
            description=(
                "Configure the native Swift/MLX TRELLIS.2 engine. This node supports macOS on "
                "Apple Silicon and does not load weights into ComfyUI's Python process."
            ),
            inputs=[
                IO.String.Input(
                    "engine_binary",
                    default="auto",
                    tooltip=f"Use 'auto' for {DEFAULT_ENGINE}, or enter an explicit executable path.",
                ),
                IO.String.Input(
                    "weights_directory",
                    default="auto",
                    tooltip=f"Use 'auto' for {DEFAULT_WEIGHTS}, or enter an explicit weights path.",
                ),
                IO.Float.Input(
                    "memory_fraction",
                    default=0.95,
                    min=0.70,
                    max=0.98,
                    step=0.01,
                    tooltip="Unified-memory budget. 0.95 is the proven 36 GB M3 Max setting.",
                ),
            ],
            outputs=[
                IO.Custom("TRELLIS2_MLX_CONFIG").Output(display_name="mlx_config"),
            ],
        )

    @classmethod
    def execute(cls, engine_binary: str, weights_directory: str, memory_fraction: float):
        engine_path = DEFAULT_ENGINE if engine_binary.strip().lower() == "auto" else Path(engine_binary).expanduser()
        weights_path = (
            DEFAULT_WEIGHTS
            if weights_directory.strip().lower() == "auto"
            else Path(weights_directory).expanduser()
        )
        config = Trellis2MLXConfig(
            engine_binary=engine_path.resolve(),
            weights_directory=weights_path.resolve(),
            memory_fraction=memory_fraction,
        )
        config.validate()
        return IO.NodeOutput(config)


class Trellis2MLXImageConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXImageConditioning",
            display_name="TRELLIS.2 MLX Image Conditioning",
            category="The Foundry/TRELLIS.2 MLX/Stages",
            description=(
                "Encode one or more ordered object views with DINOv3 and stop before sparse "
                "structure generation. Produces a reusable MLX safetensors artifact."
            ),
            is_output_node=True,
            inputs=[
                IO.Custom("TRELLIS2_MLX_CONFIG").Input("mlx_config"),
                IO.Image.Input("image_000"),
                IO.Image.Input("image_090", optional=True),
                IO.Image.Input("image_180", optional=True),
                IO.Image.Input("image_270", optional=True),
                IO.Combo.Input(
                    "matting",
                    options=["off", "on"],
                    default="off",
                    tooltip="Use off for approved RGBA views or connected masks.",
                ),
                IO.Mask.Input("mask_000", optional=True),
                IO.Mask.Input("mask_090", optional=True),
                IO.Mask.Input("mask_180", optional=True),
                IO.Mask.Input("mask_270", optional=True),
            ],
            outputs=[
                IO.Custom("TRELLIS2_MLX_CONDITIONING").Output(display_name="conditioning"),
                IO.String.Output(display_name="artifact_path"),
                IO.String.Output(display_name="conditioning_report"),
                IO.String.Output(display_name="conditioning_report_json"),
                IO.String.Output(display_name="engine_log"),
                IO.String.Output(display_name="artifact_sha256"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mlx_config: Trellis2MLXConfig,
        image_000,
        matting: str,
        image_090=None,
        image_180=None,
        image_270=None,
        mask_000=None,
        mask_090=None,
        mask_180=None,
        mask_270=None,
    ):
        candidates = [
            (0, image_000, mask_000),
            (90, image_090, mask_090),
            (180, image_180, mask_180),
            (270, image_270, mask_270),
        ]
        views = [(angle, image, mask) for angle, image, mask in candidates if image is not None]
        return _run_conditioning(mlx_config=mlx_config, views=views, matting=matting)


class Trellis2MLXImageTo3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXImageTo3D",
            display_name="TRELLIS.2 MLX Image to 3D",
            category="The Foundry/TRELLIS.2 MLX",
            description=(
                "Generate a textured GLB through the native Swift/MLX engine. The proven v1 lane "
                "is single-image 512 resolution. Cached ComfyUI models are unloaded first so the "
                "engine can use unified memory."
            ),
            inputs=[
                IO.Custom("TRELLIS2_MLX_CONFIG").Input("mlx_config"),
                IO.Image.Input("image"),
                IO.Int.Input("seed", default=0, min=0, max=2**64 - 1),
                IO.Int.Input("steps", default=12, min=1, max=50),
                IO.Combo.Input(
                    "output_mode",
                    options=["textured", "geometry_only"],
                    default="textured",
                    tooltip=(
                        "geometry_only emits a texture-free GLB with no UVs or atlas. TRELLIS "
                        "mesh extraction, remeshing, simplification, and normals still run."
                    ),
                ),
                IO.Combo.Input(
                    "matting",
                    options=["off", "on"],
                    default="off",
                    tooltip="Use 'off' with a supplied alpha mask. 'on' loads native MLX BiRefNet.",
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="ComfyUI mask semantics: white is transparent. Used as PNG alpha.",
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="model_3d"),
                IO.String.Output(display_name="artifact_path"),
                IO.String.Output(display_name="metrics_json"),
                IO.String.Output(display_name="engine_log"),
                IO.String.Output(display_name="artifact_sha256"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mlx_config: Trellis2MLXConfig,
        image,
        seed: int,
        steps: int,
        output_mode: str,
        matting: str,
        mask=None,
    ):
        return _run_generation(
            mlx_config=mlx_config,
            views=[(0, image, mask)],
            seed=seed,
            steps=steps,
            matting=matting,
            output_mode=output_mode,
        )


class Trellis2MLXMultiViewTo3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXMultiViewTo3D",
            display_name="TRELLIS.2 MLX Multi-View to 3D",
            category="The Foundry/TRELLIS.2 MLX",
            description=(
                "Generate one textured GLB from an ordered model-sheet view set. 000° is required; "
                "090°, 180°, and 270° are optional and are independently encoded by DINOv3."
            ),
            inputs=[
                IO.Custom("TRELLIS2_MLX_CONFIG").Input("mlx_config"),
                IO.Image.Input("image_000"),
                IO.Image.Input("image_090", optional=True),
                IO.Image.Input("image_180", optional=True),
                IO.Image.Input("image_270", optional=True),
                IO.Int.Input("seed", default=0, min=0, max=2**64 - 1),
                IO.Int.Input("steps", default=12, min=1, max=50),
                IO.Combo.Input(
                    "output_mode",
                    options=["textured", "geometry_only"],
                    default="textured",
                    tooltip=(
                        "geometry_only skips texture inference and emits a texture-free GLB with "
                        "no UVs or atlas for the conditioned model sheet."
                    ),
                ),
                IO.Combo.Input(
                    "matting",
                    options=["off", "on"],
                    default="off",
                    tooltip="Use 'off' for pre-matted RGBA views. 'on' mattes every raw view.",
                ),
                IO.Mask.Input("mask_000", optional=True),
                IO.Mask.Input("mask_090", optional=True),
                IO.Mask.Input("mask_180", optional=True),
                IO.Mask.Input("mask_270", optional=True),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="model_3d"),
                IO.String.Output(display_name="artifact_path"),
                IO.String.Output(display_name="metrics_json"),
                IO.String.Output(display_name="engine_log"),
                IO.String.Output(display_name="artifact_sha256"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mlx_config: Trellis2MLXConfig,
        image_000,
        seed: int,
        steps: int,
        output_mode: str,
        matting: str,
        image_090=None,
        image_180=None,
        image_270=None,
        mask_000=None,
        mask_090=None,
        mask_180=None,
        mask_270=None,
    ):
        candidates = [
            (0, image_000, mask_000),
            (90, image_090, mask_090),
            (180, image_180, mask_180),
            (270, image_270, mask_270),
        ]
        views = [(angle, image, mask) for angle, image, mask in candidates if image is not None]
        return _run_generation(
            mlx_config=mlx_config,
            views=views,
            seed=seed,
            steps=steps,
            matting=matting,
            output_mode=output_mode,
        )


class Trellis2MLXMeshReport(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXMeshReport",
            display_name="TRELLIS.2 MLX Mesh Report",
            category="The Foundry/TRELLIS.2 MLX",
            description=(
                "Inspect a generated GLB without modifying it. Reports topology, dimensions, "
                "artifact identity, and a human-review status."
            ),
            is_output_node=True,
            inputs=[
                IO.File3DGLB.Input("model_3d"),
            ],
            outputs=[
                IO.String.Output(display_name="report"),
                IO.String.Output(display_name="report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model_3d: Types.File3D):
        report = analyze_glb(model_3d.get_source())
        report_text = format_mesh_report(report)
        return IO.NodeOutput(
            report_text,
            mesh_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXRemoveFloaters(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXRemoveFloaters",
            display_name="TRELLIS.2 MLX Remove Floaters",
            category="The Foundry/TRELLIS.2 MLX/Mesh",
            description=(
                "Remove small disconnected bodies from a geometry-focused GLB. The largest body "
                "is always preserved. This does not repair O-Voxel non-manifold topology, "
                "overlapping shells, or open boundaries."
            ),
            inputs=[
                IO.File3DGLB.Input("model_3d"),
                IO.Int.Input(
                    "min_component_faces",
                    default=100,
                    min=0,
                    max=1_000_000,
                    step=10,
                    tooltip="Absolute minimum triangle count for every component except the largest.",
                ),
                IO.Float.Input(
                    "min_component_ratio",
                    default=0.001,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Minimum component size relative to the largest component. Both the "
                        "absolute and relative floors must be met."
                    ),
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="model_3d"),
                IO.String.Output(display_name="cleanup_report"),
                IO.String.Output(display_name="cleanup_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_3d: Types.File3D,
        min_component_faces: int,
        min_component_ratio: float,
    ):
        output_data, report = remove_small_components_glb(
            model_3d.get_source(),
            min_component_faces=min_component_faces,
            min_component_ratio=min_component_ratio,
        )
        temp_directory = Path(folder_paths.get_temp_directory())
        temp_directory.mkdir(parents=True, exist_ok=True)
        output_path = temp_directory / f"trellis2_mlx_floaters_{uuid.uuid4().hex}.glb"
        output_path.write_bytes(output_data)
        report_text = format_cleanup_report(report)
        return IO.NodeOutput(
            Types.File3D(str(output_path), file_format="glb"),
            report_text,
            cleanup_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXTopologyDiagnostics(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXTopologyDiagnostics",
            display_name="TRELLIS.2 MLX O-Voxel Topology Diagnostics",
            category="The Foundry/TRELLIS.2 MLX/Mesh",
            description=(
                "Read-only O-Voxel topology classification. Confirms duplicate and degenerate "
                "faces, coincident vertices, boundary and overloaded edges, then reports "
                "overlapping-component and shell-intersection candidates without modifying GLB."
            ),
            is_output_node=True,
            inputs=[
                IO.File3DGLB.Input("model_3d"),
                IO.Float.Input(
                    "coordinate_tolerance_ratio",
                    default=1e-6,
                    min=1e-9,
                    max=0.01,
                    step=1e-6,
                    tooltip=(
                        "Coordinate comparison tolerance relative to the mesh bounding-box "
                        "diagonal. Used only for diagnostic grouping."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(display_name="diagnostic_report"),
                IO.String.Output(display_name="diagnostic_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_3d: Types.File3D,
        coordinate_tolerance_ratio: float,
    ):
        report = diagnose_ovoxel_topology(
            model_3d.get_source(),
            coordinate_tolerance_ratio=coordinate_tolerance_ratio,
        )
        report_text = format_topology_diagnostics(report)
        return IO.NodeOutput(
            report_text,
            topology_diagnostics_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXTopologySanitizer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXTopologySanitizer",
            display_name="TRELLIS.2 MLX Topology Sanitizer",
            category="The Foundry/TRELLIS.2 MLX/Mesh",
            description=(
                "Deterministically weld nearly coincident vertices, remove degenerate and "
                "duplicate faces, and compact unused vertices. This geometry-only operation "
                "does not remesh or repair O-Voxel shell junctions."
            ),
            inputs=[
                IO.File3DGLB.Input("model_3d"),
                IO.Float.Input(
                    "weld_tolerance_ratio",
                    default=1e-8,
                    min=1e-9,
                    max=0.001,
                    step=1e-8,
                    tooltip=(
                        "Vertex weld tolerance relative to the mesh bounding-box diagonal. "
                        "The conservative default targets effectively coincident vertices."
                    ),
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="model_3d"),
                IO.String.Output(display_name="sanitizer_report"),
                IO.String.Output(display_name="sanitizer_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_3d: Types.File3D,
        weld_tolerance_ratio: float,
    ):
        output_data, report = sanitize_topology_glb(
            model_3d.get_source(),
            weld_tolerance_ratio=weld_tolerance_ratio,
        )
        temp_directory = Path(folder_paths.get_temp_directory())
        temp_directory.mkdir(parents=True, exist_ok=True)
        output_path = temp_directory / f"trellis2_mlx_sanitized_{uuid.uuid4().hex}.glb"
        output_path.write_bytes(output_data)
        report_text = format_sanitizer_report(report)
        return IO.NodeOutput(
            Types.File3D(str(output_path), file_format="glb"),
            report_text,
            sanitizer_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXBackgroundGeometryGuard(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXBackgroundGeometryGuard",
            display_name="TRELLIS.2 MLX Background Geometry Guard",
            category="The Foundry/TRELLIS.2 MLX/Mesh",
            description=(
                "Inspect pre-voxel bounds and planar-component evidence. The character profile "
                "blocks voxel filling when Z is not clearly dominant unless the user explicitly "
                "acknowledges the geometry. No components are deleted."
            ),
            inputs=[
                IO.File3DGLB.Input("model_3d"),
                IO.Combo.Input(
                    "profile",
                    options=["character_z_up", "generic"],
                    default="character_z_up",
                ),
                IO.Float.Input(
                    "min_z_dominance_ratio",
                    default=1.25,
                    min=1.0,
                    max=10.0,
                    step=0.05,
                ),
                IO.Float.Input(
                    "planar_flatness_ratio",
                    default=0.02,
                    min=0.0001,
                    max=0.25,
                    step=0.005,
                ),
                IO.Float.Input(
                    "large_planar_span_ratio",
                    default=0.4,
                    min=0.05,
                    max=1.0,
                    step=0.05,
                ),
                IO.Combo.Input(
                    "acknowledge_suspicious_geometry",
                    options=["no", "yes"],
                    default="no",
                    tooltip=(
                        "Leave at no for the safety gate. Set yes only after visually confirming "
                        "that non-character bounds are intentional."
                    ),
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="guarded_3d"),
                IO.String.Output(display_name="guard_report"),
                IO.String.Output(display_name="guard_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_3d: Types.File3D,
        profile: str,
        min_z_dominance_ratio: float,
        planar_flatness_ratio: float,
        large_planar_span_ratio: float,
        acknowledge_suspicious_geometry: str,
    ):
        output_data, report = inspect_background_geometry(
            model_3d.get_source(),
            profile=profile,
            min_z_dominance_ratio=min_z_dominance_ratio,
            planar_flatness_ratio=planar_flatness_ratio,
            large_planar_span_ratio=large_planar_span_ratio,
            acknowledge_suspicious_geometry=(
                acknowledge_suspicious_geometry == "yes"
            ),
        )
        report_text = format_background_geometry_guard_report(report)
        if not report["decision"]["proceed_allowed"]:
            raise ValueError(
                report_text
                + "\n\nVoxel filling stopped. Fix upstream matting or explicitly acknowledge "
                "the geometry after visual review."
            )

        temp_directory = Path(folder_paths.get_temp_directory())
        temp_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            temp_directory / f"trellis2_mlx_guarded_{uuid.uuid4().hex}.glb"
        )
        output_path.write_bytes(output_data)
        return IO.NodeOutput(
            Types.File3D(str(output_path), file_format="glb"),
            report_text,
            background_geometry_guard_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXVoxelRemeshCandidate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXVoxelRemeshCandidate",
            display_name="TRELLIS.2 MLX Watertight Voxel Remesh Candidate",
            category="The Foundry/TRELLIS.2 MLX/Mesh",
            description=(
                "Create a separate filled-voxel marching-cubes candidate. Reports topology, "
                "dimensional change, and deterministic nearest-vertex deviation. The source GLB "
                "is preserved and visual review is required."
            ),
            inputs=[
                IO.File3DGLB.Input("model_3d"),
                IO.Int.Input(
                    "target_resolution",
                    default=192,
                    min=32,
                    max=512,
                    step=16,
                    tooltip=(
                        "Voxel cells across the longest mesh dimension. Higher values preserve "
                        "more detail but use more memory and produce larger meshes."
                    ),
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="candidate_3d"),
                IO.String.Output(display_name="candidate_report"),
                IO.String.Output(display_name="candidate_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_3d: Types.File3D,
        target_resolution: int,
    ):
        output_data, report = voxel_remesh_candidate_glb(
            model_3d.get_source(),
            target_resolution=target_resolution,
        )
        temp_directory = Path(folder_paths.get_temp_directory())
        temp_directory.mkdir(parents=True, exist_ok=True)
        output_path = temp_directory / f"trellis2_mlx_voxel_{uuid.uuid4().hex}.glb"
        output_path.write_bytes(output_data)
        report_text = format_voxel_remesh_report(report)
        return IO.NodeOutput(
            Types.File3D(str(output_path), file_format="glb"),
            report_text,
            voxel_remesh_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXPostVoxelTopologyPolish(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXPostVoxelTopologyPolish",
            display_name="TRELLIS.2 MLX Post-Voxel Topology Polish",
            category="The Foundry/TRELLIS.2 MLX/Mesh",
            description=(
                "Conservatively remove exact coincident, opposite-winding internal face sheets "
                "left by marching cubes. The proposed deletion is rejected unless overloaded "
                "edges decrease without increasing boundaries or connected components."
            ),
            inputs=[IO.File3DGLB.Input("candidate_3d")],
            outputs=[
                IO.File3DGLB.Output(display_name="polished_3d"),
                IO.String.Output(display_name="polish_report"),
                IO.String.Output(display_name="polish_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, candidate_3d: Types.File3D):
        output_data, report = polish_post_voxel_glb(candidate_3d.get_source())
        temp_directory = Path(folder_paths.get_temp_directory())
        temp_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            temp_directory / f"trellis2_mlx_voxel_polished_{uuid.uuid4().hex}.glb"
        )
        output_path.write_bytes(output_data)
        report_text = format_post_voxel_polish_report(report)
        return IO.NodeOutput(
            Types.File3D(str(output_path), file_format="glb"),
            report_text,
            post_voxel_polish_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXPrintScaleFeatureGate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXPrintScaleFeatureGate",
            display_name="TRELLIS.2 MLX Print Scale & Feature Gate",
            category="The Foundry/TRELLIS.2 MLX/Print",
            description=(
                "Uniformly scale a GLB to a physical target height in millimeters and report "
                "the source voxel pitch against the selected nozzle and layer height. This is "
                "an advisory sampling gate, not local wall-thickness analysis."
            ),
            inputs=[
                IO.File3DGLB.Input("model_3d"),
                IO.Float.Input(
                    "target_height_mm",
                    default=250.0,
                    min=1.0,
                    max=5000.0,
                    step=1.0,
                ),
                IO.Combo.Input(
                    "height_axis",
                    options=["auto", "x", "y", "z"],
                    default="z",
                    tooltip=(
                        "CFP character workflows are Z-up. Auto uses the longest bounding-box "
                        "dimension and is better suited to clean, orientation-agnostic props."
                    ),
                ),
                IO.Int.Input(
                    "source_voxel_resolution",
                    default=256,
                    min=32,
                    max=512,
                    step=16,
                ),
                IO.Float.Input(
                    "nozzle_diameter_mm",
                    default=0.4,
                    min=0.1,
                    max=2.0,
                    step=0.05,
                ),
                IO.Float.Input(
                    "layer_height_mm",
                    default=0.2,
                    min=0.02,
                    max=1.0,
                    step=0.02,
                ),
            ],
            outputs=[
                IO.File3DGLB.Output(display_name="scaled_3d"),
                IO.String.Output(display_name="scale_report"),
                IO.String.Output(display_name="scale_report_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_3d: Types.File3D,
        target_height_mm: float,
        height_axis: str,
        source_voxel_resolution: int,
        nozzle_diameter_mm: float,
        layer_height_mm: float,
    ):
        output_data, report = scale_glb_for_print(
            model_3d.get_source(),
            target_height_mm=target_height_mm,
            height_axis=height_axis,
            source_voxel_resolution=source_voxel_resolution,
            nozzle_diameter_mm=nozzle_diameter_mm,
            layer_height_mm=layer_height_mm,
        )
        temp_directory = Path(folder_paths.get_temp_directory())
        temp_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            temp_directory / f"trellis2_mlx_print_scaled_{uuid.uuid4().hex}.glb"
        )
        output_path.write_bytes(output_data)
        report_text = format_print_scale_report(report)
        return IO.NodeOutput(
            Types.File3D(str(output_path), file_format="glb"),
            report_text,
            print_scale_report_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


class Trellis2MLXVoxelCandidateComparison(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MLXVoxelCandidateComparison",
            display_name="TRELLIS.2 MLX Voxel Candidate Comparison",
            category="The Foundry/TRELLIS.2 MLX/Mesh",
            description=(
                "Compare fixed-resolution voxel candidate reports by topology, triangle count, "
                "dimensional drift, and p95 nearest-vertex deviation. Recommendations remain "
                "separate by priority and never auto-promote a candidate."
            ),
            is_output_node=True,
            inputs=[
                IO.String.Input("report_128", multiline=True),
                IO.String.Input("report_192", multiline=True),
                IO.String.Input("report_256", multiline=True),
            ],
            outputs=[
                IO.String.Output(display_name="comparison_report"),
                IO.String.Output(display_name="comparison_json"),
                IO.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        report_128: str,
        report_192: str,
        report_256: str,
    ):
        report = compare_voxel_candidates(
            [report_128, report_192, report_256]
        )
        report_text = format_voxel_comparison(report)
        return IO.NodeOutput(
            report_text,
            voxel_comparison_json(report),
            report["status"],
            ui={"text": [report_text]},
        )


__all__ = [
    "Trellis2MLXInputMaskQualityGate",
    "Trellis2MLXModelSheetConsistencyGate",
    "Trellis2MLXModelSheetAlignmentReview",
    "Trellis2MLXModelSheetAlignmentCandidate",
    "Trellis2MLXModel",
    "Trellis2MLXImageTo3D",
    "Trellis2MLXMultiViewTo3D",
    "Trellis2MLXMeshReport",
    "Trellis2MLXRemoveFloaters",
    "Trellis2MLXTopologyDiagnostics",
    "Trellis2MLXTopologySanitizer",
    "Trellis2MLXBackgroundGeometryGuard",
    "Trellis2MLXVoxelRemeshCandidate",
    "Trellis2MLXPostVoxelTopologyPolish",
    "Trellis2MLXPrintScaleFeatureGate",
    "Trellis2MLXVoxelCandidateComparison",
]
