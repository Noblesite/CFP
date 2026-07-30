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

from .mesh_report import analyze_glb, format_mesh_report, mesh_report_json
from .runner import Trellis2MLXConfig, build_environment, default_paths, run_engine


log = logging.getLogger("trellis2_mlx")
NODE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_ENGINE, DEFAULT_WEIGHTS = default_paths(NODE_DIRECTORY)


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
        matting: str,
        mask=None,
    ):
        return _run_generation(
            mlx_config=mlx_config,
            views=[(0, image, mask)],
            seed=seed,
            steps=steps,
            matting=matting,
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


__all__ = [
    "Trellis2MLXModel",
    "Trellis2MLXImageTo3D",
    "Trellis2MLXMultiViewTo3D",
    "Trellis2MLXMeshReport",
]
