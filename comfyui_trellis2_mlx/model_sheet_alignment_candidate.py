from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .model_sheet_consistency import CAMERA_ANGLES, inspect_model_sheet_consistency


def align_model_sheet_candidate(
    images: dict[int, np.ndarray],
    foreground_masks: dict[int, np.ndarray],
    *,
    alignment_mode: str = "median",
    target_subject_height_fraction: float = 0.82,
    design_target_height_mm: float = 250.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    if set(images) != set(CAMERA_ANGLES) or set(foreground_masks) != set(CAMERA_ANGLES):
        raise ValueError("images and foreground_masks must contain cameras 0, 90, 180, and 270")
    if alignment_mode not in {"median", "explicit_fraction"}:
        raise ValueError("alignment_mode must be median or explicit_fraction")
    if not 0.1 <= target_subject_height_fraction <= 0.98:
        raise ValueError("target_subject_height_fraction must be between 0.1 and 0.98")
    if design_target_height_mm <= 0.0:
        raise ValueError("design_target_height_mm must be positive")

    source_data = {}
    source_heights = []
    source_baselines = []
    for angle in CAMERA_ANGLES:
        image = np.asarray(images[angle])
        foreground = np.asarray(foreground_masks[angle], dtype=np.float32).clip(0.0, 1.0)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Camera {angle:03d} image must be HWC RGB or RGBA")
        if foreground.shape != image.shape[:2]:
            raise ValueError(f"Camera {angle:03d} image and foreground dimensions differ")
        coordinates = np.argwhere(foreground >= 0.5)
        if not coordinates.size:
            raise ValueError(f"Camera {angle:03d} foreground is empty")
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        source_height, source_width = foreground.shape
        bbox_height = int(y_max - y_min + 1)
        source_heights.append(bbox_height / source_height)
        source_baselines.append(y_max / max(1, source_height - 1))
        source_data[angle] = {
            "image": image,
            "foreground": foreground,
            "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
            "bbox_height_fraction": bbox_height / source_height,
        }

    canvas_height, canvas_width = np.asarray(images[0]).shape[:2]
    if alignment_mode == "median":
        target_height_fraction = float(np.median(source_heights))
    else:
        target_height_fraction = target_subject_height_fraction
    target_baseline_fraction = float(np.median(source_baselines))
    target_height_pixels = max(1, round(target_height_fraction * canvas_height))
    target_baseline_y = round(target_baseline_fraction * max(1, canvas_height - 1))
    target_center_x = (canvas_width - 1) / 2.0

    candidate_images = {}
    candidate_foregrounds = {}
    view_reports = {}
    for angle in CAMERA_ANGLES:
        source = source_data[angle]
        x_min, y_min, x_max, y_max = source["bbox"]
        crop_rgb = source["image"][y_min : y_max + 1, x_min : x_max + 1, :3]
        crop_alpha = source["foreground"][y_min : y_max + 1, x_min : x_max + 1]
        if np.issubdtype(crop_rgb.dtype, np.floating):
            crop_rgb = (crop_rgb.clip(0.0, 1.0) * 255).round().astype(np.uint8)
        else:
            crop_rgb = crop_rgb.clip(0, 255).astype(np.uint8)

        crop_height, crop_width = crop_alpha.shape
        scale = target_height_pixels / crop_height
        scaled_width = max(1, round(crop_width * scale))
        width_limited = scaled_width > round(canvas_width * 0.98)
        if width_limited:
            scale = (canvas_width * 0.98) / crop_width
            target_view_height = max(1, round(crop_height * scale))
            scaled_width = max(1, round(crop_width * scale))
        else:
            target_view_height = target_height_pixels

        resized_rgb = Image.fromarray(crop_rgb).resize(
            (scaled_width, target_view_height), Image.Resampling.LANCZOS
        )
        resized_alpha = Image.fromarray(
            (crop_alpha * 255).round().astype(np.uint8), mode="L"
        ).resize((scaled_width, target_view_height), Image.Resampling.LANCZOS)

        paste_x = round(target_center_x - (scaled_width - 1) / 2.0)
        paste_y = target_baseline_y - target_view_height + 1
        paste_x = min(max(0, paste_x), canvas_width - scaled_width)
        paste_y = min(max(0, paste_y), canvas_height - target_view_height)

        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        layer = Image.merge("RGBA", (*resized_rgb.split(), resized_alpha))
        canvas.alpha_composite(layer, (paste_x, paste_y))
        rgba = np.asarray(canvas, dtype=np.uint8)
        foreground = rgba[..., 3].astype(np.float32) / 255.0
        candidate_images[angle] = rgba
        candidate_foregrounds[angle] = foreground
        view_reports[f"{angle:03d}"] = {
            "source_height_fraction": source["bbox_height_fraction"],
            "uniform_scale": scale,
            "candidate_width_pixels": scaled_width,
            "candidate_height_pixels": target_view_height,
            "paste_x": paste_x,
            "paste_y": paste_y,
            "width_limited": width_limited,
        }

    consistency = inspect_model_sheet_consistency(candidate_foregrounds)
    report = {
        "schema": "cfp.model-sheet-alignment-candidate.v1",
        "status": "ALIGNMENT_CANDIDATE_READY",
        "alignment_mode": alignment_mode,
        "target": {
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "subject_height_fraction": target_height_fraction,
            "baseline_fraction": target_baseline_fraction,
            "center_x_fraction": 0.5,
            "design_target_height_mm": design_target_height_mm,
            "millimeters_are_metadata_only": True,
        },
        "views": view_reports,
        "candidate_consistency_status": consistency["status"],
        "candidate_consistency": consistency["comparison"],
        "candidate_blocking_reasons": consistency["decision"]["blocking_reasons"],
        "promotion": {
            "automatic": False,
            "required_action": "Review before/after contact sheets and explicitly rewire or promote the candidate outputs.",
        },
    }
    return candidate_images, candidate_foregrounds, report


def format_alignment_candidate_report(report: dict) -> str:
    target = report["target"]
    reasons = ", ".join(report["candidate_blocking_reasons"]) or "none"
    return "\n".join(
        (
            "CFP TRELLIS.2 MLX Model-Sheet Alignment Candidate",
            "===================================================",
            f"Status: {report['status']}",
            f"Mode: {report['alignment_mode']}",
            f"Target subject height: {target['subject_height_fraction']:.2%} of canvas",
            f"Target centerline: {target['center_x_fraction']:.2%}",
            f"Target baseline: {target['baseline_fraction']:.2%}",
            f"Design target height: {target['design_target_height_mm']:.2f} mm (metadata only)",
            f"Candidate consistency: {report['candidate_consistency_status']}",
            f"Candidate blocking reasons: {reasons}",
            "Automatic promotion: No",
        )
    )


def alignment_candidate_report_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
