from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from .model_sheet_consistency import (
    CAMERA_ANGLES,
    inspect_model_sheet_consistency,
)


ANGLE_LABELS = {
    0: "000° FRONT",
    90: "090° LEFT",
    180: "180° REAR",
    270: "270° RIGHT",
}


def build_alignment_review(
    images: dict[int, np.ndarray],
    foreground_masks: dict[int, np.ndarray],
    *,
    cell_size: int = 384,
) -> tuple[np.ndarray, dict]:
    if set(images) != set(CAMERA_ANGLES):
        raise ValueError("images must contain cameras 0, 90, 180, and 270")
    if not 192 <= cell_size <= 1024:
        raise ValueError("cell_size must be between 192 and 1024")

    consistency = inspect_model_sheet_consistency(foreground_masks)
    label_height = 72
    sheet = Image.new("RGB", (cell_size * 2, (cell_size + label_height) * 2), "#202328")
    draw = ImageDraw.Draw(sheet)

    for index, angle in enumerate(CAMERA_ANGLES):
        row, column = divmod(index, 2)
        cell_x = column * cell_size
        cell_y = row * (cell_size + label_height)
        image = np.asarray(images[angle])
        foreground = np.asarray(foreground_masks[angle], dtype=np.float32).clip(0.0, 1.0)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Camera {angle:03d} image must be HWC RGB or RGBA")
        if foreground.shape != image.shape[:2]:
            raise ValueError(f"Camera {angle:03d} image and foreground dimensions differ")

        pixels = image[..., :3]
        if np.issubdtype(pixels.dtype, np.floating):
            pixels = (pixels.clip(0.0, 1.0) * 255).round().astype(np.uint8)
        else:
            pixels = pixels.clip(0, 255).astype(np.uint8)
        alpha = (foreground * 255).round().astype(np.uint8)
        rgba = Image.fromarray(np.dstack((pixels, alpha)), mode="RGBA")

        source_height, source_width = foreground.shape
        scale = min(cell_size / source_width, cell_size / source_height)
        preview_width = max(1, round(source_width * scale))
        preview_height = max(1, round(source_height * scale))
        offset_x = cell_x + (cell_size - preview_width) // 2
        offset_y = cell_y + (cell_size - preview_height) // 2
        preview = rgba.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        background = Image.new("RGBA", (preview_width, preview_height), "#343941")
        background.alpha_composite(preview)
        sheet.paste(background.convert("RGB"), (offset_x, offset_y))

        canvas_center_x = offset_x + (source_width - 1) * 0.5 * scale
        draw.line(
            (canvas_center_x, offset_y, canvas_center_x, offset_y + preview_height - 1),
            fill="#35c6e8",
            width=2,
        )

        view = consistency["views"][f"{angle:03d}"]
        if not view["foreground_empty"]:
            coordinates = np.argwhere(foreground >= 0.5)
            y_min, x_min = coordinates.min(axis=0)
            y_max, x_max = coordinates.max(axis=0)
            box = (
                offset_x + x_min * scale,
                offset_y + y_min * scale,
                offset_x + x_max * scale,
                offset_y + y_max * scale,
            )
            draw.rectangle(box, outline="#55e070", width=3)
            subject_center_x = offset_x + (x_min + x_max) * 0.5 * scale
            baseline_y = offset_y + y_max * scale
            draw.line(
                (subject_center_x, box[1], subject_center_x, box[3]),
                fill="#e653d7",
                width=2,
            )
            draw.line(
                (offset_x, baseline_y, offset_x + preview_width - 1, baseline_y),
                fill="#ffd24a",
                width=2,
            )
            metrics = (
                f"H {view['bbox_height_fraction']:.1%}  "
                f"CX {view['center_x_fraction']:.1%}  "
                f"BASE {view['baseline_fraction']:.1%}"
            )
        else:
            metrics = "EMPTY FOREGROUND"

        text_y = cell_y + cell_size + 8
        draw.text((cell_x + 10, text_y), ANGLE_LABELS[angle], fill="#ffffff")
        draw.text((cell_x + 10, text_y + 24), metrics, fill="#cbd3dd")

    report = {
        "schema": "cfp.model-sheet-alignment-review.v1",
        "status": "REVIEW_READY",
        "consistency_status": consistency["status"],
        "legend": {
            "canvas_centerline": "cyan",
            "foreground_bounds": "green",
            "subject_centerline": "magenta",
            "subject_baseline": "yellow",
        },
        "views": consistency["views"],
        "comparison": consistency["comparison"],
        "blocking_reasons": consistency["decision"]["blocking_reasons"],
    }
    return np.asarray(sheet, dtype=np.uint8), report


def format_alignment_review_report(report: dict) -> str:
    comparison = report["comparison"]
    reasons = ", ".join(report["blocking_reasons"]) or "none"
    return "\n".join(
        (
            "CFP TRELLIS.2 MLX Model-Sheet Alignment Review",
            "================================================",
            f"Status: {report['status']}",
            f"Consistency: {report['consistency_status']}",
            f"Height spread: {comparison['height_spread']:.2%}",
            f"Centerline spread: {comparison['center_x_spread']:.2%}",
            f"Vertical-center spread: {comparison['vertical_center_spread']:.2%}",
            f"Baseline spread: {comparison['baseline_spread']:.2%}",
            f"Blocking reasons: {reasons}",
            "Legend: cyan canvas center, green bounds, magenta subject center, yellow baseline",
        )
    )


def alignment_review_report_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
