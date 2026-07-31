from __future__ import annotations

import json

import numpy as np


CAMERA_ANGLES = (0, 90, 180, 270)


def inspect_model_sheet_consistency(
    foreground_masks: dict[int, np.ndarray],
    *,
    foreground_threshold: float = 0.5,
    max_height_spread: float = 0.08,
    max_center_x_spread: float = 0.05,
    max_vertical_center_spread: float = 0.05,
    max_baseline_spread: float = 0.03,
    max_opposing_width_delta: float = 0.15,
    acknowledge_inconsistent_sheet: bool = False,
) -> dict:
    if set(foreground_masks) != set(CAMERA_ANGLES):
        raise ValueError("foreground_masks must contain cameras 0, 90, 180, and 270")
    if not 0.0 < foreground_threshold < 1.0:
        raise ValueError("foreground_threshold must be between 0 and 1")
    thresholds = {
        "max_height_spread": max_height_spread,
        "max_center_x_spread": max_center_x_spread,
        "max_vertical_center_spread": max_vertical_center_spread,
        "max_baseline_spread": max_baseline_spread,
        "max_opposing_width_delta": max_opposing_width_delta,
    }
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("consistency thresholds must be between 0 and 1")

    view_reports: dict[str, dict] = {}
    shapes: set[tuple[int, int]] = set()
    blocking_reasons: list[str] = []
    for angle in CAMERA_ANGLES:
        mask = np.asarray(foreground_masks[angle], dtype=np.float32)
        if mask.ndim != 2 or not mask.size:
            raise ValueError(f"Camera {angle:03d} foreground mask must be a non-empty 2D array")
        foreground = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0) >= foreground_threshold
        height, width = foreground.shape
        shapes.add((height, width))
        coordinates = np.argwhere(foreground)
        if not coordinates.size:
            blocking_reasons.append(f"camera_{angle:03d}_foreground_empty")
            view_reports[f"{angle:03d}"] = {
                "width": width,
                "height": height,
                "foreground_empty": True,
            }
            continue

        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        bbox_height = int(y_max - y_min + 1)
        bbox_width = int(x_max - x_min + 1)
        view_reports[f"{angle:03d}"] = {
            "width": width,
            "height": height,
            "foreground_empty": False,
            "bbox_height_fraction": bbox_height / height,
            "bbox_width_fraction": bbox_width / width,
            "center_x_fraction": ((x_min + x_max) / 2.0) / max(1, width - 1),
            "vertical_center_fraction": ((y_min + y_max) / 2.0) / max(1, height - 1),
            "baseline_fraction": y_max / max(1, height - 1),
        }

    if len(shapes) != 1:
        blocking_reasons.append("canvas_dimensions_mismatch")

    populated = [report for report in view_reports.values() if not report["foreground_empty"]]

    def spread(field: str) -> float:
        values = [report[field] for report in populated]
        return max(values) - min(values) if values else 0.0

    height_spread = spread("bbox_height_fraction")
    center_x_spread = spread("center_x_fraction")
    vertical_center_spread = spread("vertical_center_fraction")
    baseline_spread = spread("baseline_fraction")

    def opposing_width_delta(first: int, second: int) -> float | None:
        first_report = view_reports[f"{first:03d}"]
        second_report = view_reports[f"{second:03d}"]
        if first_report["foreground_empty"] or second_report["foreground_empty"]:
            return None
        first_width = first_report["bbox_width_fraction"]
        second_width = second_report["bbox_width_fraction"]
        return abs(first_width - second_width) / max(first_width, second_width)

    width_delta_000_180 = opposing_width_delta(0, 180)
    width_delta_090_270 = opposing_width_delta(90, 270)

    if height_spread > max_height_spread:
        blocking_reasons.append("subject_height_mismatch")
    if center_x_spread > max_center_x_spread:
        blocking_reasons.append("subject_centerline_mismatch")
    if vertical_center_spread > max_vertical_center_spread:
        blocking_reasons.append("subject_vertical_alignment_mismatch")
    if baseline_spread > max_baseline_spread:
        blocking_reasons.append("subject_baseline_mismatch")
    if width_delta_000_180 is not None and width_delta_000_180 > max_opposing_width_delta:
        blocking_reasons.append("front_rear_width_mismatch")
    if width_delta_090_270 is not None and width_delta_090_270 > max_opposing_width_delta:
        blocking_reasons.append("left_right_width_mismatch")

    blocked = bool(blocking_reasons)
    acknowledged = blocked and acknowledge_inconsistent_sheet
    if acknowledged:
        status = "SHEET_ACKNOWLEDGED"
    elif blocked:
        status = "SHEET_BLOCKED"
    else:
        status = "SHEET_PASS"

    return {
        "schema": "cfp.model-sheet-consistency.v1",
        "status": status,
        "views": view_reports,
        "comparison": {
            "height_spread": height_spread,
            "center_x_spread": center_x_spread,
            "vertical_center_spread": vertical_center_spread,
            "baseline_spread": baseline_spread,
            "front_rear_width_delta": width_delta_000_180,
            "left_right_width_delta": width_delta_090_270,
        },
        "thresholds": {"foreground_threshold": foreground_threshold, **thresholds},
        "decision": {
            "proceed_allowed": not blocked or acknowledged,
            "acknowledged": acknowledged,
            "blocking_reasons": blocking_reasons,
        },
    }


def format_model_sheet_report(report: dict) -> str:
    comparison = report["comparison"]
    reasons = ", ".join(report["decision"]["blocking_reasons"]) or "none"
    lines = [
        "CFP TRELLIS.2 MLX Model-Sheet Consistency Gate",
        "================================================",
        f"Status: {report['status']}",
    ]
    for angle in ("000", "090", "180", "270"):
        view = report["views"][angle]
        if view["foreground_empty"]:
            lines.append(f"Camera {angle}°: EMPTY")
        else:
            lines.append(
                f"Camera {angle}°: height {view['bbox_height_fraction']:.2%}, "
                f"width {view['bbox_width_fraction']:.2%}, "
                f"center X {view['center_x_fraction']:.2%}, "
                f"baseline {view['baseline_fraction']:.2%}"
            )
    lines.extend(
        (
            f"Height spread: {comparison['height_spread']:.2%}",
            f"Centerline spread: {comparison['center_x_spread']:.2%}",
            f"Vertical-center spread: {comparison['vertical_center_spread']:.2%}",
            f"Baseline spread: {comparison['baseline_spread']:.2%}",
            f"Blocking reasons: {reasons}",
            f"Proceed to TRELLIS.2: {'Yes' if report['decision']['proceed_allowed'] else 'No'}",
        )
    )
    return "\n".join(lines)


def model_sheet_report_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
