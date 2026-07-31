from __future__ import annotations

import json

import numpy as np
from scipy import ndimage


def inspect_input_mask(
    foreground_alpha: np.ndarray,
    *,
    alpha_threshold: float = 0.5,
    min_foreground_fraction: float = 0.05,
    max_foreground_fraction: float = 0.85,
    max_border_contact_fraction: float = 0.10,
    max_minor_component_fraction: float = 0.02,
    acknowledge_suspicious_mask: bool = False,
) -> dict:
    alpha = np.asarray(foreground_alpha, dtype=np.float32)
    if alpha.ndim != 2 or not alpha.size:
        raise ValueError("foreground_alpha must be a non-empty 2D array")
    if not 0.0 < alpha_threshold < 1.0:
        raise ValueError("alpha_threshold must be between 0 and 1")
    if not 0.0 <= min_foreground_fraction < max_foreground_fraction <= 1.0:
        raise ValueError("foreground fraction limits must satisfy 0 <= min < max <= 1")
    if not 0.0 <= max_border_contact_fraction <= 1.0:
        raise ValueError("max_border_contact_fraction must be between 0 and 1")
    if not 0.0 <= max_minor_component_fraction <= 1.0:
        raise ValueError("max_minor_component_fraction must be between 0 and 1")

    alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    foreground = alpha >= alpha_threshold
    foreground_pixels = int(np.count_nonzero(foreground))
    foreground_fraction = foreground_pixels / int(foreground.size)

    border = np.concatenate(
        (foreground[0], foreground[-1], foreground[1:-1, 0], foreground[1:-1, -1])
    )
    border_contact_fraction = float(np.count_nonzero(border) / max(1, border.size))

    labels, component_count = ndimage.label(foreground)
    component_sizes = np.bincount(labels.ravel())[1:]
    component_sizes.sort()
    largest_component_pixels = int(component_sizes[-1]) if component_sizes.size else 0
    minor_component_pixels = foreground_pixels - largest_component_pixels
    minor_component_fraction = minor_component_pixels / max(1, foreground_pixels)

    blocking_reasons: list[str] = []
    warnings: list[str] = []
    if foreground_fraction < min_foreground_fraction:
        blocking_reasons.append("foreground_empty_or_too_small")
    if foreground_fraction > max_foreground_fraction:
        blocking_reasons.append("foreground_too_large_or_mask_inverted")
    if border_contact_fraction > max_border_contact_fraction:
        blocking_reasons.append("foreground_touches_too_much_of_border")
    if minor_component_fraction > max_minor_component_fraction:
        blocking_reasons.append("disconnected_mask_noise")
    elif component_count > 1:
        warnings.append("minor_disconnected_components_present")

    blocked = bool(blocking_reasons)
    acknowledged = blocked and acknowledge_suspicious_mask
    if acknowledged:
        status = "MASK_ACKNOWLEDGED"
    elif blocked:
        status = "MASK_BLOCKED"
    else:
        status = "MASK_PASS"

    return {
        "schema": "cfp.input-mask-quality.v1",
        "status": status,
        "image": {
            "width": int(alpha.shape[1]),
            "height": int(alpha.shape[0]),
        },
        "mask": {
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            "foreground_fraction": foreground_fraction,
            "border_contact_fraction": border_contact_fraction,
            "component_count": int(component_count),
            "largest_component_fraction": largest_component_pixels / max(1, foreground_pixels),
            "minor_component_fraction": minor_component_fraction,
        },
        "thresholds": {
            "alpha_threshold": alpha_threshold,
            "min_foreground_fraction": min_foreground_fraction,
            "max_foreground_fraction": max_foreground_fraction,
            "max_border_contact_fraction": max_border_contact_fraction,
            "max_minor_component_fraction": max_minor_component_fraction,
        },
        "decision": {
            "proceed_allowed": not blocked or acknowledged,
            "acknowledged": acknowledged,
            "blocking_reasons": blocking_reasons,
            "warnings": warnings,
        },
    }


def format_input_mask_report(report: dict) -> str:
    mask = report["mask"]
    decision = report["decision"]
    reasons = ", ".join(decision["blocking_reasons"]) or "none"
    warnings = ", ".join(decision["warnings"]) or "none"
    return "\n".join(
        (
            "CFP TRELLIS.2 MLX Input Mask Quality Gate",
            "==========================================",
            f"Status: {report['status']}",
            f"Image: {report['image']['width']} x {report['image']['height']}",
            f"Foreground coverage: {mask['foreground_fraction']:.2%}",
            f"Border contact: {mask['border_contact_fraction']:.2%}",
            f"Connected components: {mask['component_count']}",
            f"Minor component coverage: {mask['minor_component_fraction']:.2%}",
            f"Blocking reasons: {reasons}",
            f"Warnings: {warnings}",
            f"Proceed to TRELLIS.2: {'Yes' if decision['proceed_allowed'] else 'No'}",
        )
    )


def input_mask_report_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
