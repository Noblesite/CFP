from __future__ import annotations

import numpy as np

from comfyui_trellis2_mlx.model_sheet_alignment_review import (
    alignment_review_report_json,
    build_alignment_review,
    format_alignment_review_report,
)


def _image():
    return np.full((100, 100, 3), 0.5, dtype=np.float32)


def _mask(x_min, x_max, y_min=10, y_max=90):
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[y_min:y_max, x_min:x_max] = 1.0
    return mask


def _sheet():
    return {
        0: _mask(30, 70),
        90: _mask(37, 63),
        180: _mask(31, 69),
        270: _mask(36, 64),
    }


def test_alignment_review_builds_annotated_two_by_two_sheet_without_mutation():
    images = {angle: _image() for angle in (0, 90, 180, 270)}
    masks = _sheet()
    original = images[0].copy()

    contact_sheet, report = build_alignment_review(images, masks, cell_size=256)

    assert contact_sheet.shape == (656, 512, 3)
    assert report["status"] == "REVIEW_READY"
    assert report["consistency_status"] == "SHEET_PASS"
    assert np.array_equal(images[0], original)
    assert "Legend: cyan canvas center" in format_alignment_review_report(report)
    assert '"schema": "cfp.model-sheet-alignment-review.v1"' in alignment_review_report_json(report)


def test_alignment_review_is_still_rendered_for_inconsistent_sheet():
    images = {angle: _image() for angle in (0, 90, 180, 270)}
    masks = _sheet()
    masks[270] = _mask(55, 75, 35, 95)

    contact_sheet, report = build_alignment_review(images, masks, cell_size=192)

    assert contact_sheet.shape == (528, 384, 3)
    assert report["status"] == "REVIEW_READY"
    assert report["consistency_status"] == "SHEET_BLOCKED"
    assert report["blocking_reasons"]
