from __future__ import annotations

import numpy as np
import pytest

from comfyui_trellis2_mlx.model_sheet_alignment_candidate import (
    align_model_sheet_candidate,
    alignment_candidate_report_json,
    format_alignment_candidate_report,
)
from comfyui_trellis2_mlx.model_sheet_consistency import inspect_model_sheet_consistency


def _image(color):
    image = np.zeros((100, 100, 3), dtype=np.float32)
    image[...] = color
    return image


def _mask(x_min, x_max, y_min, y_max):
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[y_min:y_max, x_min:x_max] = 1.0
    return mask


def _inputs():
    images = {
        0: _image((0.8, 0.1, 0.1)),
        90: _image((0.1, 0.8, 0.1)),
        180: _image((0.1, 0.1, 0.8)),
        270: _image((0.7, 0.7, 0.1)),
    }
    masks = {
        0: _mask(25, 75, 5, 95),
        90: _mask(40, 65, 15, 85),
        180: _mask(30, 70, 10, 90),
        270: _mask(45, 70, 20, 90),
    }
    return images, masks


def test_median_alignment_creates_shared_center_height_and_baseline_without_mutation():
    images, masks = _inputs()
    original = images[0].copy()

    candidates, candidate_masks, report = align_model_sheet_candidate(
        images,
        masks,
        design_target_height_mm=250.0,
    )

    assert all(image.shape == (100, 100, 4) for image in candidates.values())
    assert np.array_equal(images[0], original)
    assert report["target"]["subject_height_fraction"] == pytest.approx(0.75)
    assert report["target"]["design_target_height_mm"] == 250.0
    assert report["target"]["millimeters_are_metadata_only"] is True
    assert report["promotion"]["automatic"] is False
    consistency = inspect_model_sheet_consistency(candidate_masks)
    assert consistency["comparison"]["height_spread"] <= 0.01
    assert consistency["comparison"]["center_x_spread"] <= 0.01
    assert consistency["comparison"]["baseline_spread"] <= 0.01
    assert "Automatic promotion: No" in format_alignment_candidate_report(report)
    assert '"schema": "cfp.model-sheet-alignment-candidate.v1"' in alignment_candidate_report_json(report)


def test_explicit_fraction_uses_requested_normalized_height_and_carries_mm_metadata():
    images, masks = _inputs()

    _, _, report = align_model_sheet_candidate(
        images,
        masks,
        alignment_mode="explicit_fraction",
        target_subject_height_fraction=0.82,
        design_target_height_mm=300.0,
    )

    assert report["target"]["subject_height_fraction"] == pytest.approx(0.82)
    assert report["target"]["design_target_height_mm"] == 300.0


def test_invalid_physical_height_is_rejected():
    images, masks = _inputs()

    with pytest.raises(ValueError, match="design_target_height_mm"):
        align_model_sheet_candidate(images, masks, design_target_height_mm=0.0)
