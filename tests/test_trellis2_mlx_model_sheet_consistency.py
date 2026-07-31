from __future__ import annotations

import numpy as np
import pytest

from comfyui_trellis2_mlx.model_sheet_consistency import (
    format_model_sheet_report,
    inspect_model_sheet_consistency,
    model_sheet_report_json,
)


def _mask(*, x_min=35, x_max=65, y_min=10, y_max=90, shape=(100, 100)):
    mask = np.zeros(shape, dtype=np.float32)
    mask[y_min:y_max, x_min:x_max] = 1.0
    return mask


def _aligned_sheet():
    return {
        0: _mask(x_min=30, x_max=70),
        90: _mask(x_min=37, x_max=63),
        180: _mask(x_min=31, x_max=69),
        270: _mask(x_min=36, x_max=64),
    }


def test_aligned_cardinal_sheet_passes():
    report = inspect_model_sheet_consistency(_aligned_sheet())

    assert report["status"] == "SHEET_PASS"
    assert report["decision"]["proceed_allowed"] is True
    assert "Proceed to TRELLIS.2: Yes" in format_model_sheet_report(report)
    assert '"schema": "cfp.model-sheet-consistency.v1"' in model_sheet_report_json(report)


def test_canvas_size_mismatch_is_blocked():
    sheet = _aligned_sheet()
    sheet[270] = _mask(shape=(120, 100), y_min=20, y_max=100)

    report = inspect_model_sheet_consistency(sheet)

    assert "canvas_dimensions_mismatch" in report["decision"]["blocking_reasons"]


def test_scale_centerline_and_baseline_drift_are_blocked():
    sheet = _aligned_sheet()
    sheet[270] = _mask(x_min=55, x_max=75, y_min=35, y_max=95)

    report = inspect_model_sheet_consistency(sheet)

    reasons = report["decision"]["blocking_reasons"]
    assert "subject_height_mismatch" in reasons
    assert "subject_centerline_mismatch" in reasons
    assert "subject_vertical_alignment_mismatch" in reasons
    assert "subject_baseline_mismatch" in reasons


def test_opposing_width_mismatch_is_blocked_without_comparing_front_to_side():
    sheet = _aligned_sheet()
    sheet[180] = _mask(x_min=42, x_max=58)

    report = inspect_model_sheet_consistency(sheet)

    assert "front_rear_width_mismatch" in report["decision"]["blocking_reasons"]


def test_explicit_human_acknowledgement_allows_sheet():
    sheet = _aligned_sheet()
    sheet[270] = _mask(x_min=55, x_max=75, y_min=35, y_max=95)

    report = inspect_model_sheet_consistency(
        sheet,
        acknowledge_inconsistent_sheet=True,
    )

    assert report["status"] == "SHEET_ACKNOWLEDGED"
    assert report["decision"]["proceed_allowed"] is True


def test_missing_camera_is_rejected():
    with pytest.raises(ValueError, match="0, 90, 180, and 270"):
        inspect_model_sheet_consistency({0: _mask()})
