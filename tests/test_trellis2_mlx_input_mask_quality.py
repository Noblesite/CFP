from __future__ import annotations

import numpy as np
import pytest

from comfyui_trellis2_mlx.input_mask_quality import (
    format_input_mask_report,
    input_mask_report_json,
    inspect_input_mask,
)


def _centered_character_mask() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:90, 35:65] = 1.0
    mask[30:65, 20:80] = 1.0
    return mask


def test_centered_character_mask_passes():
    report = inspect_input_mask(_centered_character_mask())

    assert report["status"] == "MASK_PASS"
    assert report["decision"]["proceed_allowed"] is True
    assert report["mask"]["border_contact_fraction"] == 0.0
    assert "Proceed to TRELLIS.2: Yes" in format_input_mask_report(report)
    assert '"schema": "cfp.input-mask-quality.v1"' in input_mask_report_json(report)


def test_empty_mask_is_blocked():
    report = inspect_input_mask(np.zeros((64, 64), dtype=np.float32))

    assert report["status"] == "MASK_BLOCKED"
    assert "foreground_empty_or_too_small" in report["decision"]["blocking_reasons"]


def test_inverted_or_full_mask_is_blocked():
    report = inspect_input_mask(np.ones((64, 64), dtype=np.float32))

    assert "foreground_too_large_or_mask_inverted" in report["decision"]["blocking_reasons"]
    assert "foreground_touches_too_much_of_border" in report["decision"]["blocking_reasons"]


def test_excessive_border_contact_is_blocked():
    mask = _centered_character_mask()
    mask[:, :8] = 1.0

    report = inspect_input_mask(mask)

    assert "foreground_touches_too_much_of_border" in report["decision"]["blocking_reasons"]


def test_disconnected_noise_is_blocked():
    mask = _centered_character_mask()
    mask[2:18, 2:18] = 1.0

    report = inspect_input_mask(mask)

    assert "disconnected_mask_noise" in report["decision"]["blocking_reasons"]


def test_explicit_acknowledgement_allows_human_override():
    report = inspect_input_mask(
        np.ones((64, 64), dtype=np.float32),
        acknowledge_suspicious_mask=True,
    )

    assert report["status"] == "MASK_ACKNOWLEDGED"
    assert report["decision"]["proceed_allowed"] is True


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("alpha_threshold", 1.0, "alpha_threshold"),
        ("min_foreground_fraction", 0.9, "foreground fraction limits"),
        ("max_border_contact_fraction", 2.0, "max_border_contact_fraction"),
        ("max_minor_component_fraction", -0.1, "max_minor_component_fraction"),
    ],
)
def test_invalid_thresholds_are_rejected(argument, value, message):
    with pytest.raises(ValueError, match=message):
        inspect_input_mask(_centered_character_mask(), **{argument: value})
