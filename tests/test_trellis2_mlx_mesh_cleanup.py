from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import trimesh

from comfyui_trellis2_mlx.mesh_cleanup import (
    cleanup_report_json,
    format_cleanup_report,
    remove_small_components_glb,
)
from comfyui_trellis2_mlx.mesh_report import analyze_glb


def _component_fixture() -> BytesIO:
    body = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    floater = trimesh.Trimesh(
        vertices=np.array(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.int64,
        ),
        process=False,
    )
    floater.apply_scale(0.02)
    floater.apply_translation([3.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([body, floater])
    return BytesIO(combined.export(file_type="glb"))


def test_remove_floaters_drops_small_component_and_reports_before_after():
    output, report = remove_small_components_glb(
        _component_fixture(),
        min_component_faces=10,
        min_component_ratio=0.01,
    )
    after = analyze_glb(BytesIO(output))

    assert report["status"] == "CHANGED"
    assert report["changes"]["components_before"] == 2
    assert report["changes"]["components_after"] == 1
    assert report["changes"]["components_removed"] == 1
    assert report["changes"]["faces_removed"] == 4
    assert after["geometry"]["connected_components"] == 1
    assert "Components: 2 → 1" in format_cleanup_report(report)
    assert '"schema": "cfp.remove-floaters.v1"' in cleanup_report_json(report)


def test_remove_floaters_preserves_components_meeting_both_thresholds():
    output, report = remove_small_components_glb(
        _component_fixture(),
        min_component_faces=4,
        min_component_ratio=0.01,
    )

    assert report["status"] == "UNCHANGED"
    assert report["changes"]["components_after"] == 2
    assert analyze_glb(BytesIO(output))["geometry"]["connected_components"] == 2


def test_remove_floaters_always_preserves_largest_component():
    output, report = remove_small_components_glb(
        _component_fixture(),
        min_component_faces=1_000_000,
        min_component_ratio=1.0,
    )

    assert report["changes"]["components_after"] == 1
    assert analyze_glb(BytesIO(output))["geometry"]["triangles"] > 0


@pytest.mark.parametrize(
    ("min_component_faces", "min_component_ratio", "message"),
    [
        (-1, 0.1, "min_component_faces"),
        (0, -0.1, "min_component_ratio"),
        (0, 1.1, "min_component_ratio"),
    ],
)
def test_remove_floaters_rejects_invalid_thresholds(
    min_component_faces,
    min_component_ratio,
    message,
):
    with pytest.raises(ValueError, match=message):
        remove_small_components_glb(
            _component_fixture(),
            min_component_faces=min_component_faces,
            min_component_ratio=min_component_ratio,
        )
