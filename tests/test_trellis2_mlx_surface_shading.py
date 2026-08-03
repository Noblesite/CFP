from __future__ import annotations

from io import BytesIO

import pytest
import trimesh

from comfyui_trellis2_mlx.surface_shading import (
    format_surface_shading_report,
    shade_surface_candidate_glb,
    surface_shading_report_json,
)


def _box() -> BytesIO:
    return BytesIO(trimesh.creation.box().export(file_type="glb"))


def test_smooth_shading_preserves_positions_faces_and_bounds():
    output, report = shade_surface_candidate_glb(_box(), mode="smooth")

    assert output
    assert report["status"] == "SHADING_CANDIDATE_PASS"
    assert report["changes"]["vertex_positions_moved"] is False
    assert report["changes"]["surface_geometry_preserved"] is True
    assert report["changes"]["vertices_before"] == report["changes"]["vertices_after"]
    assert report["changes"]["triangles_before"] == report["changes"]["triangles_after"]
    assert report["changes"]["max_bounds_delta"] == 0.0
    assert "Mode: smooth" in format_surface_shading_report(report)
    assert '"schema": "cfp.surface-shading-candidate.v1"' in (
        surface_shading_report_json(report)
    )


def test_angle_shading_splits_cube_vertices_without_moving_surface():
    _, report = shade_surface_candidate_glb(
        _box(),
        mode="angle",
        angle_degrees=30.0,
    )

    assert report["status"] == "SHADING_CANDIDATE_REVIEW"
    assert report["changes"]["surface_geometry_preserved"] is True
    assert report["changes"]["vertices_after"] > report["changes"]["vertices_before"]
    assert report["changes"]["triangles_after"] == report["changes"]["triangles_before"]
    assert report["changes"]["angle_mode_splits_vertices_at_hard_edges"] is True


def test_flat_mode_returns_exact_source_artifact():
    source = _box()
    source_bytes = source.getvalue()

    output, report = shade_surface_candidate_glb(source, mode="flat")

    assert output == source_bytes
    assert report["status"] == "UNCHANGED_FLAT"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "glossy"}, "mode"),
        ({"angle_degrees": -1.0}, "angle_degrees"),
        ({"angle_degrees": 181.0}, "angle_degrees"),
    ],
)
def test_surface_shading_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        shade_surface_candidate_glb(_box(), **kwargs)
