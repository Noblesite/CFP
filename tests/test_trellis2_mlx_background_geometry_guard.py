from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import trimesh

from comfyui_trellis2_mlx.background_geometry_guard import (
    background_geometry_guard_report_json,
    format_background_geometry_guard_report,
    inspect_background_geometry,
)


def _as_glb(mesh: trimesh.Trimesh) -> BytesIO:
    return BytesIO(mesh.export(file_type="glb"))


def test_character_guard_passes_z_dominant_geometry_without_modification():
    source = _as_glb(trimesh.creation.box(extents=[0.4, 0.3, 1.0]))
    source_bytes = source.getvalue()
    output, report = inspect_background_geometry(source)

    assert output == source_bytes
    assert report["status"] == "GUARD_PASS"
    assert report["decision"]["proceed_allowed"] is True
    assert report["bounds"]["z_dominance_ratio"] == pytest.approx(2.5)
    assert "Proceed to voxel fill: Yes" in (
        format_background_geometry_guard_report(report)
    )
    assert '"schema": "cfp.background-geometry-guard.v1"' in (
        background_geometry_guard_report_json(report)
    )


def test_character_guard_blocks_nearly_isotropic_bounds():
    _, report = inspect_background_geometry(
        _as_glb(trimesh.creation.box(extents=[1.0, 1.0, 1.05]))
    )

    assert report["status"] == "GUARD_BLOCKED"
    assert report["decision"]["proceed_allowed"] is False
    assert report["decision"]["blocking_reasons"] == [
        "character_height_not_dominant"
    ]


def test_guard_allows_explicit_human_acknowledgement():
    _, report = inspect_background_geometry(
        _as_glb(trimesh.creation.box()),
        acknowledge_suspicious_geometry=True,
    )

    assert report["status"] == "GUARD_ACKNOWLEDGED"
    assert report["decision"]["proceed_allowed"] is True


def test_generic_profile_accepts_valid_cubic_prop():
    _, report = inspect_background_geometry(
        _as_glb(trimesh.creation.box()),
        profile="generic",
    )

    assert report["status"] == "GUARD_PASS"
    assert report["decision"]["blocking_reasons"] == []


def test_guard_reports_large_planar_components_without_deleting_them():
    box = trimesh.creation.box(extents=[0.4, 0.3, 1.0])
    plane_vertices = np.array(
        [
            [-0.5, -0.5, -0.6],
            [0.5, -0.5, -0.6],
            [0.5, 0.5, -0.6],
            [-0.5, 0.5, -0.6],
        ]
    )
    plane_faces = np.array([[0, 1, 2], [0, 2, 3]])
    plane = trimesh.Trimesh(
        vertices=plane_vertices,
        faces=plane_faces,
        process=False,
    )
    combined = trimesh.util.concatenate((box, plane))
    source = _as_glb(combined)
    source_bytes = source.getvalue()

    output, report = inspect_background_geometry(source)

    assert output == source_bytes
    assert report["components"]["large_planar_component_count"] == 1
    assert "large_planar_components_present" in report["decision"]["warnings"]


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("profile", "vehicle", "profile"),
        ("min_z_dominance_ratio", 0.5, "min_z_dominance_ratio"),
        ("planar_flatness_ratio", 0.5, "planar_flatness_ratio"),
        ("large_planar_span_ratio", 0.01, "large_planar_span_ratio"),
    ],
)
def test_guard_rejects_invalid_configuration(argument, value, message):
    with pytest.raises(ValueError, match=message):
        inspect_background_geometry(
            _as_glb(trimesh.creation.box()),
            **{argument: value},
        )
