from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import trimesh

from comfyui_trellis2_mlx.topology_diagnostics import diagnose_ovoxel_topology
from comfyui_trellis2_mlx.topology_sanitizer import (
    format_sanitizer_report,
    sanitize_topology_glb,
    sanitizer_report_json,
)


def _as_glb(mesh: trimesh.Trimesh) -> BytesIO:
    return BytesIO(mesh.export(file_type="glb"))


def _dirty_tetrahedron() -> trimesh.Trimesh:
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
            [4, 2, 1],
            [4, 4, 1],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_sanitizer_welds_and_removes_duplicate_and_degenerate_faces():
    output, report = sanitize_topology_glb(_as_glb(_dirty_tetrahedron()))
    after = diagnose_ovoxel_topology(BytesIO(output))

    assert report["status"] == "CHANGED"
    assert report["changes"]["vertices_welded"] == 1
    assert report["changes"]["duplicate_faces_removed"] == 1
    assert report["changes"]["degenerate_faces_removed"] == 1
    assert report["changes"]["faces_after"] == 4
    assert after["status"] == "PASS"
    assert "Duplicate faces removed: 1" in format_sanitizer_report(report)
    assert '"schema": "cfp.topology-sanitizer.v1"' in sanitizer_report_json(report)


def test_sanitizer_leaves_clean_box_topology_unchanged():
    output, report = sanitize_topology_glb(_as_glb(trimesh.creation.box()))

    assert report["status"] == "UNCHANGED"
    assert report["changes"]["vertices_welded"] == 0
    assert report["changes"]["duplicate_faces_removed"] == 0
    assert report["changes"]["degenerate_faces_removed"] == 0
    assert diagnose_ovoxel_topology(BytesIO(output))["status"] == "PASS"


def test_sanitizer_flags_boundaries_revealed_by_stacked_duplicate_sheet():
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        process=False,
    )

    _, report = sanitize_topology_glb(_as_glb(mesh))

    assert report["status"] == "CHANGED_REVIEW"
    assert report["changes"]["topology_regressions"] == [
        "boundary_edges_increased"
    ]


@pytest.mark.parametrize("ratio", [0.0, -1e-8, 0.002])
def test_sanitizer_rejects_invalid_weld_ratio(ratio):
    with pytest.raises(ValueError, match="weld_tolerance_ratio"):
        sanitize_topology_glb(
            _as_glb(trimesh.creation.box()),
            weld_tolerance_ratio=ratio,
        )
