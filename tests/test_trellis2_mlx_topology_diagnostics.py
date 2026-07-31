from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import trimesh

from comfyui_trellis2_mlx.topology_diagnostics import (
    diagnose_ovoxel_topology,
    format_topology_diagnostics,
    topology_diagnostics_json,
)


def _as_glb(mesh: trimesh.Trimesh) -> BytesIO:
    return BytesIO(mesh.export(file_type="glb"))


def test_diagnostics_confirm_duplicate_face_overloads():
    box = trimesh.creation.box()
    faces = np.vstack((box.faces, box.faces[0]))
    mesh = trimesh.Trimesh(vertices=box.vertices, faces=faces, process=False)

    report = diagnose_ovoxel_topology(_as_glb(mesh))

    assert report["status"] == "FAIL"
    assert report["confirmed"]["duplicate_face_extras"] == 1
    assert report["confirmed"]["edge_incidence"]["overloaded_edges"] == 3
    assert (
        report["attribution_probes"]["after_deduplicating_faces"]["overloaded_edges"]
        == 0
    )
    assert "duplicate_faces" in report["classification"]["confirmed_causes"]
    assert "Duplicate face extras: 1" in format_topology_diagnostics(report)
    assert '"schema": "cfp.ovoxel-topology-diagnostics.v1"' in topology_diagnostics_json(
        report
    )


def test_diagnostics_confirm_degenerate_face():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 0, 1]], dtype=np.int64)
    report = diagnose_ovoxel_topology(
        _as_glb(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    )

    assert report["confirmed"]["degenerate_faces"] == 1
    assert "degenerate_faces" in report["classification"]["confirmed_causes"]


def test_diagnostics_mark_overlapping_component_bounds_as_candidate_not_proof():
    first = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    second = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    second.apply_translation([0.5, 0.0, 0.0])
    report = diagnose_ovoxel_topology(
        _as_glb(trimesh.util.concatenate([first, second]))
    )

    assert report["components"]["count"] == 2
    assert report["components"]["bbox_overlap_candidate_count"] == 1
    assert (
        "overlapping_disconnected_shells"
        in report["classification"]["candidate_causes"]
    )
    assert (
        report["classification"]["exact_self_intersection_test"]["status"]
        == "NOT_RUN"
    )


def test_diagnostics_pass_clean_closed_box():
    report = diagnose_ovoxel_topology(_as_glb(trimesh.creation.box()))

    assert report["status"] == "PASS"
    assert report["classification"]["confirmed_causes"] == []
    assert report["classification"]["candidate_causes"] == []


@pytest.mark.parametrize("ratio", [0.0, -1e-6, 0.02])
def test_diagnostics_reject_invalid_tolerance_ratio(ratio):
    with pytest.raises(ValueError, match="coordinate_tolerance_ratio"):
        diagnose_ovoxel_topology(
            _as_glb(trimesh.creation.box()),
            coordinate_tolerance_ratio=ratio,
        )
