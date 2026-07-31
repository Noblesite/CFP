from __future__ import annotations

from io import BytesIO

import numpy as np
import trimesh

from comfyui_trellis2_mlx.mesh_report import analyze_glb
from comfyui_trellis2_mlx.post_voxel_polish import (
    format_post_voxel_polish_report,
    polish_post_voxel_glb,
    post_voxel_polish_report_json,
)


def _export(vertices: np.ndarray, faces: np.ndarray) -> BytesIO:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return BytesIO(mesh.export(file_type="glb"))


def _box_with_internal_opposed_sheet() -> BytesIO:
    box = trimesh.creation.box()
    internal_vertices = np.array(
        [
            [-0.25, 0.0, -0.25],
            [0.25, 0.0, -0.25],
            [0.25, 0.0, 0.25],
            [-0.25, 0.0, 0.25],
        ]
    )
    offset = len(box.vertices)
    internal_faces = np.array(
        [
            [offset + 0, offset + 1, offset + 2],
            [offset + 0, offset + 2, offset + 3],
            [offset + 2, offset + 1, offset + 0],
            [offset + 3, offset + 2, offset + 0],
        ]
    )
    return _export(
        np.vstack((box.vertices, internal_vertices)),
        np.vstack((box.faces, internal_faces)),
    )


def test_polish_removes_safe_opposed_internal_sheet():
    output, report = polish_post_voxel_glb(_box_with_internal_opposed_sheet())
    geometry = analyze_glb(BytesIO(output))["geometry"]

    assert report["status"] == "POLISHED_PASS"
    assert report["rule"]["accepted"] is True
    assert report["changes"]["opposed_duplicate_groups_removed"] == 2
    assert report["changes"]["opposed_duplicate_faces_removed"] == 4
    assert report["changes"]["overloaded_edges_before"] == 1
    assert report["changes"]["overloaded_edges_after"] == 0
    assert report["changes"]["boundary_edges_after"] == 0
    assert geometry["watertight"] is True
    assert "No remeshing or vertex movement" in format_post_voxel_polish_report(
        report
    )
    assert '"schema": "cfp.post-voxel-topology-polish.v1"' in (
        post_voxel_polish_report_json(report)
    )


def test_polish_rejects_deletion_that_would_open_surface():
    box = trimesh.creation.box()
    reverse_face = box.faces[0][::-1]
    source = _export(
        np.asarray(box.vertices),
        np.vstack((box.faces, reverse_face)),
    )
    output, report = polish_post_voxel_glb(source)
    geometry = analyze_glb(BytesIO(output))["geometry"]

    assert report["status"] == "UNCHANGED_REVIEW"
    assert report["rule"]["accepted"] is False
    assert report["changes"]["opposed_duplicate_faces_removed"] == 0
    assert geometry["triangles"] == len(box.faces) + 1


def test_polish_leaves_clean_mesh_unchanged():
    box = trimesh.creation.box()
    output, report = polish_post_voxel_glb(
        _export(np.asarray(box.vertices), np.asarray(box.faces))
    )
    geometry = analyze_glb(BytesIO(output))["geometry"]

    assert report["status"] == "UNCHANGED_REVIEW"
    assert report["rule"]["candidate_groups"] == 0
    assert geometry["watertight"] is True
