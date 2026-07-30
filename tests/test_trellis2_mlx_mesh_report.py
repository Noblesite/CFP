from __future__ import annotations

from io import BytesIO

import numpy as np
import trimesh

from comfyui_trellis2_mlx.mesh_report import (
    analyze_glb,
    format_mesh_report,
    mesh_report_json,
)


def _as_glb(mesh: trimesh.Trimesh) -> BytesIO:
    return BytesIO(mesh.export(file_type="glb"))


def test_mesh_report_passes_for_closed_single_component_mesh():
    report = analyze_glb(_as_glb(trimesh.creation.box(extents=[1.0, 2.0, 3.0])))

    assert report["status"] == "PASS"
    assert report["geometry"] == {
        "vertices": 8,
        "triangles": 12,
        "connected_components": 1,
        "unique_edges": 18,
        "boundary_edges": 0,
        "non_manifold_edges": 0,
        "watertight": True,
        "bounds_min": [-0.5, -1.0, -1.5],
        "bounds_max": [0.5, 1.0, 1.5],
        "dimensions": [1.0, 2.0, 3.0],
        "units": "GLB scene units",
    }
    assert report["artifact"]["path"] == "<memory>"
    assert len(report["artifact"]["sha256"]) == 64
    assert "Status: PASS" in format_mesh_report(report)
    assert '"schema": "cfp.mesh-report.v1"' in mesh_report_json(report)


def test_mesh_report_reviews_open_mesh():
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )

    report = analyze_glb(_as_glb(mesh))

    assert report["status"] == "REVIEW"
    assert report["geometry"]["boundary_edges"] == 3
    assert report["geometry"]["non_manifold_edges"] == 0
    assert report["geometry"]["watertight"] is False


def test_mesh_report_fails_non_manifold_mesh():
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [1, 0, 3],
                [0, 1, 4],
            ]
        ),
        process=False,
    )

    report = analyze_glb(_as_glb(mesh))

    assert report["status"] == "FAIL"
    assert report["geometry"]["non_manifold_edges"] == 1
    assert "repair before print engineering" in format_mesh_report(report)
