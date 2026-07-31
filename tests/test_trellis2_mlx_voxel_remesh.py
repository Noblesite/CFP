from __future__ import annotations

from io import BytesIO

import pytest
import trimesh

from comfyui_trellis2_mlx.mesh_report import analyze_glb
from comfyui_trellis2_mlx.voxel_remesh import (
    format_voxel_remesh_report,
    voxel_remesh_candidate_glb,
    voxel_remesh_report_json,
)


def _as_glb(mesh: trimesh.Trimesh) -> BytesIO:
    return BytesIO(mesh.export(file_type="glb"))


def test_voxel_remesh_creates_watertight_candidate_and_reports_shape_change():
    output, report = voxel_remesh_candidate_glb(
        _as_glb(trimesh.creation.box(extents=[1.0, 2.0, 3.0])),
        target_resolution=48,
    )
    candidate = analyze_glb(BytesIO(output))

    assert report["status"] == "CANDIDATE_PASS"
    assert candidate["geometry"]["watertight"] is True
    assert candidate["geometry"]["boundary_edges"] == 0
    assert candidate["geometry"]["non_manifold_edges"] == 0
    assert report["configuration"]["target_resolution"] == 48
    assert report["configuration"]["pitch"] == pytest.approx(3.0 / 48.0)
    assert report["shape_comparison"]["nearest_vertex_deviation"]["p95"] >= 0
    assert "Watertight: Yes" in format_voxel_remesh_report(report)
    assert '"schema": "cfp.voxel-remesh-candidate.v1"' in voxel_remesh_report_json(
        report
    )


def test_voxel_remesh_preserves_source_artifact_as_separate_candidate():
    source = _as_glb(trimesh.creation.icosphere(subdivisions=1))
    source_bytes = source.getvalue()

    output, report = voxel_remesh_candidate_glb(source, target_resolution=40)

    assert source.getvalue() == source_bytes
    assert output != source_bytes
    assert report["limitations"][0].startswith("This is a separate candidate")


@pytest.mark.parametrize("resolution", [31, 513])
def test_voxel_remesh_rejects_resolution_outside_guardrails(resolution):
    with pytest.raises(ValueError, match="target_resolution"):
        voxel_remesh_candidate_glb(
            _as_glb(trimesh.creation.box()),
            target_resolution=resolution,
        )
