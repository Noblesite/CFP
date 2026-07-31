from __future__ import annotations

import json

import pytest

from comfyui_trellis2_mlx.voxel_comparison import (
    compare_voxel_candidates,
    format_voxel_comparison,
    voxel_comparison_json,
)


def _report(
    resolution: int,
    *,
    status: str = "CANDIDATE_PASS",
    triangles: int,
    dimension_delta: float,
    p95: float,
) -> str:
    return json.dumps(
        {
            "schema": "cfp.voxel-remesh-candidate.v1",
            "status": status,
            "configuration": {
                "target_resolution": resolution,
                "pitch": 1.0 / resolution,
            },
            "topology": {
                "after_mesh_report": {
                    "geometry": {
                        "triangles": triangles,
                        "watertight": status == "CANDIDATE_PASS",
                        "boundary_edges": 0 if status == "CANDIDATE_PASS" else 3,
                        "non_manifold_edges": 0,
                    }
                }
            },
            "shape_comparison": {
                "max_absolute_relative_dimension_delta": dimension_delta,
                "nearest_vertex_deviation": {
                    "mean": p95 / 2,
                    "p95": p95,
                },
            },
        }
    )


def test_comparison_keeps_recommendations_separate_by_priority():
    report = compare_voxel_candidates(
        [
            _report(128, triangles=100, dimension_delta=0.02, p95=0.03),
            _report(192, triangles=200, dimension_delta=0.01, p95=0.02),
            _report(256, triangles=300, dimension_delta=0.015, p95=0.01),
        ]
    )

    assert report["status"] == "ALL_PASS"
    assert report["recommendations"]["detail_priority"] == 256
    assert report["recommendations"]["dimension_priority"] == 192
    assert report["recommendations"]["lightweight_priority"] == 128
    assert report["recommendations"]["balanced_priority"] == 192
    assert report["recommendations"]["promotion"] == "HUMAN_REVIEW_REQUIRED"
    assert "Promotion: HUMAN REVIEW REQUIRED" in format_voxel_comparison(report)
    assert '"schema": "cfp.voxel-candidate-comparison.v1"' in voxel_comparison_json(
        report
    )


def test_comparison_separates_quality_ranking_from_topology_eligibility():
    report = compare_voxel_candidates(
        [
            _report(128, triangles=100, dimension_delta=0.02, p95=0.03),
            _report(
                192,
                status="CANDIDATE_REVIEW",
                triangles=200,
                dimension_delta=0.001,
                p95=0.001,
            ),
            _report(256, triangles=300, dimension_delta=0.01, p95=0.01),
        ]
    )

    assert report["status"] == "PARTIAL_PASS"
    assert report["recommendations"]["detail_priority"] == 192
    assert 192 in report["recommendations"]["balanced_rank_sum"]
    assert report["recommendations"]["eligible_for_promotion"] == [128, 256]
    formatted = format_voxel_comparison(report)
    assert "Topology-eligible resolutions: 128, 256" in formatted


def test_comparison_rejects_duplicate_resolutions():
    candidate = _report(128, triangles=100, dimension_delta=0.02, p95=0.03)

    with pytest.raises(ValueError, match="unique"):
        compare_voxel_candidates([candidate, candidate])


def test_comparison_rejects_wrong_schema():
    with pytest.raises(ValueError, match="Expected"):
        compare_voxel_candidates([json.dumps({"schema": "wrong"}), "{}"])
