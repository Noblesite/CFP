from __future__ import annotations

import json


def _candidate_summary(report_json: str) -> dict[str, object]:
    try:
        report = json.loads(report_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid voxel candidate report JSON: {error}") from error
    if report.get("schema") != "cfp.voxel-remesh-candidate.v1":
        raise ValueError("Expected cfp.voxel-remesh-candidate.v1 report")

    config = report["configuration"]
    topology = report["topology"]["after_mesh_report"]["geometry"]
    comparison = report["shape_comparison"]
    deviation = comparison["nearest_vertex_deviation"]
    return {
        "resolution": int(config["target_resolution"]),
        "status": report["status"],
        "pitch": float(config["pitch"]),
        "triangles": int(topology["triangles"]),
        "watertight": bool(topology["watertight"]),
        "boundary_edges": int(topology["boundary_edges"]),
        "non_manifold_edges": int(topology["non_manifold_edges"]),
        "max_relative_dimension_delta": float(
            comparison["max_absolute_relative_dimension_delta"]
        ),
        "mean_deviation": float(deviation["mean"]),
        "p95_deviation": float(deviation["p95"]),
    }


def _rank(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(values, key=lambda resolution: (values[resolution], resolution))
    return {resolution: index + 1 for index, resolution in enumerate(ordered)}


def compare_voxel_candidates(report_jsons: list[str]) -> dict[str, object]:
    if len(report_jsons) < 2:
        raise ValueError("At least two voxel candidate reports are required")
    candidates = sorted(
        (_candidate_summary(report_json) for report_json in report_jsons),
        key=lambda candidate: candidate["resolution"],
    )
    resolutions = [candidate["resolution"] for candidate in candidates]
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("Voxel candidate resolutions must be unique")

    passing = [
        candidate
        for candidate in candidates
        if candidate["status"] == "CANDIDATE_PASS"
    ]
    if len(passing) == len(candidates):
        status = "ALL_PASS"
    elif passing:
        status = "PARTIAL_PASS"
    else:
        status = "NO_PASS"

    recommendations: dict[str, object] = {
        "detail_priority": None,
        "dimension_priority": None,
        "lightweight_priority": None,
        "balanced_priority": None,
        "eligible_for_promotion": [
            candidate["resolution"] for candidate in passing
        ],
        "promotion": "HUMAN_REVIEW_REQUIRED",
    }
    if candidates:
        recommendations["detail_priority"] = min(
            candidates,
            key=lambda candidate: (
                candidate["p95_deviation"],
                candidate["resolution"],
            ),
        )["resolution"]
        recommendations["dimension_priority"] = min(
            candidates,
            key=lambda candidate: (
                candidate["max_relative_dimension_delta"],
                candidate["resolution"],
            ),
        )["resolution"]
        recommendations["lightweight_priority"] = min(
            candidates,
            key=lambda candidate: (
                candidate["triangles"],
                candidate["resolution"],
            ),
        )["resolution"]

        p95_ranks = _rank(
            {
                candidate["resolution"]: candidate["p95_deviation"]
                for candidate in candidates
            }
        )
        dimension_ranks = _rank(
            {
                candidate["resolution"]: candidate[
                    "max_relative_dimension_delta"
                ]
                for candidate in candidates
            }
        )
        triangle_ranks = _rank(
            {
                candidate["resolution"]: float(candidate["triangles"])
                for candidate in candidates
            }
        )
        balanced_scores = {
            candidate["resolution"]: (
                p95_ranks[candidate["resolution"]]
                + dimension_ranks[candidate["resolution"]]
                + triangle_ranks[candidate["resolution"]]
            )
            for candidate in candidates
        }
        recommendations["balanced_priority"] = min(
            balanced_scores,
            key=lambda resolution: (
                balanced_scores[resolution],
                resolution,
            ),
        )
        recommendations["balanced_rank_sum"] = balanced_scores
        recommendations["balanced_note"] = (
            "Equal rank weight is given to p95 deviation, maximum dimensional drift, and "
            "triangle count across every candidate. Topology eligibility is reported "
            "separately; this is a comparison aid, not automatic promotion."
        )

    return {
        "schema": "cfp.voxel-candidate-comparison.v1",
        "status": status,
        "candidates": candidates,
        "recommendations": recommendations,
    }


def format_voxel_comparison(report: dict[str, object]) -> str:
    lines = [
        "CFP TRELLIS.2 MLX Voxel Resolution A/B Comparison",
        "=" * 53,
        f"Status: {report['status']}",
        "",
        "Resolution | Topology | Triangles | Dim drift | P95 deviation",
        "-" * 65,
    ]
    for candidate in report["candidates"]:
        lines.append(
            f"{candidate['resolution']:>10} | "
            f"{candidate['status']:<14} | "
            f"{candidate['triangles']:>9,} | "
            f"{candidate['max_relative_dimension_delta'] * 100:>8.3f}% | "
            f"{candidate['p95_deviation']:.8g}"
        )

    recommendations = report["recommendations"]
    lines.extend(
        [
            "",
            f"Detail priority: {recommendations['detail_priority']}",
            f"Dimension priority: {recommendations['dimension_priority']}",
            f"Lightweight priority: {recommendations['lightweight_priority']}",
            f"Balanced rank priority: {recommendations['balanced_priority']}",
            "Topology-eligible resolutions: "
            + (
                ", ".join(
                    str(resolution)
                    for resolution in recommendations["eligible_for_promotion"]
                )
                or "none"
            ),
            "",
            "Promotion: HUMAN REVIEW REQUIRED",
        ]
    )
    return "\n".join(lines)


def voxel_comparison_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "compare_voxel_candidates",
    "format_voxel_comparison",
    "voxel_comparison_json",
]
