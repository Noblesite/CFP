from __future__ import annotations

import json
from io import BytesIO
from typing import BinaryIO

import numpy as np

from .mesh_cleanup import _component_face_groups
from .mesh_report import _load_mesh, _read_artifact, analyze_glb


def inspect_background_geometry(
    source: str | BinaryIO,
    *,
    profile: str = "character_z_up",
    min_z_dominance_ratio: float = 1.25,
    planar_flatness_ratio: float = 0.02,
    large_planar_span_ratio: float = 0.4,
    acknowledge_suspicious_geometry: bool = False,
) -> tuple[bytes, dict[str, object]]:
    """Inspect geometry before voxel filling without modifying the artifact."""
    if profile not in {"character_z_up", "generic"}:
        raise ValueError("profile must be character_z_up or generic")
    if not 1.0 <= min_z_dominance_ratio <= 10.0:
        raise ValueError("min_z_dominance_ratio must be between 1 and 10")
    if not 0.0001 <= planar_flatness_ratio <= 0.25:
        raise ValueError("planar_flatness_ratio must be between 0.0001 and 0.25")
    if not 0.05 <= large_planar_span_ratio <= 1.0:
        raise ValueError("large_planar_span_ratio must be between 0.05 and 1.0")

    artifact_path, input_data = _read_artifact(source)
    mesh = _load_mesh(input_data)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    dimensions = np.asarray(mesh.extents, dtype=np.float64)
    max_extent = float(np.max(dimensions))
    lateral_extent = float(max(dimensions[0], dimensions[1], 1e-12))
    z_dominance_ratio = float(dimensions[2] / lateral_extent)

    groups = _component_face_groups(faces, len(vertices))
    planar_components: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        used_vertices = np.unique(faces[group].reshape(-1))
        component_extents = np.ptp(vertices[used_vertices], axis=0)
        component_max_extent = float(np.max(component_extents))
        flatness = float(
            np.min(component_extents) / max(component_max_extent, 1e-12)
        )
        span_ratio = float(component_max_extent / max(max_extent, 1e-12))
        if (
            flatness <= planar_flatness_ratio
            and span_ratio >= large_planar_span_ratio
        ):
            planar_components.append(
                {
                    "component_index": index,
                    "faces": int(len(group)),
                    "vertices": int(len(used_vertices)),
                    "extents": component_extents.tolist(),
                    "flatness_ratio": flatness,
                    "span_ratio": span_ratio,
                }
            )

    blocking_reasons: list[str] = []
    if profile == "character_z_up" and z_dominance_ratio < min_z_dominance_ratio:
        blocking_reasons.append("character_height_not_dominant")

    warnings: list[str] = []
    if planar_components:
        warnings.append("large_planar_components_present")
    if len(groups) > 1:
        warnings.append("disconnected_surface_components_present")

    suspicious = bool(blocking_reasons)
    if suspicious and acknowledge_suspicious_geometry:
        status = "GUARD_ACKNOWLEDGED"
        proceed_allowed = True
    elif suspicious:
        status = "GUARD_BLOCKED"
        proceed_allowed = False
    else:
        status = "GUARD_PASS"
        proceed_allowed = True

    mesh_report = analyze_glb(BytesIO(input_data))
    report: dict[str, object] = {
        "schema": "cfp.background-geometry-guard.v1",
        "status": status,
        "source_path": artifact_path,
        "configuration": {
            "profile": profile,
            "min_z_dominance_ratio": min_z_dominance_ratio,
            "planar_flatness_ratio": planar_flatness_ratio,
            "large_planar_span_ratio": large_planar_span_ratio,
            "acknowledge_suspicious_geometry": acknowledge_suspicious_geometry,
        },
        "bounds": {
            "dimensions": dimensions.tolist(),
            "z_dominance_ratio": z_dominance_ratio,
            "character_height_dominant": (
                z_dominance_ratio >= min_z_dominance_ratio
            ),
        },
        "components": {
            "count": len(groups),
            "large_planar_component_count": len(planar_components),
            "large_planar_components": planar_components[:25],
            "note": (
                "TRELLIS O-Voxel surfaces commonly contain many thin disconnected patches. "
                "Planar-component findings are evidence for review and are never auto-deleted."
            ),
        },
        "decision": {
            "suspicious": suspicious,
            "proceed_allowed": proceed_allowed,
            "blocking_reasons": blocking_reasons,
            "warnings": warnings,
        },
        "mesh_report": mesh_report,
        "limitations": [
            "The source GLB is returned byte-for-byte unchanged.",
            "The character profile checks bounding proportions, not semantic identity.",
            "Large planar components are reported but never removed automatically.",
            "Use the generic profile for valid cubic or horizontal props.",
            "Upstream image matting is preferred over mesh-stage backdrop surgery.",
        ],
    }
    return input_data, report


def format_background_geometry_guard_report(report: dict[str, object]) -> str:
    bounds = report["bounds"]
    components = report["components"]
    decision = report["decision"]
    dimensions = " × ".join(f"{value:.6g}" for value in bounds["dimensions"])
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Background Geometry Guard",
            "=" * 49,
            f"Status: {report['status']}",
            "",
            f"Profile: {report['configuration']['profile']}",
            f"Dimensions: {dimensions} (GLB scene units)",
            f"Z-height dominance: {bounds['z_dominance_ratio']:.3f}",
            f"Required dominance: "
            f"{report['configuration']['min_z_dominance_ratio']:.3f}",
            "",
            f"Connected surface components: {components['count']:,}",
            f"Large planar component candidates: "
            f"{components['large_planar_component_count']:,}",
            f"Warnings: {', '.join(decision['warnings']) or 'none'}",
            f"Blocking reasons: "
            f"{', '.join(decision['blocking_reasons']) or 'none'}",
            "",
            f"Proceed to voxel fill: {'Yes' if decision['proceed_allowed'] else 'No'}",
            (
                "Set acknowledgement to yes only after visually confirming that the "
                "reported bounds are intentional."
                if not decision["proceed_allowed"]
                else "The source geometry is passed through unchanged."
            ),
        ]
    )


def background_geometry_guard_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "background_geometry_guard_report_json",
    "format_background_geometry_guard_report",
    "inspect_background_geometry",
]
