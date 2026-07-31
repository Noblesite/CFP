from __future__ import annotations

import json
from typing import BinaryIO

import numpy as np

from .mesh_cleanup import _component_face_groups
from .mesh_report import _load_mesh, _read_artifact


def _edge_incidence(faces: np.ndarray) -> dict[str, int]:
    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "unique_edges": int(len(counts)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "manifold_edges": int(np.count_nonzero(counts == 2)),
        "overloaded_edges": int(np.count_nonzero(counts > 2)),
        "max_face_incidence": int(counts.max(initial=0)),
    }


def _component_summary(
    vertices: np.ndarray,
    faces: np.ndarray,
    groups: list[np.ndarray],
    *,
    tolerance: float,
    pair_limit: int = 100,
) -> dict[str, object]:
    face_counts = [int(len(group)) for group in groups]
    inspected = groups[:pair_limit]
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    for group in inspected:
        used = np.unique(faces[group].reshape(-1))
        component_vertices = vertices[used]
        bounds.append((component_vertices.min(axis=0), component_vertices.max(axis=0)))

    overlap_candidates: list[list[int]] = []
    for left in range(len(bounds)):
        left_min, left_max = bounds[left]
        for right in range(left + 1, len(bounds)):
            right_min, right_max = bounds[right]
            if np.all(left_max + tolerance >= right_min) and np.all(
                right_max + tolerance >= left_min
            ):
                overlap_candidates.append([left, right])

    return {
        "count": len(groups),
        "face_counts_descending": face_counts[:25],
        "bbox_pair_search_components": len(inspected),
        "bbox_overlap_candidate_count": len(overlap_candidates),
        "bbox_overlap_candidate_pairs": overlap_candidates[:50],
        "bbox_overlap_note": (
            "Axis-aligned bounding-box overlap is a broad-phase candidate, not proof of "
            "triangle intersection."
        ),
    }


def diagnose_ovoxel_topology(
    source: str | BinaryIO,
    *,
    coordinate_tolerance_ratio: float = 1e-6,
) -> dict[str, object]:
    if not 0.0 < coordinate_tolerance_ratio <= 0.01:
        raise ValueError("coordinate_tolerance_ratio must be greater than 0 and at most 0.01")

    artifact_path, data = _read_artifact(source)
    mesh = _load_mesh(data)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    coordinate_tolerance = max(diagonal * coordinate_tolerance_ratio, 1e-12)
    area_tolerance = coordinate_tolerance * coordinate_tolerance

    canonical_faces = np.sort(faces, axis=1)
    _, first_indices, face_counts = np.unique(
        canonical_faces,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    duplicate_face_groups = int(np.count_nonzero(face_counts > 1))
    duplicate_face_extras = int(np.sum(np.maximum(face_counts - 1, 0)))
    keep_unique = np.zeros(len(faces), dtype=bool)
    keep_unique[first_indices] = True

    repeated_vertex_faces = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    triangle_cross = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    double_areas = np.linalg.norm(triangle_cross, axis=1)
    degenerate_faces_mask = repeated_vertex_faces | (double_areas <= area_tolerance)
    degenerate_faces = int(np.count_nonzero(degenerate_faces_mask))

    quantized = np.rint(vertices / coordinate_tolerance).astype(np.int64)
    _, coordinate_counts = np.unique(quantized, axis=0, return_counts=True)
    coincident_vertex_groups = int(np.count_nonzero(coordinate_counts > 1))
    coincident_vertex_extras = int(np.sum(np.maximum(coordinate_counts - 1, 0)))

    base_edges = _edge_incidence(faces)
    deduplicated_edges = _edge_incidence(faces[keep_unique])
    nondegenerate_edges = _edge_incidence(faces[~degenerate_faces_mask])
    sanitized_mask = keep_unique & ~degenerate_faces_mask
    sanitized_edges = _edge_incidence(faces[sanitized_mask])

    groups = _component_face_groups(faces, len(vertices))
    components = _component_summary(
        vertices,
        faces,
        groups,
        tolerance=coordinate_tolerance,
    )

    confirmed_causes: list[str] = []
    if duplicate_face_extras:
        confirmed_causes.append("duplicate_faces")
    if degenerate_faces:
        confirmed_causes.append("degenerate_faces")
    if coincident_vertex_extras:
        confirmed_causes.append("coincident_unwelded_vertices")
    if base_edges["boundary_edges"]:
        confirmed_causes.append("open_boundaries")
    if base_edges["overloaded_edges"]:
        confirmed_causes.append("edges_shared_by_more_than_two_faces")

    residual_overloaded = sanitized_edges["overloaded_edges"]
    candidate_causes: list[str] = []
    if components["bbox_overlap_candidate_count"]:
        candidate_causes.append("overlapping_disconnected_shells")
    if residual_overloaded:
        candidate_causes.append("ovoxel_shell_junction_or_self_intersection")

    if base_edges["overloaded_edges"]:
        status = "FAIL"
    elif confirmed_causes or candidate_causes or components["count"] != 1:
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "schema": "cfp.ovoxel-topology-diagnostics.v1",
        "status": status,
        "artifact": {
            "path": artifact_path,
            "bytes": len(data),
        },
        "tolerances": {
            "coordinate_tolerance_ratio": coordinate_tolerance_ratio,
            "coordinate_tolerance": coordinate_tolerance,
            "degenerate_double_area_tolerance": area_tolerance,
        },
        "confirmed": {
            "duplicate_face_groups": duplicate_face_groups,
            "duplicate_face_extras": duplicate_face_extras,
            "degenerate_faces": degenerate_faces,
            "coincident_vertex_groups": coincident_vertex_groups,
            "coincident_vertex_extras": coincident_vertex_extras,
            "edge_incidence": base_edges,
        },
        "attribution_probes": {
            "after_deduplicating_faces": deduplicated_edges,
            "after_dropping_degenerate_faces": nondegenerate_edges,
            "after_both": sanitized_edges,
            "residual_overloaded_edges": residual_overloaded,
            "note": (
                "These probes are analysis-only. They show whether duplicate or degenerate faces "
                "explain overloaded edges without modifying the artifact."
            ),
        },
        "components": components,
        "classification": {
            "confirmed_causes": confirmed_causes,
            "candidate_causes": candidate_causes,
            "exact_self_intersection_test": {
                "status": "NOT_RUN",
                "reason": (
                    "Exact triangle-triangle intersection requires a robust optional collision "
                    "backend. Residual overloaded edges and component AABB overlaps remain "
                    "candidates, not proof."
                ),
            },
        },
    }


def format_topology_diagnostics(report: dict[str, object]) -> str:
    confirmed = report["confirmed"]
    edges = confirmed["edge_incidence"]
    probes = report["attribution_probes"]
    components = report["components"]
    classification = report["classification"]

    confirmed_causes = ", ".join(classification["confirmed_causes"]) or "none"
    candidate_causes = ", ".join(classification["candidate_causes"]) or "none"
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX O-Voxel Topology Diagnostics",
            "=" * 50,
            f"Status: {report['status']}",
            "",
            "Confirmed counts:",
            f"Duplicate face extras: {confirmed['duplicate_face_extras']:,}",
            f"Degenerate faces: {confirmed['degenerate_faces']:,}",
            f"Coincident unwelded vertices: {confirmed['coincident_vertex_extras']:,}",
            f"Boundary edges: {edges['boundary_edges']:,}",
            f"Edges shared by >2 faces: {edges['overloaded_edges']:,}",
            f"Maximum faces sharing one edge: {edges['max_face_incidence']:,}",
            "",
            "Analysis-only attribution:",
            f"Overloaded edges after deduplication: "
            f"{probes['after_deduplicating_faces']['overloaded_edges']:,}",
            f"Overloaded edges after dropping degenerates: "
            f"{probes['after_dropping_degenerate_faces']['overloaded_edges']:,}",
            f"Residual after both: {probes['residual_overloaded_edges']:,}",
            "",
            f"Connected components: {components['count']:,}",
            f"Component AABB overlap candidates: "
            f"{components['bbox_overlap_candidate_count']:,}",
            "",
            f"Confirmed causes: {confirmed_causes}",
            f"Candidate causes: {candidate_causes}",
            "",
            "Exact self-intersection test: NOT RUN",
            "This node is diagnostic only and does not modify the GLB.",
        ]
    )


def topology_diagnostics_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "diagnose_ovoxel_topology",
    "format_topology_diagnostics",
    "topology_diagnostics_json",
]
