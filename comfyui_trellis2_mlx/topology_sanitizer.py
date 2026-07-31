from __future__ import annotations

import json
from io import BytesIO
from typing import BinaryIO

import numpy as np
import trimesh

from .mesh_report import _load_mesh, _read_artifact
from .topology_diagnostics import diagnose_ovoxel_topology


def sanitize_topology_glb(
    source: str | BinaryIO,
    *,
    weld_tolerance_ratio: float = 1e-8,
) -> tuple[bytes, dict[str, object]]:
    """Apply deterministic face-list sanitation without remeshing the surface."""
    if not 0.0 < weld_tolerance_ratio <= 0.001:
        raise ValueError("weld_tolerance_ratio must be greater than 0 and at most 0.001")

    artifact_path, input_data = _read_artifact(source)
    mesh = _load_mesh(input_data)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    weld_tolerance = max(diagonal * weld_tolerance_ratio, 1e-12)
    area_tolerance = weld_tolerance * weld_tolerance

    quantized = np.rint(vertices / weld_tolerance).astype(np.int64)
    _, representative_indices, old_to_welded = np.unique(
        quantized,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    welded_vertices = vertices[representative_indices]
    remapped_faces = old_to_welded[faces]
    welded_vertex_count = int(len(vertices) - len(welded_vertices))

    repeated_vertex_faces = (
        (remapped_faces[:, 0] == remapped_faces[:, 1])
        | (remapped_faces[:, 1] == remapped_faces[:, 2])
        | (remapped_faces[:, 2] == remapped_faces[:, 0])
    )
    triangle_cross = np.cross(
        welded_vertices[remapped_faces[:, 1]] - welded_vertices[remapped_faces[:, 0]],
        welded_vertices[remapped_faces[:, 2]] - welded_vertices[remapped_faces[:, 0]],
    )
    double_areas = np.linalg.norm(triangle_cross, axis=1)
    degenerate_mask = repeated_vertex_faces | (double_areas <= area_tolerance)
    degenerate_faces_removed = int(np.count_nonzero(degenerate_mask))
    nondegenerate_faces = remapped_faces[~degenerate_mask]
    if len(nondegenerate_faces) == 0:
        raise ValueError("Topology sanitation would remove every face")

    canonical_faces = np.sort(nondegenerate_faces, axis=1)
    _, first_face_indices = np.unique(
        canonical_faces,
        axis=0,
        return_index=True,
    )
    first_face_indices.sort()
    sanitized_faces = nondegenerate_faces[first_face_indices]
    duplicate_faces_removed = int(len(nondegenerate_faces) - len(sanitized_faces))

    used_vertices, compact_inverse = np.unique(
        sanitized_faces.reshape(-1),
        return_inverse=True,
    )
    compact_vertices = welded_vertices[used_vertices]
    compact_faces = compact_inverse.reshape((-1, 3))
    unreferenced_vertices_removed = int(len(welded_vertices) - len(compact_vertices))

    sanitized_mesh = trimesh.Trimesh(
        vertices=compact_vertices,
        faces=compact_faces,
        process=False,
    )
    output_data = sanitized_mesh.export(file_type="glb")
    if not isinstance(output_data, bytes):
        output_data = bytes(output_data)

    before = diagnose_ovoxel_topology(
        BytesIO(input_data),
        coordinate_tolerance_ratio=max(weld_tolerance_ratio, 1e-9),
    )
    after = diagnose_ovoxel_topology(
        BytesIO(output_data),
        coordinate_tolerance_ratio=max(weld_tolerance_ratio, 1e-9),
    )
    changed = (
        welded_vertex_count
        + degenerate_faces_removed
        + duplicate_faces_removed
        + unreferenced_vertices_removed
        > 0
    )
    before_edges = before["confirmed"]["edge_incidence"]
    after_edges = after["confirmed"]["edge_incidence"]
    topology_regressions: list[str] = []
    if after_edges["boundary_edges"] > before_edges["boundary_edges"]:
        topology_regressions.append("boundary_edges_increased")
    if after_edges["overloaded_edges"] > before_edges["overloaded_edges"]:
        topology_regressions.append("overloaded_edges_increased")
    status = (
        "CHANGED_REVIEW"
        if changed and topology_regressions
        else "CHANGED"
        if changed
        else "UNCHANGED"
    )
    report: dict[str, object] = {
        "schema": "cfp.topology-sanitizer.v1",
        "status": status,
        "source_path": artifact_path,
        "tolerances": {
            "weld_tolerance_ratio": weld_tolerance_ratio,
            "weld_tolerance": weld_tolerance,
            "degenerate_double_area_tolerance": area_tolerance,
        },
        "changes": {
            "vertices_before": int(len(vertices)),
            "vertices_after": int(len(compact_vertices)),
            "vertices_welded": welded_vertex_count,
            "unreferenced_vertices_removed": unreferenced_vertices_removed,
            "faces_before": int(len(faces)),
            "faces_after": int(len(compact_faces)),
            "degenerate_faces_removed": degenerate_faces_removed,
            "duplicate_faces_removed": duplicate_faces_removed,
            "topology_regressions": topology_regressions,
        },
        "before": before,
        "after": after,
        "limitations": [
            "No remeshing or hole filling is performed.",
            "No exact triangle-triangle self-intersection test is performed.",
            "Residual O-Voxel shell junctions and overloaded edges are preserved for review.",
            (
                "Removing stacked duplicate sheets can reveal real open boundaries that their "
                "overlap previously concealed."
            ),
            "The current export contains geometry and normals; use this on geometry-only GLBs.",
        ],
    }
    return output_data, report


def format_sanitizer_report(report: dict[str, object]) -> str:
    changes = report["changes"]
    before_edges = report["before"]["confirmed"]["edge_incidence"]
    after_edges = report["after"]["confirmed"]["edge_incidence"]
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Topology Sanitizer",
            "=" * 42,
            f"Status: {report['status']}",
            "",
            f"Vertices: {changes['vertices_before']:,} → {changes['vertices_after']:,}",
            f"Vertices welded: {changes['vertices_welded']:,}",
            f"Unreferenced vertices removed: "
            f"{changes['unreferenced_vertices_removed']:,}",
            f"Faces: {changes['faces_before']:,} → {changes['faces_after']:,}",
            f"Duplicate faces removed: {changes['duplicate_faces_removed']:,}",
            f"Degenerate faces removed: {changes['degenerate_faces_removed']:,}",
            "",
            f"Boundary edges: {before_edges['boundary_edges']:,} → "
            f"{after_edges['boundary_edges']:,}",
            f"Edges shared by >2 faces: {before_edges['overloaded_edges']:,} → "
            f"{after_edges['overloaded_edges']:,}",
            f"Review flags: "
            f"{', '.join(changes['topology_regressions']) or 'none'}",
            "",
            "This sanitizer does not remesh, fill holes, or resolve O-Voxel shell",
            "junctions and self-intersections.",
        ]
    )


def sanitizer_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "format_sanitizer_report",
    "sanitize_topology_glb",
    "sanitizer_report_json",
]
