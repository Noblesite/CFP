from __future__ import annotations

import json
from io import BytesIO
from typing import BinaryIO

import numpy as np
import trimesh

from .mesh_report import _load_mesh, _read_artifact, analyze_glb


_SHADING_MODES = ("smooth", "angle", "flat")


def _face_smoothing_groups(mesh: trimesh.Trimesh, angle_degrees: float) -> np.ndarray:
    """Return union-find face groups joined across sufficiently shallow edges."""
    face_count = len(mesh.faces)
    parent = np.arange(face_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    threshold = np.deg2rad(angle_degrees)
    for pair, angle in zip(mesh.face_adjacency, mesh.face_adjacency_angles):
        if float(angle) <= threshold:
            union(int(pair[0]), int(pair[1]))

    return np.asarray([find(index) for index in range(face_count)], dtype=np.int64)


def _angle_split_mesh(
    mesh: trimesh.Trimesh,
    angle_degrees: float,
) -> trimesh.Trimesh:
    groups = _face_smoothing_groups(mesh, angle_degrees)
    corner_vertices = np.asarray(mesh.faces, dtype=np.int64).reshape(-1)
    corner_groups = np.repeat(groups, 3)
    corner_keys = np.column_stack((corner_vertices, corner_groups))
    unique_keys, inverse = np.unique(corner_keys, axis=0, return_inverse=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[unique_keys[:, 0]]
    faces = inverse.reshape((-1, 3))
    candidate = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    candidate.vertex_normals = trimesh.geometry.weighted_vertex_normals(
        len(candidate.vertices),
        candidate.faces,
        candidate.face_normals,
        candidate.face_angles,
    )
    return candidate


def _smooth_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    candidate = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=False,
    )
    candidate.vertex_normals = trimesh.geometry.weighted_vertex_normals(
        len(candidate.vertices),
        candidate.faces,
        candidate.face_normals,
        candidate.face_angles,
    )
    return candidate


def shade_surface_candidate_glb(
    source: str | BinaryIO,
    *,
    mode: str = "smooth",
    angle_degrees: float = 30.0,
) -> tuple[bytes, dict[str, object]]:
    """Create a display-only GLB with explicit surface normals and unchanged positions."""
    normalized_mode = mode.lower()
    if normalized_mode not in _SHADING_MODES:
        raise ValueError(f"mode must be one of: {', '.join(_SHADING_MODES)}")
    if not 0.0 <= angle_degrees <= 180.0:
        raise ValueError("angle_degrees must be between 0 and 180")

    artifact_path, input_data = _read_artifact(source)
    source_mesh = _load_mesh(input_data)
    before = analyze_glb(BytesIO(input_data))

    if normalized_mode == "flat":
        output_data = input_data
        candidate_mesh = source_mesh
        status = "UNCHANGED_FLAT"
    elif normalized_mode == "angle":
        candidate_mesh = _angle_split_mesh(source_mesh, angle_degrees)
        output_data = candidate_mesh.export(file_type="glb")
        status = "SHADING_CANDIDATE_REVIEW"
    else:
        candidate_mesh = _smooth_mesh(source_mesh)
        output_data = candidate_mesh.export(file_type="glb")
        status = "SHADING_CANDIDATE_PASS"

    if not isinstance(output_data, bytes):
        output_data = bytes(output_data)
    after = analyze_glb(BytesIO(output_data))

    source_bounds = np.asarray(source_mesh.bounds, dtype=np.float64)
    candidate_bounds = np.asarray(candidate_mesh.bounds, dtype=np.float64)
    max_bounds_delta = float(np.max(np.abs(candidate_bounds - source_bounds)))
    surface_geometry_preserved = (
        before["geometry"]["triangles"] == after["geometry"]["triangles"]
        and max_bounds_delta <= 1e-12
    )

    report: dict[str, object] = {
        "schema": "cfp.surface-shading-candidate.v1",
        "status": status,
        "source_path": artifact_path,
        "configuration": {
            "mode": normalized_mode,
            "angle_degrees": angle_degrees,
        },
        "changes": {
            "vertex_positions_moved": False,
            "surface_geometry_preserved": surface_geometry_preserved,
            "triangles_before": before["geometry"]["triangles"],
            "triangles_after": after["geometry"]["triangles"],
            "vertices_before": before["geometry"]["vertices"],
            "vertices_after": after["geometry"]["vertices"],
            "max_bounds_delta": max_bounds_delta,
            "angle_mode_splits_vertices_at_hard_edges": normalized_mode == "angle",
        },
        "before": before,
        "after": after,
        "limitations": [
            "This candidate changes display normals only; it does not remove voxel stair-stepping.",
            "Angle mode duplicates vertices at hard shading boundaries because glTF normals are per vertex.",
            "Use the unchanged manufacturing artifact for topology diagnostics and print engineering.",
            "Materials and learned textures are not preserved by this geometry-only candidate.",
        ],
    }
    return output_data, report


def format_surface_shading_report(report: dict[str, object]) -> str:
    config = report["configuration"]
    changes = report["changes"]
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Surface Shading Candidate",
            "=" * 47,
            f"Status: {report['status']}",
            f"Mode: {config['mode']}",
            f"Angle threshold: {config['angle_degrees']:.1f}°",
            "",
            f"Vertex positions moved: {'Yes' if changes['vertex_positions_moved'] else 'No'}",
            f"Surface geometry preserved: "
            f"{'Yes' if changes['surface_geometry_preserved'] else 'No'}",
            f"Vertices: {changes['vertices_before']:,} → {changes['vertices_after']:,}",
            f"Triangles: {changes['triangles_before']:,} → {changes['triangles_after']:,}",
            "",
            "Display candidate only: compare shading before promotion.",
            "Voxel silhouette and print geometry are unchanged.",
        ]
    )


def surface_shading_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "format_surface_shading_report",
    "shade_surface_candidate_glb",
    "surface_shading_report_json",
]
