from __future__ import annotations

import json
import math
from io import BytesIO
from typing import BinaryIO

import numpy as np
import trimesh

from .mesh_report import _load_mesh, _read_artifact, analyze_glb


def _component_face_groups(faces: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    """Return deterministic vertex-connected face groups without requiring SciPy."""
    parent = list(range(vertex_count))
    rank = [0] * vertex_count

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for first, second, third in faces:
        union(int(first), int(second))
        union(int(first), int(third))

    grouped: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        root = find(int(face[0]))
        grouped.setdefault(root, []).append(face_index)

    groups = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]
    return sorted(groups, key=lambda indices: (-len(indices), int(indices[0])))


def _submesh(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    faces = np.asarray(mesh.faces[face_indices], dtype=np.int64)
    used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = np.asarray(mesh.vertices[used_vertices], dtype=np.float64)
    remapped_faces = inverse.reshape((-1, 3))
    return trimesh.Trimesh(vertices=vertices, faces=remapped_faces, process=False)


def remove_small_components_glb(
    source: str | BinaryIO,
    *,
    min_component_faces: int = 100,
    min_component_ratio: float = 0.001,
) -> tuple[bytes, dict[str, object]]:
    """Remove small disconnected bodies while always preserving the largest body.

    A component is retained when it meets both configured thresholds. The largest component is
    retained unconditionally so aggressive settings can never return an empty artifact.
    """
    if min_component_faces < 0:
        raise ValueError("min_component_faces must be zero or greater")
    if not 0.0 <= min_component_ratio <= 1.0:
        raise ValueError("min_component_ratio must be between 0.0 and 1.0")

    artifact_path, input_data = _read_artifact(source)
    mesh = _load_mesh(input_data)
    groups = _component_face_groups(np.asarray(mesh.faces), len(mesh.vertices))
    largest_faces = len(groups[0])
    effective_floor = max(
        min_component_faces,
        int(math.ceil(largest_faces * min_component_ratio)),
    )

    kept_groups = [
        group
        for index, group in enumerate(groups)
        if index == 0 or len(group) >= effective_floor
    ]
    removed_groups = [
        group
        for index, group in enumerate(groups)
        if index != 0 and len(group) < effective_floor
    ]
    kept_face_indices = np.concatenate(kept_groups)
    cleaned_mesh = _submesh(mesh, kept_face_indices)
    output_data = cleaned_mesh.export(file_type="glb")
    if not isinstance(output_data, bytes):
        output_data = bytes(output_data)

    before = analyze_glb(source)
    after = analyze_glb(BytesIO(output_data))
    removed_faces = sum(len(group) for group in removed_groups)
    report: dict[str, object] = {
        "schema": "cfp.remove-floaters.v1",
        "status": "CHANGED" if removed_groups else "UNCHANGED",
        "source_path": artifact_path,
        "thresholds": {
            "min_component_faces": min_component_faces,
            "min_component_ratio": min_component_ratio,
            "largest_component_faces": largest_faces,
            "effective_face_floor": effective_floor,
        },
        "changes": {
            "components_before": len(groups),
            "components_after": len(kept_groups),
            "components_removed": len(removed_groups),
            "faces_removed": removed_faces,
            "faces_preserved": int(len(cleaned_mesh.faces)),
        },
        "before": before,
        "after": after,
    }
    return output_data, report


def format_cleanup_report(report: dict[str, object]) -> str:
    thresholds = report["thresholds"]
    changes = report["changes"]
    after = report["after"]["geometry"]
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Remove Floaters",
            "=" * 38,
            f"Status: {report['status']}",
            "",
            f"Components: {changes['components_before']:,} → {changes['components_after']:,}",
            f"Components removed: {changes['components_removed']:,}",
            f"Faces removed: {changes['faces_removed']:,}",
            f"Faces preserved: {changes['faces_preserved']:,}",
            f"Effective face floor: {thresholds['effective_face_floor']:,}",
            "",
            "After cleanup:",
            f"Boundary edges: {after['boundary_edges']:,}",
            f"Non-manifold edges: {after['non_manifold_edges']:,}",
            f"Watertight: {'Yes' if after['watertight'] else 'No'}",
            "",
            "This operation removes disconnected bodies only; it does not repair O-Voxel",
            "self-intersections, overlapping shells, open boundaries, or non-manifold edges.",
        ]
    )


def cleanup_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "cleanup_report_json",
    "format_cleanup_report",
    "remove_small_components_glb",
]
