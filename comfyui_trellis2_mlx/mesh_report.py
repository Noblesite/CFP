from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import trimesh


def _read_artifact(source: str | BinaryIO) -> tuple[str, bytes]:
    if isinstance(source, str):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"3D artifact does not exist: {path}")
        return str(path), path.read_bytes()

    if hasattr(source, "seek"):
        source.seek(0)
    data = source.read()
    if hasattr(source, "seek"):
        source.seek(0)
    return "<memory>", data


def _load_mesh(data: bytes) -> trimesh.Trimesh:
    loaded = trimesh.load(
        BytesIO(data),
        file_type="glb",
        force="scene",
        process=False,
    )
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("GLB contains no mesh geometry")
        mesh = loaded.to_geometry()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported GLB payload type: {type(loaded).__name__}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("GLB contains no triangle mesh")
    return mesh


def _status_for(*, components: int, boundary_edges: int, non_manifold_edges: int) -> str:
    if non_manifold_edges > 0:
        return "FAIL"
    if components != 1 or boundary_edges > 0:
        return "REVIEW"
    return "PASS"


def _connected_components(faces: np.ndarray, vertex_count: int) -> int:
    """Count face-connected bodies without requiring SciPy."""
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

    used_vertices: set[int] = set()
    for first, second, third in faces:
        first = int(first)
        second = int(second)
        third = int(third)
        used_vertices.update((first, second, third))
        union(first, second)
        union(first, third)

    return len({find(vertex) for vertex in used_vertices})


def analyze_glb(source: str | BinaryIO) -> dict[str, object]:
    artifact_path, data = _read_artifact(source)
    mesh = _load_mesh(data)

    edge_face_counts = np.bincount(
        mesh.edges_unique_inverse,
        minlength=len(mesh.edges_unique),
    )
    boundary_edges = int(np.count_nonzero(edge_face_counts == 1))
    non_manifold_edges = int(np.count_nonzero(edge_face_counts > 2))
    components = _connected_components(mesh.faces, len(mesh.vertices))
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    dimensions = np.asarray(mesh.extents, dtype=np.float64)
    watertight = bool(mesh.is_watertight)
    status = _status_for(
        components=components,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
    )

    return {
        "schema": "cfp.mesh-report.v1",
        "status": status,
        "artifact": {
            "path": artifact_path,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "geometry": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "connected_components": components,
            "unique_edges": int(len(mesh.edges_unique)),
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold_edges,
            "watertight": watertight,
            "bounds_min": bounds[0].tolist(),
            "bounds_max": bounds[1].tolist(),
            "dimensions": dimensions.tolist(),
            "units": "GLB scene units",
        },
    }


def format_mesh_report(report: dict[str, object]) -> str:
    artifact = report["artifact"]
    geometry = report["geometry"]
    status = report["status"]
    status_note = {
        "PASS": "Closed, single-component manifold topology.",
        "REVIEW": "Usable geometry, but inspect open boundaries or disconnected components.",
        "FAIL": "Non-manifold topology detected; repair before print engineering.",
    }[status]
    dimensions = " × ".join(f"{value:.6g}" for value in geometry["dimensions"])

    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Mesh Report",
            "=" * 34,
            f"Status: {status} — {status_note}",
            "",
            f"Vertices: {geometry['vertices']:,}",
            f"Triangles: {geometry['triangles']:,}",
            f"Connected components: {geometry['connected_components']:,}",
            f"Boundary edges: {geometry['boundary_edges']:,}",
            f"Non-manifold edges: {geometry['non_manifold_edges']:,}",
            f"Watertight: {'Yes' if geometry['watertight'] else 'No'}",
            f"Dimensions: {dimensions} ({geometry['units']})",
            "",
            f"Artifact: {artifact['path']}",
            f"SHA-256: {artifact['sha256']}",
        ]
    )


def mesh_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = ["analyze_glb", "format_mesh_report", "mesh_report_json"]
