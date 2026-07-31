from __future__ import annotations

import json
from io import BytesIO
from typing import BinaryIO

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .mesh_report import _load_mesh, _read_artifact, analyze_glb
from .topology_diagnostics import diagnose_ovoxel_topology


def _deterministic_sample(points: np.ndarray, limit: int = 20_000) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
    return points[indices]


def _nearest_vertex_metrics(
    source_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    *,
    diagonal: float,
) -> dict[str, object]:
    source_sample = _deterministic_sample(source_vertices)
    candidate_sample = _deterministic_sample(candidate_vertices)
    source_to_candidate = cKDTree(candidate_vertices).query(
        source_sample,
        workers=-1,
    )[0]
    candidate_to_source = cKDTree(source_vertices).query(
        candidate_sample,
        workers=-1,
    )[0]
    combined = np.concatenate((source_to_candidate, candidate_to_source))
    safe_diagonal = max(diagonal, 1e-12)
    return {
        "method": "deterministic_bidirectional_nearest_vertex",
        "source_samples": int(len(source_sample)),
        "candidate_samples": int(len(candidate_sample)),
        "mean": float(np.mean(combined)),
        "p95": float(np.percentile(combined, 95)),
        "max": float(np.max(combined)),
        "mean_relative_to_bbox_diagonal": float(np.mean(combined) / safe_diagonal),
        "p95_relative_to_bbox_diagonal": float(
            np.percentile(combined, 95) / safe_diagonal
        ),
        "note": (
            "Nearest-vertex distance is a deterministic detail-loss proxy, not an exact "
            "point-to-surface Hausdorff measurement."
        ),
    }


def voxel_remesh_candidate_glb(
    source: str | BinaryIO,
    *,
    target_resolution: int = 192,
) -> tuple[bytes, dict[str, object]]:
    """Create a separate filled-voxel marching-cubes candidate."""
    if not 32 <= target_resolution <= 512:
        raise ValueError("target_resolution must be between 32 and 512")

    artifact_path, input_data = _read_artifact(source)
    source_mesh = _load_mesh(input_data)
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    extents = np.asarray(source_mesh.extents, dtype=np.float64)
    max_extent = float(np.max(extents))
    if max_extent <= 0:
        raise ValueError("Cannot voxel-remesh a zero-size mesh")

    pitch = max_extent / float(target_resolution)
    estimated_shape = np.ceil(extents / pitch).astype(np.int64) + 3
    estimated_voxels = int(np.prod(estimated_shape, dtype=np.int64))
    max_voxels = 512**3
    if estimated_voxels > max_voxels:
        raise ValueError(
            f"Requested voxel grid is too large: {estimated_voxels:,} > {max_voxels:,}"
        )

    voxel_grid = source_mesh.voxelized(pitch=pitch, method="subdivide")
    surface_voxels = int(voxel_grid.filled_count)
    filled_grid = voxel_grid.fill(method="holes")
    filled_voxels = int(filled_grid.filled_count)
    candidate_mesh = filled_grid.marching_cubes
    candidate_mesh.apply_transform(filled_grid.transform)
    candidate_mesh.remove_unreferenced_vertices()
    candidate_mesh.fix_normals(multibody=True)

    output_data = candidate_mesh.export(file_type="glb")
    if not isinstance(output_data, bytes):
        output_data = bytes(output_data)

    before_mesh_report = analyze_glb(BytesIO(input_data))
    after_mesh_report = analyze_glb(BytesIO(output_data))
    before_diagnostics = diagnose_ovoxel_topology(BytesIO(input_data))
    after_diagnostics = diagnose_ovoxel_topology(BytesIO(output_data))
    source_dimensions = np.asarray(
        before_mesh_report["geometry"]["dimensions"],
        dtype=np.float64,
    )
    candidate_dimensions = np.asarray(
        after_mesh_report["geometry"]["dimensions"],
        dtype=np.float64,
    )
    dimension_delta = candidate_dimensions - source_dimensions
    relative_dimension_delta = np.divide(
        dimension_delta,
        np.maximum(source_dimensions, 1e-12),
    )
    diagonal = float(np.linalg.norm(source_dimensions))
    deviation = _nearest_vertex_metrics(
        source_vertices,
        np.asarray(candidate_mesh.vertices, dtype=np.float64),
        diagonal=diagonal,
    )
    topology_pass = (
        after_mesh_report["geometry"]["watertight"]
        and after_mesh_report["geometry"]["connected_components"] == 1
        and after_mesh_report["geometry"]["boundary_edges"] == 0
        and after_mesh_report["geometry"]["non_manifold_edges"] == 0
    )

    report: dict[str, object] = {
        "schema": "cfp.voxel-remesh-candidate.v1",
        "status": "CANDIDATE_PASS" if topology_pass else "CANDIDATE_REVIEW",
        "source_path": artifact_path,
        "configuration": {
            "target_resolution": target_resolution,
            "pitch": pitch,
            "estimated_grid_shape": estimated_shape.tolist(),
            "estimated_grid_voxels": estimated_voxels,
            "surface_voxels": surface_voxels,
            "filled_voxels": filled_voxels,
        },
        "topology": {
            "before_mesh_report": before_mesh_report,
            "after_mesh_report": after_mesh_report,
            "before_diagnostics": before_diagnostics,
            "after_diagnostics": after_diagnostics,
        },
        "shape_comparison": {
            "source_dimensions": source_dimensions.tolist(),
            "candidate_dimensions": candidate_dimensions.tolist(),
            "dimension_delta": dimension_delta.tolist(),
            "relative_dimension_delta": relative_dimension_delta.tolist(),
            "max_absolute_relative_dimension_delta": float(
                np.max(np.abs(relative_dimension_delta))
            ),
            "nearest_vertex_deviation": deviation,
        },
        "limitations": [
            "This is a separate candidate; the source GLB is never overwritten.",
            "Voxel remeshing discards UVs, textures, and original topology.",
            "Thin features smaller than roughly one voxel pitch may be lost or fused.",
            "Visual comparison remains required before promotion.",
        ],
    }
    return output_data, report


def format_voxel_remesh_report(report: dict[str, object]) -> str:
    config = report["configuration"]
    topology = report["topology"]
    before = topology["before_mesh_report"]["geometry"]
    after = topology["after_mesh_report"]["geometry"]
    comparison = report["shape_comparison"]
    deviation = comparison["nearest_vertex_deviation"]
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Watertight Voxel Remesh Candidate",
            "=" * 57,
            f"Status: {report['status']}",
            "",
            f"Target resolution: {config['target_resolution']}",
            f"Voxel pitch: {config['pitch']:.8g}",
            f"Grid shape: {' × '.join(str(v) for v in config['estimated_grid_shape'])}",
            f"Filled voxels: {config['filled_voxels']:,}",
            "",
            f"Triangles: {before['triangles']:,} → {after['triangles']:,}",
            f"Connected components: {before['connected_components']:,} → "
            f"{after['connected_components']:,}",
            f"Boundary edges: {before['boundary_edges']:,} → {after['boundary_edges']:,}",
            f"Non-manifold edges: {before['non_manifold_edges']:,} → "
            f"{after['non_manifold_edges']:,}",
            f"Watertight: {'Yes' if after['watertight'] else 'No'}",
            "",
            f"Maximum relative dimension change: "
            f"{comparison['max_absolute_relative_dimension_delta'] * 100:.3f}%",
            f"Nearest-vertex mean deviation: {deviation['mean']:.8g}",
            f"Nearest-vertex p95 deviation: {deviation['p95']:.8g}",
            "",
            "Candidate only: compare surface detail before promotion.",
        ]
    )


def voxel_remesh_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "format_voxel_remesh_report",
    "voxel_remesh_candidate_glb",
    "voxel_remesh_report_json",
]
