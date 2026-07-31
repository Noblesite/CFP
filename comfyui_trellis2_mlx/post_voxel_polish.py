from __future__ import annotations

import json
from io import BytesIO
from typing import BinaryIO

import numpy as np
import trimesh

from .mesh_report import _load_mesh, _read_artifact, analyze_glb
from .topology_diagnostics import diagnose_ovoxel_topology


def _opposed_duplicate_groups(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> list[np.ndarray]:
    canonical_faces = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(
        canonical_faces,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    groups: list[np.ndarray] = []
    for group_id in np.flatnonzero(counts == 2):
        indices = np.flatnonzero(inverse == group_id)
        first, second = faces[indices]
        first_normal = np.cross(
            vertices[first[1]] - vertices[first[0]],
            vertices[first[2]] - vertices[first[0]],
        )
        second_normal = np.cross(
            vertices[second[1]] - vertices[second[0]],
            vertices[second[2]] - vertices[second[0]],
        )
        denominator = float(np.linalg.norm(first_normal) * np.linalg.norm(second_normal))
        if denominator <= 1e-24:
            continue
        cosine = float(np.dot(first_normal, second_normal) / denominator)
        if cosine < -0.999999:
            groups.append(indices)
    return groups


def polish_post_voxel_glb(
    source: str | BinaryIO,
) -> tuple[bytes, dict[str, object]]:
    """Remove opposed duplicate internal sheets when topology proves it is safe."""
    artifact_path, input_data = _read_artifact(source)
    mesh = _load_mesh(input_data)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    candidate_groups = _opposed_duplicate_groups(vertices, faces)

    before_mesh_report = analyze_glb(BytesIO(input_data))
    before_diagnostics = diagnose_ovoxel_topology(BytesIO(input_data))
    before_edges = before_diagnostics["confirmed"]["edge_incidence"]

    remove_mask = np.zeros(len(faces), dtype=bool)
    for indices in candidate_groups:
        remove_mask[indices] = True

    if candidate_groups:
        remaining_faces = faces[~remove_mask]
        used_vertices, compact_inverse = np.unique(
            remaining_faces.reshape(-1),
            return_inverse=True,
        )
        polished_mesh = trimesh.Trimesh(
            vertices=vertices[used_vertices],
            faces=compact_inverse.reshape((-1, 3)),
            process=False,
        )
        polished_mesh.fix_normals(multibody=True)
        proposed_data = polished_mesh.export(file_type="glb")
        if not isinstance(proposed_data, bytes):
            proposed_data = bytes(proposed_data)
        proposed_mesh_report = analyze_glb(BytesIO(proposed_data))
        proposed_diagnostics = diagnose_ovoxel_topology(BytesIO(proposed_data))
        proposed_edges = proposed_diagnostics["confirmed"]["edge_incidence"]
        accepted = (
            proposed_edges["overloaded_edges"] < before_edges["overloaded_edges"]
            and proposed_edges["boundary_edges"] <= before_edges["boundary_edges"]
            and proposed_mesh_report["geometry"]["connected_components"]
            <= before_mesh_report["geometry"]["connected_components"]
        )
    else:
        proposed_data = input_data
        proposed_mesh_report = before_mesh_report
        proposed_diagnostics = before_diagnostics
        proposed_edges = before_edges
        accepted = False

    if accepted:
        output_data = proposed_data
        after_mesh_report = proposed_mesh_report
        after_diagnostics = proposed_diagnostics
        status = (
            "POLISHED_PASS"
            if (
                after_mesh_report["geometry"]["watertight"]
                and proposed_edges["boundary_edges"] == 0
                and proposed_edges["overloaded_edges"] == 0
            )
            else "POLISHED_REVIEW"
        )
        rejected_reason = None
    else:
        output_data = input_data
        after_mesh_report = before_mesh_report
        after_diagnostics = before_diagnostics
        status = "UNCHANGED_REVIEW"
        rejected_reason = (
            "No opposed duplicate face pairs were found."
            if not candidate_groups
            else (
                "The proposed deletion did not reduce overloaded edges without increasing "
                "boundaries or connected components."
            )
        )

    after_edges = after_diagnostics["confirmed"]["edge_incidence"]
    report: dict[str, object] = {
        "schema": "cfp.post-voxel-topology-polish.v1",
        "status": status,
        "source_path": artifact_path,
        "rule": {
            "name": "remove_opposed_duplicate_internal_sheets",
            "candidate_groups": len(candidate_groups),
            "candidate_faces": int(np.count_nonzero(remove_mask)),
            "accepted": accepted,
            "rejected_reason": rejected_reason,
            "safety_guards": [
                "Overloaded edges must decrease.",
                "Boundary edges must not increase.",
                "Connected components must not increase.",
            ],
        },
        "changes": {
            "vertices_before": before_mesh_report["geometry"]["vertices"],
            "vertices_after": after_mesh_report["geometry"]["vertices"],
            "triangles_before": before_mesh_report["geometry"]["triangles"],
            "triangles_after": after_mesh_report["geometry"]["triangles"],
            "opposed_duplicate_groups_removed": len(candidate_groups) if accepted else 0,
            "opposed_duplicate_faces_removed": (
                int(np.count_nonzero(remove_mask)) if accepted else 0
            ),
            "boundary_edges_before": before_edges["boundary_edges"],
            "boundary_edges_after": after_edges["boundary_edges"],
            "overloaded_edges_before": before_edges["overloaded_edges"],
            "overloaded_edges_after": after_edges["overloaded_edges"],
            "watertight_before": before_mesh_report["geometry"]["watertight"],
            "watertight_after": after_mesh_report["geometry"]["watertight"],
        },
        "before": {
            "mesh_report": before_mesh_report,
            "diagnostics": before_diagnostics,
        },
        "after": {
            "mesh_report": after_mesh_report,
            "diagnostics": after_diagnostics,
        },
        "limitations": [
            "This node does not remesh, smooth, fill holes, or change vertex positions.",
            "Only exact coincident face pairs with opposite winding are considered.",
            "A proposed deletion is rejected when topology safety guards fail.",
            "Visual review remains required before promotion.",
        ],
    }
    return output_data, report


def format_post_voxel_polish_report(report: dict[str, object]) -> str:
    changes = report["changes"]
    rule = report["rule"]
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Post-Voxel Topology Polish",
            "=" * 49,
            f"Status: {report['status']}",
            "",
            f"Opposed duplicate groups found: {rule['candidate_groups']:,}",
            f"Proposed deletion accepted: {'Yes' if rule['accepted'] else 'No'}",
            f"Opposed duplicate faces removed: "
            f"{changes['opposed_duplicate_faces_removed']:,}",
            "",
            f"Vertices: {changes['vertices_before']:,} → "
            f"{changes['vertices_after']:,}",
            f"Triangles: {changes['triangles_before']:,} → "
            f"{changes['triangles_after']:,}",
            f"Boundary edges: {changes['boundary_edges_before']:,} → "
            f"{changes['boundary_edges_after']:,}",
            f"Edges shared by >2 faces: {changes['overloaded_edges_before']:,} → "
            f"{changes['overloaded_edges_after']:,}",
            f"Watertight: {'Yes' if changes['watertight_after'] else 'No'}",
            "",
            "No remeshing or vertex movement is performed.",
            "Candidate only: visually review before promotion.",
        ]
    )


def post_voxel_polish_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "format_post_voxel_polish_report",
    "polish_post_voxel_glb",
    "post_voxel_polish_report_json",
]
