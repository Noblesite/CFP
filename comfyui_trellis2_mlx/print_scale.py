from __future__ import annotations

import json
from io import BytesIO
from typing import BinaryIO

import numpy as np

from .mesh_report import _load_mesh, _read_artifact, analyze_glb


_AXIS_NAMES = ("x", "y", "z")


def _resolve_height_axis(dimensions: np.ndarray, height_axis: str) -> int:
    normalized = height_axis.lower()
    if normalized == "auto":
        return int(np.argmax(dimensions))
    if normalized not in _AXIS_NAMES:
        raise ValueError("height_axis must be one of: auto, x, y, z")
    return _AXIS_NAMES.index(normalized)


def scale_glb_for_print(
    source: str | BinaryIO,
    *,
    target_height_mm: float = 250.0,
    height_axis: str = "z",
    source_voxel_resolution: int = 256,
    nozzle_diameter_mm: float = 0.4,
    layer_height_mm: float = 0.2,
) -> tuple[bytes, dict[str, object]]:
    """Scale a GLB to a target physical height and report its voxel detail budget."""
    if not 1.0 <= target_height_mm <= 5000.0:
        raise ValueError("target_height_mm must be between 1 and 5000")
    if not 32 <= source_voxel_resolution <= 768:
        raise ValueError("source_voxel_resolution must be between 32 and 768")
    if not 0.1 <= nozzle_diameter_mm <= 2.0:
        raise ValueError("nozzle_diameter_mm must be between 0.1 and 2.0")
    if not 0.02 <= layer_height_mm <= 1.0:
        raise ValueError("layer_height_mm must be between 0.02 and 1.0")

    artifact_path, input_data = _read_artifact(source)
    mesh = _load_mesh(input_data)
    source_dimensions = np.asarray(mesh.extents, dtype=np.float64)
    axis_index = _resolve_height_axis(source_dimensions, height_axis)
    source_height = float(source_dimensions[axis_index])
    if source_height <= 0:
        raise ValueError("Selected height axis has zero extent")

    # glTF linear units are meters, while the operator-facing contract is millimeters.
    target_height_m = target_height_mm / 1000.0
    scale_factor = target_height_m / source_height
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)
    output_data = scaled_mesh.export(file_type="glb")
    if not isinstance(output_data, bytes):
        output_data = bytes(output_data)

    before = analyze_glb(BytesIO(input_data))
    after = analyze_glb(BytesIO(output_data))
    target_dimensions_mm = (
        np.asarray(after["geometry"]["dimensions"], dtype=np.float64) * 1000.0
    )
    max_extent_mm = float(np.max(target_dimensions_mm))
    voxel_pitch_mm = max_extent_mm / float(source_voxel_resolution)
    estimated_feature_floor_mm = 2.0 * voxel_pitch_mm
    nozzle_samples_per_voxel_feature = nozzle_diameter_mm / voxel_pitch_mm
    layer_samples_per_voxel_feature = layer_height_mm / voxel_pitch_mm

    if voxel_pitch_mm <= nozzle_diameter_mm * 0.5:
        detail_gate = "DETAIL_PASS"
        detail_note = (
            "Voxel pitch is at most half the nozzle diameter; the source mesh samples "
            "features more finely than a typical single extrusion width."
        )
    elif voxel_pitch_mm <= nozzle_diameter_mm:
        detail_gate = "DETAIL_REVIEW"
        detail_note = (
            "Voxel pitch is between half and one nozzle diameter. Inspect small seams and "
            "raised details at the intended print scale."
        )
    else:
        detail_gate = "DETAIL_COARSE"
        detail_note = (
            "Voxel pitch exceeds the nozzle diameter. The mesh may already have lost details "
            "that the selected nozzle could otherwise reproduce."
        )

    topology_unchanged = all(
        before["geometry"][key] == after["geometry"][key]
        for key in (
            "vertices",
            "triangles",
            "connected_components",
            "boundary_edges",
            "non_manifold_edges",
            "watertight",
        )
    )
    if before["status"] != "PASS" or after["status"] != "PASS":
        status = "SCALE_TOPOLOGY_REVIEW"
    elif detail_gate == "DETAIL_PASS":
        status = "SCALE_PASS"
    else:
        status = "SCALE_DETAIL_REVIEW"

    report: dict[str, object] = {
        "schema": "cfp.print-scale-feature-gate.v1",
        "status": status,
        "source_path": artifact_path,
        "configuration": {
            "target_height_mm": target_height_mm,
            "requested_height_axis": height_axis,
            "resolved_height_axis": _AXIS_NAMES[axis_index],
            "source_voxel_resolution": source_voxel_resolution,
            "nozzle_diameter_mm": nozzle_diameter_mm,
            "layer_height_mm": layer_height_mm,
        },
        "scaling": {
            "glb_linear_unit": "meter",
            "scale_factor": scale_factor,
            "source_dimensions_scene_units": source_dimensions.tolist(),
            "target_dimensions_m": (
                np.asarray(after["geometry"]["dimensions"], dtype=np.float64)
            ).tolist(),
            "target_dimensions_mm": target_dimensions_mm.tolist(),
            "topology_unchanged": topology_unchanged,
        },
        "feature_gate": {
            "status": detail_gate,
            "voxel_pitch_mm": voxel_pitch_mm,
            "estimated_feature_floor_mm": estimated_feature_floor_mm,
            "nozzle_diameters_per_voxel_pitch": nozzle_samples_per_voxel_feature,
            "layer_heights_per_voxel_pitch": layer_samples_per_voxel_feature,
            "note": detail_note,
            "interpretation": (
                "This estimates the source voxel sampling budget. It does not measure actual "
                "wall thickness, clearance, overhangs, or slicer toolpaths."
            ),
        },
        "topology": {
            "before": before,
            "after": after,
        },
        "limitations": [
            "The node scales uniformly and does not translate or reorient the mesh.",
            "The GLB is written in meter-based glTF units; report dimensions are millimeters.",
            "Feature-size results are sampling estimates, not local thickness measurements.",
            "Visual and slicer review remain required before manufacturing.",
        ],
    }
    return output_data, report


def format_print_scale_report(report: dict[str, object]) -> str:
    config = report["configuration"]
    scaling = report["scaling"]
    feature = report["feature_gate"]
    dimensions = " × ".join(
        f"{dimension:.2f}" for dimension in scaling["target_dimensions_mm"]
    )
    return "\n".join(
        [
            "CFP TRELLIS.2 MLX Print Scale & Feature Gate",
            "=" * 48,
            f"Status: {report['status']}",
            "",
            f"Target height: {config['target_height_mm']:.2f} mm",
            f"Height axis: {config['resolved_height_axis'].upper()} "
            f"(requested: {config['requested_height_axis']})",
            f"Uniform scale factor: {scaling['scale_factor']:.8g}",
            f"Output dimensions: {dimensions} mm",
            f"Topology unchanged: {'Yes' if scaling['topology_unchanged'] else 'No'}",
            "",
            f"Source voxel resolution: {config['source_voxel_resolution']}",
            f"Effective voxel pitch: {feature['voxel_pitch_mm']:.4f} mm",
            f"Estimated feature floor: {feature['estimated_feature_floor_mm']:.4f} mm",
            f"Nozzle diameter: {config['nozzle_diameter_mm']:.3f} mm",
            f"Layer height: {config['layer_height_mm']:.3f} mm",
            f"Feature gate: {feature['status']}",
            f"Assessment: {feature['note']}",
            "",
            "Advisory only: actual wall thickness and clearances are not measured.",
        ]
    )


def print_scale_report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "format_print_scale_report",
    "print_scale_report_json",
    "scale_glb_for_print",
]
