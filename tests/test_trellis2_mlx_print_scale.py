from __future__ import annotations

from io import BytesIO

import pytest
import trimesh

from comfyui_trellis2_mlx.mesh_report import analyze_glb
from comfyui_trellis2_mlx.print_scale import (
    format_print_scale_report,
    print_scale_report_json,
    scale_glb_for_print,
)


def _box() -> BytesIO:
    return BytesIO(
        trimesh.creation.box(extents=[1.0, 2.0, 4.0]).export(file_type="glb")
    )


def test_print_scale_uses_longest_axis_and_preserves_topology():
    output, report = scale_glb_for_print(
        _box(),
        target_height_mm=200.0,
        height_axis="auto",
        source_voxel_resolution=200,
        nozzle_diameter_mm=0.4,
        layer_height_mm=0.2,
    )
    geometry = analyze_glb(BytesIO(output))["geometry"]

    assert report["configuration"]["resolved_height_axis"] == "z"
    assert report["scaling"]["target_dimensions_mm"] == pytest.approx(
        [50.0, 100.0, 200.0]
    )
    assert geometry["dimensions"] == pytest.approx([0.05, 0.1, 0.2])
    assert report["scaling"]["topology_unchanged"] is True
    assert report["feature_gate"]["voxel_pitch_mm"] == pytest.approx(1.0)
    assert report["feature_gate"]["status"] == "DETAIL_COARSE"
    assert report["status"] == "SCALE_DETAIL_REVIEW"
    assert "Output dimensions: 50.00 × 100.00 × 200.00 mm" in (
        format_print_scale_report(report)
    )
    assert '"schema": "cfp.print-scale-feature-gate.v1"' in print_scale_report_json(
        report
    )


def test_print_scale_supports_explicit_axis():
    _, report = scale_glb_for_print(
        _box(),
        target_height_mm=100.0,
        height_axis="x",
        source_voxel_resolution=256,
    )

    assert report["configuration"]["resolved_height_axis"] == "x"
    assert report["scaling"]["target_dimensions_mm"] == pytest.approx(
        [100.0, 200.0, 400.0]
    )


def test_print_scale_can_pass_detail_gate_at_large_voxel_resolution():
    _, report = scale_glb_for_print(
        _box(),
        target_height_mm=40.0,
        source_voxel_resolution=512,
        nozzle_diameter_mm=0.4,
    )

    assert report["feature_gate"]["voxel_pitch_mm"] == pytest.approx(0.078125)
    assert report["feature_gate"]["status"] == "DETAIL_PASS"
    assert report["status"] == "SCALE_PASS"


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("target_height_mm", 0.0, "target_height_mm"),
        ("source_voxel_resolution", 12, "source_voxel_resolution"),
        ("nozzle_diameter_mm", 0.01, "nozzle_diameter_mm"),
        ("layer_height_mm", 2.0, "layer_height_mm"),
        ("height_axis", "sideways", "height_axis"),
    ],
)
def test_print_scale_rejects_invalid_configuration(argument, value, message):
    kwargs = {argument: value}
    with pytest.raises(ValueError, match=message):
        scale_glb_for_print(_box(), **kwargs)
