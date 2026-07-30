import json
import os
import time
from pathlib import Path

import pytest

from comfyui_trellis2_mlx.runner import (
    Trellis2MLXConfig,
    build_environment,
    default_paths,
    run_engine,
)


REQUIRED_WEIGHTS = {
    "dino.safetensors",
    "normalization.json",
    "shape_dec.safetensors",
    "shape_flow_512.safetensors",
    "struct_dec.safetensors",
    "struct_flow.safetensors",
    "tex_dec.safetensors",
    "tex_flow_512.safetensors",
}


def make_fake_config(tmp_path: Path, script: str) -> Trellis2MLXConfig:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    engine = engine_dir / "trellis2-run-engine"
    engine.write_text(script, encoding="utf-8")
    engine.chmod(0o755)
    (engine_dir / "mlx.metallib").write_bytes(b"metal")
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in REQUIRED_WEIGHTS:
        (weights / name).write_bytes(b"fixture")
    return Trellis2MLXConfig(engine, weights, 0.95)


def test_default_paths_are_relative_to_cfp_checkout():
    node_directory = Path("/workspace/CFP/comfyui_trellis2_mlx")

    engine, weights = default_paths(node_directory)

    assert engine == Path(
        "/workspace/CFP/mlx-trellis2-swift/.build/arm64-apple-macosx/release/trellis2-run-engine"
    )
    assert weights == Path("/workspace/CFP/trellis2-mlx")


def test_background_workflow_embeds_official_rmbg_model_metadata():
    workflow_path = (
        Path(__file__).parents[1]
        / "comfyui_trellis2_mlx"
        / "workflows"
        / "trellis2_mlx_background_clean.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    rmbg_node = next(node for node in workflow["nodes"] if node["type"] == "RMBG")
    models = rmbg_node["properties"]["models"]

    assert {model["name"] for model in models} == {
        "BiRefNet_config.py",
        "birefnet.py",
        "config.json",
        "model.safetensors",
    }
    assert all(model["directory"] == "RMBG/RMBG-2.0" for model in models)
    assert all(
        model["url"].startswith("https://huggingface.co/briaai/RMBG-2.0/resolve/main/")
        for model in models
    )

    note = next(node for node in workflow["nodes"] if node["type"] == "MarkdownNote")
    markdown = note["widgets_values"][0]
    assert "does not turn the backdrop into geometry" in markdown
    assert "https://huggingface.co/briaai/RMBG-2.0" in markdown
    assert "ComfyUI/models/RMBG/RMBG-2.0/" in markdown
    assert "Commercial use requires a separate agreement with BRIA" in markdown


def test_build_environment_preserves_parent_and_sets_proven_contract(tmp_path):
    config = Trellis2MLXConfig(
        engine_binary=Path("/engine"),
        weights_directory=Path("/weights"),
        memory_fraction=0.95,
    )

    environment = build_environment(
        {"PATH": "/usr/bin"},
        config=config,
        image_path=tmp_path / "input.png",
        output_path=tmp_path / "output.glb",
        metrics_path=tmp_path / "metrics.json",
        seed=42,
        steps=12,
        use_matting=False,
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["HR_RES"] == "512"
    assert environment["SEED"] == "42"
    assert environment["STEPS"] == "12"
    assert environment["MATTING"] == "off"
    assert environment["ENGINE_MEMORY_FRACTION"] == "0.95"


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_build_environment_rejects_invalid_seed(tmp_path, seed):
    config = Trellis2MLXConfig(Path("/engine"), Path("/weights"), 0.95)

    with pytest.raises(ValueError, match="unsigned 64-bit"):
        build_environment(
            {},
            config=config,
            image_path=tmp_path / "input.png",
            output_path=tmp_path / "output.glb",
            metrics_path=tmp_path / "metrics.json",
            seed=seed,
            steps=12,
            use_matting=False,
        )


def test_run_engine_returns_metrics_and_artifact_hash(tmp_path):
    config = make_fake_config(
        tmp_path,
        """#!/bin/sh
printf 'glTF00000000' > "$OUT_GLB"
printf '{"seed": 42}' > "$METRICS_JSON"
printf 'fake engine complete\n'
""",
    )
    output_path = tmp_path / "output.glb"
    metrics_path = tmp_path / "metrics.json"
    environment = build_environment(
        os.environ,
        config=config,
        image_path=tmp_path / "input.png",
        output_path=output_path,
        metrics_path=metrics_path,
        seed=42,
        steps=12,
        use_matting=False,
    )

    metrics, log_text = run_engine(
        config=config,
        environment=environment,
        output_path=output_path,
        metrics_path=metrics_path,
        on_log=lambda _: None,
        check_cancelled=lambda: None,
    )

    assert metrics["seed"] == 42
    assert metrics["artifact_bytes"] == 12
    assert len(metrics["artifact_sha256"]) == 64
    assert log_text == "fake engine complete"


def test_run_engine_terminates_subprocess_when_cancelled(tmp_path):
    config = make_fake_config(
        tmp_path,
        """#!/bin/sh
sleep 10
""",
    )
    output_path = tmp_path / "output.glb"
    metrics_path = tmp_path / "metrics.json"
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise InterruptedError("cancelled")

    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancelled"):
        run_engine(
            config=config,
            environment=os.environ.copy(),
            output_path=output_path,
            metrics_path=metrics_path,
            on_log=lambda _: None,
            check_cancelled=cancel,
        )

    assert time.monotonic() - started < 3
