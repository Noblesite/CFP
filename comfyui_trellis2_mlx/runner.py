from __future__ import annotations

import json
import hashlib
import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


LogCallback = Callable[[str], None]
CancelCallback = Callable[[], None]


@dataclass(frozen=True)
class Trellis2MLXConfig:
    engine_binary: Path
    weights_directory: Path
    memory_fraction: float = 0.95

    def validate(self) -> None:
        if os.uname().sysname != "Darwin" or os.uname().machine != "arm64":
            raise RuntimeError("TRELLIS.2 MLX requires macOS on Apple Silicon.")
        if not self.engine_binary.is_file():
            raise FileNotFoundError(
                f"TRELLIS.2 MLX engine not found: {self.engine_binary}\n"
                "Build it with: swift build -c release --product trellis2-run-engine"
            )
        if not os.access(self.engine_binary, os.X_OK):
            raise PermissionError(f"TRELLIS.2 MLX engine is not executable: {self.engine_binary}")
        metal_library = self.engine_binary.parent / "mlx.metallib"
        if not metal_library.is_file():
            raise FileNotFoundError(
                f"MLX Metal library not found beside the engine: {metal_library}\n"
                "Run scripts/install_trellis2_mlx_node.sh to repair the native runtime."
            )
        if not self.weights_directory.is_dir():
            raise FileNotFoundError(f"TRELLIS.2 MLX weights directory not found: {self.weights_directory}")
        required_weights = {
            "dino.safetensors",
            "normalization.json",
            "shape_dec.safetensors",
            "shape_flow_512.safetensors",
            "struct_dec.safetensors",
            "struct_flow.safetensors",
            "tex_dec.safetensors",
            "tex_flow_512.safetensors",
        }
        missing = sorted(name for name in required_weights if not (self.weights_directory / name).is_file())
        if missing:
            raise FileNotFoundError(
                f"TRELLIS.2 MLX weights directory is incomplete: {self.weights_directory}\n"
                f"Missing: {', '.join(missing)}"
            )
        if not 0.70 <= self.memory_fraction <= 0.98:
            raise ValueError("memory_fraction must be between 0.70 and 0.98")


def default_paths(node_directory: Path) -> tuple[Path, Path]:
    project_root = node_directory.resolve().parent
    engine = (
        project_root
        / "mlx-trellis2-swift"
        / ".build"
        / "arm64-apple-macosx"
        / "release"
        / "trellis2-run-engine"
    )
    weights = project_root / "trellis2-mlx"
    return engine, weights


def build_environment(
    base_environment: dict[str, str],
    *,
    config: Trellis2MLXConfig,
    image_path: Path,
    output_path: Path,
    metrics_path: Path,
    seed: int,
    steps: int,
    use_matting: bool,
    additional_views_manifest: Path | None = None,
) -> dict[str, str]:
    if not 0 <= seed <= (2**64 - 1):
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not 1 <= steps <= 50:
        raise ValueError("steps must be between 1 and 50")

    environment = dict(base_environment)
    environment.update(
        {
            "IMG": str(image_path),
            "WEIGHTS_DIR": str(config.weights_directory),
            "OUT_GLB": str(output_path),
            "METRICS_JSON": str(metrics_path),
            "HR_RES": "512",
            "STEPS": str(steps),
            "SEED": str(seed),
            "MATTING": "on" if use_matting else "off",
            "ENGINE_MEMORY_FRACTION": f"{config.memory_fraction:.2f}",
        }
    )
    if additional_views_manifest is not None:
        environment["ADDITIONAL_VIEWS_MANIFEST"] = str(additional_views_manifest)
    else:
        environment.pop("ADDITIONAL_VIEWS_MANIFEST", None)
    return environment


def run_engine(
    *,
    config: Trellis2MLXConfig,
    environment: dict[str, str],
    output_path: Path,
    metrics_path: Path,
    on_log: LogCallback,
    check_cancelled: CancelCallback,
) -> tuple[dict, str]:
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        [str(config.engine_binary)],
        cwd=str(config.engine_binary.parent),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    lines: list[str] = []
    try:
        while process.poll() is None:
            check_cancelled()
            for key, _ in selector.select(timeout=0.25):
                line = key.fileobj.readline()
                if line:
                    clean = line.rstrip()
                    lines.append(clean)
                    on_log(clean)
        for line in process.stdout:
            clean = line.rstrip()
            lines.append(clean)
            on_log(clean)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()

    log_text = "\n".join(lines)
    if process.returncode != 0:
        tail = "\n".join(lines[-40:])
        raise RuntimeError(f"TRELLIS.2 MLX engine exited with code {process.returncode}.\n{tail}")
    if not output_path.is_file() or output_path.stat().st_size < 12:
        raise RuntimeError("TRELLIS.2 MLX completed without producing a GLB artifact.")
    if output_path.read_bytes()[:4] != b"glTF":
        raise RuntimeError(f"TRELLIS.2 MLX produced an invalid GLB artifact: {output_path}")

    metrics = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["wrapper_completed_at"] = time.time()
    metrics["artifact_bytes"] = output_path.stat().st_size
    metrics["artifact_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return metrics, log_text
