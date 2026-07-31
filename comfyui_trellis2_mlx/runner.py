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
    startup_timeout_seconds: float = 60.0
    stall_timeout_seconds: float = 900.0

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
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be greater than zero")
        if self.stall_timeout_seconds <= 0:
            raise ValueError("stall_timeout_seconds must be greater than zero")


@dataclass(frozen=True)
class Trellis2MLXConditioningArtifact:
    path: Path
    view_angles: tuple[int, ...]
    tier: str
    artifact_sha256: str


@dataclass(frozen=True)
class Trellis2MLXSparseStructureArtifact:
    path: Path
    source_conditioning_sha256: str
    voxel_count: int
    artifact_sha256: str


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
    output_mode: str = "textured",
    additional_views_manifest: Path | None = None,
    engine_stage: str = "full",
    view_angles: tuple[int, ...] | None = None,
) -> dict[str, str]:
    if not 0 <= seed <= (2**64 - 1):
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not 1 <= steps <= 50:
        raise ValueError("steps must be between 1 and 50")
    if output_mode not in {"textured", "geometry_only"}:
        raise ValueError("output_mode must be 'textured' or 'geometry_only'")
    if engine_stage not in {"full", "conditioning"}:
        raise ValueError("engine_stage must be 'full' or 'conditioning'")
    if view_angles is not None and not view_angles:
        raise ValueError("view_angles must contain at least one camera angle")

    environment = dict(base_environment)
    environment.update(
        {
            "IMG": str(image_path),
            "WEIGHTS_DIR": str(config.weights_directory),
            "OUT_GLB": str(output_path),
            "OUT_CONDITIONING": str(output_path),
            "METRICS_JSON": str(metrics_path),
            "HR_RES": "512",
            "STEPS": str(steps),
            "SEED": str(seed),
            "MATTING": "on" if use_matting else "off",
            "TEXTURE": "off" if output_mode == "geometry_only" else "on",
            "ENGINE_MEMORY_FRACTION": f"{config.memory_fraction:.2f}",
            "ENGINE_STAGE": engine_stage,
        }
    )
    if view_angles is not None:
        environment["VIEW_ANGLES"] = ",".join(str(angle) for angle in view_angles)
    else:
        environment.pop("VIEW_ANGLES", None)
    if additional_views_manifest is not None:
        environment["ADDITIONAL_VIEWS_MANIFEST"] = str(additional_views_manifest)
    else:
        environment.pop("ADDITIONAL_VIEWS_MANIFEST", None)
    return environment


def build_sparse_environment(
    base_environment: dict[str, str],
    *,
    config: Trellis2MLXConfig,
    conditioning_path: Path,
    output_path: Path,
    metrics_path: Path,
    seed: int,
    steps: int,
) -> dict[str, str]:
    if not 0 <= seed <= (2**64 - 1):
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not 1 <= steps <= 50:
        raise ValueError("steps must be between 1 and 50")

    environment = dict(base_environment)
    environment.update(
        {
            "WEIGHTS_DIR": str(config.weights_directory),
            "IN_CONDITIONING": str(conditioning_path),
            "OUT_SPARSE": str(output_path),
            "METRICS_JSON": str(metrics_path),
            "HR_RES": "512",
            "STEPS": str(steps),
            "SEED": str(seed),
            "MATTING": "off",
            "TEXTURE": "off",
            "ENGINE_MEMORY_FRACTION": f"{config.memory_fraction:.2f}",
            "ENGINE_STAGE": "sparse",
        }
    )
    for key in ("IMG", "OUT_GLB", "OUT_CONDITIONING", "VIEW_ANGLES", "ADDITIONAL_VIEWS_MANIFEST"):
        environment.pop(key, None)
    return environment


def inspect_safetensors(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Safetensors artifact not found: {path}")
    artifact_bytes = path.read_bytes()
    if len(artifact_bytes) < 12:
        raise RuntimeError(f"Safetensors artifact is too small: {path}")
    header_size = int.from_bytes(artifact_bytes[:8], byteorder="little", signed=False)
    if header_size <= 2 or header_size > len(artifact_bytes) - 8:
        raise RuntimeError(f"Safetensors artifact has an invalid header: {path}")
    try:
        header = json.loads(artifact_bytes[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Safetensors artifact has an unreadable header: {path}") from error
    return header, hashlib.sha256(artifact_bytes).hexdigest()


def run_engine(
    *,
    config: Trellis2MLXConfig,
    environment: dict[str, str],
    output_path: Path,
    metrics_path: Path,
    on_log: LogCallback,
    check_cancelled: CancelCallback,
    artifact_kind: str = "glb",
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
    started_at = time.monotonic()
    last_progress_at = started_at
    received_output = False

    def stop_process() -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    try:
        while process.poll() is None:
            check_cancelled()
            events = selector.select(timeout=0.25)
            for key, _ in events:
                line = key.fileobj.readline()
                if line:
                    clean = line.rstrip()
                    lines.append(clean)
                    on_log(clean)
                    received_output = True
                    if not clean.startswith("[engine] heartbeat"):
                        last_progress_at = time.monotonic()
            now = time.monotonic()
            if not received_output and now - started_at > config.startup_timeout_seconds:
                stop_process()
                raise RuntimeError(
                    "TRELLIS.2 MLX engine produced no startup output for "
                    f"{config.startup_timeout_seconds:.1f} seconds and was terminated."
                )
            if received_output and now - last_progress_at > config.stall_timeout_seconds:
                stop_process()
                raise RuntimeError(
                    "TRELLIS.2 MLX engine made no phase progress for "
                    f"{config.stall_timeout_seconds:.1f} seconds and was terminated."
                )
        for line in process.stdout:
            clean = line.rstrip()
            lines.append(clean)
            on_log(clean)
    except BaseException:
        stop_process()
        raise
    finally:
        selector.close()
        process.stdout.close()

    log_text = "\n".join(lines)
    if process.returncode != 0:
        tail = "\n".join(lines[-40:])
        raise RuntimeError(f"TRELLIS.2 MLX engine exited with code {process.returncode}.\n{tail}")
    if artifact_kind not in {"glb", "safetensors", "sparse_safetensors"}:
        raise ValueError("artifact_kind must be 'glb', 'safetensors', or 'sparse_safetensors'")
    if not output_path.is_file() or output_path.stat().st_size < 12:
        raise RuntimeError(f"TRELLIS.2 MLX completed without producing a {artifact_kind} artifact.")
    artifact_bytes = output_path.read_bytes()
    if artifact_kind == "glb" and artifact_bytes[:4] != b"glTF":
        raise RuntimeError(f"TRELLIS.2 MLX produced an invalid GLB artifact: {output_path}")
    if artifact_kind in {"safetensors", "sparse_safetensors"}:
        header_size = int.from_bytes(artifact_bytes[:8], byteorder="little", signed=False)
        if header_size <= 2 or header_size > len(artifact_bytes) - 8:
            raise RuntimeError(f"TRELLIS.2 MLX produced an invalid safetensors artifact: {output_path}")
        try:
            header = json.loads(artifact_bytes[8 : 8 + header_size].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"TRELLIS.2 MLX produced an unreadable safetensors header: {output_path}"
            ) from error
        if "cond_512" not in header or "neg_cond_512" not in header:
            raise RuntimeError("Conditioning artifact is missing cond_512 or neg_cond_512 tensors")
        if artifact_kind == "sparse_safetensors" and "coords_32" not in header:
            raise RuntimeError("Sparse-structure artifact is missing coords_32")

    metrics = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["wrapper_completed_at"] = time.time()
    metrics["artifact_kind"] = artifact_kind
    metrics["artifact_bytes"] = output_path.stat().st_size
    metrics["artifact_sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    return metrics, log_text
