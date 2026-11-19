"""Environment-driven configuration for the Picto3D backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    version: str
    output_dir: Path
    model_id: str
    device: str
    chunk_size: int
    default_mc_resolution: int
    default_target_faces: int
    default_foreground_ratio: float
    image_size: int
    segmentation_model: str
    max_upload_bytes: int
    max_images: int
    file_ttl_seconds: int
    preload_model: bool
    to_y_up: bool
    public_base_url: str | None
    triposr_path: str | None
    mc_cpu_fallback: bool
    warmup_on_load: bool

    @property
    def max_upload_mb(self) -> int:
        return max(1, self.max_upload_bytes // (1024 * 1024))


def load_settings() -> Settings:
    """Read settings from PICTO3D_* environment variables."""

    output_dir = Path(
        _env_str("PICTO3D_OUTPUT_DIR", str(BASE_DIR / "outputs"))
    ).expanduser()

    public_base_url = os.getenv("PICTO3D_PUBLIC_URL")
    if public_base_url:
        public_base_url = public_base_url.strip().rstrip("/") or None
    else:
        public_base_url = None

    return Settings(
        app_name="Picto3D Backend",
        version="1.0.0",
        output_dir=output_dir,
        model_id=_env_str("PICTO3D_MODEL_ID", "stabilityai/TripoSR"),
        device=_env_str("PICTO3D_DEVICE", "auto"),
        chunk_size=max(256, _env_int("PICTO3D_CHUNK_SIZE", 8192)),
        default_mc_resolution=min(
            384, max(128, _env_int("PICTO3D_MC_RESOLUTION", 256))
        ),
        default_target_faces=min(
            100_000, max(10_000, _env_int("PICTO3D_TARGET_FACES", 40000))
        ),
        default_foreground_ratio=min(
            1.0, max(0.5, _env_float("PICTO3D_FOREGROUND_RATIO", 0.8))
        ),
        image_size=min(1024, max(256, _env_int("PICTO3D_IMAGE_SIZE", 512))),
        segmentation_model=_env_str("PICTO3D_SEG_MODEL", "isnet-general-use"),
        max_upload_bytes=max(1, _env_int("PICTO3D_MAX_UPLOAD_MB", 25)) * 1024 * 1024,
        max_images=max(1, _env_int("PICTO3D_MAX_IMAGES", 8)),
        file_ttl_seconds=max(60, _env_int("PICTO3D_FILE_TTL", 3600)),
        preload_model=_env_bool("PICTO3D_PRELOAD", True),
        to_y_up=_env_bool("PICTO3D_Y_UP", True),
        public_base_url=public_base_url,
        triposr_path=os.getenv("PICTO3D_TRIPOSR_PATH") or None,
        mc_cpu_fallback=_env_bool("PICTO3D_MC_CPU_FALLBACK", True),
        warmup_on_load=_env_bool("PICTO3D_WARMUP", True),
    )
