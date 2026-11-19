"""TripoSR inference wrapper.

TripoSR is a single-image feed-forward reconstructor.  Multi-view uploads are
ranked in :mod:`picto3d.preprocessing` and the strongest view is reconstructed.

The official repository is not a pip package, so we add its checkout to
``sys.path`` (``PICTO3D_TRIPOSR_PATH``, ``/content/TripoSR`` or ``../TripoSR``).
If ``torchmcubes`` is missing we optionally shim it with PyMCubes on CPU so the
server still works (slower marching cubes).
"""

from __future__ import annotations

import logging
import sys
import threading
import types
from pathlib import Path
from typing import Any

import PIL.Image
import trimesh

from ..config import BASE_DIR, Settings
from ..errors import GpuOutOfMemoryError, ReconstructionError
from ..runtime import release_memory, resolve_device

logger = logging.getLogger("picto3d.engine")


def _is_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


class TripoSREngine:
    name = "triposr"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self.device = resolve_device(settings.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ setup
    def _ensure_import_path(self) -> None:
        candidates = [
            self._settings.triposr_path,
            "/content/TripoSR",
            str(BASE_DIR.parent / "TripoSR"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            root = Path(candidate).expanduser()
            if (root / "tsr" / "system.py").is_file():
                resolved = str(root.resolve())
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)
                    logger.info("TripoSR checkout: %s", resolved)
                return
        raise ReconstructionError(
            "Không tìm thấy mã nguồn TripoSR. Hãy clone "
            "https://github.com/VAST-AI-Research/TripoSR và đặt biến môi trường "
            "PICTO3D_TRIPOSR_PATH trỏ tới thư mục đó."
        )

    def _install_marching_cubes_fallback(self) -> None:
        try:
            import torchmcubes  # noqa: F401

            return
        except Exception:
            logger.debug("torchmcubes unavailable", exc_info=True)

        if not self._settings.mc_cpu_fallback:
            return

        try:
            import mcubes
            import numpy as np
            import torch
        except Exception:
            logger.warning(
                "torchmcubes is missing and PyMCubes is unavailable. "
                "Install: pip install git+https://github.com/tatsy/torchmcubes.git"
            )
            return

        module = types.ModuleType("torchmcubes")
        module.__dict__["__picto3d_fallback__"] = True

        def marching_cubes(level: Any, isovalue: float) -> tuple[Any, Any]:
            volume = np.asarray(level.detach().to("cpu").numpy())
            verts, faces = mcubes.marching_cubes(volume, float(isovalue))
            return (
                torch.as_tensor(verts, dtype=torch.float32),
                torch.as_tensor(faces, dtype=torch.int64),
            )

        module.marching_cubes = marching_cubes  # type: ignore[attr-defined]
        sys.modules["torchmcubes"] = module
        logger.warning(
            "torchmcubes not found: using CPU PyMCubes fallback "
            "(marching cubes will be slower)."
        )

    # ------------------------------------------------------------------- load
    def load(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            self._ensure_import_path()
            self._install_marching_cubes_fallback()

            try:
                from tsr.system import TSR
            except Exception as exc:  # pragma: no cover - environment issue
                raise ReconstructionError(
                    "Không import được TripoSR/PyTorch. Hãy cài đặt theo "
                    "backend/README.md (torch, torchmcubes, TripoSR)."
                ) from exc

            logger.info(
                "loading TripoSR model=%s device=%s chunk=%d",
                self._settings.model_id,
                self.device,
                self._settings.chunk_size,
            )
            try:
                model = TSR.from_pretrained(
                    self._settings.model_id,
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                model.renderer.set_chunk_size(self._settings.chunk_size)
                model.to(self.device)
                model.eval()
                if self._settings.warmup_on_load:
                    self._warmup(model)
            except Exception as exc:
                release_memory()
                raise ReconstructionError(
                    f"Không nạp được mô hình TripoSR: {exc}"
                ) from exc

            self._model = model
            logger.info("TripoSR ready on %s", self.device)

    def _warmup(self, model: Any) -> None:
        import torch

        logger.info("warming up TripoSR (first run only)…")
        warm = PIL.Image.new("RGB", (512, 512), (127, 127, 127))
        with torch.no_grad():
            scene_codes = model([warm], device=self.device)
        del scene_codes, warm
        release_memory()

    # ------------------------------------------------------------ inference
    def reconstruct(
        self,
        image: PIL.Image.Image,
        *,
        mc_resolution: int,
    ) -> trimesh.Trimesh:
        self.load()
        assert self._model is not None

        import torch

        try:
            with torch.no_grad():
                scene_codes = self._model([image], device=self.device)
                meshes = self._model.extract_mesh(
                    scene_codes,
                    True,
                    resolution=int(mc_resolution),
                )
            if not meshes:
                raise ReconstructionError("Mô hình AI không sinh ra lưới 3D.")
            mesh = meshes[0]
            del scene_codes, meshes
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ReconstructionError("Lưới 3D sinh ra không hợp lệ.")
            return mesh
        except GpuOutOfMemoryError:
            raise
        except Exception as exc:
            if _is_oom(exc):
                release_memory()
                raise GpuOutOfMemoryError() from exc
            if isinstance(exc, ReconstructionError):
                raise
            raise ReconstructionError(f"Lỗi suy luận 3D: {exc}") from exc
        finally:
            release_memory()

    def unload(self) -> None:
        self._model = None
        release_memory()
        logger.info("TripoSR unloaded")
