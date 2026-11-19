"""End-to-end reconstruction pipeline shared by sync and async endpoints."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from .config import Settings
from .engines.base import ReconstructionEngine
from .errors import GpuOutOfMemoryError, Picto3DError, ReconstructionError
from .jobs import JobStore
from .mesh_ops import export_glb, postprocess
from .preprocessing import preprocess_images
from .runtime import release_memory, timer
from .schemas import JobState, ReconstructOptions, ReconstructResult, Stage

logger = logging.getLogger("picto3d.pipeline")


@dataclass(frozen=True)
class PipelineRequest:
    job_id: str
    files: list[tuple[str, bytes]]
    options: ReconstructOptions
    base_url: str


def _friendly_error(exc: BaseException) -> str:
    if isinstance(exc, Picto3DError):
        return str(exc)
    text = str(exc).strip()
    if "out of memory" in text.lower():
        return str(GpuOutOfMemoryError())
    return f"Không thể dựng mô hình 3D: {text or exc.__class__.__name__}"


class ReconstructionPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        engine: ReconstructionEngine,
        store: JobStore,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._store = store
        # The GPU and the rembg session are both single-worker resources.
        self._run_lock = threading.Lock()

    def run(self, request: PipelineRequest) -> ReconstructResult:
        started = time.perf_counter()
        job_id = request.job_id

        self._store.update(
            job_id,
            status=JobState.RUNNING,
            stage=Stage.PREPROCESS,
            progress=0.04,
            message="Đang tách nền và căn chỉnh vật thể…",
        )

        try:
            with self._run_lock:
                with timer() as preprocess_timer:
                    pre = preprocess_images(
                        request.files,
                        self._settings,
                        foreground_ratio=request.options.foreground_ratio,
                        remove_background=request.options.remove_background,
                    )

                self._store.update(
                    job_id,
                    stage=Stage.INFERENCE,
                    progress=0.30,
                    message=(
                        "Đang nạp mô hình AI (lần đầu có thể mất 1-2 phút)…"
                        if not self._engine.is_loaded
                        else "Đang dựng hình học 3D…"
                    ),
                )

                with timer() as inference_timer:
                    self._engine.load()
                    mesh = self._engine.reconstruct(
                        pre.selected.image,
                        mc_resolution=request.options.mc_resolution,
                    )

                self._store.update(
                    job_id,
                    stage=Stage.MESH,
                    progress=0.72,
                    message="Đang làm sạch, làm mượt và tối ưu lưới…",
                )

                with timer() as mesh_timer:
                    mesh = postprocess(
                        mesh,
                        target_faces=request.options.target_faces,
                        y_up=self._settings.to_y_up,
                    )

                self._store.update(
                    job_id,
                    stage=Stage.EXPORT,
                    progress=0.90,
                    message="Đang xuất file GLB…",
                )

                file_name = f"picto3d_{job_id}.glb"
                file_path = self._settings.output_dir / file_name
                stats = export_glb(mesh, file_path)

                texture = "vertex_color"
                try:
                    if mesh.visual.kind != "vertex":
                        texture = "none"
                except Exception:
                    texture = "none"

                elapsed = round(time.perf_counter() - started, 2)
                base = request.base_url.rstrip("/")
                result = ReconstructResult(
                    job_id=job_id,
                    glb_url=f"{base}/files/{file_name}",
                    download_url=f"{base}/api/download/{job_id}",
                    file_name=file_name,
                    elapsed_seconds=elapsed,
                    preprocess_seconds=preprocess_timer["seconds"],
                    inference_seconds=inference_timer["seconds"],
                    mesh_seconds=mesh_timer["seconds"],
                    vertices=stats.vertices,
                    faces=stats.faces,
                    texture=texture,
                    input_views=len(pre.views),
                    selected_view=pre.selected.index + 1,
                    device=self._engine.device,
                )

                self._store.complete(job_id, result)
                logger.info(
                    "job %s done in %.1fs (%d verts, %d faces, view %d/%d)",
                    job_id,
                    elapsed,
                    stats.vertices,
                    stats.faces,
                    pre.selected.index + 1,
                    len(pre.views),
                )
                return result

        except Exception as exc:
            message = _friendly_error(exc)
            logger.exception("job %s failed: %s", job_id, message)
            self._store.fail(job_id, message)
            if isinstance(exc, Picto3DError):
                raise
            if "out of memory" in str(exc).lower():
                raise GpuOutOfMemoryError() from exc
            raise ReconstructionError(message) from exc
        finally:
            release_memory()
