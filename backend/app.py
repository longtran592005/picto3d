"""FastAPI application for Picto3D."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from picto3d.config import Settings, load_settings
from picto3d.engines import ReconstructionEngine, TripoSREngine
from picto3d.errors import Picto3DError
from picto3d.jobs import JobStore
from picto3d.pipeline import PipelineRequest, ReconstructionPipeline
from picto3d.preprocessing import get_rembg_session
from picto3d.runtime import gpu_summary, release_memory
from picto3d.schemas import (
    HealthResponse,
    JobState,
    JobStatus,
    ReconstructOptions,
    ReconstructResult,
)

logger = logging.getLogger("picto3d.app")

CLEANUP_INTERVAL_SECONDS = 300


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    _configure_logging()
    settings = settings or load_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    store = JobStore()
    engine: ReconstructionEngine = TripoSREngine(settings)
    pipeline = ReconstructionPipeline(settings=settings, engine=engine, store=store)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="picto3d-job")
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleanup_task = asyncio.create_task(_cleanup_loop(store, settings))

        preload_thread: threading.Thread | None = None
        if settings.preload_model:
            preload_thread = threading.Thread(
                target=_preload,
                args=(engine, settings),
                name="picto3d-preload",
                daemon=True,
            )
            preload_thread.start()

        logger.info(
            "%s %s | device=%s | output=%s",
            settings.app_name,
            settings.version,
            engine.device,
            settings.output_dir,
        )
        try:
            yield
        finally:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
            executor.shutdown(wait=False, cancel_futures=True)
            engine.unload()
            release_memory()
            logger.info("Picto3D backend stopped")

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Chuyển ảnh (một hoặc nhiều góc nhìn) thành mô hình 3D GLB bằng TripoSR."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.engine = engine
    app.state.pipeline = pipeline
    app.state.executor = executor
    app.state.started_at = started_at

    # CORS must wrap every response, including compressed ones.
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Length"],
        max_age=86400,
    )

    app.mount(
        "/files",
        StaticFiles(directory=str(settings.output_dir), check_dir=False),
        name="files",
    )

    # ------------------------------------------------------------- endpoints
    @app.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "health": "/api/health",
            "docs": "/docs",
            "reconstruct": "POST /api/reconstruct",
        }

    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        gpu, vram_total, vram_allocated = gpu_summary(engine.device)
        return HealthResponse(
            app=settings.app_name,
            version=settings.version,
            device=engine.device,
            gpu=gpu,
            vram_total_mb=vram_total,
            vram_allocated_mb=vram_allocated,
            model_id=settings.model_id,
            model_loaded=engine.is_loaded,
            queue_depth=store.queue_depth(),
            uptime_seconds=round(time.time() - started_at, 1),
            max_images=settings.max_images,
            max_upload_mb=settings.max_upload_mb,
        )

    @app.post(
        "/api/reconstruct",
        response_model=ReconstructResult | JobStatus,
        tags=["reconstruct"],
        summary="Upload 1..N ảnh và nhận file GLB",
    )
    async def reconstruct(
        request: Request,
        response: Response,
        files: Annotated[list[UploadFile], File(description="Ảnh các góc nhìn")],
        foreground_ratio: Annotated[float | None, Form(ge=0.5, le=1.0)] = None,
        mc_resolution: Annotated[int | None, Form(ge=128, le=384)] = None,
        target_faces: Annotated[int | None, Form(ge=10000, le=100000)] = None,
        remove_background: Annotated[bool, Form()] = True,
        wait: Annotated[
            bool, Query(description="false = trả job_id để polling")
        ] = True,
    ) -> ReconstructResult | JobStatus:
        if not files:
            raise HTTPException(status_code=422, detail="Chưa có ảnh đầu vào.")
        if len(files) > settings.max_images:
            raise HTTPException(
                status_code=400,
                detail=f"Tối đa {settings.max_images} ảnh mỗi lần.",
            )

        payloads: list[tuple[str, bytes]] = []
        for upload in files:
            name = upload.filename or "image"
            if upload.size is not None and upload.size > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(f"Ảnh '{name}' vượt quá {settings.max_upload_mb} MB."),
                )
            try:
                data = await upload.read(settings.max_upload_bytes + 1)
            finally:
                await upload.close()
            if not data:
                raise HTTPException(status_code=422, detail=f"Ảnh '{name}' rỗng.")
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Ảnh '{name}' vượt quá {settings.max_upload_mb} MB.",
                )
            payloads.append((name, data))

        record = store.create(
            input_views=len(payloads),
            file_path=settings.output_dir / f"picto3d_placeholder_{time.time_ns()}.glb",
            file_name="",
        )
        file_name = f"picto3d_{record.id}.glb"
        record.file_path = settings.output_dir / file_name
        record.file_name = f"picto3d_{record.id}.glb"

        base_url = settings.public_base_url or str(request.base_url).rstrip("/")
        job_request = PipelineRequest(
            job_id=record.id,
            files=payloads,
            options=ReconstructOptions(
                foreground_ratio=(
                    foreground_ratio
                    if foreground_ratio is not None
                    else settings.default_foreground_ratio
                ),
                mc_resolution=(
                    mc_resolution
                    if mc_resolution is not None
                    else settings.default_mc_resolution
                ),
                target_faces=(
                    target_faces
                    if target_faces is not None
                    else settings.default_target_faces
                ),
                remove_background=remove_background,
            ),
            base_url=base_url,
        )

        if not wait:
            future = executor.submit(pipeline.run, job_request)
            future.add_done_callback(_log_future_failure)
            response.status_code = 202
            return store.to_status(record)

        try:
            result = await run_in_threadpool(pipeline.run, job_request)
        except Picto3DError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("unexpected reconstruction failure")
            raise HTTPException(
                status_code=500, detail="Lỗi máy chủ khi dựng mô hình."
            ) from exc
        return result

    @app.get("/api/jobs/{job_id}", response_model=JobStatus, tags=["reconstruct"])
    async def get_job(job_id: str) -> JobStatus:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy job.")
        return store.to_status(record)

    @app.get("/api/download/{job_id}", tags=["reconstruct"])
    async def download(job_id: str) -> FileResponse:
        record = store.get(job_id)
        if (
            record is None
            or record.status != JobState.COMPLETED
            or not record.file_path.exists()
        ):
            raise HTTPException(
                status_code=404, detail="File không tồn tại hoặc đã hết hạn."
            )
        return FileResponse(
            path=record.file_path,
            media_type="model/gltf-binary",
            filename=record.file_name or f"picto3d-{job_id}.glb",
            headers={"Cache-Control": "no-store"},
        )

    return app


def _log_future_failure(future: Future[ReconstructResult]) -> None:
    try:
        exc = future.exception()
    except Exception:  # pragma: no cover - cancelled future
        return
    if exc is not None:
        logger.error("background reconstruction failed: %s", exc, exc_info=exc)


def _preload(engine: ReconstructionEngine, settings: Settings) -> None:
    try:
        get_rembg_session(settings.segmentation_model)
    except Exception:
        logger.warning("rembg preload failed", exc_info=True)
    try:
        engine.load()
    except Exception:
        logger.exception("model preload failed; it will retry on first request")


async def _cleanup_loop(store: JobStore, settings: Settings) -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            store.cleanup(settings.file_ttl_seconds)
        except Exception:
            logger.warning("job cleanup failed", exc_info=True)


app = create_app()
