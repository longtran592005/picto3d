"""In-memory, thread-safe job registry with TTL cleanup."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import JobState, JobStatus, ReconstructResult, Stage

logger = logging.getLogger("picto3d.jobs")


@dataclass
class JobRecord:
    id: str
    input_views: int
    file_path: Path
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: JobState = JobState.QUEUED
    stage: Stage = Stage.QUEUED
    progress: float = 0.0
    message: str = "Đang chờ xử lý…"
    result: ReconstructResult | None = None
    error: str | None = None
    file_name: str = ""


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}

    def create(self, *, input_views: int, file_path: Path, file_name: str) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id,
            input_views=input_views,
            file_path=file_path,
            file_name=file_name,
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

    def update(
        self,
        job_id: str,
        *,
        status: JobState | None = None,
        stage: Stage | None = None,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if status is not None:
                record.status = status
            if stage is not None:
                record.stage = stage
            if progress is not None:
                record.progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                record.message = message
            record.updated_at = time.time()

    def complete(self, job_id: str, result: ReconstructResult) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = JobState.COMPLETED
            record.stage = Stage.DONE
            record.progress = 1.0
            record.message = "Hoàn tất"
            record.result = result
            record.error = None
            record.updated_at = time.time()

    def fail(self, job_id: str, message: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = JobState.FAILED
            record.stage = Stage.ERROR
            record.message = "Xử lý thất bại"
            record.error = message
            record.updated_at = time.time()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record

    def to_status(self, record: JobRecord) -> JobStatus:
        return JobStatus(
            job_id=record.id,
            status=record.status,
            stage=record.stage,
            progress=round(record.progress, 4),
            message=record.message,
            created_at=record.created_at,
            updated_at=record.updated_at,
            input_views=record.input_views,
            result=record.result,
            error=record.error,
        )

    def queue_depth(self) -> int:
        with self._lock:
            return sum(
                1
                for record in self._jobs.values()
                if record.status in (JobState.QUEUED, JobState.RUNNING)
            )

    def cleanup(self, ttl_seconds: int) -> int:
        cutoff = time.time() - max(60, ttl_seconds)
        removed = 0
        with self._lock:
            expired = [
                job_id
                for job_id, record in self._jobs.items()
                if record.status in (JobState.COMPLETED, JobState.FAILED)
                and record.updated_at < cutoff
            ]
            for job_id in expired:
                record = self._jobs.pop(job_id)
                try:
                    record.file_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("could not delete %s", record.file_path, exc_info=True)
                removed += 1
        if removed:
            logger.info("cleaned up %d expired job(s)", removed)
        return removed
