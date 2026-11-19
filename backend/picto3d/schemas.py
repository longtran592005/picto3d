"""Pydantic API schemas shared by the endpoints and the job store."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Stage(str, Enum):
    QUEUED = "queued"
    PREPROCESS = "preprocess"
    INFERENCE = "inference"
    MESH = "mesh"
    EXPORT = "export"
    DONE = "done"
    ERROR = "error"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReconstructOptions(BaseModel):
    foreground_ratio: float = Field(0.8, ge=0.5, le=1.0)
    mc_resolution: int = Field(256, ge=128, le=384)
    target_faces: int = Field(40000, ge=10000, le=100000)
    remove_background: bool = True


class ReconstructResult(BaseModel):
    job_id: str
    status: Literal["completed"] = "completed"
    glb_url: str = Field(description="Public URL of the generated GLB file")
    download_url: str = Field(description="URL that forces a file download")
    file_name: str
    elapsed_seconds: float
    preprocess_seconds: float
    inference_seconds: float
    mesh_seconds: float
    vertices: int
    faces: int
    texture: str = "vertex_color"
    input_views: int
    selected_view: int = Field(
        description="1-based index of the view used to reconstruct"
    )
    device: str


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    stage: Stage
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    created_at: float
    updated_at: float
    input_views: int
    result: ReconstructResult | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    device: str
    gpu: str | None = None
    vram_total_mb: int | None = None
    vram_allocated_mb: int | None = None
    model_id: str
    model_loaded: bool
    queue_depth: int
    uptime_seconds: float
    max_images: int
    max_upload_mb: int
