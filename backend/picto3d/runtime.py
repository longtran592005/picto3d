"""Small runtime helpers (timers, memory release, device detection)."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("picto3d.runtime")


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Measure elapsed wall-clock seconds into ``state['seconds']``."""

    state = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["seconds"] = round(time.perf_counter() - start, 3)


def release_memory() -> None:
    """Drop Python references, collect garbage and empty the CUDA cache."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - best effort only
        logger.debug("CUDA cache release skipped", exc_info=True)


def resolve_device(requested: str) -> str:
    """Turn ``auto`` into ``cuda`` when a GPU is available."""

    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard runtime dep
        return "cpu"

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_summary(device: str) -> tuple[str | None, int | None, int | None]:
    """Return (gpu name, vram total MB, vram allocated MB)."""

    try:
        import torch

        if not torch.cuda.is_available() or not device.startswith("cuda"):
            return None, None, None
        index = 0
        if ":" in device:
            try:
                index = int(device.split(":", 1)[1])
            except ValueError:
                index = 0
        props = torch.cuda.get_device_properties(index)
        allocated = torch.cuda.memory_allocated(index) // (1024 * 1024)
        return props.name, props.total_memory // (1024 * 1024), int(allocated)
    except Exception:  # pragma: no cover - diagnostics only
        return None, None, None
