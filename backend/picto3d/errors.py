"""Domain errors that map onto friendly HTTP responses."""

from __future__ import annotations


class Picto3DError(Exception):
    """Base class for expected, user-facing pipeline errors."""

    status_code = 400


class ImageDecodeError(Picto3DError):
    status_code = 422


class PreprocessError(Picto3DError):
    status_code = 422


class ReconstructionError(Picto3DError):
    status_code = 500


class GpuOutOfMemoryError(Picto3DError):
    status_code = 503

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "GPU không đủ VRAM để dựng mô hình. Hãy thử giảm chất lượng lưới "
            "hoặc giải phóng GPU rồi chạy lại."
        )
