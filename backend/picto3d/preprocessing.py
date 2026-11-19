"""Image decoding, background removal and model-ready framing.

TripoSR expects a single RGB image of a centered subject on a neutral gray
background.  This module turns arbitrary uploads into exactly that, and scores
each view so the best (usually the front) view is used for reconstruction.
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import ImageDecodeError, PreprocessError

logger = logging.getLogger("picto3d.preprocess")

# Guard against decompression bombs while still allowing 40 MP photos.
Image.MAX_IMAGE_PIXELS = 50_000_000

_NEUTRAL_GRAY = (127, 127, 127)
_MAX_SOURCE_EDGE = 2048

# Pillow >= 9.1 exposes the resampling enums; older aliases are being removed.
_LANCZOS = Image.Resampling.LANCZOS
_BILINEAR = Image.Resampling.BILINEAR

_session_lock = threading.Lock()
_sessions: dict[str, object] = {}


@dataclass(frozen=True)
class PreprocessedView:
    index: int
    image: Image.Image
    score: float
    coverage: float
    sharpness: float

    @property
    def label(self) -> str:
        return f"view {self.index + 1}"


@dataclass(frozen=True)
class PreprocessOutput:
    views: list[PreprocessedView]
    selected: PreprocessedView
    background_removed: bool


def decode_image(data: bytes, *, name: str = "upload") -> Image.Image:
    """Validate and decode an uploaded image into an upright RGB image."""

    if not data:
        raise ImageDecodeError(f"Tệp '{name}' rỗng.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.load()
            image = ImageOps.exif_transpose(probe)
            has_alpha = "A" in image.getbands() or (
                image.mode == "P" and "transparency" in image.info
            )
            image = image.convert("RGBA" if has_alpha else "RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError(
            f"Không đọc được ảnh '{name}'. Hãy dùng PNG, JPG hoặc WEBP hợp lệ."
        ) from exc

    if max(image.size) > _MAX_SOURCE_EDGE:
        image.thumbnail((_MAX_SOURCE_EDGE, _MAX_SOURCE_EDGE), _LANCZOS)
    return image


def get_rembg_session(model_name: str) -> object | None:
    """Create (and cache) a rembg session, falling back to known models."""

    try:
        from rembg import new_session
    except Exception as exc:  # pragma: no cover - dependency error
        raise PreprocessError(
            "Thiếu thư viện rembg. Hãy chạy: pip install rembg"
        ) from exc

    candidates = [model_name, "isnet-general-use", "u2net"]
    seen: set[str] = set()
    with _session_lock:
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in _sessions:
                return _sessions[candidate]
            try:
                session = new_session(candidate)
            except Exception:
                logger.warning("rembg model '%s' unavailable", candidate)
                continue
            _sessions[candidate] = session
            logger.info("rembg session ready: %s", candidate)
            return session
    return None


def remove_background(image: Image.Image, session: object | None) -> Image.Image:
    """Return an RGBA cutout. Existing soft alpha is kept as-is."""

    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        low, high = alpha.getextrema()
        if low < 250 and high > 0:
            return image

    try:
        from rembg import remove
    except Exception as exc:  # pragma: no cover
        raise PreprocessError("Không thể tách nền: thiếu thư viện rembg.") from exc

    try:
        cutout = remove(
            image.convert("RGB"),
            session=session,
            post_process_mask=False,
        )
    except Exception as exc:
        raise PreprocessError(
            "Tách nền thất bại. Hãy thử ảnh rõ hơn hoặc tắt tách nền nếu ảnh đã "
            "có nền trong suốt."
        ) from exc

    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    return cutout


def _score_alpha(rgba: Image.Image) -> tuple[float, float, float]:
    """Heuristic quality score: coverage, bbox fill, sharpness, centering."""

    sample = rgba.resize((256, 256), _BILINEAR)
    arr = np.asarray(sample, dtype=np.float32)
    alpha = arr[..., 3] / 255.0
    mask = alpha > 0.12
    coverage = float(mask.mean())
    if coverage <= 0.0 or not mask.any():
        return 0.0, 0.0, 0.0

    ys, xs = np.nonzero(mask)
    w = float(xs.max() - xs.min() + 1) / 256.0
    h = float(ys.max() - ys.min() + 1) / 256.0
    bbox_fill = float(np.clip(coverage / max(w * h, 1e-4), 0.0, 1.0))

    rgb = arr[..., :3] / 255.0
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gy, gx = np.gradient(luma)
    sharpness = float(np.var(gx[mask]) + np.var(gy[mask]))
    sharp_norm = float(np.clip(sharpness / 0.0015, 0.0, 1.0))

    cx = float(xs.mean()) / 256.0
    cy = float(ys.mean()) / 256.0
    center_offset = min(1.0, (abs(cx - 0.5) + abs(cy - 0.5)) * 2.0)
    center_score = 1.0 - center_offset

    coverage_score = float(np.clip(coverage / 0.45, 0.0, 1.0))
    score = (
        0.42 * coverage_score
        + 0.23 * bbox_fill
        + 0.25 * sharp_norm
        + 0.10 * center_score
    )
    return round(float(score), 4), round(coverage, 4), round(sharpness, 6)


def _center_foreground(
    rgba: Image.Image,
    ratio: float,
    size: int,
    background: tuple[int, int, int] = _NEUTRAL_GRAY,
) -> Image.Image:
    """Crop to alpha, pad to a square (``ratio`` of its edge) and composite."""

    arr = np.asarray(rgba.convert("RGBA"))
    alpha = arr[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    if len(xs) == 0:
        raise PreprocessError(
            "Không tìm thấy vật thể sau khi tách nền. Hãy dùng ảnh có chủ thể rõ."
        )

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    fg = arr[y0:y1, x0:x1]
    h, w = fg.shape[:2]

    side = int(np.ceil(max(h, w) / max(ratio, 0.05)))
    side = max(side, 64)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = fg

    framed = Image.fromarray(canvas).resize((size, size), _LANCZOS)
    plate = Image.new("RGB", (size, size), background)
    plate.paste(framed, (0, 0), framed)
    return plate


def _letterbox_square(
    image: Image.Image,
    size: int,
    background: tuple[int, int, int] = _NEUTRAL_GRAY,
) -> Image.Image:
    """Resize a pre-processed image into a square gray plate."""

    plate = Image.new("RGB", (size, size), background)
    copy = image.copy()
    copy.thumbnail((size, size), _LANCZOS)
    offset = ((size - copy.width) // 2, (size - copy.height) // 2)
    if copy.mode == "RGBA":
        plate.paste(copy, offset, copy)
    else:
        plate.paste(copy.convert("RGB"), offset)
    return plate


def preprocess_images(
    payloads: Iterable[tuple[str, bytes]],
    settings: Settings,
    *,
    foreground_ratio: float,
    remove_background: bool,
) -> PreprocessOutput:
    """Decode, segment, frame and rank every uploaded view."""

    items = list(payloads)
    if not items:
        raise PreprocessError("Chưa có ảnh đầu vào.")

    session = (
        get_rembg_session(settings.segmentation_model) if remove_background else None
    )
    views: list[PreprocessedView] = []

    for index, (name, payload) in enumerate(items):
        image = decode_image(payload, name=name)

        if remove_background:
            cutout = remove_background(image, session)
            score, coverage, sharpness = _score_alpha(cutout)
            framed = _center_foreground(
                cutout,
                ratio=foreground_ratio,
                size=settings.image_size,
            )
        else:
            score, coverage, sharpness = 0.5, 1.0, 0.0
            framed = _letterbox_square(image, settings.image_size)

        views.append(
            PreprocessedView(
                index=index,
                image=framed,
                score=score,
                coverage=coverage,
                sharpness=sharpness,
            )
        )

    if not views:
        raise PreprocessError("Không có ảnh hợp lệ nào được xử lý.")

    # TripoSR is single-image: pick the strongest view, but keep the first
    # upload unless another angle is clearly better (front views are usually
    # uploaded first and reconstruct more faithfully).
    best = max(views, key=lambda view: view.score)
    if views[0].score >= best.score * 0.92:
        best = views[0]

    logger.info(
        "preprocessed %d view(s); selected=%s score=%.3f coverage=%.3f",
        len(views),
        best.label,
        best.score,
        best.coverage,
    )
    return PreprocessOutput(
        views=views,
        selected=best,
        background_removed=remove_background,
    )


def save_debug_image(image: Image.Image, path: Path) -> None:
    """Optional helper used while tuning preprocessing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
