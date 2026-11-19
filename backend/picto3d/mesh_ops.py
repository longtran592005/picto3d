"""Mesh cleanup, smoothing, decimation and GLB export."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .errors import ReconstructionError

logger = logging.getLogger("picto3d.mesh")


@dataclass(frozen=True)
class MeshStats:
    vertices: int
    faces: int
    file_size_bytes: int


def orient_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """TripoSR produces Z-up geometry; glTF and Three.js expect Y-up."""

    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return mesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop degenerate/duplicate faces and unreferenced vertices."""

    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ReconstructionError("Mô hình 3D rỗng, không thể xuất file.")

    try:
        mesh.remove_infinite_values()
    except Exception:
        logger.debug("remove_infinite_values skipped", exc_info=True)

    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        logger.debug("nondegenerate_faces skipped", exc_info=True)

    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        logger.debug("unique_faces skipped", exc_info=True)

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        logger.debug("remove_unreferenced_vertices skipped", exc_info=True)

    if mesh.faces.size == 0:
        raise ReconstructionError("Mô hình 3D không còn mặt nào sau khi làm sạch.")
    return mesh


def _read_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    try:
        if mesh.visual.kind != "vertex":
            return None
        colors = np.asarray(mesh.visual.vertex_colors)
    except Exception:
        return None

    if colors.ndim != 2 or colors.shape[0] != len(mesh.vertices) or colors.shape[1] < 3:
        return None
    colors = (
        colors[:, :4]
        if colors.shape[1] >= 4
        else np.hstack(
            [colors[:, :3], np.full((colors.shape[0], 1), 255, dtype=colors.dtype)]
        )
    )
    if colors.dtype != np.uint8:
        values = colors.astype(np.float32)
        if values.max(initial=0.0) <= 1.0:
            values = values * 255.0
        colors = np.clip(values, 0, 255).astype(np.uint8)
    return colors


def _transfer_vertex_colors(
    source_vertices: np.ndarray,
    source_colors: np.ndarray,
    target_vertices: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Nearest-neighbour (inverse distance) colour transfer after decimation."""

    from scipy.spatial import cKDTree

    if len(source_vertices) == 0:
        return np.full((len(target_vertices), 4), 180, dtype=np.uint8)

    tree = cKDTree(source_vertices)
    neighbours = min(k, len(source_vertices))
    distances, indices = tree.query(target_vertices, k=neighbours)

    if neighbours == 1:
        return source_colors[np.asarray(indices, dtype=np.int64)]

    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    mixed = (source_colors[indices].astype(np.float64) * weights[..., None]).sum(axis=1)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 2) -> trimesh.Trimesh:
    """Light Taubin smoothing: shrinks noise but preserves overall volume."""

    if iterations <= 0 or len(mesh.vertices) < 32:
        return mesh
    try:
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=0.5,
            nu=0.53,
            iterations=iterations,
        )
    except Exception:
        logger.warning("Taubin smoothing failed, keeping raw mesh", exc_info=True)
    return mesh


def decimate_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int,
    *,
    keep_colors: bool = True,
) -> trimesh.Trimesh:
    """Quadric decimation with vertex-colour transfer for the web budget."""

    face_count = len(mesh.faces)
    if target_faces <= 0 or face_count <= target_faces:
        return mesh

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    colors = _read_vertex_colors(mesh) if keep_colors else None

    try:
        import fast_simplification

        reduction = 1.0 - (float(target_faces) / float(face_count))
        reduction = min(0.9999, max(0.0001, reduction))
        new_vertices, new_faces = fast_simplification.simplify(
            vertices,
            faces,
            target_reduction=reduction,
            agg=5,
        )
        new_vertices = np.asarray(new_vertices, dtype=np.float64)
        new_faces = np.asarray(new_faces, dtype=np.int64)
    except Exception:
        logger.warning(
            "fast-simplification failed, trying trimesh decimation", exc_info=True
        )
        try:
            reduced = mesh.simplify_quadric_decimation(face_count=int(target_faces))
            new_vertices = np.asarray(reduced.vertices, dtype=np.float64)
            new_faces = np.asarray(reduced.faces, dtype=np.int64)
        except Exception as exc:
            logger.warning("Decimation unavailable, exporting full mesh", exc_info=True)
            raise ReconstructionError(
                "Không thể giảm số mặt lưới. Hãy cài fast-simplification."
            ) from exc

    if len(new_faces) == 0 or len(new_vertices) == 0:
        logger.warning("Decimation returned an empty mesh, keeping original")
        return mesh

    simple = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False,
    )

    if colors is not None:
        try:
            simple.visual = trimesh.visual.ColorVisuals(
                mesh=simple,
                vertex_colors=_transfer_vertex_colors(vertices, colors, new_vertices),
            )
        except Exception:
            logger.warning("Vertex colour transfer failed", exc_info=True)

    try:
        simple.remove_unreferenced_vertices()
    except Exception:
        logger.debug("remove_unreferenced_vertices skipped", exc_info=True)

    logger.info(
        "decimated %d -> %d faces (%d verts)",
        face_count,
        len(simple.faces),
        len(simple.vertices),
    )
    return simple


def normalize_for_web(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center X/Z, put the feet on Y=0 and fit inside a 1 m cube."""

    mesh.apply_translation(-mesh.bounding_box.centroid)
    extents = mesh.extents
    largest = float(max(extents)) if len(extents) else 0.0
    if largest > 0:
        mesh.apply_scale(1.0 / largest)

    bounds = mesh.bounds
    mesh.apply_translation(
        [
            -(bounds[0][0] + bounds[1][0]) * 0.5,
            -bounds[0][1],
            -(bounds[0][2] + bounds[1][2]) * 0.5,
        ]
    )
    return mesh


def export_glb(mesh: trimesh.Trimesh, path: Path) -> MeshStats:
    """Export a binary glTF and return final geometry statistics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.export(str(path), file_type="glb")
    except Exception as exc:
        raise ReconstructionError(f"Xuất file GLB thất bại: {exc}") from exc

    if not path.exists() or path.stat().st_size == 0:
        raise ReconstructionError("File GLB được tạo nhưng rỗng.")

    stats = MeshStats(
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        file_size_bytes=int(path.stat().st_size),
    )
    logger.info(
        "exported %s (%d verts, %d faces, %.2f MB)",
        path.name,
        stats.vertices,
        stats.faces,
        stats.file_size_bytes / (1024 * 1024),
    )
    return stats


def postprocess(
    mesh: trimesh.Trimesh,
    *,
    target_faces: int,
    y_up: bool = True,
) -> trimesh.Trimesh:
    if y_up:
        mesh = orient_y_up(mesh)
    mesh = clean_mesh(mesh)
    mesh = decimate_mesh(mesh, target_faces=target_faces, keep_colors=True)
    mesh = smooth_mesh(mesh, iterations=2)
    mesh = clean_mesh(mesh)
    mesh = normalize_for_web(mesh)
    return mesh
