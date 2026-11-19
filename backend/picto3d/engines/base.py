"""Reconstruction engine interface."""

from __future__ import annotations

from typing import Protocol

import PIL.Image
import trimesh


class ReconstructionEngine(Protocol):
    name: str
    device: str

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def reconstruct(
        self,
        image: PIL.Image.Image,
        *,
        mc_resolution: int,
    ) -> trimesh.Trimesh: ...

    def unload(self) -> None: ...
