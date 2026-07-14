"""BVH motion interpolation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from broom.bvh.io import (
    load_bvh_document,
    write_bvh_with_motion_values,
)
from broom.bvh.schemas import BVHDocument
from broom.bvh.interpolation.config import (
    DEFAULT_PRECISION,
    DEFAULT_STATIC_THRESHOLD,
    DEFAULT_STATIC_WINDOW,
)
from broom.bvh.interpolation.utils import (
    interpolate_motion_values,
    trim_trailing_static_frames,
)




class Interpolation:
    """BVH interpolation service."""

    def __init__(
        self,
        min_frames: int,
        bvh_template_path: str | Path | None = None,
        check_last_duplicates: bool = False,
        root_name: str = "Hips",
        static_threshold: float = DEFAULT_STATIC_THRESHOLD,
        static_window: int = DEFAULT_STATIC_WINDOW,
        precision: int = DEFAULT_PRECISION,
    ):
        self.min_frames = int(min_frames)
        self.check_last_duplicates = bool(check_last_duplicates)
        self.root_name = root_name
        self.static_threshold = float(static_threshold)
        self.static_window = int(static_window)
        self.precision = int(precision)
        #TODO thgink about: interpolation and application on different template.
        self.template_document = (
            load_bvh_document(bvh_template_path, root_name=None)
            if bvh_template_path is not None
            else None
        )

    def load_bvh_frames(
        self,
        filename: str | Path,
        root_name: str | None = None,
    ) -> tuple[BVHDocument, np.ndarray]:
        """Load a BVH file and return its document plus full motion values."""

        document = load_bvh_document(
            filename,
            root_name=root_name if root_name is not None else self.root_name,
        )
        motion_values = document.motion_values.copy()
        motion_values[np.isnan(motion_values)] = 0.0
        return document, motion_values

    def save_bvh(
        self,
        mocap_template: BVHDocument,
        frames: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """Save interpolated frames using the template BVH hierarchy."""

        write_bvh_with_motion_values(
            document=mocap_template,
            output_path=output_path,
            motion_values=frames,
            precision=self.precision,
        )

    def interpolate_animations(
        self,
        mocap1: BVHDocument,
        frames1: np.ndarray,
        mocap2: BVHDocument,
        frames2: np.ndarray,
        transition_frames: int = 30,
    ) -> np.ndarray:
        """Interpolate two loaded BVH motions."""

        return interpolate_motion_values(
            first_document=mocap1,
            first_values=frames1,
            second_document=mocap2,
            second_values=frames2,
            transition_frames=transition_frames,
        )

    def interpolate(
        self,
        first_animation: str | Path,
        second_animation: str | Path,
        transition_frames: int,
    ) -> tuple[BVHDocument, np.ndarray]:
        """Interpolate two BVH files."""

        document1, frames1 = self.load_bvh_frames(first_animation)
        document2, frames2 = self.load_bvh_frames(second_animation)
        frames1 = self._trim_static_frames(frames1)
        frames2 = self._trim_static_frames(frames2)

        motion = self.interpolate_animations(
            mocap1=document1,
            frames1=frames1,
            mocap2=document2,
            frames2=frames2,
            transition_frames=transition_frames,
        )
        return document1, motion

    def batch_interpolate(
        self,
        animation_list: Sequence[str | Path],
        transition_frames: int,
    ) -> tuple[BVHDocument, np.ndarray]:
        """Sequentially interpolate a list of BVH files."""

        if not animation_list:
            raise ValueError("Animation list is empty.")

        first_document, motion = self.load_bvh_frames(animation_list[0])
        motion = self._trim_static_frames(motion)

        for animation_path in animation_list[1:]:
            next_document, next_motion = self.load_bvh_frames(animation_path)
            next_motion = self._trim_static_frames(next_motion)
            motion = self.interpolate_animations(
                mocap1=first_document,
                frames1=motion,
                mocap2=next_document,
                frames2=next_motion,
                transition_frames=transition_frames,
            )

        return first_document, motion

    def process(
        self,
        first_animation: str | Path,
        second_animation: str | Path,
        transition_frames: int,
        output_path: str | Path,
    ) -> None:
        """Interpolate two files and write the resulting BVH."""

        document, motion = self.interpolate(
            first_animation=first_animation,
            second_animation=second_animation,
            transition_frames=transition_frames,
        )
        self.save_bvh(
            mocap_template=document,
            frames=motion,
            output_path=output_path,
        )

    def process_batch(
        self,
        animation_list: Sequence[str | Path],
        transition_frames: int,
        output_path: str | Path,
    ) -> None:
        """Interpolate multiple files and write the resulting BVH."""

        document, motion = self.batch_interpolate(
            animation_list=animation_list,
            transition_frames=transition_frames,
        )
        self.save_bvh(
            mocap_template=document,
            frames=motion,
            output_path=output_path,
        )

    def _trim_static_frames(self, frames: np.ndarray) -> np.ndarray:
        """Optionally remove static frames from the end of a motion."""

        if not self.check_last_duplicates:
            return np.asarray(frames, dtype=np.float64).copy()
        return trim_trailing_static_frames(
            motion_values=frames,
            min_frames=self.min_frames,
            threshold=self.static_threshold,
            window=self.static_window,
        )
