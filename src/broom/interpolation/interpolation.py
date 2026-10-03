"""BVH motion interpolation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from broom import Motion
from broom.interpolation.config import (
    DEFAULT_PRECISION,
    DEFAULT_STATIC_THRESHOLD,
    DEFAULT_STATIC_WINDOW,
)
from broom.interpolation.utils import interpolate_motion
from broom.io import load_bvh, write_bvh
from broom.ops.motion_editing import trim_trailing_static_frames


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
    ) -> None:
        self.min_frames = int(min_frames)
        self.check_last_duplicates = bool(check_last_duplicates)
        self.root_name = root_name
        self.static_threshold = float(static_threshold)
        self.static_window = int(static_window)
        self.precision = int(precision)
        #TODO thgink about: interpolation and application on different template.
        self.template_motion = (
            load_bvh(bvh_template_path, root_name=None)
            if bvh_template_path is not None
            else None
        )

    def load_bvh_frames(
        self,
        filename: str | Path,
        root_name: str | None = None,
    ) -> Motion:
        """Load a BVH file into a Motion."""

        return load_bvh(
            filename,
            root_name=root_name if root_name is not None else self.root_name,
        )

    def save_bvh(self, motion: Motion, output_path: str | Path) -> None:
        """Save a Motion as a BVH file."""

        write_bvh(motion, output_path, precision=self.precision)

    def interpolate_animations(
        self,
        first_motion: Motion,
        second_motion: Motion,
        transition_frames: int = 30,
    ) -> Motion:
        """Interpolate two loaded BVH motions."""

        return interpolate_motion(
            first_motion,
            second_motion,
            transition_frames,
        )

    def interpolate(
        self,
        first_animation: str | Path,
        second_animation: str | Path,
        transition_frames: int,
    ) -> Motion:
        """Interpolate two BVH files."""
        #TODO: why trim static frames? maybe it should be optional?
        first_motion = self._trim_static_frames(
            self.load_bvh_frames(first_animation)
        )
        second_motion = self._trim_static_frames(
            self.load_bvh_frames(second_animation)
        )
        return self.interpolate_animations(
            first_motion=first_motion,
            second_motion=second_motion,
            transition_frames=transition_frames,
        )

    def batch_interpolate(
        self,
        animation_list: Sequence[str | Path],
        transition_frames: int,
    ) -> Motion:
        """Sequentially interpolate a list of BVH files."""

        if not animation_list:
            raise ValueError("Animation list is empty.")

        #TODO: why trim static frames? maybe it should be optional?
        motion = self._trim_static_frames(self.load_bvh_frames(animation_list[0]))
        for animation_path in animation_list[1:]:
            #TODO: why trim static frames? maybe it should be optional?
            next_motion = self._trim_static_frames(self.load_bvh_frames(animation_path))
            motion = self.interpolate_animations(
                first_motion=motion,
                second_motion=next_motion,
                transition_frames=transition_frames,
            )
        return motion

    def process(
        self,
        first_animation: str | Path,
        second_animation: str | Path,
        transition_frames: int,
        output_path: str | Path,
    ) -> None:
        """Interpolate two files and write the resulting BVH."""

        self.save_bvh(
            motion=self.interpolate(
                first_animation=first_animation,
                second_animation=second_animation,
                transition_frames=transition_frames,
            ),
            output_path=output_path,
        )

    def process_batch(
        self,
        animation_list: Sequence[str | Path],
        transition_frames: int,
        output_path: str | Path,
    ) -> None:
        """Interpolate multiple files and write the resulting BVH."""

        self.save_bvh(
            motion=self.batch_interpolate(
                animation_list=animation_list,
                transition_frames=transition_frames,
            ),
            output_path=output_path,
        )

    def _trim_static_frames(self, motion: Motion) -> Motion:
        """Optionally remove static frames from the end of a motion."""

        if not self.check_last_duplicates:
            return motion.with_values(motion.values)

        values = trim_trailing_static_frames(
            motion_values=motion.values,
            min_frames=self.min_frames,
            threshold=self.static_threshold,
            window=self.static_window,
        )
        return motion.with_values(values)
