"""In-place BVH transform."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from broom import Motion
from broom.bvh.in_place.schemas import InPlacePCAResult
from broom.bvh.in_place.config import (
    DEFAULT_BODY_JOINT_WEIGHTS,
    DEFAULT_ROOT_AXES,
    InPlacePCAConfig,
    PCASource,
    PCA_SOURCE_BODY,
    PCA_SOURCE_ROOT,
)

from broom.math_helpers import principal_direction
from broom.kinematics import compute_global_positions
from broom.channels import (
    position_channel_dimension,
    root_points,
    select_body_joints,
    weighted_body_points
)
from broom.io import load_bvh, write_bvh
from broom.foot_lock import apply_root_foot_lock

from broom.bvh.in_place.utils import (
    remove_smoothed_pca_trend,
    validate_axis_set,
)


class InPlaceConverter:
    """Converter for in-place BVH transformations."""

    def __init__(
        self,
        root_name: str | None,
        axes: Sequence[str],
        source: PCASource,
        smooth_window: int,
        zero_origin: bool,
        precision: int,
        foot_blend_frames: int,
        foot_height_threshold: float | None,
        foot_velocity_threshold: float | None,
        body_joint_weights: Sequence[tuple[str, float]],
        use_all_joints: bool,
    ):
        self.root_name = root_name
        self.axes = axes
        self.source = self._validate_source(source)
        self.smooth_window = smooth_window
        self.zero_origin = zero_origin
        self.precision = precision
        self.foot_blend_frames = foot_blend_frames
        self.foot_height_threshold = foot_height_threshold
        self.foot_velocity_threshold = foot_velocity_threshold
        self.body_joint_weights = body_joint_weights
        self.use_all_joints = use_all_joints
        validate_axis_set(
            axes=self.axes,
            expected_axes=DEFAULT_ROOT_AXES,
            context="PCA in-place conversion",
        )

    @staticmethod
    def _validate_source(source: str) -> PCASource:
        """Validate and normalize the configured PCA source."""

        if source not in (PCA_SOURCE_ROOT, PCA_SOURCE_BODY):
            raise ValueError(
                f"Unsupported PCA source: {source!r}. "
                f"Expected {PCA_SOURCE_ROOT!r} or {PCA_SOURCE_BODY!r}."
            )
        return source

    @classmethod
    def load(cls, config: InPlacePCAConfig) -> InPlaceConverter:
        """Load configuration from an InPlacePCAConfig dataclass."""

        return cls(
            root_name=config.root_name,
            axes=config.axes,
            source=config.source,
            smooth_window=config.smooth_window,
            zero_origin=config.zero_origin,
            precision=config.precision,
            foot_blend_frames=config.foot_blend_frames,
            foot_height_threshold=config.foot_height_threshold,
            foot_velocity_threshold=config.foot_velocity_threshold,
            body_joint_weights=config.body_joint_weights,
            use_all_joints=config.use_all_joints,
        )

    def transform_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        foot_lock: bool,
    ) -> InPlacePCAResult:
        """Read a BVH file, transform it in-place, and write a new BVH."""

        motion = load_bvh(input_path, root_name=self.root_name)
        root_points, result_data = self.transform_motion(
            motion=motion,
            foot_lock=foot_lock,
        )
        axis_to_column = {
            axis: index for index, axis in enumerate(DEFAULT_ROOT_AXES)
        }
        root_channel_values = np.column_stack(
            [root_points[:, axis_to_column[axis]] for axis in self.axes]
        )
        values = motion.values.copy()
        root_name = motion.hierarchy.root_name
        root_channel_indices = [
            motion.hierarchy.channel_index(root_name, axis) for axis in self.axes
        ]
        values[:, root_channel_indices] = root_channel_values
        output_motion = motion.with_values(values)
        write_bvh(
            output_motion,
            output_path,
            precision=self.precision,
        )

        return InPlacePCAResult(
            input_path=Path(input_path),
            output_path=Path(output_path),
            root_name=motion.hierarchy.root_name,
            source=self.source,
            frames=motion.frame_count,
            dominant_direction=(
                float(result_data["direction"][0]),
                float(result_data["direction"][1]),
            ),
            explained_ratio=float(result_data["explained_ratio"]),
            selected_joints=tuple(result_data["selected_joints"]),
            foot_lock=foot_lock,
        )

    def transform_motion(
        self, motion: Motion, foot_lock: bool
    ) -> tuple[np.ndarray, dict]:
        """Transform an already loaded Motion.

        :param motion: Motion whose root translation is transformed.
        """

        original_root_points = root_points(
            motion=motion,
            axes=DEFAULT_ROOT_AXES,
        )
        global_positions = None
        selected_joints: tuple[str, ...] = ()

        if self.source == PCA_SOURCE_ROOT:
            sensor_points = original_root_points
        else:
            (
                sensor_points,
                global_positions,
                selected_joints,
            ) = self._body_sensor_data(motion)

        direction, explained_ratio = principal_direction(sensor_points)
        transformed_root_points = remove_smoothed_pca_trend(
            root_points=original_root_points,
            sensor_points=sensor_points,
            direction=direction,
            smooth_window=self.smooth_window,
            zero_origin=self.zero_origin,
        )

        if foot_lock:
            if global_positions is None:
                global_positions = compute_global_positions(motion)
            transformed_root_points = apply_root_foot_lock(
                root_points=transformed_root_points,
                original_root_points=original_root_points,
                global_positions=global_positions,
                hierarchy=motion.hierarchy,
                blend_frames=self.foot_blend_frames,
                height_threshold=self.foot_height_threshold,
                velocity_threshold=self.foot_velocity_threshold,
            )

        return transformed_root_points, {
            "direction": direction,
            "explained_ratio": explained_ratio,
            "selected_joints": selected_joints,
        }

    def _body_sensor_data(
        self, motion: Motion
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Build weighted body sensor points and metadata for PCA."""

        global_positions = compute_global_positions(motion)
        selected_indices, weights = select_body_joints(
            joints=motion.hierarchy.joints,
            weighted_targets=self.body_joint_weights,
            use_all_joints=self.use_all_joints,
        )
        sensor_points = weighted_body_points(
            global_positions=global_positions,
            selected_indices=selected_indices,
            weights=weights,
            dimensions=tuple(
                position_channel_dimension(axis) for axis in DEFAULT_ROOT_AXES
            ),
        )
        selected_joints = tuple(
            motion.hierarchy.joints[int(index)].name for index in selected_indices
        )
        return sensor_points, global_positions, selected_joints
