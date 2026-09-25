"""Motion editing operations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from broom import Motion, Hierarchy
from broom.ops.skeleton_editing import scale_offsets


def scale_skeleton(motion: Motion, factor: float) -> Motion:
    """Return a motion with its offsets and translation channels scaled.

    The returned Motion scales joint and End Site offsets and every
    ``Xposition``, ``Yposition`` and ``Zposition`` channel, including
    non-root joints. Rotation channels and frame time are preserved.
    """
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")

    scaled_hierarchy = scale_offsets(motion.hierarchy, factor)
    values = motion.values.copy()
    position_indices = motion.hierarchy.position_channel_indices()
    if position_indices:
        values[:, position_indices] *= float(factor)
    return Motion(scaled_hierarchy, values, motion.frame_time)


def repeat_pose(
    hierarchy: Hierarchy,
    pose: np.ndarray | Sequence[float] | None,
    frame_count: int,
    frame_time: float,
) -> Motion:
    """Return a Motion containing a repeated pose or zero channel values."""
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy.")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("frame_count must be a positive integer.")
    if frame_count <= 0:
        raise ValueError("frame_count must be a positive integer.")
    if pose is None:
        values = np.zeros((frame_count, hierarchy.total_channels), dtype=np.float64)
    else:
        pose_values = np.asarray(pose, dtype=np.float64)
        if pose_values.ndim != 1 or pose_values.shape[0] != hierarchy.total_channels:
            raise ValueError(
                f"pose must have shape ({hierarchy.total_channels},)."
            )
        values = np.repeat(pose_values[None, :], frame_count, axis=0)
    return Motion(hierarchy, values, frame_time)


def trim_frames(
    motion: Motion,
    start: int | None = None,
    stop: int | None = None,
) -> Motion:
    """Return a non-empty frame slice using Python start/stop semantics."""
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")
    normalized_start, normalized_stop, _ = slice(start, stop).indices(
        motion.frame_count
    )
    if normalized_stop <= normalized_start:
        raise ValueError("Frame slice cannot be empty.")
    return motion.with_values(motion.values[normalized_start:normalized_stop])


def reverse(
    motion: Motion,
    keep_root_start: bool = False,
) -> Motion:
    """Return a Motion with frames in reverse order."""
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")
    values = motion.values[::-1].copy()
    # TODO: Decide whether implicit root-start compensation belongs in reverse.
    if keep_root_start:
        indices = motion.hierarchy.position_channel_indices(motion.hierarchy.root_name)
        if indices:
            offset = motion.values[0, indices] - values[0, indices]
            values[:, indices] += offset
    return motion.with_values(values)


# TODO: Define [start, stop) timestamp selection before adding slice_by_time.
# def slice_by_time(
#     document: BVHDocument,
#     start_seconds: float | None = None,
#     end_seconds: float | None = None,
# ) -> BVHDocument:
#     """Return a copy of the document sliced by seconds.

#     end_seconds is exclusive. The start time is floored to the nearest
#     source frame and the end time is ceiled so the requested time range is not
#     accidentally shortened.
#     """

#     frame_time = _require_frame_time(document)
#     start_frame = (
#         None
#         if start_seconds is None
#         else int(np.floor(float(start_seconds) / frame_time))
#     )
#     end_frame = (
#         None if end_seconds is None else int(np.ceil(float(end_seconds) / frame_time))
#     )
#     return trim_frames(
#         document=document,
#         start_frame=start_frame,
#         end_frame=end_frame,
#     )

# TODO: Define zero_root_translation and rebase_root_position contracts separately.
# def zero_origin(
#     document: BVHDocument,
#     axes: Sequence[str] | None = None,
# ) -> BVHDocument:
#     """Return a bvh animation where position channels start at zero.

#     :param document: BVH document to modify.
#     :param axes: Optional list of root position channel names to zero.
#     """

#     indices = _root_position_indices_for_axes(document=document, axes=axes)
#     motion_values = document.motion_values.copy()
#     if indices and motion_values.shape[0] > 0:
#         motion_values[:, indices] -= motion_values[0, indices]
#     return with_motion_values(document=document, motion_values=motion_values)

# def _root_position_indices_for_axes(
#     document: BVHDocument,
#     axes: Sequence[str] | None,
# ) -> list[int]:
    
#     """Return absolute root position channel indices for selected axes.
    
#     :param document: BVH document to query for root channels.
#     :param axes: Optional list of root position channel names to select.
#     """

#     root_joint = next(
#         (joint for joint in document.joints if joint.parent == -1),
#         None,
#     )
#     if root_joint is None:
#         return []

#     if axes is None:
#         return [
#             root_joint.channel_start + offset
#             for offset, channel in enumerate(root_joint.channels)
#             if channel.endswith("position")
#         ]

#     indices = []
#     for axis in axes:
#         if axis not in root_joint.channels:
#             raise ValueError(
#                 f"Root channel '{axis}' was not found. "
#                 f"Available root channels: {', '.join(root_joint.channels)}"
#             )
#         indices.append(
#             root_joint.channel_start + root_joint.channels.index(axis)
#             )
#     return indices
