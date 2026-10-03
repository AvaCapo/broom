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


def subtract_root_offset_from_translation(motion: Motion) -> Motion:
    """Return a Motion whose root position channels are offset-relative.

    Subtracts each component of ``hierarchy.root_joint.offset`` from the
    matching root position channel. Use this when an external source stores
    root position samples as absolute coordinates rather than Broom's
    offset-plus-translation representation. The hierarchy is unchanged.
    """
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")

    hierarchy = motion.hierarchy
    root = hierarchy.root_joint
    root_start = hierarchy.channel_start(hierarchy.root)
    values = motion.values.copy()
    for channel_offset, channel_name in enumerate(root.channels):
        if not channel_name.endswith("position"):
            continue
        axis_index = "XYZ".index(channel_name[0])
        values[:, root_start + channel_offset] -= root.offset[axis_index]
    return motion.with_values(values)


def add_root_offset_to_translation(motion: Motion) -> Motion:
    """Return a Motion whose root position channels include its root offset.

    Adds each component of ``hierarchy.root_joint.offset`` to the matching
    root position channel. This is the inverse of
    :func:`subtract_root_offset_from_translation`.
    """
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")

    hierarchy = motion.hierarchy
    root = hierarchy.root_joint
    root_start = hierarchy.channel_start(hierarchy.root)
    values = motion.values.copy()
    for channel_offset, channel_name in enumerate(root.channels):
        if not channel_name.endswith("position"):
            continue
        axis_index = "XYZ".index(channel_name[0])
        values[:, root_start + channel_offset] += root.offset[axis_index]
    return motion.with_values(values)


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


# TODO: Should we adapt the below functions to Motion class or still using ndarray?
def count_trailing_static_frames(
    motion_values: np.ndarray,
    threshold: float,
    window: int,
) -> int:
    """Count redundant static frames at the end of motion data.

    The first frame of the static segment is kept as an anchor; only frames
    after it are counted as removable.
    """

    values = np.asarray(motion_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Motion values must be a 2D array.")
    if values.shape[0] < 2:
        return 0

    threshold = float(threshold)
    window = max(1, int(window))
    frame_deltas = np.max(np.abs(np.diff(values, axis=0)), axis=1)

    static_diffs = 0
    for delta in frame_deltas[::-1]:
        if float(delta) <= threshold:
            static_diffs += 1
            continue
        break

    if static_diffs < window:
        return 0
    return static_diffs


def trim_trailing_static_frames(
    motion_values: np.ndarray,
    min_frames: int,
    threshold: float,
    window: int,
) -> np.ndarray:
    """Remove redundant trailing static frames while keeping min_frames."""

    values = np.asarray(motion_values, dtype=np.float64)
    removable = count_trailing_static_frames(
        motion_values=values,
        threshold=threshold,
        window=window,
    )
    if removable == 0:
        return values.copy()

    keep_count = max(int(min_frames), values.shape[0] - removable)
    return values[:keep_count].copy()


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
