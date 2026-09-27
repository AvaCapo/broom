"""Root-motion transfer helpers for retargeting."""

from collections.abc import Mapping

import numpy as np

from broom import Hierarchy, Motion
from broom.kinematics import compute_global_positions
from broom.ops.skeleton_geometry import estimate_height


# TODO: Review name-based joint selection; these substrings are not universal
# skeleton semantics and can exclude relevant joints from floor estimation.
_FLOOR_JOINT_NAME_MARKERS = ("foot", "toe", "ankle", "heel")


#TODO: Consider renaming function because of possible confusion with scale_skeleton from bvh.ops.skeleton
def skeleton_scale(
    source_hierarchy: Hierarchy,
    target_hierarchy: Hierarchy,
) -> float:
    """Return target/source skeleton size ratio for root translation."""

    source_size = estimate_height(source_hierarchy, include_end_sites=False)
    target_size = estimate_height(target_hierarchy, include_end_sites=False)
    if source_size <= 1.0e-8 or target_size <= 1.0e-8:
        return 1.0
    return float(target_size / source_size)


def transfer_root_translation(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    target_motion: np.ndarray,
    scale: float,
) -> None:
    """Copy source root translation into the output target motion."""

    source_root = _root_position_channels(source_motion.hierarchy)
    target_root = _root_position_channels(target_hierarchy)
    if not source_root or not target_root:
        return

    source_positions = _extract_positions(
        motion_values=source_motion.values,
        channels=source_root,
    )
    root_delta = (source_positions - source_positions[0:1]) * float(scale)
    _write_positions(
        motion_values=target_motion,
        channels=target_root,
        positions=root_delta,
    )


# TODO: Unify floor estimation in analysis and retargeting, with analysis as
# the source of truth. Keep the retargeting implementation for now.
def estimate_floor_level(
    motion: Motion,
    *,
    use_rest_pose: bool = True,
    first_frame_only: bool = False,
) -> float:
    """Return the minimum world Y of foot/toe/ankle/heel origins in the selected poses.

    Use the whole clip unless first_frame_only selects just its first frame.
    Fall back to all joints when no foot/toe/ankle/heel names exist. With use_rest_pose,
    also include a pose with zero rotations and the first frame's translations.
    Without animation frames, use zero channels for that rest pose.
    The motion is not modified; the returned value is a level, not a shift.
    """
    hierarchy = motion.hierarchy
    if not hierarchy.joints:
        raise ValueError("Floor estimation requires at least one joint.")

    # TODO: Clarify the original use case for first-frame-only floor estimation.
    values = motion.values[:1] if first_frame_only else motion.values
    if use_rest_pose:
        rest = (
            values[:1].copy() if values.shape[0]
            else np.zeros((1, hierarchy.total_channels), dtype=values.dtype)
        )
        for joint_index, joint in enumerate(hierarchy.joints):
            for offset, channel in enumerate(joint.channels):
                if channel.endswith("rotation"):
                    rest[0, hierarchy.channel_start(joint_index) + offset] = 0.0
        values = np.concatenate((values, rest), axis=0)
    positions = compute_global_positions(motion.with_values(values))
    floor_indices = [
        index
        for index, joint in enumerate(hierarchy.joints)
        if any(marker in joint.name.casefold() for marker in _FLOOR_JOINT_NAME_MARKERS)
    ]
    if not floor_indices:
        floor_indices = list(range(hierarchy.joint_count))

    return float(positions[:, floor_indices, 1].min())


def align_root_to_floor(
    motion: Motion,
    floor_height: float = 0.0,
    *,
    use_rest_pose: bool = True,
    first_frame_only: bool = False,
) -> Motion:
    """Return a copy shifted by one constant root Y translation to the floor.

    Estimate over all frames or just the first, optionally including rest pose
    as in estimate_floor_level. Shift the entire clip in either case; preserve
    frame count, timing and all other channels. Requires nonempty motion.
    """
    if motion.values.shape[0] == 0:
        raise ValueError("Floor alignment requires at least one motion frame.")
    if not np.isfinite(floor_height):
        raise ValueError("floor_height must be finite.")
    root_y_channel = _root_position_channels(motion.hierarchy).get(1)
    if root_y_channel is None:
        raise ValueError("Floor alignment requires a root Yposition channel.")
    level = estimate_floor_level(
        motion, use_rest_pose=use_rest_pose, first_frame_only=first_frame_only,
    )
    values = motion.values.copy()
    values[:, root_y_channel] += float(floor_height) - level
    return motion.with_values(values)


def _root_position_channels(hierarchy: Hierarchy) -> dict[int, int]:
    root_joint = hierarchy.root_joint

    result = {}
    for offset, channel in enumerate(root_joint.channels):
        if not channel.endswith("position"):
            continue
        dimension = {"X": 0, "Y": 1, "Z": 2}[channel[0]]
        result[dimension] = hierarchy.channel_start(hierarchy.root) + offset
    return result


def _extract_positions(
    motion_values: np.ndarray,
    channels: Mapping[int, int],
) -> np.ndarray:
    positions = np.zeros((motion_values.shape[0], 3), dtype=np.float64)
    for dimension, channel_index in channels.items():
        positions[:, dimension] = motion_values[:, channel_index]
    return positions


def _write_positions(
    motion_values: np.ndarray,
    channels: Mapping[int, int],
    positions: np.ndarray,
) -> None:
    for dimension, channel_index in channels.items():
        motion_values[:, channel_index] = positions[:, dimension]
