"""Root-motion transfer helpers for retargeting."""

import numpy as np

from broom import Hierarchy, Motion
from broom.kinematics import compute_global_positions


# TODO: Review name-based joint selection; these substrings are not universal
# skeleton semantics and can exclude relevant joints from floor estimation.
_FLOOR_JOINT_NAME_MARKERS = ("foot", "toe", "ankle", "heel")


def transfer_root_translation(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    target_motion: np.ndarray,
    scale: float,
) -> None:
    """Copy source root translation into the output target motion."""

    source_hierarchy = source_motion.hierarchy
    source_root_name = source_hierarchy.root_name
    target_root_name = target_hierarchy.root_name
    if not source_hierarchy.position_channel_indices(source_root_name):
        return

    for axis in "XYZ":
        try:
            target_index = target_hierarchy.channel_index(
                target_root_name, f"{axis}position"
            )
        except KeyError:
            continue
        try:
            source_index = source_hierarchy.channel_index(
                source_root_name, f"{axis}position"
            )
        except KeyError:
            target_motion[:, target_index] = 0.0
            continue

        source_values = source_motion.values[:, source_index]
        target_motion[:, target_index] = (
            source_values - source_values[0]
        ) * float(scale)


# TODO: Unify floor estimation in analysis and retargeting, with analysis as
# the source of truth. Keep the retargeting implementation for now.
def estimate_floor_level(
    motion: Motion,
    *,
    up_axis: str = "Y",
    use_rest_pose: bool = True,
    first_frame_only: bool = False,
) -> float:
    """Return the lowest selected joint coordinate along an up axis.

    Use the whole clip unless first_frame_only selects just its first frame.
    Fall back to all joints when no foot/toe/ankle/heel names exist. With use_rest_pose,
    also include a pose with zero rotations and the first frame's translations.
    Without animation frames, use zero channels for that rest pose.
    The motion is not modified; the returned value is a level, not a shift.
    """
    hierarchy = motion.hierarchy
    if not hierarchy.joints:
        raise ValueError("Floor estimation requires at least one joint.")
    if not isinstance(up_axis, str) or up_axis.upper() not in "XYZ":
        raise ValueError("up_axis must be one of 'X', 'Y', or 'Z'.")
    axis_index = "XYZ".index(up_axis.upper())

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

    return float(positions[:, floor_indices, axis_index].min())


def align_root_to_floor(
    motion: Motion,
    floor_height: float = 0.0,
    *,
    up_axis: str = "Y",
    use_rest_pose: bool = True,
    first_frame_only: bool = False,
) -> Motion:
    """Return a copy shifted along an up axis to the requested floor level.

    Estimate over all frames or just the first, optionally including rest pose
    as in estimate_floor_level. Shift the entire clip in either case; preserve
    frame count, timing and all other channels. ``floor_height`` is a coordinate
    along ``up_axis``. Requires nonempty motion.
    """
    if motion.values.shape[0] == 0:
        raise ValueError("Floor alignment requires at least one motion frame.")
    if not np.isfinite(floor_height):
        raise ValueError("floor_height must be finite.")
    if not isinstance(up_axis, str) or up_axis.upper() not in "XYZ":
        raise ValueError("up_axis must be one of 'X', 'Y', or 'Z'.")
    axis = up_axis.upper()
    try:
        root_channel = motion.hierarchy.channel_index(
            motion.hierarchy.root_name, f"{axis}position"
        )
    except KeyError as error:
        raise ValueError(
            f"Floor alignment requires a root {axis}position channel."
        ) from error
    level = estimate_floor_level(
        motion,
        up_axis=axis,
        use_rest_pose=use_rest_pose,
        first_frame_only=first_frame_only,
    )
    values = motion.values.copy()
    values[:, root_channel] += float(floor_height) - level
    return motion.with_values(values)
