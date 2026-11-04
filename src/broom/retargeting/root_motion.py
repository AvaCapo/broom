"""Root-motion transfer helpers for retargeting."""

from collections.abc import Mapping

import numpy as np

from broom import Hierarchy, Motion
from broom.kinematics import compute_global_positions, compute_global_transforms, compute_global_transforms_from_local


# TODO: Review name-based joint selection; these substrings are not universal
# skeleton semantics and can exclude relevant joints from floor estimation.
_FLOOR_JOINT_NAME_MARKERS = ("foot", "toe", "ankle", "heel")


def transfer_mapped_translations(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    source_to_target: Mapping[str, str],
    target_values: np.ndarray,
    *,
    scale: float,
    rotation_correction: str = "local_orientation",
    source_root_height: float | None = None,
    target_mapped_root_height: float | None = None,
    up_axis: str = "Y",
) -> None:
    """Write mapped animation displacements into a target sample array in place.

    Parameters
    ----------
    source_motion : Motion
        Source samples. Position channels encode ``d`` in ``offset + d``,
        expressed in parent axes, or skeleton axes for the root.
    target_hierarchy : Hierarchy
        Target channel layout and rest transforms in the same skeleton space.
    source_to_target : Mapping[str, str]
        Validated, injective mapping of source joint names to target names.
    target_values : numpy.ndarray
        Writable floating-point array of shape
        ``(source_motion.frame_count, target_hierarchy.total_channels)``.
        Only transferred position columns are written; other columns retain
        their existing values. Rotations must already be transferred: the
        mapped root recipient uses animated target parent FK. Do not alias
        the source sample array.
    scale : float
        Translation multiplier, validated as finite and positive by the caller.
    rotation_correction : {"local_orientation", "rest_pose", "none"}, default="local_orientation"
        For non-root source joints, ``local_orientation`` converts vectors between global
        parent rest bases before scaling. ``none`` and ``rest_pose`` copy
        matching XYZ position channels. The caller must validate the mode.
    source_root_height : float or None, default=None
        Source root standing height above zero ground, in source units.
        None uses its global rest coordinate along ``up_axis``.
    target_mapped_root_height : float or None, default=None
        Standing height of the source root's mapped target joint, in target
        units. None uses that joint's global rest coordinate. This joint need
        not be the target root. Heights must be finite; no floor is inferred.
    up_axis : {"X", "Y", "Z"}, default="Y"
        Vertical axis, case-insensitive. Other coordinates are only scaled.

    Returns
    -------
    None
        The supplied target array is modified in place.

    Raises
    ------
    KeyError
        If a mapped joint name does not exist in its hierarchy.

    Notes
    -----
    The mapped source root transfers its skeleton-space trajectory
    ``p = scale * (source_offset + d_source)`` plus the vertical correction
    ``target_height - scale * source_height``. No first sample is subtracted.
    Default heights assume standing rest poses with ground at zero; explicit
    heights support other baseline encodings without changing the inputs.
    For the mapped target recipient,
    ``d_target = R_parent(t).T @ (p - p_parent(t)) - target_offset``.
    Its parent's actual animated transforms are used, with identity/zero for
    a target root. No source-root mapping means no trajectory transfer.

    Other source joints use
    ``d_target = scale * B_target_parent.T @ B_source_parent @ d_source``
    in ``local_orientation`` mode. These bases are global rest rotations,
    with identity for a virtual root parent. Other modes copy matching axes.
    This non-root transfer does not compensate for animated parent differences.

    Missing source components are zero; a non-root source joint with no
    position channels leaves its target unchanged. A mapped source root
    without channels still contributes its static rest position. Missing
    target components are discarded, so motion may be lost. Target offsets
    are never modified. Source and target must already share coordinate axes.
    """

    if not isinstance(up_axis, str) or up_axis.upper() not in {"X", "Y", "Z"}:
        raise ValueError("up_axis must be one of 'X', 'Y', or 'Z'.")
    up_axis_index = "XYZ".index(up_axis.upper())
    for name, height in (("source_root_height", source_root_height),
                         ("target_mapped_root_height", target_mapped_root_height)):
        if height is not None and (not np.isscalar(height) or
                                   isinstance(height, (str, bool, np.bool_)) or
                                   not np.isreal(height) or not np.isfinite(height)):
            raise ValueError(f"{name} must be a finite number or None.")
    source_hierarchy = source_motion.hierarchy
    scale = float(scale)
    if rotation_correction == "local_orientation":
        _, source_rest = compute_global_transforms_from_local(source_hierarchy)
        _, target_rest = compute_global_transforms_from_local(target_hierarchy)
        # Parent index -1 selects the virtual skeleton-space parent's identity.
        source_parent_bases = np.concatenate((source_rest[0], np.eye(3)[None]))
        target_parent_bases = np.concatenate((target_rest[0], np.eye(3)[None]))

    for source_name, target_name in source_to_target.items():
        if source_name == source_hierarchy.root_name:
            continue  # Transfer the root trajectory after target parents are ready.
        source_index = source_hierarchy.joint_index(source_name)
        target_index = target_hierarchy.joint_index(target_name)
        source_joint = source_hierarchy.joints[source_index]
        target_joint = target_hierarchy.joints[target_index]
        source_columns = {
            channel[0]: source_hierarchy.channel_index(source_name, channel)
            for channel in source_joint.channels if channel.endswith("position")
        }
        if not source_columns:
            continue
        target_columns = {
            channel[0]: target_hierarchy.channel_index(target_name, channel)
            for channel in target_joint.channels if channel.endswith("position")
        }
        # TODO: Compensate for unrepresentable displacement with IK or another
        # retargeting algorithm; for now, missing target channels lose motion.
        if rotation_correction == "local_orientation":
            translations = np.zeros((source_motion.frame_count, 3), dtype=np.float64)
            for axis, column in source_columns.items():
                translations[:, "XYZ".index(axis)] = source_motion.values[:, column]
            correction = (
                target_parent_bases[target_joint.parent].T
                @ source_parent_bases[source_joint.parent]
            )
            translations = scale * (translations @ correction.T)
            for axis, column in target_columns.items():
                target_values[:, column] = translations[:, "XYZ".index(axis)]
        else:
            for axis, column in target_columns.items():
                if axis in source_columns:
                    target_values[:, column] = source_motion.values[:, source_columns[axis]] * scale

    target_name = source_to_target.get(source_hierarchy.root_name)
    if target_name is None:
        return
    target_joint = target_hierarchy.joints[target_hierarchy.joint_index(target_name)]
    target_columns = {
        channel[0]: target_hierarchy.channel_index(target_name, channel)
        for channel in target_joint.channels if channel.endswith("position")
    }
    if not target_columns:
        return
    translations = np.zeros((source_motion.frame_count, 3), dtype=np.float64)
    for channel in source_hierarchy.root_joint.channels:
        if channel.endswith("position"):
            column = source_hierarchy.channel_index(source_hierarchy.root_name, channel)
            translations[:, "XYZ".index(channel[0])] = source_motion.values[:, column]
    desired_positions = scale * (np.asarray(source_hierarchy.root_joint.offset) + translations)
    if source_root_height is None:
        source_root_height = source_hierarchy.root_joint.offset[up_axis_index]
    if target_mapped_root_height is None:
        rest_positions, _ = compute_global_transforms_from_local(target_hierarchy)
        target_mapped_root_height = rest_positions[0, target_hierarchy.joint_index(target_name), up_axis_index]
    desired_positions[:, up_axis_index] += float(target_mapped_root_height) - scale * float(source_root_height)
    if target_joint.parent == -1:
        local_positions = desired_positions
    else:
        positions, rotations = compute_global_transforms(
            Motion(target_hierarchy, target_values, source_motion.frame_time)
        )
        parent = target_joint.parent
        local_positions = np.einsum(
            "fji,fj->fi", rotations[:, parent], desired_positions - positions[:, parent]
        )
    displacement = local_positions - np.asarray(target_joint.offset)
    # TODO: Recover components unsupported by target channels via IK or another
    # algorithm. For now, write only available axes, allowing trajectory loss.
    for axis, column in target_columns.items():
        target_values[:, column] = displacement[:, "XYZ".index(axis)]


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
