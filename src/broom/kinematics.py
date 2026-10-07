"""Forward kinematics for Broom motions and local transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from broom.hierarchy import Hierarchy
from broom.motion import Motion

_ROTATION_TOLERANCE = 1e-6
_AXIS_DIMENSIONS = {"X": 0, "Y": 1, "Z": 2}


def _validate_motion_values(motion: Motion) -> np.ndarray:
    """Return current Motion values after boundary validation."""
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")
    values = np.asarray(motion.values, dtype=np.float64)
    expected_shape = (motion.frame_count, motion.hierarchy.total_channels)
    if values.ndim != 2 or values.shape != expected_shape:
        raise ValueError("Motion values no longer match the Hierarchy layout.")
    if not np.isfinite(values).all():
        raise ValueError("Forward kinematics requires finite Motion values.")
    return values


def _decode_local_transforms(motion: Motion) -> tuple[np.ndarray, np.ndarray]:
    """Decode animation deltas without applying offsets or rest orientations."""
    values = _validate_motion_values(motion)
    hierarchy = motion.hierarchy
    frame_count = motion.frame_count
    joint_count = hierarchy.joint_count
    local_translations = np.empty((frame_count, joint_count, 3), dtype=np.float64)
    local_rotations = np.empty((frame_count, joint_count, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)

    for index, joint in enumerate(hierarchy.joints):
        translation = np.zeros((frame_count, 3), dtype=np.float64)
        axes: list[str] = []
        angles: list[np.ndarray] = []
        start = hierarchy.channel_start(index)
        for channel_offset, channel in enumerate(joint.channels):
            channel_values = values[:, start + channel_offset]
            if channel.endswith("position"):
                translation[:, _AXIS_DIMENSIONS[channel[0]]] += channel_values
            else:
                axes.append(channel[0])
                angles.append(channel_values)

        if axes:
            euler_angles = np.column_stack(angles)
            rotation_angles: np.ndarray = (
                euler_angles[:, 0] if len(axes) == 1 else euler_angles
            )
            channel_rotations = Rotation.from_euler(
                "".join(axes), rotation_angles, degrees=True
            ).as_matrix()
        else:
            channel_rotations = np.broadcast_to(identity, (frame_count, 3, 3)).copy()

        local_translations[:, index] = translation
        local_rotations[:, index] = channel_rotations

    return local_translations, local_rotations


def _validated_local_inputs(
    hierarchy: Hierarchy,
    local_rotations: Any,
    local_translations: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate batched local transform arrays for one hierarchy."""
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy.")
    rotations = (
        None if local_rotations is None
        else np.asarray(local_rotations, dtype=np.float64)
    )
    translations = (
        None if local_translations is None
        else np.asarray(local_translations, dtype=np.float64)
    )
    joint_count = hierarchy.joint_count
    if rotations is not None and (
        rotations.ndim != 4 or rotations.shape[1:] != (joint_count, 3, 3)
    ):
        raise ValueError(
            "local_rotations must have shape (frames, joints, 3, 3) for hierarchy."
        )
    if translations is not None and (
        translations.ndim != 3 or translations.shape[1:] != (joint_count, 3)
    ):
        raise ValueError(
            "local_translations must have shape (frames, joints, 3) for hierarchy."
        )
    frame_count = (
        rotations.shape[0] if rotations is not None
        else translations.shape[0] if translations is not None else 1
    )
    if frame_count == 0 or (
        translations is not None and translations.shape[0] != frame_count
    ):
        raise ValueError("Local transforms must have the same non-empty frame axis.")
    if rotations is None:
        rotations = np.broadcast_to(np.eye(3), (frame_count, joint_count, 3, 3))
    if translations is None:
        translations = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        raise ValueError("Local transforms must contain only finite values.")

    identity = np.eye(3, dtype=np.float64)
    gram = rotations.swapaxes(-1, -2) @ rotations
    if not np.allclose(gram, identity, rtol=0.0, atol=_ROTATION_TOLERANCE):
        raise ValueError("local_rotations must contain proper orthonormal matrices.")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, rtol=0.0, atol=_ROTATION_TOLERANCE):
        raise ValueError("local_rotations must contain proper rotations with determinant +1.")
    return rotations, translations


def compute_global_transforms_from_local(
    hierarchy: Hierarchy,
    local_rotations: np.ndarray | None = None,
    local_translations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply local animation deltas to a hierarchy's rest pose and compute FK.

    Parameters
    ----------
    hierarchy:
        Defines topology, joint order and rest pose. Each joint's offset and
        static ``local_orientation`` are applied exactly once.
    local_rotations:
        Finite animation rotation matrices of shape ``(F, J, 3, 3)``, relative
        to each joint's rest orientation: ``R_local = O @ local_rotations``.
        Do not include ``O`` in the input. None means identity animation
        rotations for all joints.
    local_translations:
        Finite additive displacements of shape ``(F, J, 3)`` in parent axes
        (skeleton axes for the root): ``t_local = offset + local_translations``.
        Do not include offsets in the input. Neither the joint's rest nor
        animation rotation rotates its own displacement. Units must match
        hierarchy offsets. None means zero displacement for all joints.

    Returns
    -------
    world_positions, world_rotations:
        Float64 arrays of shapes ``(F, J, 3)`` and ``(F, J, 3, 3)``, in
        ``hierarchy.joints`` order. Positions use the input length units;
        rotations map joint-local column vectors into skeleton/world axes.
        Results own their data and do not share storage with the inputs.

    Raises
    ------
    TypeError
        If ``hierarchy`` is not a Hierarchy.
    ValueError
        If array shapes or frame counts disagree, frames are empty, values
        are non-finite, or matrices are not proper rotations. Both
        ``R.T @ R == I`` and ``det(R) == 1`` are checked with absolute
        tolerance ``1e-6`` and zero relative tolerance. Invalid matrices are
        rejected rather than orthogonalized.

    Notes
    -----
    Supplied arrays must have the same ``F >= 1`` and
    ``J == hierarchy.joint_count``; a single pose still requires a frame axis.
    A missing array uses the other's frame count; both None return one frame
    of the rest pose. Identity rotations and zero displacements also recover
    the rest pose. Supplied arrays cover all joints; use identity/zero for
    unanimated joints. No broadcasting of supplied arrays is performed.
    The caller is responsible for joint correspondence and coordinate-frame
    semantics; matching shapes cannot detect transforms from another skeleton.

    Inputs are not modified. Channel declarations do not restrict the supplied
    deltas; this function neither decodes channels nor applies joint limits. End Sites are not additional joints in the results. For a
    child, ``p_world = p_parent + R_parent @ t_local`` and
    ``R_world = R_parent @ R_local``; root transforms are already in world axes.
    """
    rotations, translations = _validated_local_inputs(
        hierarchy, local_rotations, local_translations
    )
    frame_count = rotations.shape[0]
    joint_count = hierarchy.joint_count
    world_positions = np.empty((frame_count, joint_count, 3), dtype=np.float64)
    world_rotations = np.empty((frame_count, joint_count, 3, 3), dtype=np.float64)

    for index, joint in enumerate(hierarchy.joints):
        local_translation = np.asarray(joint.offset) + translations[:, index]
        w, x, y, z = joint.local_orientation
        rest_rotation = Rotation.from_quat((x, y, z, w)).as_matrix()
        local_rotation = rest_rotation @ rotations[:, index]
        if joint.parent == -1:
            world_positions[:, index] = local_translation
            world_rotations[:, index] = local_rotation
            continue
        parent_position = world_positions[:, joint.parent]
        parent_rotation = world_rotations[:, joint.parent]
        world_positions[:, index] = parent_position + np.einsum(
            "fij,fj->fi", parent_rotation, local_translation
        )
        world_rotations[:, index] = parent_rotation @ local_rotation

    return world_positions, world_rotations


def compute_rest_joint_positions(hierarchy: Hierarchy) -> np.ndarray:
    """Return world-space joint positions in the hierarchy rest pose."""
    positions, _ = compute_global_transforms_from_local(hierarchy)
    return positions[0]


def compute_global_transforms(motion: Motion) -> tuple[np.ndarray, np.ndarray]:
    """Decode a Motion and return world positions and rotation matrices.

    Parameters
    ----------
    motion:
        Motion whose current samples are decoded using its Hierarchy. Position
        channels form an additive displacement ``d`` in parent axes (skeleton
        axes for the root): ``t_local = offset + d``. Rotation channels are
        Euler angles in degrees, composed in declared intrinsic axis order.
        ``R_local = O @ R_channels``, where ``O`` is the joint's static
        ``local_orientation`` quaternion in wxyz order. Missing position
        channels contribute zero; no rotation channels means ``R_channels = I``.

    Returns
    -------
    world_positions, world_rotations:
        Independently owned float64 arrays of shapes ``(F, J, 3)`` and
        ``(F, J, 3, 3)``, in ``motion.hierarchy.joints`` order. Positions use
        the same length units as offsets and position channels. Rotations map
        joint-local column vectors into skeleton/world axes. End Sites are
        not additional joints in these arrays.

    Raises
    ------
    TypeError
        If ``motion`` is not a Motion.
    ValueError
        If current samples do not match the hierarchy layout, contain
        non-finite values, or decoded transforms fail validation by
        :func:`compute_global_transforms_from_local`.

    Notes
    -----
    Samples are validated on each call because Motion values are writable.
    The function does not mutate Motion, apply joint limits or cache results.
    Timing does not affect FK; each frame is evaluated independently.
    """
    local_translations, local_rotations = _decode_local_transforms(motion)
    return compute_global_transforms_from_local(
        motion.hierarchy, local_rotations, local_translations
    )


def compute_global_positions(motion: Motion) -> np.ndarray:
    """Decode a Motion and return world-space joint positions.

    This convenience wrapper has the same validation and coordinate
    conventions as :func:`compute_global_transforms`. The returned float64
    array has shape ``(F, J, 3)`` in ``motion.hierarchy.joints`` order.
    """
    world_positions, _ = compute_global_transforms(motion)
    return world_positions
