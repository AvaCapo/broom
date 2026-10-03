"""Rotation transfer helpers for FK retargeting."""

import numpy as np
from scipy.spatial.transform import Rotation

from broom import Hierarchy, Motion
from broom.rotations.euler import unwrap_euler_degrees, wrap_degrees
from broom.math_helpers import normalize_vectors
from broom.kinematics import compute_rest_joint_positions


def transfer_fk_rotations(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    source_to_target: dict[str, str],
    target_motion: np.ndarray,
    rotation_correction: str,
) -> None:
    """Transfer source Euler rotations onto the target skeleton."""

    source_hierarchy = source_motion.hierarchy
    target_to_source = {
        target_name: source_name
        for source_name, target_name in source_to_target.items()
    }
    source_rest_frames = estimate_rest_frames(source_hierarchy)
    target_rest_frames = estimate_rest_frames(target_hierarchy)

    for target_name, source_name in target_to_source.items():
        if (
            source_name not in source_hierarchy.joint_names
            or target_name not in target_hierarchy.joint_names
        ):
            continue

        source_index = source_hierarchy.joint_index(source_name)
        target_index = target_hierarchy.joint_index(target_name)
        source_indices = source_hierarchy.rotation_channel_indices(source_name)
        target_indices = target_hierarchy.rotation_channel_indices(target_name)
        if len(source_indices) != 3 or len(target_indices) != 3:
            continue

        source_order = "".join(
            channel[0]
            for channel in source_hierarchy.joints[source_index].channels
            if channel.endswith("rotation")
        )
        target_order = "".join(
            channel[0]
            for channel in target_hierarchy.joints[target_index].channels
            if channel.endswith("rotation")
        )

        source_angles = source_motion.values[:, list(source_indices)]
        converted = convert_euler_degrees(
            angles=source_angles,
            source_order=source_order,
            target_order=target_order,
            source_rest_frame=source_rest_frames[source_index],
            target_rest_frame=target_rest_frames[target_index],
            rotation_correction=rotation_correction,
        )
        target_motion[:, list(target_indices)] = converted

# TODO: Separate generic Euler representation conversion from retarget-specific
# rest-frame correction. Define correction through Joint.local_orientation and
# canonical FK/rest transforms instead of heuristic bone-direction frames.
# Reuse the existing kinematics and rotations APIs for rest-pose computation.
def convert_euler_degrees(
    angles: np.ndarray,
    source_order: str,
    target_order: str,
    source_rest_frame: np.ndarray | None = None,
    target_rest_frame: np.ndarray | None = None,
    rotation_correction: str = "none",
) -> np.ndarray:
    """Convert Euler channels between source and target BVH conventions.

    ``source_order`` and ``target_order`` must use the same uppercase axis
    convention as the BVH channel list, e.g. ``XYZ`` or ``ZYX``.
    """

    if source_order == target_order and rotation_correction == "none":
        return np.nan_to_num(angles, nan=0.0).copy()

    rotations = Rotation.from_euler(
        source_order,
        np.nan_to_num(angles, nan=0.0),
        degrees=True,
    )
    if (
        rotation_correction == "rest_pose"
        and source_rest_frame is not None
        and target_rest_frame is not None
    ):
        correction = target_rest_frame.T @ source_rest_frame
        matrices = rotations.as_matrix()
        matrices = np.einsum(
            "ij,fjk,lk->fil",
            correction,
            matrices,
            correction,
        )
        rotations = Rotation.from_matrix(matrices)

    converted = wrap_degrees(rotations.as_euler(target_order, degrees=True))
    return unwrap_euler_degrees(converted)


def estimate_rest_frames(hierarchy: Hierarchy) -> np.ndarray:
    """Estimate a stable rest-pose orientation frame for each joint."""

    positions = compute_rest_joint_positions(hierarchy)
    frames = np.zeros((hierarchy.joint_count, 3, 3), dtype=np.float64)

    for index, joint in enumerate(hierarchy.joints):
        child_indices = hierarchy.children(index)
        if len(child_indices) == 1:
            direction = positions[child_indices[0]] - positions[index]
        elif joint.parent != -1:
            direction = positions[index] - positions[joint.parent]
        else:
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        frames[index] = _frame_from_direction(direction)

    return frames

def _frame_from_direction(direction: np.ndarray) -> np.ndarray:
    y_axis = normalize_vectors(direction, fallback=np.array([0.0, 1.0, 0.0]))
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(reference @ y_axis)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = normalize_vectors(np.cross(reference, y_axis))
    z_axis = normalize_vectors(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)
