"""Rotation transfer helpers for FK BVH retargeting."""

import numpy as np
from scipy.spatial.transform import Rotation

from bvh.interpolation.utils import wrap_degrees
from bvh.math_helpers import normalize_vectors
from bvh.ops.skeleton import compute_rest_joint_positions
from bvh.schemas import BVHDocument
from bvh.retargeting.mapping import invert_mapping
from bvh.retargeting.schemas import RotationChannels


def rotation_channels_by_name(document: BVHDocument) -> dict[str, RotationChannels]:
    """Return BVH rotation channel metadata keyed by joint name.

    The returned Euler order preserves BVH channel order exactly, for example
    ``XYZ`` or ``ZYX``. The letter case matters: our BVH FK implementation
    matches SciPy's uppercase convention, not lowercase.
    """

    channels = {}
    for joint in document.joints:
        rotation_offsets = [
            offset
            for offset, channel in enumerate(joint.channels)
            if channel.endswith("rotation")
        ]
        if len(rotation_offsets) != 3:
            continue

        indices = tuple(joint.channel_start + offset for offset in rotation_offsets)
        order = "".join(
            joint.channels[offset][0] for offset in rotation_offsets
        )
        channels[joint.name] = RotationChannels(
            indices=indices,
            order=order,
        )
    return channels


def transfer_fk_rotations(
    source_document: BVHDocument,
    target_document: BVHDocument,
    source_to_target: dict[str, str],
    target_motion: np.ndarray,
    rotation_correction: str,
) -> None:
    """Transfer source Euler rotations onto the target skeleton."""

    source_rotations = rotation_channels_by_name(source_document)
    target_rotations = rotation_channels_by_name(target_document)
    target_to_source = invert_mapping(source_to_target)
    source_rest_frames = estimate_rest_frames(source_document)
    target_rest_frames = estimate_rest_frames(target_document)

    for target_name, source_name in target_to_source.items():
        source_channels = source_rotations.get(source_name)
        target_channels = target_rotations.get(target_name)
        if source_channels is None or target_channels is None:
            continue

        source_angles = source_document.motion_values[:, list(source_channels.indices)]
        converted = convert_euler_degrees(
            angles=source_angles,
            source_order=source_channels.order,
            target_order=target_channels.order,
            source_rest_frame=source_rest_frames[
                source_document.joint_index[source_name]
            ],
            target_rest_frame=target_rest_frames[
                target_document.joint_index[target_name]
            ],
            rotation_correction=rotation_correction,
        )
        target_motion[:, list(target_channels.indices)] = converted


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


def unwrap_euler_degrees(angles: np.ndarray) -> np.ndarray:
    """Keep Euler channels continuous over time to avoid 360-degree jumps."""

    if angles.shape[0] <= 1:
        return angles.copy()
    radians = np.deg2rad(angles)
    return np.rad2deg(np.unwrap(radians, axis=0))


def estimate_rest_frames(document: BVHDocument) -> np.ndarray:
    """Estimate a stable rest-pose orientation frame for each joint."""

    positions = compute_rest_joint_positions(document.joints)
    children = _children_by_parent(document)
    frames = np.zeros((len(document.joints), 3, 3), dtype=np.float64)

    for index, joint in enumerate(document.joints):
        child_indices = children.get(index, ())
        if len(child_indices) == 1:
            direction = positions[child_indices[0]] - positions[index]
        elif joint.parent != -1:
            direction = positions[index] - positions[joint.parent]
        else:
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        frames[index] = _frame_from_direction(direction)

    return frames


def _children_by_parent(document: BVHDocument) -> dict[int, tuple[int, ...]]:
    children: dict[int, list[int]] = {}
    for index, joint in enumerate(document.joints):
        if joint.parent == -1:
            continue
        children.setdefault(joint.parent, []).append(index)
    return {parent: tuple(indices) for parent, indices in children.items()}


def _frame_from_direction(direction: np.ndarray) -> np.ndarray:
    y_axis = normalize_vectors(direction, fallback=np.array([0.0, 1.0, 0.0]))
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(reference @ y_axis)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = normalize_vectors(np.cross(reference, y_axis))
    z_axis = normalize_vectors(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)
