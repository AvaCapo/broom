"""Retarget BVH joint rotations to the combined SMPL-X pose layout."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from broom import Motion
from broom.channels import position_channel_dimension
from broom.kinematics import compute_global_transforms
from broom.ops.resample import resample_fps
from broom.rotations import quat


SMPLX_BODY_JOINT_COUNT = 22
SMPLX_POSE_SIZE = 165


def retarget_bvh_to_smplx(
    motion: Motion,
    source_joint_names: Sequence[str],
    smplx_to_source: Sequence[tuple[str, str, int]],
    *,
    source_global_rotation_offsets: np.ndarray | None = None,
    root_translation_scale: float = 1.0,
    coordinate_transform: np.ndarray | None = None,
    target_fps: float | None = None,
    betas: np.ndarray | None = None,
    hand_pose: np.ndarray | None = None,
    gender: str = "neutral",
) -> dict[str, np.ndarray]:
    """Retarget a Motion to a compact AMASS-style SMPL-X dictionary.

    ``source_joint_names`` defines the source order used by
    ``source_global_rotation_offsets``. ``smplx_to_source`` defines the target
    SMPL-X order as ``(target_name, source_name, target_parent_index)`` tuples.
    The mapping must contain the root followed by all 21 SMPL-X body joints.

    Rotation offsets re-express source global joint frames in the SMPL-X
    standard T-pose frame. Each matrix is applied as
    ``source_global @ offset.T``. Identity offsets may be omitted when the BVH
    already uses compatible standard joint frames.

    ``coordinate_transform`` is an optional proper 3D rotation applied in
    world space to the root orientation and translation. The returned keys are
    exactly ``poses``, ``trans``, ``gender``, ``mocap_framerate``, and
    ``betas``. ``poses`` follows the standard 165D SMPL-X order: root, body,
    jaw, eyes, left hand, right hand.
    """

    source_names = tuple(str(name) for name in source_joint_names)
    mapping = tuple(
        (str(target), str(source), int(parent))
        for target, source, parent in smplx_to_source
    )
    _validate_inputs(motion, source_names, mapping)
    if target_fps is not None and (
        not np.isfinite(float(target_fps)) or float(target_fps) <= 0.0
    ):
        raise ValueError("target_fps must be finite and positive")
    root_translation_scale = float(root_translation_scale)
    if not np.isfinite(root_translation_scale):
        raise ValueError("root_translation_scale must be finite")

    motion = motion if target_fps is None else resample_fps(motion, target_fps)
    _, source_global = compute_global_transforms(motion)
    source_indices = {
        name: motion.hierarchy.joint_index(name) for name in source_names
    }
    selected_global = source_global[:, [source_indices[name] for name in source_names]]

    if source_global_rotation_offsets is not None:
        offsets = np.asarray(source_global_rotation_offsets, dtype=np.float64)
        expected_shape = (len(source_names), 3, 3)
        if offsets.shape != expected_shape:
            raise ValueError(
                "source_global_rotation_offsets must have shape "
                f"{expected_shape}, got {offsets.shape}"
            )
        if not np.isfinite(offsets).all():
            raise ValueError("source_global_rotation_offsets contains NaN or infinite values")
        selected_global = selected_global @ np.swapaxes(offsets, -1, -2)

    source_order = {name: index for index, name in enumerate(source_names)}
    target_global = np.stack(
        [selected_global[:, source_order[source_name]] for _, source_name, _ in mapping],
        axis=1,
    )
    target_local = np.empty_like(target_global)
    for target_index, (_, _, parent_index) in enumerate(mapping):
        if parent_index < 0:
            target_local[:, target_index] = target_global[:, target_index]
        else:
            target_local[:, target_index] = (
                np.swapaxes(target_global[:, parent_index], -1, -2)
                @ target_global[:, target_index]
            )

    root_source_name = mapping[0][1]
    translations = _translation_channels_to_joint(
        motion,
        motion.hierarchy.joint_index(root_source_name),
    )
    translations *= root_translation_scale

    if coordinate_transform is not None:
        transform = np.asarray(coordinate_transform, dtype=np.float64)
        if transform.shape != (3, 3):
            raise ValueError(f"coordinate_transform must have shape (3, 3), got {transform.shape}")
        if not np.isfinite(transform).all():
            raise ValueError("coordinate_transform contains NaN or infinite values")
        if not np.allclose(transform.T @ transform, np.eye(3), atol=1.0e-6) or not np.isclose(
            np.linalg.det(transform), 1.0, atol=1.0e-6
        ):
            raise ValueError("coordinate_transform must be a proper rotation matrix")
        target_local[:, 0] = transform @ target_local[:, 0]
        translations = translations @ transform.T

    frame_count = motion.frame_count
    axis_angles = _matrices_to_axis_angles(target_local).reshape(frame_count, -1)
    jaw_and_eyes = np.zeros((frame_count, 9), dtype=np.float64)
    hands = _frame_pose(hand_pose, frame_count, 90, "hand_pose")
    poses = np.concatenate((axis_angles, jaw_and_eyes, hands), axis=1)
    if poses.shape != (frame_count, SMPLX_POSE_SIZE):
        raise AssertionError(f"Combined SMPL-X poses have unexpected shape {poses.shape}")

    shape = (
        np.zeros(16, dtype=np.float32)
        if betas is None
        else np.asarray(betas, dtype=np.float32).reshape(-1)
    )
    if shape.shape != (16,):
        raise ValueError(f"betas must contain 16 values, got shape {shape.shape}")
    if not np.isfinite(shape).all():
        raise ValueError("betas contains NaN or infinite values")

    fps = 1.0 / float(motion.frame_time)
    rounded_fps = round(fps)
    if abs(fps - rounded_fps) < 0.05:
        fps = float(rounded_fps)
    return {
        "poses": poses.astype(np.float64, copy=False),
        "trans": translations.astype(np.float32),
        "gender": np.asarray(str(gender)),
        "mocap_framerate": np.asarray(fps, dtype=np.float64),
        "betas": shape,
    }


def _validate_inputs(
    motion: Motion,
    source_joint_names: tuple[str, ...],
    mapping: tuple[tuple[str, str, int], ...],
) -> None:
    if len(source_joint_names) != len(set(source_joint_names)):
        raise ValueError("source_joint_names must not contain duplicates")
    missing = [
        name for name in source_joint_names if name not in motion.hierarchy.joint_names
    ]
    if missing:
        raise ValueError(f"BVH is missing configured source joints: {', '.join(missing)}")
    if len(mapping) != SMPLX_BODY_JOINT_COUNT:
        raise ValueError(
            f"smplx_to_source must define {SMPLX_BODY_JOINT_COUNT} body joints, got {len(mapping)}"
        )
    target_names = [target for target, _, _ in mapping]
    if len(target_names) != len(set(target_names)):
        raise ValueError("smplx_to_source target names must not contain duplicates")
    configured_sources = set(source_joint_names)
    unknown_sources = [source for _, source, _ in mapping if source not in configured_sources]
    if unknown_sources:
        raise ValueError(
            "smplx_to_source references joints absent from source_joint_names: "
            + ", ".join(unknown_sources)
        )
    if mapping[0][2] != -1:
        raise ValueError("The first smplx_to_source entry must be the root with parent -1")
    for index, (_, _, parent) in enumerate(mapping[1:], start=1):
        if parent < 0 or parent >= index:
            raise ValueError(
                f"SMPL-X joint {index} has invalid parent {parent}; parents must precede children"
            )
    if motion.frame_time <= 0.0:
        raise ValueError("Motion must have a positive frame_time")
    if motion.frame_count <= 0:
        raise ValueError("Motion must contain at least one motion frame")


def _translation_channels_to_joint(motion: Motion, joint_index: int) -> np.ndarray:
    hierarchy = motion.hierarchy
    translations = np.zeros((motion.frame_count, 3), dtype=np.float64)
    path = []
    while joint_index >= 0:
        path.append(joint_index)
        joint_index = hierarchy.joints[joint_index].parent

    for index in reversed(path):
        joint = hierarchy.joints[index]
        for channel_offset, channel in enumerate(joint.channels):
            if channel.endswith("position"):
                values = motion.values[
                    :, hierarchy.channel_start(index) + channel_offset
                ]
                translations[:, position_channel_dimension(channel)] += np.nan_to_num(values, nan=0.0)
    return translations


def _matrices_to_axis_angles(matrices: np.ndarray) -> np.ndarray:
    quaternions = quat.from_matrix(matrices)
    quaternions = np.where(quaternions[..., :1] < 0.0, -quaternions, quaternions)
    return quat.to_scaled_angle_axis(quaternions)


def _frame_pose(value: np.ndarray | None, frame_count: int, width: int, label: str) -> np.ndarray:
    if value is None:
        return np.zeros((frame_count, width), dtype=np.float64)
    pose = np.asarray(value, dtype=np.float64)
    if not np.isfinite(pose).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    if pose.shape == (width,):
        return np.broadcast_to(pose, (frame_count, width)).copy()
    if pose.shape == (frame_count, width):
        return pose.copy()
    raise ValueError(
        f"{label} must have shape ({width},) or ({frame_count}, {width}), got {pose.shape}"
    )
