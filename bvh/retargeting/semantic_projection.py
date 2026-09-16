"""Reference-motion projection between semantically corresponding BVH chains."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from broom.bvh.kinematics import compute_global_transforms
from broom.bvh.ops.motion import with_motion_values
from broom.bvh.ops.skeleton import compute_rest_joint_positions
from broom.bvh.retargeting.root_motion import skeleton_scale, transfer_root_translation
from broom.bvh.retargeting.rotation_transfer import (
    matrices_to_euler_near_reference,
    transfer_fk_rotations,
    unwrap_euler_degrees,
)
from broom.bvh.schemas import BVHDocument

# Ordered from the root towards leaves so reconstructed local rotations have a
# reconstructed parent when a chain is processed.
_TEMPLATE_BRANCHES = (
    ("Hips", "Spine", "Spine1", "Spine2"),
    ("Spine2", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder", "RightArm", "RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg", "RightLeg", "RightFoot"),
    ("Spine2", "Neck", "Head"),
)


def semantic_skeleton_projection(
    source_document: BVHDocument,
    target_document: BVHDocument,
    joint_mapping: Mapping[str, tuple[str | None, str | None]],
    *,
    root_scale: float | None = None,
    rotation_correction: str = "none",
) -> BVHDocument:
    """Project source motion onto semantic chains of a target BVH skeleton.

    ``joint_mapping`` maps template-joint names to ``(source, target)`` names,
    for example ``{"Hips": ("mixamorig:Hips", "Hips")}``. Use ``None`` for
    a missing source or target anchor.

    The projection assumes the common BVH local XYZ convention. It transfers
    equal-length chains directly and redistributes accumulated global rotation
    over unequal chains by normalized rest-pose arc length.
    """

    if source_document.frame_count <= 0:
        raise ValueError("Semantic projection requires source motion frames.")
    if rotation_correction not in {"none", "rest_pose"}:
        raise ValueError("rotation_correction must be 'none' or 'rest_pose'.")
    joint_mapping = _resolve_mapping(source_document, target_document, joint_mapping)
    frame_count = source_document.frame_count
    motion = np.zeros((frame_count, target_document.total_channels), dtype=np.float64)
    scale = skeleton_scale(source_document, target_document) if root_scale is None else float(root_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("root_scale must be finite and positive.")
    transfer_root_translation(source_document, target_document, motion, scale)

    source_to_target = {
        source: target for source, target in joint_mapping.values()
        if source is not None and target is not None
    }
    if len(set(source_to_target.values())) != len(source_to_target):
        raise ValueError("Semantic anchors cannot map multiple source joints to one target joint.")
    # Reuse the existing direct transfer for anchors and equal paths.
    transfer_fk_rotations(
        source_document, target_document, source_to_target, motion, rotation_correction
    )

    source_positions, source_global = compute_global_transforms(source_document)
    del source_positions
    segments = _semantic_segments(source_document, target_document, joint_mapping)
    for source_path, target_path in segments:
        if _paths_are_direct(source_document, target_document, source_path, target_path):
            transfer_fk_rotations(
                source_document,
                target_document,
                {source_document.joints[s].name: target_document.joints[t].name
                 for s, t in zip(source_path, target_path)},
                motion,
                rotation_correction,
            )
            continue
        _resample_chain(source_document, target_document, motion, source_global, source_path, target_path)

    return with_motion_values(target_document, motion, frame_time=source_document.frame_time)


def _resolve_mapping(source, target, mapping):
    resolved = {}
    for template_name, pair in mapping.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Template joint {template_name!r} must map to (source_name, target_name).")
        source_name, target_name = pair
        if source_name is not None and source_name not in source.joint_index:
            raise ValueError(f"Unknown source template joint {source_name!r}.")
        if target_name is not None and target_name not in target.joint_index:
            raise ValueError(f"Unknown target joint {target_name!r} for {source_name!r}.")
        resolved[template_name] = (source_name, target_name)
    return resolved


def _semantic_segments(source, target, joint_mapping):
    segments = []
    seen = set()
    for branch in _TEMPLATE_BRANCHES:
        common = [name for name in branch if name in joint_mapping and all(joint_mapping[name])]
        for first, second in zip(common, common[1:]):
            source_start, target_start = joint_mapping[first]
            source_end, target_end = joint_mapping[second]
            source_path = _path(source, source_start, source_end)
            target_path = _path(target, target_start, target_end)
            key = (tuple(source_path), tuple(target_path))
            if key not in seen:
                segments.append(key)
                seen.add(key)
    return segments


def _path(document, start_name, end_name):
    start, end = document.joint_index[start_name], document.joint_index[end_name]
    path = [end]
    while path[-1] != start and document.joints[path[-1]].parent != -1:
        path.append(document.joints[path[-1]].parent)
    if path[-1] != start:
        raise ValueError(f"{end_name!r} is not a descendant of {start_name!r}.")
    return tuple(reversed(path))


def _paths_are_direct(source, target, source_path, target_path):
    return len(source_path) == len(target_path) and all(
        source.joints[s].channels == target.joints[t].channels
        for s, t in zip(source_path, target_path)
    )


def _resample_chain(source, target, motion, source_global, source_path, target_path):
    _require_three_axis_rotations(target, target_path[1:])
    source_s = _arc_coordinates(source, source_path)
    target_s = _arc_coordinates(target, target_path)
    source_start = source_path[0]
    relative = source_global[:, source_start].swapaxes(1, 2)[:, None] @ source_global[:, source_path]
    sampled = _sample_rotations(relative, source_s, target_s)

    # Re-evaluate target FK after prior chains: shared anchors supply the start
    # orientation and each reconstructed node immediately becomes its child's parent.
    _, target_global = compute_global_transforms(replace(target, motion_values=motion))
    desired = target_global[:, target_path[0], None] @ sampled
    desired_by_joint = {joint: desired[:, index] for index, joint in enumerate(target_path)}
    for joint in target_path[1:]:
        parent = target.joints[joint].parent
        parent_rotation = desired_by_joint.get(parent, target_global[:, parent])
        local = parent_rotation.swapaxes(1, 2) @ desired_by_joint[joint]
        _write_local_rotation(target, motion, joint, local)
        _, target_global = compute_global_transforms(replace(target, motion_values=motion))


def _arc_coordinates(document, path):
    positions = compute_rest_joint_positions(document.joints)
    lengths = np.linalg.norm(np.diff(positions[list(path)], axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        raise ValueError("Semantic chain has zero rest-pose length.")
    return np.r_[0.0, np.cumsum(lengths) / total]


def _sample_rotations(samples, coordinates, queries):
    result = np.empty((samples.shape[0], len(queries), 3, 3))
    for target_index, coordinate in enumerate(queries):
        right = min(int(np.searchsorted(coordinates, coordinate, side="right")), len(coordinates) - 1)
        left = max(right - 1, 0)
        if left == right:
            result[:, target_index] = samples[:, left]
            continue
        alpha = (coordinate - coordinates[left]) / (coordinates[right] - coordinates[left])
        delta = samples[:, left].swapaxes(1, 2) @ samples[:, right]
        rotvec = Rotation.from_matrix(delta).as_rotvec()
        result[:, target_index] = samples[:, left] @ Rotation.from_rotvec(alpha * rotvec).as_matrix()
    return result


def _require_three_axis_rotations(document, joints):
    for index in joints:
        channels = [channel for channel in document.joints[index].channels if channel.endswith("rotation")]
        if len(channels) != 3 or len({channel[0] for channel in channels}) != 3:
            raise ValueError(
                f"Semantic resampling requires three distinct rotation channels: "
                f"{document.joints[index].name!r}."
            )


def _write_local_rotation(document, motion, joint_index, matrices):
    joint = document.joints[joint_index]
    offsets = [offset for offset, channel in enumerate(joint.channels) if channel.endswith("rotation")]
    indices = np.asarray([joint.channel_start + offset for offset in offsets])
    order = "".join(joint.channels[offset][0] for offset in offsets)
    reference = np.deg2rad(motion[:, indices])
    angles = matrices_to_euler_near_reference(matrices, order, reference)
    motion[:, indices] = unwrap_euler_degrees(np.rad2deg(angles))
