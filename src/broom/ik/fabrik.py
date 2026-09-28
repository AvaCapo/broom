"""FABRIK inverse kinematics solver implemented with NumPy."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from broom import Hierarchy, Motion
from broom.channels import position_channel_dimension
from broom.kinematics import compute_global_transforms
from broom.rotations.euler import axis_rotation_matrices, wrap_degrees


def solve_fabrik(
    joint_positions: np.ndarray,
    target_position: np.ndarray,
    *,
    lengths: np.ndarray | None = None,
    max_iterations: int = 10,
    threshold: float = 1.0e-3,
    pole_target: np.ndarray | None = None,
    pole_weight: float = 1.0,
) -> np.ndarray:
    """Solve one joint chain with FABRIK and return solved positions.

    Parameters
    ----------
    joint_positions:
        Chain joint positions with shape ``(N, 3)``.
    target_position:
        Desired end-effector target position with shape ``(3,)``.
    lengths:
        Optional segment lengths with shape ``(N - 1,)``. If omitted, they are
        computed from ``joint_positions``.
    pole_target:
        Optional pole target used to stabilize the bend direction.
    """

    positions = np.asarray(joint_positions, dtype=np.float64).copy()
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("joint_positions must have shape (N, 3).")
    if positions.shape[0] < 2:
        raise ValueError("FABRIK requires at least two joints.")

    target = np.asarray(target_position, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError("target_position must have shape (3,).")

    if lengths is None:
        segment_vectors = positions[1:] - positions[:-1]
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    else:
        segment_lengths = np.asarray(lengths, dtype=np.float64)
        if segment_lengths.shape != (positions.shape[0] - 1,):
            raise ValueError("lengths must have shape (N - 1,).")

    root_position = positions[0].copy()
    total_length = float(segment_lengths.sum())
    root_to_target = target - root_position
    target_distance = float(np.linalg.norm(root_to_target))

    if target_distance >= total_length and target_distance > 1.0e-12:
        direction = root_to_target / target_distance
        solved = np.empty_like(positions)
        solved[0] = root_position
        for joint_index in range(1, positions.shape[0]):
            solved[joint_index] = (
                solved[joint_index - 1]
                + direction * segment_lengths[joint_index - 1]
            )
        return solved

    pole = None if pole_target is None else np.asarray(pole_target, dtype=np.float64)
    if pole is None and positions.shape[0] > 2:
        pole = _infer_pole_target(positions)
    if pole is not None and pole.shape != (3,):
        raise ValueError("pole_target must have shape (3,).")

    threshold_sq = float(threshold) * float(threshold)
    for _ in range(int(max_iterations)):
        positions[-1] = target
        for joint_index in range(positions.shape[0] - 2, -1, -1):
            delta = positions[joint_index] - positions[joint_index + 1]
            positions[joint_index] = positions[joint_index + 1] + (
                _normalize(delta) * segment_lengths[joint_index]
            )

        positions[0] = root_position
        for joint_index in range(1, positions.shape[0]):
            delta = positions[joint_index] - positions[joint_index - 1]
            positions[joint_index] = positions[joint_index - 1] + (
                _normalize(delta) * segment_lengths[joint_index - 1]
            )

        if pole is not None and pole_weight > 0.0:
            _apply_pole_constraint(
                positions=positions,
                target=target,
                pole=pole,
                weight=float(pole_weight),
            )

        distance_sq = float(np.sum((positions[-1] - target) ** 2))
        if distance_sq <= threshold_sq:
            break

    return positions


def solve_fabrik_chain(
    joint_positions: np.ndarray,
    chain_indices: np.ndarray | list[int] | tuple[int, ...],
    target_position: np.ndarray,
    *,
    max_iterations: int = 10,
    threshold: float = 1.0e-3,
    pole_target: np.ndarray | None = None,
    pole_weight: float = 1.0,
) -> np.ndarray:
    """Apply FABRIK to one chain inside a full joint-position array."""

    positions = np.asarray(joint_positions, dtype=np.float64).copy()
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("joint_positions must have shape (N, 3).")

    chain = np.asarray(chain_indices, dtype=np.int64)
    if chain.ndim != 1 or chain.shape[0] < 2:
        raise ValueError("chain_indices must be a 1D sequence with at least 2 items.")

    solved_chain = solve_fabrik(
        joint_positions=positions[chain],
        target_position=target_position,
        max_iterations=max_iterations,
        threshold=threshold,
        pole_target=pole_target,
        pole_weight=pole_weight,
    )
    positions[chain] = solved_chain
    return positions


def solve_fabrik_bvh_frame(
    motion: Motion,
    frame_index: int,
    *,
    target_position: np.ndarray,
    chain_joint_names: list[str] | tuple[str, ...] | None = None,
    chain_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
    max_iterations: int = 10,
    threshold: float = 1.0e-3,
    pole_target: np.ndarray | None = None,
    pole_weight: float = 1.0,
) -> np.ndarray:
    """Solve one Motion frame with FABRIK and return updated channel values."""

    # TODO: Decode and encode rotations relative to Joint.local_orientation.
    # FABRIK solves full local rotations; writing BVH channels must remove the
    # static rest orientation before matrix-to-Euler conversion.

    chain = _resolve_chain_indices(
        hierarchy=motion.hierarchy,
        chain_joint_names=chain_joint_names,
        chain_indices=chain_indices,
    )
    if chain.shape[0] < 2:
        raise ValueError("FABRIK chain must contain at least two joints.")

    frame = motion.frame(frame_index).copy()
    frame_motion = motion.with_values(frame[None, :])
    world_positions, world_rotations = compute_global_transforms(frame_motion)
    _, local_positions, local_rotations = (
        _compute_frame_rotations_and_local_positions(
            hierarchy=motion.hierarchy,
            frame_values=frame,
        )
    )
    world_positions = world_positions[0]
    world_rotations = world_rotations[0]

    solved_positions = solve_fabrik(
        joint_positions=world_positions[chain],
        target_position=np.asarray(target_position, dtype=np.float64),
        max_iterations=max_iterations,
        threshold=threshold,
        pole_target=(
            None
            if pole_target is None
            else np.asarray(pole_target, dtype=np.float64)
        ),
        pole_weight=pole_weight,
    )

    updated_world_positions = world_positions.copy()
    updated_world_rotations = world_rotations.copy()
    updated_world_positions[chain] = solved_positions

    for chain_offset in range(chain.shape[0] - 1):
        joint_index = int(chain[chain_offset])
        child_index = int(chain[chain_offset + 1])
        joint = motion.hierarchy.joints[joint_index]

        rotation_offsets = [
            offset
            for offset, channel in enumerate(joint.channels)
            if channel.endswith("rotation")
        ]
        if len(rotation_offsets) != 3:
            continue

        if joint.parent == -1:
            parent_world_rotation = np.eye(3, dtype=np.float64)
        else:
            parent_world_rotation = updated_world_rotations[joint.parent]

        desired_world = (
            updated_world_positions[child_index] - updated_world_positions[joint_index]
        )
        desired_local = parent_world_rotation.T @ desired_world
        base_local = local_positions[child_index]
        current_local_rotation = local_rotations[joint_index]
        current_local_direction = current_local_rotation @ base_local

        local_rotation = (
            _align_vectors(current_local_direction, desired_local)
            @ current_local_rotation
        )
        updated_world_rotations[joint_index] = parent_world_rotation @ local_rotation

        order = "".join(joint.channels[offset][0] for offset in rotation_offsets)
        angles = wrap_degrees(
            Rotation.from_matrix(local_rotation).as_euler(order, degrees=True)
        )
        for angle_offset, channel_offset in enumerate(rotation_offsets):
            frame[
                motion.hierarchy.channel_start(joint_index) + channel_offset
            ] = angles[angle_offset]

    return frame


def solve_fabrik_bvh_clip(
    motion: Motion,
    *,
    target_positions: np.ndarray,
    chain_joint_names: list[str] | tuple[str, ...] | None = None,
    chain_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
    max_iterations: int = 10,
    threshold: float = 1.0e-3,
    pole_targets: np.ndarray | None = None,
    pole_weight: float = 1.0,
) -> Motion:
    """Solve FABRIK for one chain across a whole Motion clip."""

    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")
    values = motion.values.copy()

    targets = np.asarray(target_positions, dtype=np.float64)
    if targets.shape != (motion.frame_count, 3):
        raise ValueError(
            "target_positions must have shape "
            f"({motion.frame_count}, 3), got {targets.shape}."
        )

    poles = _resolve_pole_targets(
        pole_targets=pole_targets,
        frame_count=motion.frame_count,
    )

    for frame_index in range(motion.frame_count):
        values[frame_index] = solve_fabrik_bvh_frame(
            motion=motion.with_values(values),
            frame_index=frame_index,
            target_position=targets[frame_index],
            chain_joint_names=chain_joint_names,
            chain_indices=chain_indices,
            max_iterations=max_iterations,
            threshold=threshold,
            pole_target=None if poles is None else poles[frame_index],
            pole_weight=pole_weight,
        )

    return motion.with_values(values)


def _apply_pole_constraint(
    *,
    positions: np.ndarray,
    target: np.ndarray,
    pole: np.ndarray,
    weight: float,
) -> None:
    root_position = positions[0]
    chain_axis = target - root_position
    chain_length = float(np.linalg.norm(chain_axis))
    if chain_length < 1.0e-6:
        return

    chain_direction = chain_axis / chain_length
    pole_relative = pole - root_position
    pole_projected = pole_relative - (
        np.dot(pole_relative, chain_direction) * chain_direction
    )
    pole_projected_norm = float(np.linalg.norm(pole_projected))
    if pole_projected_norm < 1.0e-6:
        return
    pole_projected = pole_projected / pole_projected_norm

    for joint_index in range(1, positions.shape[0] - 1):
        joint_relative = positions[joint_index] - root_position
        joint_projected = joint_relative - (
            np.dot(joint_relative, chain_direction) * chain_direction
        )
        joint_projected_norm = float(np.linalg.norm(joint_projected))
        if joint_projected_norm < 1.0e-6:
            continue

        joint_projected = joint_projected / joint_projected_norm
        angle = _signed_angle(joint_projected, pole_projected, chain_direction)
        rotated = _rotate_around_axis(
            vector=joint_relative,
            axis=chain_direction,
            angle_radians=angle * weight,
        )
        positions[joint_index] = root_position + rotated


def _compute_frame_rotations_and_local_positions(
    *,
    hierarchy: Hierarchy,
    frame_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_count = hierarchy.joint_count
    world_rotations = np.zeros((joint_count, 3, 3), dtype=np.float64)
    local_positions = np.zeros((joint_count, 3), dtype=np.float64)
    local_rotations = np.zeros((joint_count, 3, 3), dtype=np.float64)

    identity = np.eye(3, dtype=np.float64)
    for joint_index, joint in enumerate(hierarchy.joints):
        local_position = np.asarray(joint.offset, dtype=np.float64).copy()
        local_rotation = identity.copy()

        for channel_offset, channel in enumerate(joint.channels):
            value = float(
                frame_values[hierarchy.channel_start(joint_index) + channel_offset]
            )
            if channel.endswith("position"):
                local_position[position_channel_dimension(channel)] += value
            elif channel.endswith("rotation"):
                rotation_matrix = axis_rotation_matrices(
                    channel[0],
                    np.array([np.deg2rad(value)], dtype=np.float64),
                )[0]
                local_rotation = local_rotation @ rotation_matrix

        local_positions[joint_index] = local_position
        local_rotations[joint_index] = local_rotation
        if joint.parent == -1:
            world_rotations[joint_index] = local_rotation
        else:
            world_rotations[joint_index] = (
                world_rotations[joint.parent] @ local_rotation
            )

    return world_rotations, local_positions, local_rotations


def _resolve_chain_indices(
    *,
    hierarchy: Hierarchy,
    chain_joint_names: list[str] | tuple[str, ...] | None,
    chain_indices: np.ndarray | list[int] | tuple[int, ...] | None,
) -> np.ndarray:
    if chain_joint_names is None and chain_indices is None:
        raise ValueError("Pass either chain_joint_names or chain_indices.")
    if chain_joint_names is not None and chain_indices is not None:
        raise ValueError("Pass either chain_joint_names or chain_indices, not both.")

    if chain_joint_names is not None:
        return np.asarray(
            [hierarchy.joint_index(name) for name in chain_joint_names],
            dtype=np.int64,
        )

    chain = np.asarray(chain_indices, dtype=np.int64)
    if chain.ndim != 1:
        raise ValueError("chain_indices must be a 1D sequence.")
    return chain


def _resolve_pole_targets(
    *,
    pole_targets: np.ndarray | None,
    frame_count: int,
) -> np.ndarray | None:
    if pole_targets is None:
        return None

    poles = np.asarray(pole_targets, dtype=np.float64)
    if poles.shape == (3,):
        return np.repeat(poles[None, :], frame_count, axis=0)
    if poles.shape != (frame_count, 3):
        raise ValueError(
            "pole_targets must have shape (3,) or "
            f"({frame_count}, 3), got {poles.shape}."
        )
    return poles


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = _normalize(np.asarray(source, dtype=np.float64))
    target_norm = _normalize(np.asarray(target, dtype=np.float64))
    if np.linalg.norm(source_norm) < 1.0e-12 or np.linalg.norm(target_norm) < 1.0e-12:
        return np.eye(3, dtype=np.float64)

    cross = np.cross(source_norm, target_norm)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(source_norm, target_norm), -1.0, 1.0))

    if cross_norm < 1.0e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = _orthogonal_vector(source_norm)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()

    axis = cross / cross_norm
    angle = float(np.arctan2(cross_norm, dot))
    return Rotation.from_rotvec(angle * axis).as_matrix()


def _infer_pole_target(positions: np.ndarray) -> np.ndarray | None:
    if positions.shape[0] <= 2:
        return None
    middle_index = positions.shape[0] // 2
    return np.asarray(positions[middle_index], dtype=np.float64).copy()


def _orthogonal_vector(vector: np.ndarray) -> np.ndarray:
    if abs(vector[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        other = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _normalize(np.cross(vector, other))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


def _signed_angle(first: np.ndarray, second: np.ndarray, axis: np.ndarray) -> float:
    cross = np.cross(first, second)
    sin_value = float(np.dot(axis, cross))
    cos_value = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.arctan2(sin_value, cos_value))


def _rotate_around_axis(
    *,
    vector: np.ndarray,
    axis: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    cos_angle = float(np.cos(angle_radians))
    sin_angle = float(np.sin(angle_radians))
    axis_dot = float(np.dot(vector, axis))
    return (
        cos_angle * vector
        + sin_angle * np.cross(axis, vector)
        + (1.0 - cos_angle) * axis_dot * axis
    )
