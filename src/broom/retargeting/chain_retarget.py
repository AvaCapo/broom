"""Build target reference motion from source chain geometry and orientations.

This is a non-iterative initializer, not IK: target link lengths are preserved,
so endpoint positions and contacts need not match the sampled source positions.
Source and target must already use compatible world axes and orientation frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from broom import Hierarchy, Motion
from broom.kinematics import compute_global_transforms, compute_rest_joint_positions
from broom.retargeting.mapping import validate_joint_mapping
from broom.retargeting.retarget import retarget_mapped_motion
from broom.rotations.euler import matrices_to_euler_near_reference


# Explicit preset for Broom's generated skeleton; not an implicit chain policy.
DEFAULT_SOURCE_CHAINS = (
    ("Hips", "Spine2"),
    ("Spine2", "LeftHand"),
    ("Spine2", "RightHand"),
    ("Hips", "LeftFoot"),
    ("Hips", "RightFoot"),
    ("Spine2", "Head"),
)


def retarget_mapped_chains(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    joint_map: Mapping[str, str],
    *,
    chains: Sequence[tuple[str, str]],
    scale: float = 1.0,
    rotation_correction: str = "none",
) -> Motion:
    """Compose explicit mapped transfer and geometric chain refinement.

    ``chains`` lists source (start, end) pairs. Both endpoints need mappings;
    mapped intermediate joints automatically split the full hierarchy path. ``scale`` affects transferred translations, not the
    target rest geometry. No automatic mapping, scale estimation or floor shift
    is performed. Only ``rotation_correction='none'`` is supported: heuristic
    rest-frame corrections are not calibration for this geometric algorithm.

    See :func:`refine_mapped_chains` for geometric and rotation-DOF restrictions.
    """
    if rotation_correction != "none":
        raise ValueError("Chain projection requires rotation_correction='none' and compatible frames.")
    mapping = dict(joint_map)
    validate_joint_mapping(source_motion.hierarchy, target_hierarchy, mapping)
    # Validate editable source samples before the legacy transfer can sanitize them.
    compute_global_transforms(source_motion)
    for source_name, target_name in mapping.items():
        source_dof = len(source_motion.hierarchy.rotation_channel_indices(source_name))
        target_dof = len(target_hierarchy.rotation_channel_indices(target_name))
        if source_dof and (source_dof != 3 or target_dof != 3):
            raise ValueError("Mapped chain initialization supports three-axis rotation pairs only.")
    initial = retarget_mapped_motion(
        source_motion, target_hierarchy, mapping,
        scale=scale, rotation_correction="none",
    )
    return refine_mapped_chains(source_motion, initial, mapping, chains=chains)


def refine_mapped_chains(
    source_motion: Motion,
    initial_target_motion: Motion,
    joint_map: Mapping[str, str],
    *,
    chains: Sequence[tuple[str, str]],
) -> Motion:
    """Project source chain shape onto an independently owned target Motion.

    Source positions are sampled along the moving polyline at fixed normalized
    *rest-length* coordinates of target joints. Each target link is aimed along
    the corresponding sampled chord. Interpolated source global orientations
    supply a twist reference; the minimum swing aligns each outgoing link.
    Terminal orientations are copied from the source reference where the target
    terminal has three rotation channels. A channel-less terminal is left alone.

    Each chain is a source (start, end) pair. Mapped intermediate source joints
    become segment boundaries; their target pairs must lie in the same order
    on the target endpoint path. Unmapped intermediate joints remain part of
    the source geometry. No matches are guessed or added.

    Start joints can rotate. Their origins follow the existing parent motion;
    all translation channels, offsets, timing and rotations outside the selected
    paths are preserved. Consequently descendants outside the paths can move.
    Source and initial must have identical frame count and frame_time.

    Shared boundary joints are allowed. Multiple outgoing directions at a shared
    joint are fitted jointly with equal weights, so incompatible branch angles
    yield an approximation, independently of branch order. Overlapping interiors
    and conflicting mappings are rejected. Nonterminal joints need three distinct
    rotation axes. Zero rest links and collapsed sampled/target links are errors.

    No endpoint-position guarantee or temporal smoothing is provided. Shortest
    SO(3) interpolation does not preserve multi-turn winding. Exactly antiparallel
    swings use a deterministic axis from the orientation reference; continuity
    near this ambiguity cannot be guaranteed. Space-time may refine the output
    using constraints built from the original source motion.
    """
    if source_motion.frame_count != initial_target_motion.frame_count:
        raise ValueError("Source and target must have the same frame count.")
    if source_motion.frame_time != initial_target_motion.frame_time:
        raise ValueError("Source and target must have the same frame_time.")
    source = source_motion.hierarchy
    target = initial_target_motion.hierarchy
    mapping = dict(joint_map)
    validate_joint_mapping(source, target, mapping)
    segments = _chain_segments(source, target, mapping, chains)
    source_positions, source_rotations = compute_global_transforms(source_motion)
    target_positions, target_rotations = compute_global_transforms(initial_target_motion)
    output = initial_target_motion.with_values(initial_target_motion.values)
    # Constraints on each joint are accumulated before any rotations are written.
    priors: dict[int, np.ndarray] = {}
    directions: dict[int, list[tuple[int, np.ndarray]]] = {}
    source_rest = compute_rest_joint_positions(source)
    target_rest = compute_rest_joint_positions(target)
    for source_path, target_path in segments:
        source_u = _arc_coordinates(source_rest, source_path)
        target_u = _arc_coordinates(target_rest, target_path)
        left, right, alpha = _sample_indices(source_u, target_u)
        points = source_positions[:, source_path]
        sampled_points = (
            (1 - alpha)[None, :, None] * points[:, left]
            + alpha[None, :, None] * points[:, right]
        )
        sampled_rotations = _sample_rotations(
            source_rotations[:, source_path], left, right, alpha
        )
        chords = _unit(sampled_points[:, 1:] - sampled_points[:, :-1], "source sampled chord")
        for index, joint in enumerate(target_path):
            dof = len(target.rotation_channel_indices(target.joints[joint].name))
            if dof != 3 and (index < len(target_path) - 1 or dof != 0):
                raise ValueError(f"Chain joint {target.joints[joint].name!r} needs three rotation axes.")
            prior = sampled_rotations[:, index]
            if joint in priors and not np.allclose(priors[joint], prior, atol=1e-8, rtol=0):
                raise ValueError("Shared target anchor has conflicting source orientation references.")
            priors[joint] = prior
            if index < len(target_path) - 1:
                directions.setdefault(joint, []).append((target_path[index + 1], chords[:, index]))

    # Parent-before-child hierarchy order makes shared anchors deterministic.
    actual_global = np.empty_like(target_rotations)
    for joint_index, joint in enumerate(target.joints):
        parent_global = np.eye(3) if joint.parent == -1 else actual_global[:, joint.parent]
        initial_local = (
            target_rotations[:, joint_index] if joint.parent == -1 else
            target_rotations[:, joint.parent].swapaxes(-1, -2) @ target_rotations[:, joint_index]
        )
        if joint_index not in priors or not target.rotation_channel_indices(joint.name):
            actual_global[:, joint_index] = parent_global @ initial_local
            continue
        prior = priors[joint_index]
        outgoing = directions.get(joint_index, [])
        if outgoing:
            local_links, desired = [], []
            for child, direction in sorted(outgoing, key=lambda item: item[0]):
                # Includes animated child translation in the parent's frame.
                link = np.einsum(
                    "fji,fj->fi", target_rotations[:, joint_index],
                    target_positions[:, child] - target_positions[:, joint_index],
                )
                local_links.append(_unit(link, "target link"))
                desired.append(direction)
            global_rotation = _fit_directions(prior, np.stack(local_links, axis=1), np.stack(desired, axis=1))
        else:
            global_rotation = prior
        local = np.swapaxes(parent_global, -1, -2) @ global_rotation
        _write_local_rotation(target, output.values, joint_index, local)
        actual_global[:, joint_index] = global_rotation
    return output


def _chain_segments(source, target, mapping, chains):
    """Resolve endpoint pairs and split paths at existing intermediate matches."""
    if isinstance(chains, str):
        raise ValueError("chains must contain source (start, end) pairs.")
    segments = set()
    for chain_index, endpoints in enumerate(chains):
        if isinstance(endpoints, str):
            raise ValueError(f"Chain {chain_index} must be a (start, end) pair.")
        names = tuple(endpoints)
        if len(names) != 2 or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Chain {chain_index} needs exactly two string endpoint names.")
        start, end = names
        if start == end:
            raise ValueError(f"Chain {chain_index} endpoints must be distinct.")
        for name in names:
            if name not in mapping:
                raise ValueError(f"Source chain endpoint {name!r} is absent from joint_map.")
        source_path = _path(source, start, end)
        target_path = _path(target, mapping[start], mapping[end])
        target_order = {joint: index for index, joint in enumerate(target_path)}
        boundaries = []
        for source_offset, source_index in enumerate(source_path):
            source_name = source.joints[source_index].name
            if source_name not in mapping:
                continue
            target_index = target.joint_index(mapping[source_name])
            if target_index not in target_order:
                raise ValueError(
                    f"Mapped intermediate joint {source_name!r} -> {mapping[source_name]!r} "
                    f"is outside target chain {mapping[start]!r} -> {mapping[end]!r}."
                )
            target_offset = target_order[target_index]
            if boundaries and target_offset <= boundaries[-1][1]:
                raise ValueError(f"Mapped joints are out of order on chain {start!r} -> {end!r}.")
            boundaries.append((source_offset, target_offset))
        for (s0, t0), (s1, t1) in zip(boundaries, boundaries[1:]):
            segments.add((source_path[s0:s1 + 1], target_path[t0:t1 + 1]))
    segments = sorted(segments, key=lambda paths: (paths[1], paths[0]))
    for i, (s1, t1) in enumerate(segments):
        for s2, t2 in segments[i + 1:]:
            for first, second in ((s1, s2), (t1, t2)):
                shared = set(first) & set(second)
                boundaries = {first[0], first[-1]} & {second[0], second[-1]}
                if shared - boundaries or len(shared) > 1:
                    raise ValueError("Chains have conflicting overlapping paths; split them at shared mappings.")
    return tuple(segments)


def _path(hierarchy, start_name, end_name):
    start, end = hierarchy.joint_index(start_name), hierarchy.joint_index(end_name)
    path = [end]
    while path[-1] != start and hierarchy.joints[path[-1]].parent != -1:
        path.append(hierarchy.joints[path[-1]].parent)
    if path[-1] != start:
        raise ValueError(f"{end_name!r} is not a descendant of {start_name!r}.")
    return tuple(reversed(path))


def _unit(vectors, label):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-12) or not np.isfinite(lengths).all():
        raise ValueError(f"Degenerate {label}: cannot determine a direction.")
    return vectors / lengths


def _arc_coordinates(rest_positions, path):
    lengths = np.linalg.norm(np.diff(rest_positions[list(path)], axis=0), axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Chain rest links must have positive lengths.")
    return np.r_[0.0, np.cumsum(lengths) / lengths.sum()]


def _sample_indices(coordinates, queries):
    right = np.clip(np.searchsorted(coordinates, queries, side="right"), 1, len(coordinates) - 1)
    left = right - 1
    alpha = (queries - coordinates[left]) / (coordinates[right] - coordinates[left])
    return left, right, alpha


def _sample_rotations(samples, left, right, alpha):
    delta = samples[:, left].swapaxes(-1, -2) @ samples[:, right]
    rotvec = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(*delta.shape[:2], 3)
    interpolated = Rotation.from_rotvec((alpha[None, :, None] * rotvec).reshape(-1, 3)).as_matrix()
    return samples[:, left] @ interpolated.reshape(delta.shape)


def _fit_directions(prior, local_links, desired):
    current = np.einsum("fij,fkj->fki", prior, local_links)
    if current.shape[1] == 1:
        a, b = current[:, 0], desired[:, 0]
        cross = np.cross(a, b)
        sine = np.linalg.norm(cross, axis=1)
        cosine = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
        rotvec = np.zeros_like(a)
        regular = sine > 1e-10
        rotvec[regular] = cross[regular] / sine[regular, None] * np.arctan2(sine[regular], cosine[regular])[:, None]
        for frame in np.flatnonzero(~regular & (cosine < 0)):
            # The reference provides a deterministic axis in the ambiguous plane.
            axis_index = np.argmin(np.abs(prior[frame].T @ a[frame]))
            axis = np.cross(a[frame], prior[frame, :, axis_index])
            rotvec[frame] = np.pi * axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(rotvec).as_matrix() @ prior
    # Wahba fit of unit directions. Tiny identity preference resolves free twist
    # towards the prior for rank-deficient shared chains without random axes.
    covariance = np.einsum("fki,fkj->fij", desired, current) + 1e-10 * np.eye(3)
    u, _, vh = np.linalg.svd(covariance)
    sign = np.ones((len(prior), 3))
    sign[:, -1] = np.linalg.det(u @ vh)
    correction = (u * sign[:, None, :]) @ vh
    return correction @ prior


def _write_local_rotation(hierarchy, values, joint_index, matrices):
    joint = hierarchy.joints[joint_index]
    indices = list(hierarchy.rotation_channel_indices(joint.name))
    order = "".join(channel[0] for channel in joint.channels if channel.endswith("rotation"))
    orientation = Rotation.from_quat(np.asarray(joint.local_orientation)[[1, 2, 3, 0]]).as_matrix()
    channels = orientation.T @ matrices
    # Track an equivalent Euler branch over time, not just against each seed row.
    reference = np.deg2rad(values[0, indices])[None]
    for frame in range(len(values)):
        reference = matrices_to_euler_near_reference(channels[frame:frame + 1], order, reference)
        values[frame, indices] = np.rad2deg(reference[0])
