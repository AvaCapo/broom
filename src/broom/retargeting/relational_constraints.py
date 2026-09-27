"""Precompute source-defined pair-distance constraints for space-time retargeting."""

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from broom import Hierarchy, Motion
from broom.kinematics import compute_global_positions, compute_rest_joint_positions

DEFAULT_RELATIONAL_JOINTS = ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head")


def build_relational_constraints(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    joint_mapping: Mapping[str, tuple[str | None, str | None]],
    *,
    joints: Sequence[str] = DEFAULT_RELATIONAL_JOINTS,
    tau_on: float = 0.15,
    tau_off: float = 0.30,
    weight: float = 1.0,
) -> list[dict[str, object]]:
    """Build source-normalized, target-space relational constraint dictionaries."""
    # TODO: Validate mapping names here too, so this public helper does not
    # rely on semantic_skeleton_projection having validated them first.
    if not 0 <= tau_on < tau_off or not np.isfinite((tau_on, tau_off, weight)).all() or weight < 0:
        raise ValueError("Require finite weight and 0 <= tau_on < tau_off.")
    endpoints = {
        name: joint_mapping[name] for name in joints
        if name in joint_mapping and all(joint_mapping[name])
    }
    source_hierarchy = source_motion.hierarchy
    source_positions = compute_global_positions(source_motion)
    constraints = []
    for first, second in combinations(endpoints, 2):
        source_name, target_name = endpoints[first]
        source_index = source_hierarchy.joint_index(source_name)
        target_index = target_hierarchy.joint_index(target_name)
        source_length = _path_length(
            source_hierarchy,
            source_index,
            source_hierarchy.joint_index(endpoints[second][0]),
        )
        target_length = _path_length(
            target_hierarchy,
            target_index,
            target_hierarchy.joint_index(endpoints[second][1]),
        )
        # TODO: Revisit path-length normalization for skeletons with different
        # proportions. A distance relative to each skeleton's rest pose may
        # better distinguish intentional approach from naturally long limbs.
        rho = np.linalg.norm(
            source_positions[:, source_index]
            - source_positions[
                :, source_hierarchy.joint_index(endpoints[second][0])
            ],
            axis=1,
        ) / source_length
        activation = _activation(rho, tau_on, tau_off)
        frames = np.flatnonzero(activation > 0)
        if frames.size:
            constraints.append({
                "type": "relational", "name": f"relation_{first}_{second}",
                "joint_a": target_name, "joint_b": endpoints[second][1], "frames": frames,
                "source_normalized_distance": rho[frames], "activation": activation[frames],
                "target_path_length": target_length, "weight": weight,
            })
    return constraints


def _path_length(hierarchy, first, second):
    rest = compute_rest_joint_positions(hierarchy)
    first_ancestors = {first}
    current = first
    while hierarchy.joints[current].parent != -1:
        current = hierarchy.joints[current].parent
        first_ancestors.add(current)
    length = 0.0
    current = second
    while current not in first_ancestors:
        parent = hierarchy.joints[current].parent
        length += float(np.linalg.norm(rest[current] - rest[parent]))
        current = parent
        if current == -1:
            raise ValueError("Joints do not share a hierarchy root.")
    lca = current
    current = first
    while current != lca:
        parent = hierarchy.joints[current].parent
        length += float(np.linalg.norm(rest[current] - rest[parent]))
        current = parent
    if length <= 1e-12:
        raise ValueError("Relational endpoints have zero kinematic path length.")
    return length


def _activation(rho, tau_on, tau_off):
    x = np.clip((rho - tau_on) / (tau_off - tau_on), 0.0, 1.0)
    smooth = 6 * x**5 - 15 * x**4 + 10 * x**3
    return np.where(rho <= tau_on, 1.0, np.where(rho >= tau_off, 0.0, 1.0 - smooth))
