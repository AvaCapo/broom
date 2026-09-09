"""Diagnostics for evaluating retargeted BVH motion."""

from __future__ import annotations

import numpy as np

from broom.bvh.kinematics import compute_global_positions
from broom.bvh.schemas import BVHDocument


def mapped_joint_pairs(
    source_document: BVHDocument,
    target_document: BVHDocument,
    joint_map: dict[str, str],
) -> list[tuple[str, int, str, int]]:
    """Return mapped source/target joints with their hierarchy indices."""

    pairs: list[tuple[str, int, str, int]] = []
    for source_name, target_name in joint_map.items():
        source_index = source_document.joint_index.get(source_name)
        target_index = target_document.joint_index.get(target_name)
        if source_index is not None and target_index is not None:
            pairs.append((source_name, source_index, target_name, target_index))
    return pairs


def compute_mapped_position_error(
    source_document: BVHDocument,
    target_document: BVHDocument,
    joint_map: dict[str, str],
    *,
    root_relative: bool = True,
    scale: float = 1.0,
) -> dict[str, np.ndarray | float | tuple[str, ...]]:
    """Measure world-position error for joints resolved by ``joint_map``.

    ``scale`` is applied to source positions. With ``root_relative=True``,
    each clip is first expressed relative to its own root position.
    """

    pairs = mapped_joint_pairs(source_document, target_document, joint_map)
    if not pairs:
        raise ValueError("joint_map does not contain any resolvable joint pairs.")

    source_positions = compute_global_positions(source_document)
    target_positions = compute_global_positions(target_document)

    source_root_index = source_document.joint_index[source_document.root_name]
    target_root_index = target_document.joint_index[target_document.root_name]
    source_selected = source_positions[:, [pair[1] for pair in pairs], :]
    target_selected = target_positions[:, [pair[3] for pair in pairs], :]

    if root_relative:
        source_selected = (
            source_selected - source_positions[:, source_root_index, None, :]
        ) * float(scale)
        target_selected = target_selected - target_positions[:, target_root_index, None, :]
    else:
        source_selected = source_selected * float(scale)

    joint_error = np.linalg.norm(target_selected - source_selected, axis=-1)
    frame_error = joint_error.mean(axis=1)

    return {
        "joint_names": tuple(pair[2] for pair in pairs),
        "frame_error": frame_error,
        "joint_mean_error": joint_error.mean(axis=0),
        "mean_error": float(frame_error.mean()),
        "max_error": float(frame_error.max()),
    }
