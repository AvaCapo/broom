"""Motion retargeting between different skeleton hierarchies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from broom import Hierarchy, Motion
from broom.io import load_bvh, write_bvh
from broom.ops.skeleton_geometry import estimate_skeleton_scale_ratio
from broom.retargeting.mapping import (
    load_joint_mapping,
    map_joints,
    validate_joint_mapping,
)
from broom.retargeting.root_motion import (
    align_root_to_floor,
    transfer_mapped_translations,
)
from broom.retargeting.rotation_transfer import transfer_fk_rotations
from broom.retargeting.schemas import RetargetResult


def retarget_mapped_motion(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    joint_map: Mapping[str, str],
    *,
    scale: float = 1.0,
    rotation_correction: str = "local_orientation",
    floor_height: float | None = None,
    up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
    source_root_height: float | None = None,
    target_mapped_root_height: float | None = None,
) -> Motion:
    """Transfer animation using an explicit source-to-target joint mapping.

    Parameters
    ----------
    source_motion : Motion
        Source animation. Source and target must use a common coordinate space.
    target_hierarchy : Hierarchy
        Target skeleton, including offsets, rest orientations and channel layout.
        Its geometry is retained; offsets are not scaled by this function.
    joint_map : Mapping[str, str]
        Source joint names mapped to target names. The mapping may be partial,
        but names must exist and target names must be unique. No additional
        correspondences are inferred.
    scale : float, default=1.0
        Finite positive multiplier for transferred translations.
        Source root placement is scaled too; it is not anchored to frame zero.
    rotation_correction : {"rest_pose", "local_orientation", "none"}, default="local_orientation"
        ``rest_pose`` estimates joint bases from rest-position directions.
        ``local_orientation`` uses global rest orientations computed by FK and
        also converts non-root source translations between parent rest bases.
        ``none`` converts Euler order without rest-basis correction.
        Rotations are transferred only when both joints have three rotation
        channels. The other groups are left at zero.
    floor_height : float or None, default=None
        If supplied, shift the whole output clip so its estimated floor reaches
        this coordinate along ``up_axis``, in target length units.
        This is constant alignment, not foot-contact optimization.
    up_axis : {"X", "Y", "Z"}, default="Y"
        Vertical axis for height transfer and optional floor alignment.
    floor_use_rest_pose : bool, default=False
        Include a zero-channel-rotation pose with first-frame translations in
        the floor estimate. Static joint orientations remain applied.
    floor_first_frame_only : bool, default=True
        Estimate the animated floor from the first frame instead of all frames.
        The resulting constant shift still applies to the entire clip.
    source_root_height : float or None, default=None
        Source root standing height above zero ground, in source units.
        None uses its global rest coordinate along ``up_axis``.
    target_mapped_root_height : float or None, default=None
        Standing height of the source root's mapped target joint, in target
        units. None uses that joint's global rest coordinate. This joint need
        not be the target root. Heights must be finite; no floor is inferred.

    Returns
    -------
    Motion
        New motion with the target hierarchy and source frame count/timing.
        Unmapped samples start at zero. Neither input is modified.

    Raises
    ------
    ValueError
        If the mapping, scale, correction mode, axis or height is invalid,
        or floor alignment cannot be performed.

    Notes
    -----
    Rest-basis correction transfers animation deltas, not the source rest pose:
    zero rotations remain zero. It does not compensate for omitted animated
    ancestors or solve different chain topologies.

    The mapped source root follows its scaled skeleton-space trajectory,
    with vertical correction ``target_height - scale * source_height``.
    The first sample is never subtracted; horizontal placement is scaled.
    Its target channels account for target offset and animated parent FK.
    No mapping means no root trajectory transfer. Root orientation does not
    rotate this skeleton-space trajectory.

    Non-root source translations use parent rest-basis conversion in
    ``local_orientation`` mode, or matching axes in the other modes.
    Missing target position channels discard components without an error;
    unsupported rotation groups are also skipped. This can lose motion.
    """

    validate_joint_mapping(
        source_motion.hierarchy, target_hierarchy, joint_map
    )
    try:
        scale = float(scale)
    except (TypeError, ValueError) as error:
        raise ValueError("scale must be a finite positive number.") from error
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a finite positive number.")

    if not isinstance(rotation_correction, str):
        raise ValueError(
            "rotation_correction must be 'rest_pose', 'local_orientation', or 'none'."
        )
    rotation_correction = rotation_correction.lower()
    if rotation_correction not in {"rest_pose", "local_orientation", "none"}:
        raise ValueError(
            "rotation_correction must be 'rest_pose', 'local_orientation', or 'none'."
        )

    target_values = np.zeros(
        (source_motion.frame_count, target_hierarchy.total_channels),
        dtype=np.float64,
    )
    source_to_target = dict(joint_map)
    transfer_fk_rotations(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        source_to_target=source_to_target,
        target_motion=target_values,
        rotation_correction=rotation_correction,
    )
    transfer_mapped_translations(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        source_to_target=source_to_target,
        target_values=target_values,
        scale=scale,
        rotation_correction=rotation_correction,
        up_axis=up_axis,
        source_root_height=source_root_height,
        target_mapped_root_height=target_mapped_root_height,
    )
    output_motion = Motion(
        target_hierarchy, target_values, source_motion.frame_time
    )
    if floor_height is not None:
        output_motion = align_root_to_floor(
            output_motion,
            floor_height=floor_height,
            up_axis=up_axis,
            use_rest_pose=floor_use_rest_pose,
            first_frame_only=floor_first_frame_only,
        )
    return output_motion


def retarget_motion(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    known_joint_map: Mapping[str, str] | None = None,
    *,
    min_score: float = 0.65,
    scale: float | None = None,
    auto_scale: bool = True,
    rotation_correction: str = "local_orientation",
    floor_align: bool = True,
    floor_height: float = 0.0,
    strict: bool = False,
    up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
    source_root_height: float | None = None,
    target_mapped_root_height: float | None = None,
) -> RetargetResult:
    """Suggest joint correspondences and retarget a motion onto a target skeleton.

    Parameters
    ----------
    source_motion : Motion
        Source animation in the same coordinate space as the target.
    target_hierarchy : Hierarchy
        Target skeleton and channel layout; its rest geometry is preserved.
    known_joint_map : Mapping[str, str] or None, default=None
        Explicit source-to-target pairs retained while the mapper attempts to
        match remaining joints. Use ``retarget_mapped_motion`` instead when
        only an approved mapping should be used.
    min_score : float, default=0.65
        Minimum score for accepting an automatically suggested joint pair.
    scale : float or None, default=None
        Explicit positive translation scale. Requires ``auto_scale=False``.
    auto_scale : bool, default=True
        Estimate the target/source rest-height ratio. Requires ``scale=None``.
        With this option disabled, an explicit scale is required.
    rotation_correction : {"rest_pose", "local_orientation", "none"}, default="local_orientation"
        ``rest_pose`` corrects rotations using bases estimated from joint
        positions. ``local_orientation`` uses FK-derived global rest bases
        and converts non-root source translations between parent rest bases.
        ``none`` transfers rotations with Euler-order conversion only.
        See ``retarget_mapped_motion`` for channel support and limitations.
    floor_align : bool, default=True
        Apply a constant vertical shift after transfer to align the estimated
        floor. This does not lock feet or optimize contacts.
    floor_height : float, default=0.0
        Desired floor coordinate in target length units; ignored when
        ``floor_align=False``.
    strict : bool, default=False
        Reject a mapping if any source or target joint remains unmatched.
        This checks mapping completeness, not kinematic equivalence.
    up_axis : {"X", "Y", "Z"}, default="Y"
        Vertical axis for height transfer and optional floor alignment.
    floor_use_rest_pose : bool, default=False
        Include a zero-channel-rotation pose with first-frame translations in
        the floor estimate, retaining static joint orientations.
    floor_first_frame_only : bool, default=True
        Estimate the animated floor from the first frame rather than all frames.
    source_root_height : float or None, default=None
        Source root standing height above zero ground, in source units.
        None uses its global rest coordinate along ``up_axis``.
    target_mapped_root_height : float or None, default=None
        Standing height of the source root's mapped target joint, in target
        units. None uses that joint's global rest coordinate. This joint need
        not be the target root. Heights must be finite; no floor is inferred.

    Returns
    -------
    RetargetResult
        New animation in ``motion``, the resolved ``joint_map``,
        ``unmapped_source_joints``, ``unmapped_target_joints``, and ``scale``.
        The animation retains source timing; neither input is modified.

    Raises
    ------
    ValueError
        If the scaling policy is inconsistent, strict mapping is incomplete,
        or mapping/transfer/height/floor-alignment validation fails.

    Notes
    -----
    Automatic matching is heuristic. The transfer has the same channel and
    rest-basis limitations as ``retarget_mapped_motion``; it does not perform
    chain refinement or space-time optimization.
    """

    if not isinstance(auto_scale, bool):
        raise ValueError("auto_scale must be a bool.")
    if auto_scale and scale is not None:
        raise ValueError("Pass either auto_scale=True or an explicit scale, not both.")
    if not auto_scale and scale is None:
        raise ValueError("scale is required when auto_scale is False.")

    mapping_result = map_joints(
        source_hierarchy=source_motion.hierarchy,
        target_hierarchy=target_hierarchy,
        known_joint_map=known_joint_map,
        min_score=min_score,
    )
    if strict and (
        mapping_result.unmapped_sources or mapping_result.unmapped_targets
    ):
        raise ValueError(
            "Retargeting mapping is incomplete. "
            "Unmapped source joints: "
            f"{', '.join(mapping_result.unmapped_sources) or 'none'}. "
            "Unmapped target joints: "
            f"{', '.join(mapping_result.unmapped_targets) or 'none'}."
        )

    resolved_scale = (
        estimate_skeleton_scale_ratio(
            source_hierarchy=source_motion.hierarchy,
            target_hierarchy=target_hierarchy,
        )
        if auto_scale
        else scale
    )
    output_motion = retarget_mapped_motion(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        joint_map=mapping_result.mapping,
        scale=resolved_scale,
        rotation_correction=rotation_correction,
        floor_height=floor_height if floor_align else None,
        up_axis=up_axis,
        floor_use_rest_pose=floor_use_rest_pose,
        floor_first_frame_only=floor_first_frame_only,
        source_root_height=source_root_height,
        target_mapped_root_height=target_mapped_root_height,
    )
    return RetargetResult(
        motion=output_motion,
        joint_map=mapping_result.mapping,
        unmapped_source_joints=mapping_result.unmapped_sources,
        unmapped_target_joints=mapping_result.unmapped_targets,
        scale=float(resolved_scale),
    )


def retarget_bvh_file(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    known_joint_map: Mapping[str, str] | None = None,
    mapping_path: str | Path | None = None,
    min_score: float = 0.65,
    scale: float | None = None,
    auto_scale: bool = True,
    rotation_correction: str = "local_orientation",
    floor_align: bool = True,
    floor_height: float = 0.0,
    strict: bool = False,
    precision: int = 6,
    *,
    up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
    source_root_height: float | None = None,
    target_mapped_root_height: float | None = None,
) -> RetargetResult:
    """Load BVH motions, retarget onto the target hierarchy, and write BVH.

    ``source_root_height`` and ``target_mapped_root_height`` override the
    standing baselines along ``up_axis`` without modifying either hierarchy.
    Parameters follow ``retarget_motion``; floor alignment runs afterwards.
    """

    if known_joint_map is not None and mapping_path is not None:
        raise ValueError("Pass either known_joint_map or mapping_path, not both.")

    source_motion = load_bvh(source_path)
    target_hierarchy = load_bvh(target_path).hierarchy
    mapping = load_joint_mapping(mapping_path) if mapping_path else known_joint_map

    result = retarget_motion(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        known_joint_map=mapping,
        min_score=min_score,
        scale=scale,
        auto_scale=auto_scale,
        rotation_correction=rotation_correction,
        floor_align=floor_align,
        floor_height=floor_height,
        strict=strict,
        up_axis=up_axis,
        floor_use_rest_pose=floor_use_rest_pose,
        floor_first_frame_only=floor_first_frame_only,
        source_root_height=source_root_height,
        target_mapped_root_height=target_mapped_root_height,
    )
    write_bvh(result.motion, output_path, precision=precision)
    return result
