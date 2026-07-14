"""BVH document helpers and lightweight motion utilities."""

from bvh.channels import (
    joint_name_matches,
    position_channel_dimension,
    root_channel_indices,
    root_points,
    select_body_joints,
    weighted_body_points,
)
from bvh.io import (
    load_bvh_document,
    update_prefix_frame_count,
    validate_motion_values,
    write_bvh_with_channel_values,
    write_bvh_with_motion_values,
    write_bvh_with_root_channels,
)
from bvh.joint_limits import (
    DEFAULT_JOINT_LIMITS_PATH,
    JointLimit,
    load_joint_limits,
    resolve_joint_limit,
)
from bvh.kinematics import compute_global_positions
from bvh.schemas import BVHDocument, BVHJoint

__all__ = (
    "BVHDocument",
    "BVHJoint",
    "DEFAULT_JOINT_LIMITS_PATH",
    "JointLimit",
    "compute_global_positions",
    "joint_name_matches",
    "load_bvh_document",
    "load_joint_limits",
    "position_channel_dimension",
    "resolve_joint_limit",
    "root_channel_indices",
    "root_points",
    "select_body_joints",
    "update_prefix_frame_count",
    "validate_motion_values",
    "weighted_body_points",
    "write_bvh_with_channel_values",
    "write_bvh_with_motion_values",
    "write_bvh_with_root_channels",
)
