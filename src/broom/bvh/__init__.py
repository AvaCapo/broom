"""BVH document helpers and lightweight motion utilities."""

from broom.bvh.channels import (
    joint_name_matches,
    position_channel_dimension,
    root_channel_indices,
    root_points,
    select_body_joints,
    weighted_body_points,
)
from broom.bvh.joint_limits import (
    DEFAULT_JOINT_LIMITS_PATH,
    JointLimit,
    load_joint_limits,
    resolve_joint_limit,
)
from broom.bvh.kinematics import compute_global_positions, compute_global_transforms
from broom.bvh.schemas import BVHDocument, BVHJoint

__all__ = (
    "BVHDocument",
    "BVHJoint",
    "DEFAULT_JOINT_LIMITS_PATH",
    "JointLimit",
    "compute_global_positions",
    "compute_global_transforms",
    "joint_name_matches",
    "load_joint_limits",
    "position_channel_dimension",
    "resolve_joint_limit",
    "root_channel_indices",
    "root_points",
    "select_body_joints",
    "weighted_body_points",
)
