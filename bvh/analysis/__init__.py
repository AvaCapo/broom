"""Analysis helpers built on top of BVH documents."""

from broom.bvh.analysis.calculus import (
    derivative,
    drift_correct_to_reference,
    integrate,
    integrate_rect,
    integrate_simpson,
    integrate_trapezoid,
)
from broom.bvh.analysis.filters import lowpass_butter
from broom.bvh.analysis.foot_contacts import (
    contact_segments,
    detect_foot_contacts,
    estimate_floor_height,
)
from broom.bvh.analysis.center_of_mass import (
    compute_center_velocity,
    compute_com_outside_support_distance,
    compute_com_support_offset,
    compute_margin_of_stability,
    compute_xcom,
    get_weighted_joint_center,
)
from broom.bvh.analysis.kinematics import (
    compute_world_kinematics,
    compute_world_kinematics_many,
    reconstruct,
)
from broom.bvh.analysis.plotting import (
    PANEL_PRESETS,
    SERIES,
    PlotConfig,
    plot_joint_debug,
    plot_joint_debug_multi,
    plot_joints_panel,
    plot_joints_panel_multi,
    plot_trajectory_planes,
    plot_trajectory_planes_multi,
)
from broom.bvh.analysis.schemas import (
    ReconstructionResult,
    WorldKinematicsMeta,
    WorldKinematicsResult,
)
from broom.bvh.analysis.root_motion import (
    PlanarTrendResult,
    ReconstructionErrorResult,
    decompose_joint_planar_motion,
    decompose_planar_trajectory,
    decompose_weighted_joint_center_motion,
    fit_planar_polynomial_trend,
    reconstruction_error,
    reconstruction_error_summary,
    remove_joint_planar_trend_from_clip,
    remove_weighted_joint_center_trend_from_clip,
)
from broom.bvh.analysis.support_polygon import SupportBounds, compute_support_bounds
from broom.bvh.analysis.ui import (
    MotionCompareUI,
    MotionPlotUI,
)

__all__ = (
    "compute_world_kinematics",
    "compute_world_kinematics_many",
    "compute_center_velocity",
    "compute_com_outside_support_distance",
    "compute_com_support_offset",
    "compute_margin_of_stability",
    "compute_support_bounds",
    "compute_xcom",
    "contact_segments",
    "decompose_joint_planar_motion",
    "decompose_planar_trajectory",
    "decompose_weighted_joint_center_motion",
    "detect_foot_contacts",
    "derivative",
    "drift_correct_to_reference",
    "estimate_floor_height",
    "fit_planar_polynomial_trend",
    "get_weighted_joint_center",
    "integrate",
    "integrate_rect",
    "integrate_simpson",
    "integrate_trapezoid",
    "lowpass_butter",
    "MotionCompareUI",
    "MotionPlotUI",
    "PANEL_PRESETS",
    "PlotConfig",
    "PlanarTrendResult",
    "plot_joint_debug",
    "plot_joint_debug_multi",
    "plot_joints_panel",
    "plot_joints_panel_multi",
    "plot_trajectory_planes",
    "plot_trajectory_planes_multi",
    "reconstruct",
    "ReconstructionResult",
    "ReconstructionErrorResult",
    "reconstruction_error",
    "reconstruction_error_summary",
    "remove_joint_planar_trend_from_clip",
    "remove_weighted_joint_center_trend_from_clip",
    "SERIES",
    "SupportBounds",
    "WorldKinematicsMeta",
    "WorldKinematicsResult",
)
