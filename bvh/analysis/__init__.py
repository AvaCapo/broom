"""Analysis helpers built on top of BVH documents."""

from bvh.analysis.calculus import (
    derivative,
    drift_correct_to_reference,
    integrate,
    integrate_rect,
    integrate_simpson,
    integrate_trapezoid,
)
from bvh.analysis.filters import lowpass_butter
from bvh.analysis.kinematics import (
    compute_world_kinematics,
    compute_world_kinematics_many,
    reconstruct,
)
from bvh.analysis.plotting import (
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
from bvh.analysis.schemas import (
    ReconstructionResult,
    WorldKinematicsMeta,
    WorldKinematicsResult,
)
from bvh.analysis.ui import (
    MotionCompareUI,
    MotionPlotUI,
)

__all__ = (
    "compute_world_kinematics",
    "compute_world_kinematics_many",
    "derivative",
    "drift_correct_to_reference",
    "integrate",
    "integrate_rect",
    "integrate_simpson",
    "integrate_trapezoid",
    "lowpass_butter",
    "MotionCompareUI",
    "MotionPlotUI",
    "PANEL_PRESETS",
    "PlotConfig",
    "plot_joint_debug",
    "plot_joint_debug_multi",
    "plot_joints_panel",
    "plot_joints_panel_multi",
    "plot_trajectory_planes",
    "plot_trajectory_planes_multi",
    "reconstruct",
    "ReconstructionResult",
    "SERIES",
    "WorldKinematicsMeta",
    "WorldKinematicsResult",
)
