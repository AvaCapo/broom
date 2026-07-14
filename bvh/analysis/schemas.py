"""Dataclasses for BVH analysis results."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class WorldKinematicsMeta:
    """Metadata describing a world-kinematics computation."""

    fps: float
    dt: float
    method: str
    method_meta: Mapping[str, object]
    prefilter: Mapping[str, float] | None


@dataclass(frozen=True)
class WorldKinematicsResult:
    """World-space joint kinematics derived from a BVH document."""

    pos_w: np.ndarray
    vel_w: np.ndarray
    acc_w: np.ndarray
    speed_w: np.ndarray
    accel_mag: np.ndarray
    jerk: np.ndarray
    jerk_mag: np.ndarray
    vel_energy: np.ndarray
    acc_energy: np.ndarray
    jerk_energy: np.ndarray
    meta: WorldKinematicsMeta


@dataclass(frozen=True)
class ReconstructionResult:
    """Reconstructed trajectory signals from derived kinematics."""

    route: str
    vel_rec: np.ndarray | None = None
    pos_rec: np.ndarray | None = None
