"""World-space kinematics analysis over BVH documents."""

from collections.abc import Mapping, Sequence

import numpy as np

from broom.bvh.analysis.calculus import (
    derivative,
    drift_correct_to_reference,
    integrate,
)
from broom.bvh.analysis.filters import lowpass_butter
from broom.bvh.analysis.schemas import (
    ReconstructionResult,
    WorldKinematicsMeta,
    WorldKinematicsResult,
)
from broom.bvh.kinematics import compute_global_positions
from broom.bvh.schemas import BVHDocument


def compute_world_kinematics(
    document: BVHDocument,
    method: str = "gradient",
    *,
    sg_window: int = 9,
    sg_polyorder: int = 3,
    sg_mode: str = "interp",
    prefilter: Mapping[str, float] | None = None,
) -> WorldKinematicsResult:
    """Compute world-space position derivatives for a BVH document."""

    fps = _fps_from_document(document)
    dt = 1.0 / fps
    pos_w = compute_global_positions(document).astype(np.float64, copy=False)

    if prefilter is not None:
        filter_type = str(prefilter.get("type", "")).lower()
        if filter_type == "lowpass_butter":
            pos_w = lowpass_butter(
                pos_w,
                fps=fps,
                cutoff_hz=float(prefilter.get("cutoff_hz", 6.0)),
                order=int(prefilter.get("order", 2)),
            )

    method_normalized = str(method).lower()
    method_key = _method_key(
        method_normalized,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        sg_mode=sg_mode,
    )
    method_meta: dict[str, object] = {"method": method_normalized}
    if method_normalized == "savgol":
        method_meta.update(
            {
                "sg_window": int(sg_window),
                "sg_polyorder": int(sg_polyorder),
                "sg_mode": str(sg_mode),
            }
        )

    vel_w = derivative(
        pos_w,
        dt,
        order=1,
        method=method_normalized,
        window=sg_window,
        polyorder=sg_polyorder,
        mode=sg_mode,
    )
    acc_w = derivative(
        pos_w,
        dt,
        order=2,
        method=method_normalized,
        window=sg_window,
        polyorder=sg_polyorder,
        mode=sg_mode,
    )
    jerk_w = derivative(
        pos_w,
        dt,
        order=3,
        method=method_normalized,
        window=sg_window,
        polyorder=sg_polyorder,
        mode=sg_mode,
    )

    return WorldKinematicsResult(
        pos_w=pos_w,
        vel_w=vel_w,
        acc_w=acc_w,
        speed_w=np.linalg.norm(vel_w, axis=-1),
        accel_mag=np.linalg.norm(acc_w, axis=-1),
        jerk=jerk_w,
        jerk_mag=np.linalg.norm(jerk_w, axis=-1),
        vel_energy=np.sum(np.sum(vel_w**2, axis=-1), axis=0) * dt,
        acc_energy=np.sum(np.sum(acc_w**2, axis=-1), axis=0) * dt,
        jerk_energy=np.sum(np.sum(jerk_w**2, axis=-1), axis=0) * dt,
        meta=WorldKinematicsMeta(
            fps=fps,
            dt=dt,
            method=method_key,
            method_meta=method_meta,
            prefilter=None if prefilter is None else dict(prefilter),
        ),
    )


def compute_world_kinematics_many(
    document: BVHDocument,
    methods: Sequence[str] = ("gradient", "five_point", "savgol"),
    *,
    sg_window: int = 9,
    sg_polyorder: int = 3,
    sg_mode: str = "interp",
    prefilter: Mapping[str, float] | None = None,
) -> dict[str, WorldKinematicsResult]:
    """Compute multiple world-kinematics variants for one BVH document."""

    output: dict[str, WorldKinematicsResult] = {}
    for method in methods:
        result = compute_world_kinematics(
            document,
            method=method,
            sg_window=sg_window,
            sg_polyorder=sg_polyorder,
            sg_mode=sg_mode,
            prefilter=prefilter,
        )
        output[result.meta.method] = result
    return output


def reconstruct(
    kinematics: WorldKinematicsResult,
    route: str,
    *,
    method: str = "trapezoid",
    rect_variant: str = "left",
    anchor_pos0: bool = True,
    anchor_vel0: bool = True,
    drift_correct: bool = True,
    drift_strength: float = 1.0,
) -> ReconstructionResult:
    """Reconstruct signals from velocity or acceleration estimates."""

    dt = float(kinematics.meta.dt)
    pos_w = kinematics.pos_w
    vel_w = kinematics.vel_w
    acc_w = kinematics.acc_w

    if route == "acc_to_vel":
        v0 = vel_w[0] if anchor_vel0 else np.zeros_like(vel_w[0])
        vel_rec = integrate(
            acc_w,
            dt,
            method=method,
            axis=0,
            y0=v0,
            variant=rect_variant,
        )
        return ReconstructionResult(route=route, vel_rec=vel_rec)

    if route == "vel_to_pos":
        x0 = pos_w[0] if anchor_pos0 else np.zeros_like(pos_w[0])
        pos_rec = integrate(
            vel_w,
            dt,
            method=method,
            axis=0,
            y0=x0,
            variant=rect_variant,
        )
        if drift_correct:
            pos_rec = drift_correct_to_reference(
                pos_rec,
                pos_w,
                strength=drift_strength,
            )
        return ReconstructionResult(route=route, pos_rec=pos_rec)

    if route == "acc_to_pos":
        v0 = vel_w[0] if anchor_vel0 else np.zeros_like(vel_w[0])
        x0 = pos_w[0] if anchor_pos0 else np.zeros_like(pos_w[0])
        vel_rec = integrate(
            acc_w,
            dt,
            method=method,
            axis=0,
            y0=v0,
            variant=rect_variant,
        )
        pos_rec = integrate(
            vel_rec,
            dt,
            method=method,
            axis=0,
            y0=x0,
            variant=rect_variant,
        )
        if drift_correct:
            pos_rec = drift_correct_to_reference(
                pos_rec,
                pos_w,
                strength=drift_strength,
            )
        return ReconstructionResult(
            route=route,
            vel_rec=vel_rec,
            pos_rec=pos_rec,
        )

    raise ValueError("route must be 'acc_to_vel' | 'vel_to_pos' | 'acc_to_pos'.")


def _fps_from_document(document: BVHDocument) -> float:
    if document.frame_time is None or document.frame_time <= 0.0:
        raise ValueError("BVH document does not have a positive frame_time.")
    return 1.0 / float(document.frame_time)


def _method_key(
    method: str,
    *,
    sg_window: int,
    sg_polyorder: int,
    sg_mode: str,
) -> str:
    if method == "savgol":
        return (
            f"savgol(w={int(sg_window)},p={int(sg_polyorder)},"
            f"mode={str(sg_mode)})"
        )
    if method in ("five_point", "five-point", "fivepoint"):
        return "five_point"
    if method == "gradient":
        return "gradient"
    return method
