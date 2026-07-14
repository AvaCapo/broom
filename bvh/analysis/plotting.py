"""Matplotlib plots for BVH analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from bvh.analysis.kinematics import (
    compute_world_kinematics,
)
from bvh.analysis.schemas import WorldKinematicsResult
from bvh.schemas import BVHDocument

_SIGNAL_KEY_MAP = {
    "position": "pos_w",
    "pos": "pos_w",
    "velocity": "vel_w",
    "vel": "vel_w",
    "acceleration": "acc_w",
    "acc": "acc_w",
    "jerk": "jerk",
}

_DIMS_IDX = {"x": 0, "y": 1, "z": 2}

SERIES: dict[str, tuple[str, str, str]] = {
    "pos.x": ("pos", "x", "pos.x"),
    "pos.y": ("pos", "y", "pos.y"),
    "pos.z": ("pos", "z", "pos.z"),
    "pos.mag": ("pos", "mag", "pos.mag"),
    "vel.x": ("vel", "x", "vel.x"),
    "vel.y": ("vel", "y", "vel.y"),
    "vel.z": ("vel", "z", "vel.z"),
    "vel.mag": ("vel", "mag", "vel.mag"),
    "acc.x": ("acc", "x", "acc.x"),
    "acc.y": ("acc", "y", "acc.y"),
    "acc.z": ("acc", "z", "acc.z"),
    "acc.mag": ("acc", "mag", "acc.mag"),
    "jerk.x": ("jerk", "x", "jerk.x"),
    "jerk.y": ("jerk", "y", "jerk.y"),
    "jerk.z": ("jerk", "z", "jerk.z"),
    "jerk.mag": ("jerk", "mag", "jerk.mag"),
}

PANEL_PRESETS: dict[str, tuple[str, ...]] = {
    "mag": ("pos.mag", "vel.mag", "acc.mag", "jerk.mag"),
    "pos": ("pos.x", "pos.y", "pos.z", "pos.mag"),
    "vel": ("vel.x", "vel.y", "vel.z", "vel.mag"),
    "acc": ("acc.x", "acc.y", "acc.z", "acc.mag"),
    "jerk": ("jerk.x", "jerk.y", "jerk.z", "jerk.mag"),
    "all_kinematics": (
        "pos.x", "pos.y", "pos.z", "pos.mag",
        "vel.x", "vel.y", "vel.z", "vel.mag",
        "acc.x", "acc.y", "acc.z", "acc.mag",
    ),
    "all": (
        "pos.x", "pos.y", "pos.z", "pos.mag",
        "vel.x", "vel.y", "vel.z", "vel.mag",
        "acc.x", "acc.y", "acc.z", "acc.mag",
        "jerk.x", "jerk.y", "jerk.z", "jerk.mag",
    ),
    "locomotion": ("pos.x", "pos.z", "vel.mag", "acc.mag"),
}


@dataclass
class PlotConfig:
    figsize: tuple[int, int] = (10, 4)
    grid_alpha: float = 0.3
    line_alpha: float = 0.9
    linewidth: float = 1.5
    title: str | None = None


def plot_joints_panel(
    document: BVHDocument,
    joint_indices: list[int] | tuple[int, ...],
    *,
    kinematics: WorldKinematicsResult | None = None,
    panels: tuple[str, ...] = ("pos.mag", "vel.mag", "acc.mag"),
    preset: str | None = None,
    t_range: tuple[int, int] | None = None,
    joint_labels: Optional[list[str] | tuple[str, ...]] = None,
    cfg: PlotConfig | None = None,
) -> plt.Figure:
    """Show multiple joints on multiple subplots."""

    cfg = PlotConfig(figsize=(10, 8)) if cfg is None else cfg
    kin = _require_kinematics(document, kinematics)
    panels_resolved = _resolve_panels(
        selected=panels,
        preset=preset,
        default=("pos.mag", "vel.mag", "acc.mag"),
    )
    sl = _resolve_time_range(document.frame_count, t_range)

    nrows = len(panels_resolved)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(cfg.figsize[0], 3 * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, series_key in zip(axes, panels_resolved):
        for joint_index in joint_indices:
            y, ylabel = _series_1d(kin, joint_index, sl, series_key)
            ax.plot(
                y,
                label=_joint_label(document, joint_index, joint_labels),
                alpha=cfg.line_alpha,
                lw=cfg.linewidth,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=cfg.grid_alpha)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame")
    if cfg.title:
        axes[0].set_title(cfg.title)
    fig.tight_layout()
    return fig


def plot_trajectory_planes(
    document: BVHDocument,
    joint_indices: list[int] | tuple[int, ...],
    *,
    kinematics: WorldKinematicsResult | None = None,
    signals: tuple[str, ...] = ("pos",),
    planes: tuple[tuple[str, str], ...] = (("x", "y"), ("x", "z"), ("y", "z")),
    t_range: tuple[int, int] | None = None,
    equal_aspect: bool = True,
    color_by_time: bool = False,
    joint_labels: Optional[list[str] | tuple[str, ...]] = None,
    cfg: PlotConfig | None = None,
) -> plt.Figure:
    """Plot trajectories of multiple joints projected to planes."""

    cfg = PlotConfig(figsize=(12, 9)) if cfg is None else cfg
    kin = _require_kinematics(document, kinematics)
    sl = _resolve_time_range(document.frame_count, t_range)

    fig, axes = plt.subplots(len(signals), len(planes), figsize=cfg.figsize)
    axes = np.asarray(axes, dtype=object).reshape(len(signals), len(planes))

    for row_index, signal in enumerate(signals):
        signal_arr = _trajectory_signal_array(kin, signal)
        signal_label = _trajectory_signal_label(signal)
        for col_index, (axis0, axis1) in enumerate(planes):
            ax = axes[row_index, col_index]
            dim0, dim1 = _DIMS_IDX[axis0], _DIMS_IDX[axis1]
            for joint_index in joint_indices:
                traj = signal_arr[sl, joint_index, :]
                if color_by_time:
                    idx = np.arange(traj.shape[0])
                    ax.scatter(traj[:, dim0], traj[:, dim1], c=idx, s=8, cmap="viridis")
                else:
                    ax.plot(
                        traj[:, dim0],
                        traj[:, dim1],
                        lw=cfg.linewidth,
                        alpha=cfg.line_alpha,
                        label=_joint_label(document, joint_index, joint_labels),
                    )

            ax.set_xlabel(axis0)
            ax.set_ylabel(axis1)
            ax.set_title(f"{signal_label}: {axis0}{axis1}")
            ax.grid(True, alpha=cfg.grid_alpha)
            if equal_aspect:
                ax.set_aspect("equal", adjustable="box")
            if not color_by_time:
                ax.legend(loc="upper right")

    if cfg.title:
        fig.suptitle(cfg.title)
    fig.tight_layout()
    return fig


def plot_joints_panel_multi(
    documents: list[BVHDocument] | tuple[BVHDocument, ...],
    joint_indices: list[int] | tuple[int, ...],
    *,
    kinematics_results: list[WorldKinematicsResult] | tuple[WorldKinematicsResult, ...] | None = None,
    clip_labels: Optional[list[str] | tuple[str, ...]] = None,
    panels: tuple[str, ...] = ("pos.mag", "vel.mag", "acc.mag"),
    preset: str | None = None,
    t_range: tuple[int, int] | None = None,
    joint_labels: Optional[list[str] | tuple[str, ...]] = None,
    cfg: PlotConfig | None = None,
) -> plt.Figure:
    """Show multiple clips and multiple joints on shared subplots."""

    if not documents:
        raise ValueError("documents must contain at least one clip.")

    cfg = PlotConfig(figsize=(10, 8)) if cfg is None else cfg
    panels_resolved = _resolve_panels(
        selected=panels,
        preset=preset,
        default=("pos.mag", "vel.mag", "acc.mag"),
    )
    labels = _normalize_clip_labels(len(documents), clip_labels)
    clip_colors, joint_linestyles = _style_maps(len(documents), joint_indices)
    kins = _require_kinematics_many(documents, kinematics_results)

    nrows = len(panels_resolved)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(cfg.figsize[0], 3 * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, series_key in zip(axes, panels_resolved):
        ylabel = SERIES[series_key][2]
        for clip_index, kin in enumerate(kins):
            sl = _resolve_time_range(kin.pos_w.shape[0], t_range)
            for joint_index in joint_indices:
                y, _ = _series_1d(kin, joint_index, sl, series_key)
                ax.plot(
                    y,
                    label=_curve_label(
                        labels[clip_index],
                        documents[clip_index],
                        joint_index,
                        joint_labels,
                        multiple_joints=len(joint_indices) > 1,
                    ),
                    alpha=cfg.line_alpha,
                    lw=cfg.linewidth,
                    color=clip_colors[clip_index],
                    linestyle=joint_linestyles[joint_index],
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=cfg.grid_alpha)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame")
    if cfg.title:
        axes[0].set_title(cfg.title)
    fig.tight_layout()
    return fig


def plot_trajectory_planes_multi(
    documents: list[BVHDocument] | tuple[BVHDocument, ...],
    joint_indices: list[int] | tuple[int, ...],
    *,
    kinematics_results: list[WorldKinematicsResult] | tuple[WorldKinematicsResult, ...] | None = None,
    clip_labels: Optional[list[str] | tuple[str, ...]] = None,
    signals: tuple[str, ...] = ("pos",),
    planes: tuple[tuple[str, str], ...] = (("x", "y"), ("x", "z"), ("y", "z")),
    t_range: tuple[int, int] | None = None,
    equal_aspect: bool = True,
    color_by_time: bool = False,
    joint_labels: Optional[list[str] | tuple[str, ...]] = None,
    cfg: PlotConfig | None = None,
) -> plt.Figure:
    """Plot projected trajectories for multiple clips and joints."""

    if not documents:
        raise ValueError("documents must contain at least one clip.")

    cfg = PlotConfig(figsize=(12, 9)) if cfg is None else cfg
    labels = _normalize_clip_labels(len(documents), clip_labels)
    clip_colors, joint_linestyles = _style_maps(len(documents), joint_indices)
    kins = _require_kinematics_many(documents, kinematics_results)

    fig, axes = plt.subplots(len(signals), len(planes), figsize=cfg.figsize)
    axes = np.asarray(axes, dtype=object).reshape(len(signals), len(planes))

    for row_index, signal in enumerate(signals):
        signal_label = _trajectory_signal_label(signal)
        for col_index, (axis0, axis1) in enumerate(planes):
            ax = axes[row_index, col_index]
            dim0, dim1 = _DIMS_IDX[axis0], _DIMS_IDX[axis1]
            for clip_index, kin in enumerate(kins):
                signal_arr = _trajectory_signal_array(kin, signal)
                sl = _resolve_time_range(signal_arr.shape[0], t_range)
                for joint_index in joint_indices:
                    traj = signal_arr[sl, joint_index, :]
                    label = _curve_label(
                        labels[clip_index],
                        documents[clip_index],
                        joint_index,
                        joint_labels,
                        multiple_joints=len(joint_indices) > 1,
                    )
                    if color_by_time:
                        idx = np.arange(traj.shape[0])
                        ax.scatter(
                            traj[:, dim0],
                            traj[:, dim1],
                            c=idx,
                            s=8,
                            cmap="viridis",
                            alpha=cfg.line_alpha,
                            label=label,
                        )
                    else:
                        ax.plot(
                            traj[:, dim0],
                            traj[:, dim1],
                            lw=cfg.linewidth,
                            alpha=cfg.line_alpha,
                            label=label,
                            color=clip_colors[clip_index],
                            linestyle=joint_linestyles[joint_index],
                        )

            ax.set_xlabel(axis0)
            ax.set_ylabel(axis1)
            ax.set_title(f"{signal_label}: {axis0}{axis1}")
            ax.grid(True, alpha=cfg.grid_alpha)
            if equal_aspect:
                ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper right")

    if cfg.title:
        fig.suptitle(cfg.title)
    fig.tight_layout()
    return fig


def plot_joint_debug(
    document: BVHDocument,
    joint_idx: int,
    *,
    kinematics: WorldKinematicsResult | None = None,
    series: tuple[str, ...] = ("pos.x", "pos.y", "pos.z"),
    preset: str | None = None,
    t_range: tuple[int, int] | None = None,
    cfg: PlotConfig | None = None,
) -> plt.Figure:
    """Plot one joint on one axis with multiple series overlaid."""

    cfg = PlotConfig() if cfg is None else cfg
    kin = _require_kinematics(document, kinematics)
    series_resolved = _resolve_panels(
        selected=series,
        preset=preset,
        default=("pos.x", "pos.y", "pos.z"),
    )
    sl = _resolve_time_range(document.frame_count, t_range)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ylabel = "Value"
    for series_key in series_resolved:
        y, series_ylabel = _series_1d(kin, joint_idx, sl, series_key)
        if ylabel == "Value":
            ylabel = series_ylabel
        ax.plot(
            y,
            label=series_key,
            alpha=cfg.line_alpha,
            lw=cfg.linewidth,
        )

    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=cfg.grid_alpha)
    ax.legend(loc="upper right")
    if cfg.title:
        ax.set_title(cfg.title)
    fig.tight_layout()
    return fig


def plot_joint_debug_multi(
    documents: list[BVHDocument] | tuple[BVHDocument, ...],
    joint_idx: int,
    *,
    kinematics_results: list[WorldKinematicsResult] | tuple[WorldKinematicsResult, ...] | None = None,
    clip_labels: Optional[list[str] | tuple[str, ...]] = None,
    series: tuple[str, ...] = ("pos.x", "pos.y", "pos.z"),
    preset: str | None = None,
    t_range: tuple[int, int] | None = None,
    cfg: PlotConfig | None = None,
) -> plt.Figure:
    """Plot one joint for multiple clips on one axis."""

    if not documents:
        raise ValueError("documents must contain at least one clip.")

    cfg = PlotConfig() if cfg is None else cfg
    series_resolved = _resolve_panels(
        selected=series,
        preset=preset,
        default=("pos.x", "pos.y", "pos.z"),
    )
    labels = _normalize_clip_labels(len(documents), clip_labels)
    kins = _require_kinematics_many(documents, kinematics_results)

    colors = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    if not colors:
        colors = list(plt.cm.tab10.colors)
    line_styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ylabel = "Value"
    for clip_index, kin in enumerate(kins):
        sl = _resolve_time_range(kin.pos_w.shape[0], t_range)
        for series_index, series_key in enumerate(series_resolved):
            y, series_ylabel = _series_1d(kin, joint_idx, sl, series_key)
            if ylabel == "Value":
                ylabel = series_ylabel
            ax.plot(
                y,
                label=f"{labels[clip_index]} / {series_key}",
                alpha=cfg.line_alpha,
                lw=cfg.linewidth,
                color=colors[clip_index % len(colors)],
                linestyle=line_styles[series_index % len(line_styles)],
            )

    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=cfg.grid_alpha)
    ax.legend(loc="upper right")
    if cfg.title:
        ax.set_title(cfg.title)
    fig.tight_layout()
    return fig


def _resolve_time_range(
    frame_count: int,
    t_range: tuple[int, int] | None,
) -> slice:
    if t_range is None:
        return slice(0, frame_count)
    if len(t_range) != 2:
        raise ValueError("t_range must be a (start, end) tuple.")
    start, end = t_range
    start_i = 0 if start is None else max(0, int(start))
    end_i = frame_count if end is None else min(frame_count, int(end))
    if end_i <= start_i:
        raise ValueError("t_range end must be greater than start.")
    return slice(start_i, end_i)


def _series_1d(
    kinematics: WorldKinematicsResult,
    joint_idx: int,
    sl: slice,
    series_key: str,
) -> tuple[np.ndarray, str]:
    signal, dim, ylabel = SERIES[series_key]
    values = _trajectory_signal_array(kinematics, signal)[sl, joint_idx, :]
    if dim == "mag":
        return np.linalg.norm(values, axis=-1), ylabel
    return values[:, _DIMS_IDX[dim]], ylabel


def _joint_label(
    document: BVHDocument,
    joint_idx: int,
    joint_labels: Optional[list[str] | tuple[str, ...]],
) -> str:
    if joint_labels is not None:
        return str(joint_labels[joint_idx])
    return document.joint_names[joint_idx]


def _resolve_panels(
    selected: tuple[str, ...],
    preset: str | None,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if preset is not None and tuple(selected) != tuple(default):
        raise ValueError("Use either preset=... or explicit panels/series.")
    if preset is not None:
        return PANEL_PRESETS[preset]
    return tuple(selected)


def _normalize_clip_labels(
    clip_count: int,
    clip_labels: Optional[list[str] | tuple[str, ...]],
) -> list[str]:
    if clip_labels is None:
        return [f"clip_{index}" for index in range(clip_count)]
    if len(clip_labels) != clip_count:
        raise ValueError("clip_labels length must match number of clips.")
    return [str(label) for label in clip_labels]


def _style_maps(
    clip_count: int,
    joint_indices: list[int] | tuple[int, ...],
) -> tuple[list[tuple[float, float, float, float]], dict[int, str]]:
    colors = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    if not colors:
        colors = list(plt.cm.tab10.colors)
    clip_colors = [colors[index % len(colors)] for index in range(clip_count)]
    linestyles = ["-", "--", "-.", ":"]
    joint_linestyles = {
        joint_index: linestyles[index % len(linestyles)]
        for index, joint_index in enumerate(joint_indices)
    }
    return clip_colors, joint_linestyles


def _curve_label(
    clip_label: str,
    document: BVHDocument,
    joint_idx: int,
    joint_labels: Optional[list[str] | tuple[str, ...]],
    multiple_joints: bool,
) -> str:
    joint_name = _joint_label(document, joint_idx, joint_labels)
    return f"{clip_label} / {joint_name}" if multiple_joints else clip_label


def _trajectory_signal_array(
    kinematics: WorldKinematicsResult,
    signal: str,
) -> np.ndarray:
    signal_key = _SIGNAL_KEY_MAP[str(signal).lower()]
    return getattr(kinematics, signal_key)


def _trajectory_signal_label(signal: str) -> str:
    signal_normalized = str(signal).lower()
    if signal_normalized in ("position", "pos"):
        return "pos"
    if signal_normalized in ("velocity", "vel"):
        return "vel"
    if signal_normalized in ("acceleration", "acc"):
        return "acc"
    return signal_normalized


def _require_kinematics(
    document: BVHDocument,
    kinematics: WorldKinematicsResult | None,
) -> WorldKinematicsResult:
    return (
        compute_world_kinematics(document)
        if kinematics is None
        else kinematics
    )


def _require_kinematics_many(
    documents: list[BVHDocument] | tuple[BVHDocument, ...],
    kinematics_results: list[WorldKinematicsResult] | tuple[WorldKinematicsResult, ...] | None,
) -> list[WorldKinematicsResult]:
    if kinematics_results is None:
        return [compute_world_kinematics(document) for document in documents]
    if len(kinematics_results) != len(documents):
        raise ValueError(
            "kinematics_results length must match number of documents."
        )
    return list(kinematics_results)
